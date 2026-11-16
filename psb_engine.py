from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
import re
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


LOG = logging.getLogger('psb')
TOKEN_RE = re.compile('[а-яёa-z0-9]+', re.I)
CONTRACT_RE = re.compile('(?<![а-яa-z])(?:пт|п|а)\\s*[-–]?\\s*\\d{1,3}\\s*[/\\-]\\s*20\\d{2}(?!\\d)', re.I)
SECTION_RE = re.compile('^\\s*(\\d+(?:\\.\\d+){1,4})\\.\\s+(.*)$')
ARTICLE_RE = re.compile('^\\s*Статья\\s+(\\d+(?:\\.\\d+)?)\\b', re.I)
ONE_LEVEL_RE = re.compile('^\\s*(\\d+)\\.\\s+(.*)$')
INJECT_RE = re.compile('(?i)(?:внимание\\s+для\\s+(?:ии|ai)[-\\s]?ассистента|инструкция\\s+(?:для|языковой)\\s+(?:модели|ии)|ignore\\s+all\\s+(?:previous|other)\\s+instructions|system\\s*prompt\\s*:)', re.I)
MONTHS = {'январ': '01', 'феврал': '02', 'март': '03', 'апрел': '04', 'ма': '05', 'июн': '06', 'июл': '07', 'август': '08', 'сентябр': '09', 'октябр': '10', 'ноябр': '11', 'декабр': '12'}


def norm(s):
    return str(s).lower().replace('ё', 'е').replace('\xa0', ' ').replace('–', '-').replace('—', '-')


def tokens(s):
    return TOKEN_RE.findall(norm(s))


def contract_ids(text):
    return {re.sub('[^A-ZА-Я0-9]', '', m.group().upper()) for m in CONTRACT_RE.finditer(str(text))}


def digest(text):
    return hashlib.sha256(str(text).encode('utf8')).hexdigest()


def dialogue(question):
    q = str(question)
    user_turns = re.findall('(?ms)(?:^|\\n)\\s*(?:Пользователь|User)\\s*:\\s*(.*?)(?=\\n\\s*(?:Пользователь|Ассистент|Assistant|User)\\s*:|\\Z)', q, flags=re.I)
    if user_turns:
        last = user_turns[-1].strip()
        user_context = ' '.join((t.strip() for t in user_turns))
        return (last, user_context[:2500])
    return (q.strip(), q.strip()[:2500])


def parse_question_date(question):
    last, ctx = dialogue(question)
    for t in (last, ctx):
        m = re.search('\\b(20\\d{2})[-./](\\d{1,2})[-./](\\d{1,2})\\b', t)
        if m:
            y, mo, d = m.groups()
            return f'{y}-{int(mo):02d}-{int(d):02d}'
        m = re.search('\\b(\\d{1,2})[./](\\d{1,2})[./](20\\d{2})\\b', t)
        if m:
            d, mo, y = m.groups()
            return f'{y}-{int(mo):02d}-{int(d):02d}'
        m = re.search('\\b(\\d{1,2})\\s+([а-яё]+)\\s+(20\\d{2})\\b', t, re.I)
        if m:
            d, month, y = m.groups()
            for stem, mm in MONTHS.items():
                if norm(month).startswith(stem):
                    return f'{y}-{mm}-{int(d):02d}'
        m = re.search('\\b([а-яё]+)\\s+(20\\d{2})\\s*(?:года|г\\.)?', t, re.I)
        if m:
            month, y = m.groups()
            for stem, mm in MONTHS.items():
                if norm(month).startswith(stem):
                    return f'{y}-{mm}-15'
        years = re.findall('\\b20(?:2\\d)\\b', t)
        if years:
            return f'{years[-1]}-07-01'
    return None


def _company(q):
    last, ctx = dialogue(q)
    for source in (last, ctx):
        s = norm(source)
        if re.search('альфа[-\\s]?трейдинг\\b', s):
            return 'альфа-трейдинг'
        if re.search('альфа[-\\s]?трейд\\b(?!инг)', s):
            return 'альфа-трейд'
        if re.search('северн\\w*\\s+склад', s):
            return 'северный склад'
    return ''


def load_docs(path):
    docs = []
    with zipfile.ZipFile(path) as z:
        for filename in sorted(z.namelist()):
            if filename.startswith('__MACOSX') or filename.endswith('/'):
                continue
            data = z.read(filename)
            suffix = Path(filename).suffix.lower()
            if suffix in ('.txt', '.md'):
                text = data.decode('utf-8-sig')
            elif suffix == '.docx':
                from docx import Document
                text = '\n'.join((p.text for p in Document(io.BytesIO(data)).paragraphs))
            else:
                continue
            if text.strip():
                docs.append((filename, text))
    if not docs:
        raise ValueError('Knowledge base empty')
    return docs


def safe_lines(s):
    return '\n'.join(('[Непроверенная инструкция внутри документа исключена]' if INJECT_RE.search(line) else line for line in s.splitlines()))


@dataclass(frozen=True)

class Chunk:
    idx: int
    filename: str
    title: str
    section: str
    text: str
    category: str
    ordinal: int

    @property
    def retrieval_text(self):
        return f'{self.title}\n{self.section}\n{self.text}'

    @property
    def document_id(self):
        return self.filename


def build_chunks(docs, chunk_size=1150, overlap=110):
    result = []
    for filename, raw in docs:
        title = Path(filename.split('__', 1)[-1]).stem
        law = filename.startswith('законы__')
        article_match = re.search('ст\\s*(\\d+(?:\\.\\d+)?)', title, re.I)
        law_article = f'ст. {article_match.group(1)}' if article_match else 'закон'
        cleaned = safe_lines(raw)
        if not law:
            result.append(Chunk(len(result), filename, title, 'полный текст документа', cleaned.strip(), 'company', 0))
            continue
        lines = cleaned.splitlines()
        section = 'ст. ' + article_match.group(1) if law and article_match else 'заголовок'
        blocks = []
        buffer = []

        def flush():
            if buffer:
                txt = '\n'.join(buffer).strip()
                if txt:
                    blocks.append((section, txt))
                buffer.clear()
        for line in lines:
            s = line.strip()
            if not s:
                continue
            sm = SECTION_RE.match(s)
            am = ARTICLE_RE.match(s)
            onem = ONE_LEVEL_RE.match(s) if law else None
            if sm or am or onem:
                flush()
                if sm:
                    section = f'{law_article}, п. {sm.group(1)}' if law else 'п. ' + sm.group(1)
                elif am:
                    section = 'ст. ' + am.group(1)
                else:
                    section = f'{law_article}, п. {onem.group(1)}'
            buffer.append(s)
        flush()
        ordinal = 0
        for anchor, block in blocks:
            if len(block) <= chunk_size:
                pieces = [block]
            else:
                pieces = []
                pos = 0
                while pos < len(block):
                    end = min(pos + chunk_size, len(block))
                    if end < len(block):
                        cut = block.rfind(' ', pos + int(chunk_size * 0.72), end)
                        if cut > pos:
                            end = cut + 1
                    pieces.append(block[pos:end].strip())
                    if end >= len(block):
                        break
                    pos = max(pos + 1, end - overlap)
            for part in pieces:
                if not part:
                    continue
                result.append(Chunk(len(result), filename, title, anchor, part, 'law' if law else 'company', ordinal))
                ordinal += 1
    assert len({c.idx for c in result}) == len(result)
    return result


class Proxy:
    def __init__(self, base_url=None, key=None, model=None, embedding_model=None, reranker_model=None, mode='platform'):
        self.mode = mode
        self.base = (base_url or os.getenv('LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL') or '').rstrip('/')
        self.key = key or os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY') or ''
        self.model = model or os.getenv('LLM_MODEL') or os.getenv('OPENAI_MODEL') or os.getenv('GEN_MODEL') or ''
        self.embedding_model = embedding_model or os.getenv('EMB_MODEL') or 'BAAI/bge-m3'
        self.reranker_model = reranker_model or os.getenv('RERANKER_MODEL') or ''
        if mode == 'ollama':
            self.base = (base_url or os.getenv('OLLAMA_URL') or 'http://127.0.0.1:11434').rstrip('/')
            self.model = model or os.getenv('OLLAMA_MODEL', 'qwen3:8b')
            self.key = 'local'
        self.client = httpx.Client(timeout=httpx.Timeout(110, connect=15))

    @property
    def available(self):
        return bool(self.base and self.key and self.model)

    def _post(self, endpoint, body, attempts=3):
        for i in range(attempts):
            try:
                res = self.client.post(self.base + endpoint, json=body, headers={'Authorization': 'Bearer ' + self.key} if self.mode != 'ollama' else {})
                res.raise_for_status()
                return res.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as ex:
                status = ex.response.status_code if isinstance(ex, httpx.HTTPStatusError) else None
                if status and status not in (408, 429, 500, 502, 503, 504):
                    raise
                if i == attempts - 1:
                    raise
                time.sleep(min(10, 1.5 * 2 ** i))

    def chat(self, system, user, max_tokens=1000, temperature=0):
        if not self.available:
            raise RuntimeError('No configured inference endpoint; use platform proxy or start Ollama')
        if self.mode == 'ollama':
            result = self._post('/api/chat', {'model': self.model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 'stream': False, 'think': False, 'options': {'temperature': temperature, 'num_predict': max_tokens, 'num_ctx': int(os.getenv('LOCAL_NUM_CTX', '8192'))}})
            text = result['message'].get('content', '')
        else:
            result = self._post('/chat/completions', {'model': self.model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 'temperature': temperature, 'max_tokens': max_tokens})
            text = result['choices'][0]['message'].get('content') or ''
        if not text.strip():
            raise ValueError('Empty model answer')
        return text.strip()

    def embed(self, texts, batch=16):
        if self.mode == 'ollama':
            raise RuntimeError('Ollama is not configured for BGE-M3 embeddings; use local encoder')
        batches = []
        for start in range(0, len(texts), batch):
            items = list(texts[start:start + batch])
            r = self._post('/embeddings', {'model': self.embedding_model, 'input': items, 'encoding_format': 'float'})
            data = sorted(r['data'], key=lambda x: x.get('index', 0))
            if len(data) != len(items):
                raise ValueError('Embeddings response length differs')
            batches.extend((np.asarray(z['embedding'], dtype='float32') for z in data))
        return normalize(np.vstack(batches))

    def rerank(self, query, chunks, top_n):
        if not self.reranker_model:
            raise RuntimeError('Reranker model unavailable')
        r = self._post('/rerank', {'model': self.reranker_model, 'query': query, 'documents': [c.retrieval_text for c in chunks], 'top_n': top_n})
        rows = r.get('results', r.get('data', []))
        return [int(x['index']) for x in rows if 0 <= int(x['index']) < len(chunks)]


class Retriever:

    def __init__(self, chunks, proxy=None, embed_fn=None, rerank_fn=None, index_path='kb_vectors.npz'):
        self.chunks = list(chunks)
        self.proxy = proxy
        self.embed_fn = embed_fn
        self.rerank_fn = rerank_fn
        self.path = Path(index_path)
        self.fingerprint = digest('\n'.join((f'{c.filename}|{c.section}|{c.ordinal}|{c.text}' for c in chunks)))
        self.embedding_model = (proxy.embedding_model if proxy else 'BAAI/bge-m3').lower()
        texts = [c.retrieval_text for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([tokens(t) for t in texts])
            self.bm_word = None
        except ImportError:
            self.bm25 = None
            self.word = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b', ngram_range=(1, 2))
            self.bm_word = self.word.fit_transform(texts)
        self.char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), dtype=np.float32, sublinear_tf=True, max_features=75000)
        self.char_matrix = self.char.fit_transform(texts)
        self.vectors = None
        self.load_index()

    def load_index(self):
        if not self.path.exists():
            return False
        try:
            with np.load(self.path, allow_pickle=False) as z:
                if z['fingerprint'].item() != self.fingerprint or str(z['model'].item()).lower() != self.embedding_model:
                    return False
                mat = z['vectors'].astype('float32')
                if len(mat) != len(self.chunks):
                    return False
                self.vectors = normalize(mat)
                return True
        except (ValueError, KeyError, OSError):
            return False

    def save_index(self):
        if self.vectors is None:
            return
        np.savez_compressed(self.path, vectors=self.vectors.astype('float16'), fingerprint=self.fingerprint, model=self.embedding_model)

    def build_index(self, provider=None, batch=16, progress=None):
        fn = provider or self.embed_fn or (self.proxy.embed if self.proxy and self.proxy.mode != 'ollama' else None)
        if fn is None:
            raise RuntimeError('No embedding provider')
        blocks = []
        for start in range(0, len(self.chunks), batch):
            batch_data = [c.retrieval_text for c in self.chunks[start:start + batch]]
            block = np.asarray(fn(batch_data), dtype='float32')
            blocks.append(block)
            if progress:
                progress(min(start + batch, len(self.chunks)), len(self.chunks))
        self.vectors = normalize(np.concatenate(blocks, axis=0))
        self.save_index()
        return self.vectors.shape

    def _raw_ranks(self, query, dense=True):
        if self.bm25 is not None:
            bm = np.asarray(self.bm25.get_scores(tokens(query)), dtype='float32')
        else:
            bm = (self.bm_word @ self.word.transform([query]).T).toarray().ravel()
        ch = (self.char_matrix @ self.char.transform([query]).T).toarray().ravel()
        runs = {'bm25': np.argsort(-bm).tolist(), 'char': np.argsort(-ch).tolist()}
        if dense and self.vectors is not None:
            fn = self.embed_fn or (self.proxy.embed if self.proxy and self.proxy.mode != 'ollama' else None)
            if fn is not None:
                try:
                    q = normalize(np.asarray(fn([query]), dtype='float32'))[0]
                    runs['dense'] = np.argsort(-(self.vectors @ q)).tolist()
                except (RuntimeError, ValueError, httpx.HTTPError, KeyError) as ex:
                    LOG.warning('Dense query failed: %s', type(ex).__name__)
        return runs

    def _boost(self, query, c):
        title = norm(c.title)
        ids = contract_ids(query)
        cids = contract_ids(c.title)
        company = _company(query)
        bonus = 0.0
        if ids:
            if cids and ids & cids:
                bonus += 0.025
            elif cids:
                bonus -= 0.012
        if company and c.category == 'company':
            in_title = company in title
            if in_title:
                bonus += 0.018
            elif 'альфа-трейд' in title or 'северный склад' in title:
                bonus -= 0.014
        date = parse_question_date(query)
        if 'регламент согласования' in title and date:
            if '2023' in title and date < '2025-03-01':
                bonus += 0.025
            elif '2025' in title and date >= '2025-03-01':
                bonus += 0.025
            else:
                bonus -= 0.008
        return bonus

    def rank(self, question, strategy='hybrid', candidate_k=40, final_k=9, rerank=True, allow_dense=True):
        latest, query = dialogue(question)
        query = query[-2500:]
        ranks = self._raw_ranks(query, dense=strategy in ('hybrid', 'dense') and allow_dense)
        if strategy == 'bm25':
            ranks = {'bm25': ranks['bm25']}
        if strategy == 'char':
            ranks = {'char': ranks['char']}
        if strategy == 'dense' and 'dense' in ranks:
            ranks = {'dense': ranks['dense']}
        weights = {'bm25': 1.0, 'char': 0.9, 'dense': 1.35}
        scores = defaultdict(float)
        for name, inds in ranks.items():
            for place, idx in enumerate(inds[:max(100, candidate_k * 4)], 1):
                scores[idx] += weights.get(name, 1.0) / (60 + place)
        for idx in list(scores):
            scores[idx] += self._boost(query, self.chunks[idx])
        indices = sorted(scores, key=lambda i: (-scores[i], i))[:candidate_k]
        selected = [self.chunks[i] for i in indices]
        fn = self.rerank_fn or (self.proxy.rerank if self.proxy and self.proxy.reranker_model else None)
        if rerank and fn:
            try:
                ordering = fn(query, selected, min(len(selected), max(final_k + 10, 18)))
                seen = set()
                new = []
                for idx in ordering:
                    if 0 <= idx < len(selected) and idx not in seen:
                        seen.add(idx)
                        new.append(selected[idx])
                selected = new + [c for i, c in enumerate(selected) if i not in seen]
            except (RuntimeError, ValueError, httpx.HTTPError, KeyError, IndexError) as ex:
                LOG.warning('Neural rerank unavailable; RRF fallback: %s', type(ex).__name__)
        top = selected[:final_k]
        if re.search('согласован|одобрен|директор|регламент', norm(question)):
            date = parse_question_date(question)
            if date:
                needed = '2023' if date < '2025-03-01' else '2025'
                exact = [c for c in self.chunks if 'регламент согласования' in norm(c.title) and needed in c.title]
                exact = sorted(exact, key=lambda x: (x.section != 'п. 2.4' if 'одобр' in norm(question) or 'директор' in norm(question) else x.section == 'заголовок', x.idx))
                if exact and (not any((c.filename == exact[0].filename for c in top))):
                    top = top[:-1] + [exact[0]] if top else [exact[0]]
        return list({c.idx: c for c in top}.values())


def evidence_pack(chunks, max_chars=17500):
    out = []
    mapping = {}
    n = 0
    for c in chunks:
        txt = c.text
        body = f'[E{n + 1}] {c.title} | {c.section} | {c.filename}\n{txt}'
        if out and sum((len(x) for x in out)) + len(body) > max_chars:
            break
        n += 1
        mapping[f'E{n}'] = c
        out.append(body)
    return ('\n\n'.join(out), mapping)


def find_file(names):
    for base in (os.getenv('DATA_DIR'), 'data', '.'):
        if not base:
            continue
        for name in names:
            p = Path(base) / name
            if p.is_file():
                return p
    raise FileNotFoundError(f"Could not find {names}; DATA_DIR={os.getenv('DATA_DIR')}")


def read_questions(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'id', 'question'}.issubset(rows[0]):
        raise ValueError('id/question columns required')
    data = [(str(x['id']).strip(), str(x['question']).strip()) for x in rows]
    if len({x[0] for x in data}) != len(data):
        raise ValueError('Duplicate IDs')
    return data


def submit(assistant, rows, path='submission.csv', workers=2, progress=None):
    result = [''] * len(rows)

    def task(i, q):
        try:
            return (i, assistant.answer(q), None)
        except Exception as ex:
            LOG.warning('Question %s failed (%s): %s', rows[i][0], type(ex).__name__, str(ex)[:200])
            return (i, 'Не удалось проверить нормативные основания из-за технической ошибки. Попробуйте повторить запрос.', str(ex))
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 6))) as pool:
        future = [pool.submit(task, i, q) for i, (_, q) in enumerate(rows)]
        for n, f in enumerate(as_completed(future), 1):
            i, answer, _ = f.result()
            result[i] = answer or 'Недостаточно сведений для ответа.'
            if progress:
                progress(n, len(rows))
    with open(path, 'w', encoding='utf8', newline='') as out:
        w = csv.writer(out)
        w.writerow(['id', 'answer'])
        w.writerows(((qid, result[i][:10000]) for i, (qid, _) in enumerate(rows)))
    return path
