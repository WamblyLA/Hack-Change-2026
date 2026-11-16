import re
from collections import defaultdict

from psb_engine import contract_ids, evidence_pack

ENTITY = re.compile(r'(?i)\b(?:ООО|АО|ПАО)\s+[«"]([^»"\n]{3,85})[»"]')
EDITION = re.compile(r"\s+ред(?:акция)?\.?\s+\d{4}\b", re.I)
DATE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]20\d{2}\b|(?<![/\-])\b20\d{2}\s+(?:году|года|г\.)\b")
COMPARE = re.compile(r"(?i)сравн|редакци|различи|отлича|изменил|вступил|утратил|порог")
APPROVAL = re.compile(r"(?i)регламент|согласован|одобрен|(?:финансов|генеральн)\w*\s+директор")
ROLE = re.compile(r"(?im)^\s*(Пользователь|User|Ассистент|Assistant)\s*:")
TRANSLATE = str.maketrans("ё–—‑", "е---")


def normalize(text):
    return text.casefold().translate(TRANSLATE)


def user_turns(question):
    turns = ROLE.split(question)
    return [text.strip() for role, text in zip(turns[1::2], turns[2::2])
            if role.casefold() in {"пользователь", "user"}] or [question]


class EvidenceBundles:
    def __init__(self, chunks):
        self.contracts = defaultdict(list)
        self.editions = defaultdict(list)
        aliases = defaultdict(set)
        for chunk in chunks:
            if chunk.category != "company":
                continue
            identifiers = contract_ids(chunk.title)
            for identifier in identifiers:
                self.contracts[identifier].append(chunk)
            if identifiers:
                for name in ENTITY.findall(chunk.text[:1400]):
                    aliases[normalize(name)].update(identifiers)
            edition = EDITION.search(chunk.title)
            if edition:
                self.editions[normalize(chunk.title[:edition.start()])].append(chunk)
        universe = set(self.contracts)
        self.aliases = [(re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)"), identifiers)
                        for name, identifiers in aliases.items()
                        if len(universe) == 1 or identifiers != universe]

    def match(self, turn):
        text = normalize(turn)
        explicit = contract_ids(text)
        for alias, identifiers in self.aliases:
            if alias.search(text):
                explicit.update(identifiers)
        return [chunk for identifier in sorted(explicit) for chunk in self.contracts.get(identifier, ())]

    def pack(self, question, ranked, budget=18000, extra=4500):
        initial, initial_map = evidence_pack(ranked, max_chars=budget)
        turns = user_turns(question)
        latest = turns[-1]
        required = []
        has_contract = False
        for turn in reversed(turns):
            required = self.match(turn)
            has_contract = bool(required or contract_ids(normalize(turn)))
            if has_contract:
                break
        if not has_contract and APPROVAL.search(latest) and (DATE.search(latest) or COMPARE.search(latest)):
            present = {chunk.idx for chunk in initial_map.values()}
            for family in self.editions.values():
                if any(chunk.idx in present for chunk in family):
                    required.extend(family)
        seen = {chunk.idx for chunk in initial_map.values()}
        for chunk in required:
            if chunk.idx in seen:
                continue
            label = f"E{len(initial_map) + 1}"
            entry = f"[{label}] {chunk.title} | {chunk.section} | {chunk.filename}\n{chunk.text}"
            if initial:
                entry = "\n\n" + entry
            if len(initial) + len(entry) <= budget + extra:
                initial += entry
                initial_map[label] = chunk
                seen.add(chunk.idx)
        return initial, initial_map
