"use strict";

const form = document.querySelector("#chat");
const input = document.querySelector("#question");
const send = document.querySelector("#send");
const reset = document.querySelector("#reset");
const log = document.querySelector("#messages");
const intro = document.querySelector("#intro");
const statusLine = document.querySelector("#status");
const errorLine = document.querySelector("#error");
const storageKey = "psb-legal-dialog-v1";
let turns = [];
let busy = false;
let readiness = "Проверяем подключение…";

try {
  const saved = JSON.parse(sessionStorage.getItem(storageKey) || "{}");
  if (Array.isArray(saved.turns)) turns = saved.turns;
  if (typeof saved.draft === "string") input.value = saved.draft;
} catch {
  // The dialog still works when browser storage is unavailable.
}

function persist() {
  try {
    sessionStorage.setItem(storageKey, JSON.stringify({ turns, draft: input.value }));
  } catch {
    try { sessionStorage.removeItem(storageKey); } catch { /* Storage is unavailable. */ }
  }
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function addMessage(role, text) {
  const article = node("article", undefined, `message ${role}`);
  article.append(node("h3", role === "user" ? "Вы" : "Помощник"), node("p", text, "answer"));
  log.append(article);
  intro.hidden = true;
  return article;
}

function addAnswer(turn) {
  const answer = addMessage("assistant", turn.answer);
  const panel = node("div", undefined, "sources");
  if (!turn.sources.length) {
    panel.append(node("p", "В ответе нет подтверждённых ссылок на фрагменты базы.", "no-sources"));
  } else {
    panel.append(node("h4", "Источники"));
    for (const source of turn.sources) {
      const card = node("div", undefined, "source");
      const title = node("p", source.title, "source-title");
      title.append(node("span", ` · ${source.section}`));
      const disclosure = node("details");
      disclosure.append(
        node("summary", "Читать текст документа"),
        node("div", source.text, "source-text"),
      );
      card.append(title, node("p", source.fragment, "source-fragment"), disclosure);
      panel.append(card);
    }
  }
  answer.append(panel);
  return answer;
}

for (const turn of turns) {
  addMessage("user", turn.question);
  addAnswer(turn);
}

function update() {
  send.disabled = busy;
  reset.disabled = busy || (!turns.length && !input.value.trim());
  input.readOnly = busy;
  statusLine.textContent = busy ? "Ищем источники и ждём ответ модели…" : readiness;
  form.setAttribute("aria-busy", String(busy));
}

function showError(text = "") {
  errorLine.hidden = !text;
  errorLine.textContent = text;
}

async function checkStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error();
    const state = await response.json();
    readiness = !state.knowledge_base ? "База знаний не загружена." :
      !state.proxy_configured ? "База загружена · прокси не настроен" : "База загружена · прокси настроен";
  } catch {
    readiness = "Нет связи с сервером.";
  }
  update();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || busy) return;
  busy = true;
  showError();
  update();
  const pending = addMessage("user", question);
  pending.scrollIntoView({ block: "nearest" });
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: turns.slice(-12).map(turn => turn.question) }),
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(typeof data?.detail === "string" ? data.detail :
        response.status === 422 ? "Введите вопрос длиной от 1 до 2000 символов." : "Сервер не смог обработать запрос.");
    }
    if (typeof data?.answer !== "string" || !Array.isArray(data.sources)) throw new Error("Сервер вернул некорректный ответ.");
    const turn = { question, answer: data.answer, sources: data.sources };
    addAnswer(turn).scrollIntoView({ block: "start" });
    turns.push(turn);
    input.value = "";
  } catch (error) {
    pending.remove();
    intro.hidden = Boolean(turns.length);
    showError(error instanceof TypeError ? "Не удалось связаться с сервером. Попробуйте отправить вопрос ещё раз." : error.message);
  } finally {
    busy = false;
    persist();
    update();
  }
});

input.addEventListener("input", () => { persist(); update(); });
reset.addEventListener("click", () => {
  if (busy) return;
  turns = [];
  log.replaceChildren();
  intro.hidden = false;
  input.value = "";
  showError();
  persist();
  update();
  input.focus();
});
update();
checkStatus();
