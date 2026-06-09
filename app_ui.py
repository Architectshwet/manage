from __future__ import annotations

import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

BASE_DIR = Path(__file__).resolve().parent
EXAM_DIR = BASE_DIR / "databricks_generative_ai"

OPTION_COLUMNS = ("Option A", "Option B", "Option C", "Option D", "Option E", "Option F")
ALLOWED_FILES = {
    #"main.csv": "main.csv", #123
    #"3.csv": "3.csv", #116
    #"4.csv": "4.csv", #40
    #"6realudemy.csv": "6_real_udemy.csv", #100
    #"7realudemy.csv": "7_real_udemy.csv", #200
    #"8realudemy.csv": "8_real_udemy.csv", #200
    "1.csv": "1.csv",
    "2.csv": "2.csv",
    "3.csv": "3.csv",
    "4.csv": "4.csv",
    "5.csv": "5.csv",
    "6.csv": "6.csv"
}


def normalize_answer_tokens(raw_answer: str) -> list[str]:
    if not raw_answer:
        return []
    tokens: list[str] = []
    for part in str(raw_answer).split(","):
        clean = part.strip().upper()
        if clean.startswith("OPTION "):
            clean = clean.replace("OPTION ", "", 1).strip()
        if clean and clean[0] in "ABCDEF":
            tokens.append(clean[0])
    return sorted(set(tokens))


def extract_options(row: pd.Series) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for col in OPTION_COLUMNS:
        value = str(row.get(col, "")).strip()
        if value:
            options.append({"key": col.split()[-1], "text": value})
    return options


@lru_cache(maxsize=len(ALLOWED_FILES))
def load_dataframe(source_alias: str) -> pd.DataFrame:
    filename = ALLOWED_FILES.get(source_alias)
    if filename is None:
        raise HTTPException(status_code=404, detail="Source is not allowed")

    file_path = EXAM_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing file: {filename}")

    df = pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "Question" not in df.columns or "Answer" not in df.columns:
        raise HTTPException(status_code=500, detail=f"Invalid CSV schema in {filename}")
    return df


@lru_cache(maxsize=1)
def build_question_index() -> list[dict[str, object]]:
    index: list[dict[str, object]] = []
    for source_alias, filename in ALLOWED_FILES.items():
        df = load_dataframe(source_alias)
        for row_index in range(len(df)):
            row = df.iloc[row_index]
            options = extract_options(row)
            answer_keys = normalize_answer_tokens(str(row.get("Answer", "")))
            searchable_text = " ".join(
                [str(row.get("Question", "")).strip(), *[opt["text"] for opt in options]]
            ).lower()

            index.append(
                {
                    "id": f"{source_alias}:{row_index}",
                    "source": source_alias,
                    "source_file": filename,
                    "index": row_index + 1,
                    "question": str(row.get("Question", "")).strip(),
                    "options": options,
                    "answer_keys": answer_keys,
                    "searchable_text": searchable_text,
                }
            )
    return index


@asynccontextmanager
async def lifespan(_: FastAPI):
    missing = [name for name in ALLOWED_FILES.values() if not (EXAM_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"Required CSV files missing: {', '.join(missing)}")
    yield


app = FastAPI(
    title="NVIDIA GenAI Study Deck",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=800)


@app.middleware("http")
async def add_common_headers(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/web", status_code=307)


@app.get("/api/sources")
async def get_sources():
    sources = []
    for alias, filename in ALLOWED_FILES.items():
        total = len(load_dataframe(alias))
        sources.append(
            {
                "id": alias,
                "label": alias,
                "actual_filename": filename,
                "total_questions": total,
            }
        )
    return {"sources": sources}


@app.get("/api/questions")
async def get_questions(
    source: str = Query(..., description="Allowed source alias"),
    offset: int = Query(0, ge=0),
    limit: int = Query(5, ge=1, le=20),
):
    if source not in ALLOWED_FILES:
        raise HTTPException(status_code=404, detail="Source is not allowed")

    df = load_dataframe(source)
    total = len(df)
    if offset >= total:
        return {
            "source": source,
            "offset": offset,
            "count": 0,
            "total": total,
            "has_more": False,
            "items": [],
        }

    end = min(offset + limit, total)
    items: list[dict[str, object]] = []

    for row_index in range(offset, end):
        row = df.iloc[row_index]
        options = extract_options(row)
        answer_keys = normalize_answer_tokens(str(row.get("Answer", "")))
        option_lookup = {opt["key"]: opt["text"] for opt in options}
        answers = [{"key": key, "text": option_lookup.get(key, "")} for key in answer_keys]

        items.append(
            {
                "id": f"{source}:{row_index}",
                "index": row_index + 1,
                "question": str(row.get("Question", "")).strip(),
                "options": options,
                "answer_keys": answer_keys,
                "answers": answers,
            }
        )

    return {
        "source": source,
        "offset": offset,
        "count": len(items),
        "total": total,
        "has_more": end < total,
        "items": items,
    }


@app.get("/api/search")
async def search_questions(
    q: str = Query(..., min_length=1, description="Keyword query"),
    source: str | None = Query(None, description="Optional source alias filter"),
):
    query = q.strip().lower()
    if not query:
        return {"query": q, "source": source, "count": 0, "items": []}

    results: list[dict[str, object]] = []
    for record in build_question_index():
        if source and record["source"] != source:
            continue
        text = str(record["searchable_text"])
        if query not in text:
            continue

        results.append(
            {
                "id": record["id"],
                "source": record["source"],
                "source_file": record["source_file"],
                "index": record["index"],
                "question": record["question"],
                "options": record["options"],
                "answer_keys": record["answer_keys"],
            }
        )

    results.sort(
        key=lambda item: (
            str(item["source"]),
            int(item["index"]),
        )
    )

    return {"query": q, "source": source, "count": len(results), "items": results}


HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NVIDIA GenAI Study Deck</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,700&display=swap" rel="stylesheet">
    <style>
        :root {
            --panel: #ffffff;
            --ink: #1a2540;
            --muted: #556078;
            --accent: #1f6feb;
            --good: #198754;
            --good-soft: #e8f7ef;
            --border: #dce4f3;
            --shadow: 0 14px 36px rgba(32, 44, 74, 0.16);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--ink);
            background:
                radial-gradient(circle at 12% 8%, #dff0ff 0%, rgba(223, 240, 255, 0) 42%),
                radial-gradient(circle at 88% 2%, #d7e3ff 0%, rgba(215, 227, 255, 0) 40%),
                linear-gradient(180deg, #f7fbff 0%, #eef4ff 100%);
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
            min-height: 100vh;
        }
        .shell {
            width: calc(100vw - 24px);
            margin: 0 auto;
            padding: 10px 0 16px;
        }
        .topbar {
            width: 100%;
            border-radius: 22px;
            padding: 1.05rem 1.35rem;
            background: linear-gradient(130deg, #0e234f 0%, #173d83 55%, #2054b4 100%);
            color: #f7f9ff;
            box-shadow: var(--shadow);
        }
        .topbar h1 { margin: 0 0 .38rem; font-size: clamp(1.38rem, 2vw, 2rem); }
        .topbar p { margin: 0; color: #d9e6ff; font-size: .95rem; }
        .topbar-meta {
            margin-top: .7rem;
            display: flex;
            gap: .55rem;
            flex-wrap: wrap;
        }
        .meta-pill {
            border: 1px solid rgba(227, 235, 255, .34);
            color: #dbe9ff;
            padding: .28rem .58rem;
            border-radius: 999px;
            font-size: .82rem;
            background: rgba(255, 255, 255, .07);
        }
        .layout {
            width: 100%;
            margin: .82rem auto 0;
            display: grid;
            grid-template-columns: 320px minmax(0, 1fr);
            gap: .85rem;
            align-items: start;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(39, 62, 118, 0.1);
        }
        .controls {
            padding: .95rem;
            position: sticky;
            top: 10px;
            height: calc(100vh - 34px);
            overflow: auto;
        }
        .group { margin-bottom: .86rem; }
        label { display: block; margin-bottom: .35rem; color: var(--muted); font-size: .9rem; }
        select {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #bfcae1;
            padding: .62rem .72rem;
            font-family: inherit;
            font-size: .94rem;
        }
        .fixed-size {
            border: 1px dashed #b8c9e7;
            border-radius: 12px;
            background: #f3f8ff;
            padding: .72rem .76rem;
            color: #284f94;
            font-weight: 700;
            font-size: .92rem;
        }
        .search-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            margin: 0 0 .86rem;
            border-radius: 10px;
            border: 1px solid #b8c9e7;
            background: linear-gradient(120deg, #eef4ff 0%, #e6efff 100%);
            color: #234784;
            font-weight: 700;
            font-size: .9rem;
            padding: .6rem .72rem;
            text-decoration: none;
        }
        .search-link:hover { filter: brightness(1.03); }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: .4rem;
            margin-bottom: .84rem;
        }
        .stat {
            border: 1px solid #d9e5fb;
            border-radius: 11px;
            padding: .46rem .52rem;
            background: #f7faff;
        }
        .stat-title {
            font-size: .74rem;
            color: #607296;
            margin-bottom: .18rem;
        }
        .stat-value {
            font-size: .95rem;
            font-weight: 700;
            color: #1d437f;
        }
        .jump-wrap {
            border: 1px solid #d6e0f5;
            border-radius: 14px;
            background: #fbfdff;
            padding: .7rem;
        }
        .jump-title {
            margin: 0 0 .45rem;
            color: #2a4f92;
            font-size: .86rem;
            font-weight: 700;
        }
        .jump-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: .34rem;
        }
        .jump-btn {
            border: 1px solid #c6d5f2;
            --progress: 0;
            background:
                linear-gradient(
                    90deg,
                    rgba(24, 56, 108, 0.45) 0%,
                    rgba(24, 56, 108, 0.45) calc(var(--progress) * 1%),
                    #edf3ff calc(var(--progress) * 1%),
                    #edf3ff 100%
                );
            color: #214989;
            border-radius: 9px;
            padding: .38rem 0;
            font-size: .82rem;
            font-weight: 700;
            cursor: pointer;
            transition: border-color .15s ease, transform .1s ease;
        }
        .jump-btn:hover {
            filter: brightness(1.03);
            transform: translateY(-1px);
        }
        .jump-btn.active {
            border-color: #1f6feb;
            box-shadow: 0 0 0 2px rgba(31, 111, 235, 0.18);
            color: #173f7f;
        }
        .jump-btn:disabled { opacity: .55; cursor: wait; }
        .status {
            margin-top: .74rem;
            font-size: .86rem;
            color: var(--muted);
            line-height: 1.46;
            border-top: 1px solid #e3ebfa;
            padding-top: .62rem;
        }
        .feed {
            padding: 1rem;
            min-height: calc(100vh - 120px);
            display: flex;
            flex-direction: column;
        }
        .feed-head {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: .6rem;
            margin-bottom: .65rem;
        }
        .feed-title { margin: 0; color: #153a7a; font-size: 1.05rem; }
        .feed-subtitle { margin: .16rem 0 0; color: #5e7091; font-size: .85rem; }
        .question-feed {
            display: grid;
            gap: .85rem;
            flex: 1;
            align-content: start;
        }
        .card {
            border: 1px solid #d8e2f5;
            border-radius: 14px;
            padding: .92rem;
            background: linear-gradient(180deg, #ffffff 0%, #fcfdff 100%);
        }
        .card.known {
            border-color: #7295cc;
            background: linear-gradient(180deg, #e8f0ff 0%, #dde8ff 100%);
            box-shadow: 0 0 0 1px #c7d8f6 inset;
        }
        .card-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: .6rem;
            margin-bottom: .55rem;
        }
        .q-title { margin: 0; color: #0e2b5f; font-size: 1rem; }
        .q-index { color: #366fd3; font-weight: 700; margin-right: .42rem; }
        .q-text { font-family: "Source Serif 4", Georgia, serif; font-size: 1.08rem; line-height: 1.52; }
        .known-toggle {
            border: 1px solid #c4d2ea;
            background: #f2f7ff;
            color: #2b4f8e;
            border-radius: 999px;
            padding: .3rem .62rem;
            font-family: inherit;
            font-size: .78rem;
            font-weight: 700;
            cursor: pointer;
            white-space: nowrap;
            line-height: 1.2;
        }
        .known-toggle.active {
            border-color: #204682;
            background: #d4e2fb;
            color: #123565;
        }
        .options { list-style: none; margin: .8rem 0 .68rem; padding: 0; display: grid; gap: .4rem; }
        .option {
            border: 1px solid #dde5f5;
            border-radius: 9px;
            padding: .52rem .64rem;
            background: #f8fbff;
            line-height: 1.44;
        }
        .option strong { color: #2b4f8e; }
        .option.correct {
            border-color: #74c69d;
            background: var(--good-soft);
            color: #0f5132;
        }
        .center { text-align: center; padding: .6rem; color: var(--muted); }
        .hidden { display: none; }
        .pager {
            margin-top: .7rem;
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: .6rem;
            align-items: center;
            border-top: 1px solid #e4ebfb;
            padding-top: .7rem;
        }
        .pager-btn {
            border-radius: 10px;
            border: 1px solid #bfcae1;
            background: linear-gradient(180deg, #edf4ff 0%, #e4eeff 100%);
            color: #214989;
            padding: .56rem .7rem;
            font-family: inherit;
            font-size: .94rem;
            font-weight: 600;
            cursor: pointer;
        }
        .pager-btn:hover { filter: brightness(1.03); }
        .pager-btn:disabled { opacity: .45; cursor: not-allowed; }
        .pager-info {
            font-size: .9rem;
            color: #395988;
            font-weight: 700;
            white-space: nowrap;
            text-align: center;
        }
        @media (max-width: 1120px) {
            .layout { grid-template-columns: 280px minmax(0, 1fr); }
            .jump-grid { grid-template-columns: repeat(5, minmax(0, 1fr)); }
        }
        @media (max-width: 900px) {
            .shell { width: calc(100vw - 14px); }
            .layout { grid-template-columns: 1fr; }
            .controls {
                position: static;
                height: auto;
            }
            .feed { min-height: auto; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <header class="topbar">
            <h1>NVIDIA Generative AI Multimodal</h1>
            <p>Study in focused chunks of 5. Jump directly to any page using the numbered bars in the sidebar.</p>
            <div class="topbar-meta">
                <span class="meta-pill">Chunk Size: 5 Questions</span>
                <span class="meta-pill">Direct Page Jump Enabled</span>
                <span class="meta-pill">Blue Reviewed State</span>
            </div>
        </header>

        <main class="layout">
            <section class="panel controls">
                <div class="group">
                    <label for="source-select">Question File</label>
                    <select id="source-select"></select>
                </div>
                <a class="search-link" href="/search">Open Keyword Search</a>
                <div class="group fixed-size">Questions Per Page: 5</div>

                <div class="stats">
                    <div class="stat">
                        <div class="stat-title">Total</div>
                        <div id="stat-total" class="stat-value">0</div>
                    </div>
                    <div class="stat">
                        <div class="stat-title">Pages</div>
                        <div id="stat-pages" class="stat-value">0</div>
                    </div>
                    <div class="stat">
                        <div class="stat-title">Current</div>
                        <div id="stat-current" class="stat-value">0</div>
                    </div>
                    <div class="stat">
                        <div class="stat-title">Known</div>
                        <div id="stat-known" class="stat-value">0</div>
                    </div>
                </div>

                <div class="jump-wrap">
                    <p class="jump-title">Quick Jump (Pages)</p>
                    <div id="jump-grid" class="jump-grid"></div>
                </div>
                <p id="status" class="status">Loading source list...</p>
            </section>

            <section class="panel feed">
                <div class="feed-head">
                    <div>
                        <h2 class="feed-title">Question Deck</h2>
                        <p class="feed-subtitle">Each page shows exactly 5 questions for fast memorization.</p>
                    </div>
                </div>
                <div id="question-feed" class="question-feed"></div>
                <div id="empty" class="center hidden">No questions loaded yet.</div>
                <div id="loading" class="center hidden">Loading questions...</div>
                <div id="pager" class="pager hidden">
                    <button id="prev-btn" class="pager-btn" type="button">Previous</button>
                    <div id="page-info" class="pager-info"></div>
                    <button id="next-btn" class="pager-btn" type="button">Next</button>
                </div>
            </section>
        </main>
    </div>

    <script>
        const PAGE_SIZE = 5;
        const KNOWN_STORE_KEY = "nvidia_genai_known_runtime";
        const sourceSelect = document.getElementById("source-select");
        const questionFeed = document.getElementById("question-feed");
        const statusEl = document.getElementById("status");
        const emptyEl = document.getElementById("empty");
        const loadingEl = document.getElementById("loading");
        const pagerEl = document.getElementById("pager");
        const prevBtn = document.getElementById("prev-btn");
        const nextBtn = document.getElementById("next-btn");
        const pageInfoEl = document.getElementById("page-info");
        const jumpGrid = document.getElementById("jump-grid");
        const statTotal = document.getElementById("stat-total");
        const statPages = document.getElementById("stat-pages");
        const statCurrent = document.getElementById("stat-current");
        const statKnown = document.getElementById("stat-known");

        let selectedSource = "";
        let currentOffset = 0;
        let currentTotal = 0;
        let hasMore = false;
        let isLoading = false;
        let currentPage = 0;
        let totalPages = 0;
        let knownIds = new Set();

        function setStatus(text) { statusEl.textContent = text; }
        function showLoading(show) { loadingEl.classList.toggle("hidden", !show); }

        function clearFeed() {
            questionFeed.innerHTML = "";
            emptyEl.classList.add("hidden");
        }

        function initKnownStore() {
            try {
                const nav = performance.getEntriesByType("navigation");
                if (Array.isArray(nav) && nav.length > 0 && nav[0].type === "reload") {
                    sessionStorage.removeItem(KNOWN_STORE_KEY);
                }
            } catch (error) {
            }
        }

        function readKnownStore() {
            try {
                const raw = sessionStorage.getItem(KNOWN_STORE_KEY);
                if (!raw) return {};
                const parsed = JSON.parse(raw);
                return parsed && typeof parsed === "object" ? parsed : {};
            } catch (error) {
                return {};
            }
        }

        function writeKnownStore(store) {
            try {
                sessionStorage.setItem(KNOWN_STORE_KEY, JSON.stringify(store));
            } catch (error) {
            }
        }

        function loadKnownMarks() {
            const store = readKnownStore();
            const existing = store[selectedSource];
            knownIds = new Set(Array.isArray(existing) ? existing : []);
        }

        function saveKnownMarks() {
            const store = readKnownStore();
            store[selectedSource] = Array.from(knownIds);
            writeKnownStore(store);
        }

        function pruneKnownMarks() {
            if (!selectedSource || currentTotal <= 0) return;

            const filtered = new Set();
            for (const questionId of knownIds) {
                if (!questionId.startsWith(`${selectedSource}:`)) continue;
                const rowPart = questionId.split(":")[1];
                const rowIndex = Number(rowPart);
                if (Number.isInteger(rowIndex) && rowIndex >= 0 && rowIndex < currentTotal) {
                    filtered.add(questionId);
                }
            }
            knownIds = filtered;
            saveKnownMarks();
        }

        function parseQuestionRowIndex(questionId) {
            if (typeof questionId !== "string" || !questionId.startsWith(`${selectedSource}:`)) {
                return -1;
            }
            const rowPart = questionId.split(":")[1];
            const rowIndex = Number(rowPart);
            return Number.isInteger(rowIndex) ? rowIndex : -1;
        }

        function getQuestionCountForPage(pageNumber) {
            const start = (pageNumber - 1) * PAGE_SIZE;
            if (start >= currentTotal) return 0;
            return Math.min(PAGE_SIZE, currentTotal - start);
        }

        function getKnownCountForPage(pageNumber) {
            const start = (pageNumber - 1) * PAGE_SIZE;
            const end = start + PAGE_SIZE;
            let count = 0;

            for (const questionId of knownIds) {
                const rowIndex = parseQuestionRowIndex(questionId);
                if (rowIndex >= start && rowIndex < end) {
                    count += 1;
                }
            }
            return count;
        }

        function renderPageJumps() {
            jumpGrid.innerHTML = "";
            if (totalPages === 0) {
                return;
            }

            for (let p = 1; p <= totalPages; p += 1) {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = p === currentPage ? "jump-btn active" : "jump-btn";
                btn.textContent = String(p);
                const knownCount = getKnownCountForPage(p);
                const pageQuestionCount = getQuestionCountForPage(p);
                const progressPercent = pageQuestionCount > 0
                    ? Math.round((knownCount / pageQuestionCount) * 100)
                    : 0;
                btn.style.setProperty("--progress", String(progressPercent));
                btn.title = `Page ${p}: ${knownCount}/${pageQuestionCount} known`;
                btn.disabled = isLoading;
                btn.addEventListener("click", async () => {
                    await loadQuestions((p - 1) * PAGE_SIZE);
                });
                jumpGrid.appendChild(btn);
            }
        }

        function updatePager() {
            const hasItems = currentTotal > 0;
            pagerEl.classList.toggle("hidden", !hasItems);
            currentPage = hasItems ? Math.floor(currentOffset / PAGE_SIZE) + 1 : 0;
            totalPages = hasItems ? Math.ceil(currentTotal / PAGE_SIZE) : 0;

            pageInfoEl.textContent = hasItems ? `Page ${currentPage} / ${totalPages}` : "";
            prevBtn.disabled = isLoading || currentOffset === 0;
            nextBtn.disabled = isLoading || !hasMore;

            statTotal.textContent = String(currentTotal);
            statPages.textContent = String(totalPages);
            statCurrent.textContent = String(currentPage);
            statKnown.textContent = String(knownIds.size);

            renderPageJumps();
        }

        function escapeHtml(text) {
            const div = document.createElement("div");
            div.textContent = text;
            return div.innerHTML;
        }

        async function fetchJson(url) {
            const res = await fetch(url);
            if (!res.ok) {
                throw new Error(`Request failed (${res.status})`);
            }
            return res.json();
        }

        function renderCard(item) {
            const card = document.createElement("article");
            const isKnown = knownIds.has(item.id);
            card.className = isKnown ? "card known" : "card";

            const header = document.createElement("div");
            header.className = "card-top";

            const title = document.createElement("h2");
            title.className = "q-title";
            title.innerHTML = `<span class="q-index">Q${item.index}.</span><span class="q-text">${escapeHtml(item.question)}</span>`;

            const knownBtn = document.createElement("button");
            knownBtn.type = "button";
            knownBtn.className = isKnown ? "known-toggle active" : "known-toggle";
            knownBtn.textContent = isKnown ? "✓ Known" : "Mark Known";
            knownBtn.addEventListener("click", () => {
                const nowKnown = !knownIds.has(item.id);
                if (nowKnown) {
                    knownIds.add(item.id);
                } else {
                    knownIds.delete(item.id);
                }
                saveKnownMarks();
                knownBtn.className = nowKnown ? "known-toggle active" : "known-toggle";
                knownBtn.textContent = nowKnown ? "✓ Known" : "Mark Known";
                card.className = nowKnown ? "card known" : "card";
                updatePager();
            });

            header.appendChild(title);
            header.appendChild(knownBtn);
            card.appendChild(header);

            const options = document.createElement("ul");
            options.className = "options";
            const answerKeys = new Set(item.answer_keys || []);

            for (const option of item.options || []) {
                const li = document.createElement("li");
                li.className = answerKeys.has(option.key) ? "option correct" : "option";
                li.innerHTML = `<strong>${escapeHtml(option.key)}.</strong> ${escapeHtml(option.text)}`;
                options.appendChild(li);
            }
            card.appendChild(options);

            return card;
        }

        async function loadSources() {
            const data = await fetchJson("/api/sources");
            sourceSelect.innerHTML = "";

            for (const source of data.sources) {
                const option = document.createElement("option");
                option.value = source.id;
                option.textContent = `${source.label} (${source.actual_filename})`;
                sourceSelect.appendChild(option);
            }

            if (!data.sources.length) {
                throw new Error("No allowed files found.");
            }

            selectedSource = data.sources[0].id;
            sourceSelect.value = selectedSource;
        }

        async function loadQuestions(offsetToLoad = 0) {
            if (isLoading || !selectedSource) return;
            const safeOffset = Math.max(0, Number(offsetToLoad) || 0);

            isLoading = true;
            showLoading(true);
            prevBtn.disabled = true;
            nextBtn.disabled = true;
            setStatus("Loading questions...");
            clearFeed();
            renderPageJumps();

            try {
                const query = new URLSearchParams({
                    source: selectedSource,
                    offset: String(safeOffset),
                    limit: String(PAGE_SIZE)
                });
                const data = await fetchJson(`/api/questions?${query.toString()}`);
                currentOffset = safeOffset;
                currentTotal = data.total || 0;
                pruneKnownMarks();

                if (data.count === 0 && safeOffset === 0) {
                    emptyEl.classList.remove("hidden");
                    setStatus("No questions in this file.");
                    hasMore = false;
                    updatePager();
                    return;
                }

                for (const item of data.items) {
                    questionFeed.appendChild(renderCard(item));
                }

                hasMore = data.has_more;
                const start = data.count ? safeOffset + 1 : 0;
                const end = safeOffset + data.count;
                setStatus(`Showing ${start}-${end} of ${data.total}.`);
                updatePager();
            } catch (error) {
                setStatus(`Error: ${error.message}`);
                currentTotal = 0;
                hasMore = false;
                updatePager();
            } finally {
                isLoading = false;
                showLoading(false);
                updatePager();
            }
        }

        sourceSelect.addEventListener("change", async () => {
            selectedSource = sourceSelect.value;
            loadKnownMarks();
            await loadQuestions(0);
        });

        prevBtn.addEventListener("click", async () => {
            if (currentOffset > 0) {
                await loadQuestions(Math.max(0, currentOffset - PAGE_SIZE));
            }
        });

        nextBtn.addEventListener("click", async () => {
            if (hasMore) {
                await loadQuestions(currentOffset + PAGE_SIZE);
            }
        });

        window.addEventListener("DOMContentLoaded", async () => {
            try {
                initKnownStore();
                await loadSources();
                loadKnownMarks();
                await loadQuestions(0);
            } catch (error) {
                setStatus(`Startup error: ${error.message}`);
            }
        });
    </script>
</body>
</html>
"""

SEARCH_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keyword Search - NVIDIA GenAI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,700&display=swap" rel="stylesheet">
    <style>
        :root {
            --panel: #ffffff;
            --ink: #1a2540;
            --muted: #556078;
            --border: #dce4f3;
            --good: #198754;
            --good-soft: #e8f7ef;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--ink);
            background:
                radial-gradient(circle at 15% 6%, #e5f2ff 0%, rgba(229, 242, 255, 0) 36%),
                linear-gradient(180deg, #f7fbff 0%, #edf4ff 100%);
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
        }
        .shell {
            width: calc(100vw - 26px);
            margin: 0 auto;
            padding: 10px 0 16px;
        }
        .topbar {
            background: linear-gradient(130deg, #0e234f 0%, #173d83 55%, #2054b4 100%);
            color: #f7f9ff;
            border-radius: 20px;
            padding: 1rem 1.2rem;
        }
        .topbar h1 { margin: 0 0 .34rem; font-size: clamp(1.2rem, 2vw, 1.8rem); }
        .topbar p { margin: 0; color: #d9e6ff; }
        .toolbar {
            margin-top: .8rem;
            display: grid;
            grid-template-columns: minmax(240px, 1fr) 220px 140px auto;
            gap: .5rem;
            align-items: center;
        }
        .toolbar input,
        .toolbar select,
        .toolbar button,
        .back-link {
            border-radius: 10px;
            border: 1px solid #b7c8e8;
            padding: .58rem .72rem;
            font-family: inherit;
            font-size: .92rem;
        }
        .toolbar input:focus,
        .toolbar select:focus {
            outline: 2px solid #b7ccf7;
            outline-offset: 1px;
        }
        .toolbar button {
            border: 0;
            color: #fff;
            background: linear-gradient(120deg, #1a5fd0 0%, #2e7bf8 100%);
            font-weight: 700;
            cursor: pointer;
        }
        .back-link {
            text-decoration: none;
            text-align: center;
            color: #214989;
            font-weight: 700;
            background: #eef4ff;
        }
        .meta-row {
            margin-top: .72rem;
            display: flex;
            justify-content: space-between;
            gap: .6rem;
            font-size: .86rem;
            color: #d5e5ff;
        }
        .content {
            margin-top: .8rem;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: .9rem;
            min-height: calc(100vh - 180px);
        }
        .status { color: #5e7091; margin-bottom: .72rem; font-size: .9rem; }
        .result-list {
            display: grid;
            gap: .8rem;
        }
        .card {
            border: 1px solid #d8e2f5;
            border-radius: 14px;
            background: linear-gradient(180deg, #ffffff 0%, #fcfdff 100%);
            padding: .84rem;
        }
        .card.known {
            border-color: #7295cc;
            background: linear-gradient(180deg, #e8f0ff 0%, #dde8ff 100%);
            box-shadow: 0 0 0 1px #c7d8f6 inset;
        }
        .card-top {
            display: flex;
            justify-content: space-between;
            gap: .6rem;
            margin-bottom: .58rem;
        }
        .source-pill {
            display: inline-flex;
            border: 1px solid #b8cbec;
            color: #234782;
            background: #eef4ff;
            border-radius: 999px;
            padding: .2rem .56rem;
            font-size: .76rem;
            font-weight: 700;
            margin-right: .45rem;
        }
        .q-title { margin: .28rem 0 0; color: #0e2b5f; font-size: .99rem; }
        .q-text { font-family: "Source Serif 4", Georgia, serif; font-size: 1.05rem; line-height: 1.5; }
        .known-toggle {
            border: 1px solid #c4d2ea;
            background: #f2f7ff;
            color: #2b4f8e;
            border-radius: 999px;
            padding: .3rem .62rem;
            font-family: inherit;
            font-size: .78rem;
            font-weight: 700;
            cursor: pointer;
            white-space: nowrap;
            line-height: 1.2;
        }
        .known-toggle.active {
            border-color: #204682;
            background: #d4e2fb;
            color: #123565;
        }
        .options { list-style: none; margin: .75rem 0 0; padding: 0; display: grid; gap: .4rem; }
        .option {
            border: 1px solid #dde5f5;
            border-radius: 9px;
            padding: .48rem .6rem;
            background: #f8fbff;
        }
        .option.correct {
            border-color: #74c69d;
            background: var(--good-soft);
            color: #0f5132;
        }
        .hl { background: #fff3b6; border-radius: 4px; padding: 0 .08rem; }
        @media (max-width: 980px) {
            .toolbar { grid-template-columns: 1fr; }
            .meta-row { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <header class="topbar">
            <h1>Keyword Search Across All Files</h1>
            <p>Search in questions and options, find related items instantly, and mark known from here too.</p>
            <div class="toolbar">
                <input id="search-input" type="text" placeholder="Try: active learning, tokenization, embedding..." />
                <select id="source-filter">
                    <option value="">All Files</option>
                </select>
                <button id="search-btn" type="button">Search</button>
                <a class="back-link" href="/web">Back To Study Deck</a>
            </div>
            <div class="meta-row">
                <span id="result-meta">Enter a keyword to start.</span>
                <span id="known-meta">Known in results: 0</span>
            </div>
        </header>

        <main class="content">
            <div id="status" class="status">Ready.</div>
            <section id="result-list" class="result-list"></section>
        </main>
    </div>

    <script>
        const KNOWN_STORE_KEY = "nvidia_genai_known_runtime";
        const searchInput = document.getElementById("search-input");
        const sourceFilter = document.getElementById("source-filter");
        const searchBtn = document.getElementById("search-btn");
        const statusEl = document.getElementById("status");
        const resultMeta = document.getElementById("result-meta");
        const knownMeta = document.getElementById("known-meta");
        const resultList = document.getElementById("result-list");

        let currentQuery = "";
        let currentResults = [];
        let sourceMap = {};

        function setStatus(text) { statusEl.textContent = text; }

        function initKnownStore() {
            try {
                const nav = performance.getEntriesByType("navigation");
                if (Array.isArray(nav) && nav.length > 0 && nav[0].type === "reload") {
                    sessionStorage.removeItem(KNOWN_STORE_KEY);
                }
            } catch (error) {
            }
        }

        function readKnownStore() {
            try {
                const raw = sessionStorage.getItem(KNOWN_STORE_KEY);
                if (!raw) return {};
                const parsed = JSON.parse(raw);
                return parsed && typeof parsed === "object" ? parsed : {};
            } catch (error) {
                return {};
            }
        }

        function writeKnownStore(store) {
            try {
                sessionStorage.setItem(KNOWN_STORE_KEY, JSON.stringify(store));
            } catch (error) {
            }
        }

        function isKnown(questionId) {
            const source = questionId.split(":")[0];
            const store = readKnownStore();
            const knownForSource = Array.isArray(store[source]) ? store[source] : [];
            return knownForSource.includes(questionId);
        }

        function setKnown(questionId, known) {
            const source = questionId.split(":")[0];
            const store = readKnownStore();
            const set = new Set(Array.isArray(store[source]) ? store[source] : []);
            if (known) set.add(questionId);
            else set.delete(questionId);
            store[source] = Array.from(set);
            writeKnownStore(store);
        }

        function getKnownCountInResults() {
            let count = 0;
            for (const item of currentResults) {
                if (isKnown(item.id)) count += 1;
            }
            return count;
        }

        function updateKnownMeta() {
            const knownCount = getKnownCountInResults();
            knownMeta.textContent = `Known in results: ${knownCount}/${currentResults.length}`;
        }

        function escapeHtml(text) {
            const div = document.createElement("div");
            div.textContent = text;
            return div.innerHTML;
        }

        function escapeRegExp(value) {
            const specialChars = new Set([".", "*", "+", "?", "^", "$", "{", "}", "(", ")", "|", "[", "]", String.fromCharCode(92)]);
            let out = "";
            for (const ch of value) {
                out += specialChars.has(ch) ? String.fromCharCode(92) + ch : ch;
            }
            return out;
        }

        function highlight(text) {
            const safe = escapeHtml(text);
            if (!currentQuery.trim()) return safe;

            const phrase = currentQuery.trim();
            const pattern = new RegExp(`(${escapeRegExp(phrase)})`, "ig");
            return safe.replace(pattern, '<span class="hl">$1</span>');
        }

        async function fetchJson(url) {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Request failed (${response.status})`);
            return response.json();
        }

        async function loadSources() {
            const data = await fetchJson("/api/sources");
            sourceMap = {};
            sourceFilter.innerHTML = '<option value="">All Files</option>';
            for (const source of data.sources) {
                sourceMap[source.id] = `${source.id} (${source.actual_filename})`;
                const option = document.createElement("option");
                option.value = source.id;
                option.textContent = sourceMap[source.id];
                sourceFilter.appendChild(option);
            }
        }

        function renderResults(items) {
            resultList.innerHTML = "";
            currentResults = items;
            if (!items.length) {
                resultList.innerHTML = '<div class="status">No matches found for this keyword.</div>';
                updateKnownMeta();
                return;
            }

            for (const item of items) {
                const card = document.createElement("article");
                const known = isKnown(item.id);
                card.className = known ? "card known" : "card";

                const header = document.createElement("div");
                header.className = "card-top";

                const left = document.createElement("div");
                const sourceLabel = sourceMap[item.source] || `${item.source} (${item.source_file})`;
                left.innerHTML = `<span class="source-pill">${escapeHtml(sourceLabel)}</span><span class="source-pill">Q${item.index}</span><h2 class="q-title"><span class="q-text">${highlight(item.question)}</span></h2>`;

                const knownBtn = document.createElement("button");
                knownBtn.type = "button";
                knownBtn.className = known ? "known-toggle active" : "known-toggle";
                knownBtn.textContent = known ? "✓ Known" : "Mark Known";
                knownBtn.addEventListener("click", () => {
                    const nowKnown = !isKnown(item.id);
                    setKnown(item.id, nowKnown);
                    card.className = nowKnown ? "card known" : "card";
                    knownBtn.className = nowKnown ? "known-toggle active" : "known-toggle";
                    knownBtn.textContent = nowKnown ? "✓ Known" : "Mark Known";
                    updateKnownMeta();
                });

                header.appendChild(left);
                header.appendChild(knownBtn);
                card.appendChild(header);

                const optionList = document.createElement("ul");
                optionList.className = "options";
                const answerSet = new Set(item.answer_keys || []);
                for (const option of item.options || []) {
                    const li = document.createElement("li");
                    li.className = answerSet.has(option.key) ? "option correct" : "option";
                    li.innerHTML = `<strong>${escapeHtml(option.key)}.</strong> ${highlight(option.text)}`;
                    optionList.appendChild(li);
                }
                card.appendChild(optionList);
                resultList.appendChild(card);
            }

            updateKnownMeta();
        }

        async function runSearch() {
            const query = searchInput.value.trim();
            currentQuery = query;
            if (!query) {
                currentResults = [];
                resultList.innerHTML = "";
                resultMeta.textContent = "Enter a keyword to start.";
                knownMeta.textContent = "Known in results: 0";
                setStatus("Please type a keyword.");
                return;
            }

            try {
                setStatus("Searching...");
                const params = new URLSearchParams({ q: query });
                if (sourceFilter.value) params.set("source", sourceFilter.value);
                const data = await fetchJson(`/api/search?${params.toString()}`);
                resultMeta.textContent = `Results for "${query}": ${data.count} question(s)`;
                renderResults(data.items || []);
                setStatus("Search complete.");
            } catch (error) {
                setStatus(`Search error: ${error.message}`);
            }
        }

        searchBtn.addEventListener("click", async () => { await runSearch(); });
        searchInput.addEventListener("keydown", async (event) => {
            if (event.key === "Enter") await runSearch();
        });

        window.addEventListener("DOMContentLoaded", async () => {
            initKnownStore();
            try {
                await loadSources();
                setStatus("Ready. Search by keyword across all question files.");
            } catch (error) {
                setStatus(`Startup error: ${error.message}`);
            }
        });
    </script>
</body>
</html>
"""


@app.get("/web", response_class=HTMLResponse)
async def serve_web():
    return HTMLResponse(content=HTML_CONTENT)


@app.get("/search", response_class=HTMLResponse)
async def serve_search():
    return HTMLResponse(content=SEARCH_HTML)


if __name__ == "__main__":
    print("Starting app_ui server at http://127.0.0.1:8003/web")
    uvicorn.run("app_ui:app", host="127.0.0.1", port=8003, reload=False)
