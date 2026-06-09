import os
import random

import pandas as pd
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- App Configuration & Middleware ---
app = FastAPI(title="Databricks Certified Generative AI Engineer Associate")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()
QUESTION_SOURCES = [
    #{"key": "file1", "path": "exam_questions/1.csv", "pick_count": 0},
    #{"key": "file2", "path": "exam_questions/2.csv", "pick_count": 0},
    #{"key": "main", "path": "exam_questions/main_questions.csv", "pick_count": 0}, #main 123
    #{"key": "file3", "path": "exam_questions/3.csv", "pick_count": 0}, #main 40
    #{"key": "file4", "path": "exam_questions/4.csv", "pick_count": 0}, #main 116
    #{"key": "file5", "path": "exam_questions/5.csv", "pick_count": 0},
    #{"key": "file6", "path": "exam_questions/udemy_1.csv", "pick_count": 0},
    #{"key": "file7", "path": "exam_questions/6_real_udemy.csv", "pick_count": 0}, #main 100
    #{"key": "file8", "path": "exam_questions/7_real_udemy.csv", "pick_count": 0}, #main 200
    #{"key": "file9", "path": "exam_questions/8_real_udemy.csv", "pick_count": 0}, #main 200


    {"key": "file10", "path": "databricks_generative_ai/1.csv", "pick_count": 0},
    {"key": "file11", "path": "databricks_generative_ai/2.csv", "pick_count": 0},
    {"key": "file12", "path": "databricks_generative_ai/3.csv", "pick_count": 20}, #main 40
    {"key": "file13", "path": "databricks_generative_ai/4.csv", "pick_count": 0}, #main 116
    {"key": "file14", "path": "databricks_generative_ai/5.csv", "pick_count": 0},
    {"key": "file15", "path": "databricks_generative_ai/6.csv", "pick_count": 0},
    # {"key": "file16", "path": "exam_questions/6_real_udemy.csv", "pick_count": 0}, #main 100
    #{"key": "file8", "path": "exam_questions/7_real_udemy.csv", "pick_count": 0}, #main 200
    #{"key": "file9", "path": "exam_questions/8_real_udemy.csv", "pick_count": 0} #main 200

]
EXAM_QUESTION_COUNT = sum(source["pick_count"] for source in QUESTION_SOURCES)
EXAM_DURATION_SECONDS = 90 * 60
OPTION_COLUMNS = ["Option A", "Option B", "Option C", "Option D", "Option E", "Option F"]


# --- Models ---
class SubmitRequest(BaseModel):
    question_ids: list[str]
    user_answers: dict[str, list[str] | str]


# --- Helper Functions ---
def ensure_times_asked_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Times Asked" not in df.columns:
        df["Times Asked"] = 0
    df["Times Asked"] = pd.to_numeric(df["Times Asked"], errors="coerce").fillna(0)
    return df


def parse_option_keys(raw_value) -> list[str]:
    if pd.isna(raw_value):
        return []
    return [
        token.strip().upper()
        for token in str(raw_value).split(",")
        if token and token.strip()
    ]


def normalize_user_answer(raw_value) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        tokens = raw_value
    else:
        tokens = str(raw_value).split(",")

    normalized = []
    for token in tokens:
        token_clean = str(token).strip().upper()
        if token_clean:
            normalized.append(token_clean)
    return sorted(set(normalized))


def extract_options(row: pd.Series) -> dict[str, str]:
    options: dict[str, str] = {}
    for col in OPTION_COLUMNS:
        value = row.get(col)
        if pd.notna(value):
            value_clean = str(value).strip()
            if value_clean:
                option_key = col.split()[-1]
                options[option_key] = value_clean
    return options


def keys_to_display(keys: list[str], options: dict[str, str]) -> str:
    if not keys:
        return "None"
    rendered = []
    for key in keys:
        option_text = options.get(key)
        if option_text:
            rendered.append(f"{key}. {option_text}")
        else:
            rendered.append(key)
    return ", ".join(rendered)


def load_prioritized_questions_from_file(source_key: str, csv_path: str, pick_count: int):
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path)
    df = ensure_times_asked_column(df)
    df["row_id"] = df.index

    # Prioritize low "Times Asked"; randomize within equal-priority buckets.
    prioritized = (
        df.sample(frac=1)
        .sort_values(by="Times Asked", ascending=True, kind="mergesort")
        .head(pick_count)
    )

    if prioritized.empty:
        return []

    questions = []
    for _, row in prioritized.iterrows():
        options = extract_options(row)
        answer_keys = parse_option_keys(row.get("Answer", ""))
        questions.append(
            {
                "id": f"{source_key}:{int(row['row_id'])}",
                "question": str(row["Question"]).strip(),
                "options": options,
                "allow_multiple": len(answer_keys) > 1,
            }
        )

    return questions


def load_and_prioritize_questions():
    all_questions = []
    for source in QUESTION_SOURCES:
        all_questions.extend(
            load_prioritized_questions_from_file(
                source_key=source["key"],
                csv_path=source["path"],
                pick_count=source["pick_count"],
            )
        )

    random.shuffle(all_questions)
    return all_questions


def parse_question_id(question_id: str):
    if not isinstance(question_id, str) or ":" not in question_id:
        return None, None
    source_key, row_id_raw = question_id.split(":", 1)
    try:
        return source_key, int(row_id_raw)
    except ValueError:
        return None, None

# --- HTML Frontend Embedded ---
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Databricks Certified Generative AI Engineer Associate</title>
    <style>
        :root {
            --nv-green: #76b900;
            --nv-dark: #1a1a1a;
            --nv-light: #f4f4f4;
            --nv-gray: #333;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--nv-light);
            color: var(--nv-dark);
            margin: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        header {
            background-color: var(--nv-dark);
            color: white;
            width: 100%;
            padding: 20px 0;
            text-align: center;
            border-bottom: 4px solid var(--nv-green);
        }
        header h1 { margin: 0; font-size: 24px; }
        .container {
            background: white;
            width: 90%;
            max-width: 1200px;
            margin: 40px auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            padding: 30px;
        }
        .top-strip {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }
        .progress {
            font-weight: bold;
            color: var(--nv-gray);
            font-size: 14px;
        }
        .timer-pill {
            background: var(--nv-dark);
            color: #fff;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 14px;
            letter-spacing: 0.5px;
        }
        .quiz-layout {
            display: flex;
            gap: 24px;
            align-items: flex-start;
        }
        .question-panel {
            flex: 1;
            min-width: 0;
        }
        .navigator-panel {
            width: 280px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px;
            background: #fafafa;
        }
        .navigator-title {
            font-size: 14px;
            font-weight: 700;
            color: var(--nv-gray);
            margin-bottom: 10px;
        }
        .question-nav-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 8px;
        }
        .qnav-btn {
            border: 1px solid #d1d5db;
            border-radius: 6px;
            height: 34px;
            cursor: pointer;
            font-weight: 700;
            font-size: 12px;
            color: #1f2937;
        }
        .qnav-unseen { background: #e5e7eb; }
        .qnav-seen { background: #fde68a; }
        .qnav-attempted { background: var(--nv-green); color: #fff; border-color: #5d9300; }
        .qnav-current { outline: 2px solid var(--nv-dark); }
        .legend {
            display: flex;
            gap: 10px;
            margin-top: 12px;
            flex-wrap: wrap;
            font-size: 12px;
            color: #4b5563;
        }
        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .legend-dot {
            width: 11px;
            height: 11px;
            border-radius: 50%;
            border: 1px solid #9ca3af;
        }
        .question-text {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            line-height: 1.5;
        }
        .options {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 30px;
        }
        .option-label {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            border: 2px solid #ddd;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .option-label:hover { border-color: var(--nv-green); }
        .option-label input { margin-right: 15px; transform: scale(1.2); cursor: pointer; }
        .option-label.selected { border-color: var(--nv-green); background-color: #eef7e5; }
        
        .nav-buttons {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }
        button {
            padding: 10px 24px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-secondary { background-color: #ddd; color: var(--nv-dark); }
        .btn-secondary:hover:not(:disabled) { background-color: #ccc; }
        .btn-primary { background-color: var(--nv-green); color: white; }
        .btn-primary:hover { background-color: #5d9300; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        
        /* Results Styles */
        .result-container { display: none; }
        .score-banner {
            background: var(--nv-dark); color: white;
            padding: 20px; text-align: center; border-radius: 6px;
            margin-bottom: 30px; border-left: 5px solid var(--nv-green);
        }
        .review-item { margin-bottom: 20px; padding: 15px; border-radius: 6px; border: 1px solid #ddd; }
        .correct-ans { background-color: #eef7e5; border-color: var(--nv-green); }
        .wrong-ans { background-color: #fceceb; border-color: #d9534f; }
        .review-ans-text { font-size: 14px; margin-top: 10px; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-size: 12px; margin-right: 10px; }
        .bg-green { background: var(--nv-green); }
        .bg-red { background: #d9534f; }

        @media (max-width: 980px) {
            .quiz-layout {
                flex-direction: column;
            }
            .navigator-panel {
                width: 100%;
            }
            .question-nav-grid {
                grid-template-columns: repeat(10, minmax(0, 1fr));
            }
        }
        @media (max-width: 700px) {
            .question-nav-grid {
                grid-template-columns: repeat(6, minmax(0, 1fr));
            }
        }
    </style>
</head>
<body>

    <header>
        <h1>Databricks Certified Generative AI Engineer Associate</h1>
    </header>

    <div class="container" id="exam-container">
        <div id="loading">Loading exam questions...</div>
        
        <div id="quiz-ui" style="display: none;">
            <div class="top-strip">
                <div class="progress" id="progress-text">Question 1 of 50</div>
                <div class="timer-pill" id="timer-text">60:00</div>
            </div>

            <div class="quiz-layout">
                <div class="question-panel">
                    <div class="question-text" id="question-text"></div>
                    <div id="question-hint" style="margin-bottom: 10px; color: #666; font-size: 14px;"></div>
                    <div class="options" id="options-container"></div>
                    
                    <div class="nav-buttons">
                        <button class="btn-secondary" id="btn-prev" onclick="prevQuestion()">Previous</button>
                        <button class="btn-primary" id="btn-next" onclick="nextQuestion()">Next</button>
                        <button class="btn-primary" id="btn-submit" onclick="submitExam(false)" style="display:none;">Submit Exam</button>
                    </div>
                </div>

                <aside class="navigator-panel">
                    <div class="navigator-title">Question Navigator</div>
                    <div class="question-nav-grid" id="question-nav-grid"></div>
                    <div class="legend">
                        <span class="legend-item"><span class="legend-dot" style="background:#76b900;"></span>Attempted</span>
                        <span class="legend-item"><span class="legend-dot" style="background:#fde68a;"></span>Seen</span>
                        <span class="legend-item"><span class="legend-dot" style="background:#e5e7eb;"></span>Unseen</span>
                    </div>
                </aside>
            </div>
        </div>

        <div class="result-container" id="result-ui">
            <div class="score-banner">
                <h2 id="score-text"></h2>
            </div>
            <div id="review-container"></div>
            <button class="btn-primary" onclick="location.reload()" style="margin-top: 20px;">Take Another Exam</button>
        </div>
    </div>

    <script>
        const EXAM_DURATION_SECONDS = 90 * 60;
        let questions = [];
        let currentIdx = 0;
        let userAnswers = {}; // Format: { "question_id": ["A"] } or ["A", "C"]
        let seenQuestions = new Set();
        let remainingSeconds = EXAM_DURATION_SECONDS;
        let timerInterval = null;
        let isSubmitting = false;

        // Fetch Exam Data
        window.onload = async () => {
            try {
                const res = await fetch('/api/exam');
                questions = await res.json();
                
                if (questions.length === 0) {
                    document.getElementById('loading').innerText = "No questions found in the CSV files.";
                    return;
                }

                document.getElementById('loading').style.display = 'none';
                document.getElementById('quiz-ui').style.display = 'block';
                renderQuestionNavigator();
                startTimer();
                renderQuestion();
            } catch (err) {
                document.getElementById('loading').innerText = "Error loading exam. Make sure backend is running and CSV files exist.";
            }
        };

        function formatRemainingTime(totalSeconds) {
            const minutes = Math.floor(totalSeconds / 60);
            const seconds = totalSeconds % 60;
            return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        }

        function updateTimerUI() {
            document.getElementById('timer-text').innerText = formatRemainingTime(remainingSeconds);
        }

        function handleTimerExpired() {
            if (timerInterval) {
                clearInterval(timerInterval);
                timerInterval = null;
            }
            if (!isSubmitting) {
                alert("Time is up. Submitting your exam now.");
                submitExam(true);
            }
        }

        function startTimer() {
            updateTimerUI();
            timerInterval = setInterval(() => {
                if (remainingSeconds <= 0) {
                    handleTimerExpired();
                    return;
                }

                remainingSeconds -= 1;
                updateTimerUI();

                if (remainingSeconds <= 0) {
                    handleTimerExpired();
                }
            }, 1000);
        }

        function goToQuestion(index) {
            if (index < 0 || index >= questions.length) {
                return;
            }
            currentIdx = index;
            renderQuestion();
        }

        function renderQuestionNavigator() {
            const grid = document.getElementById('question-nav-grid');
            grid.innerHTML = '';

            questions.forEach((q, idx) => {
                const answered = Array.isArray(userAnswers[q.id]) && userAnswers[q.id].length > 0;
                const seen = seenQuestions.has(q.id);

                let stateClass = 'qnav-unseen';
                if (answered) {
                    stateClass = 'qnav-attempted';
                } else if (seen) {
                    stateClass = 'qnav-seen';
                }

                const button = document.createElement('button');
                button.type = 'button';
                button.className = `qnav-btn ${stateClass} ${idx === currentIdx ? 'qnav-current' : ''}`;
                button.innerText = String(idx + 1);
                button.onclick = () => goToQuestion(idx);
                grid.appendChild(button);
            });
        }

        function renderQuestion() {
            const q = questions[currentIdx];
            seenQuestions.add(q.id);

            document.getElementById('progress-text').innerText = `Question ${currentIdx + 1} of ${questions.length}`;
            document.getElementById('question-text').innerText = q.question;

            const optionsContainer = document.getElementById('options-container');
            optionsContainer.innerHTML = '';
            document.getElementById('question-hint').innerText = q.allow_multiple
                ? 'Choose all correct options.'
                : 'Choose one option.';

            const selectedKeys = userAnswers[q.id] || [];

            for (const [key, text] of Object.entries(q.options)) {
                const isSelected = selectedKeys.includes(key);

                const label = document.createElement('label');
                label.className = `option-label ${isSelected ? 'selected' : ''}`;

                const input = document.createElement('input');
                input.type = q.allow_multiple ? 'checkbox' : 'radio';
                input.name = q.allow_multiple ? `option-${q.id}-${key}` : `option-${q.id}`;
                input.value = key;
                input.checked = isSelected;
                input.onchange = () => handleSelect(q.id, key, q.allow_multiple);

                label.appendChild(input);
                label.appendChild(document.createTextNode(` ${key}. ${text}`));
                optionsContainer.appendChild(label);
            }

            document.getElementById('btn-prev').disabled = currentIdx === 0;

            if (currentIdx === questions.length - 1) {
                document.getElementById('btn-next').style.display = 'none';
                document.getElementById('btn-submit').style.display = 'block';
            } else {
                document.getElementById('btn-next').style.display = 'block';
                document.getElementById('btn-submit').style.display = 'none';
            }

            renderQuestionNavigator();
        }

        function handleSelect(qId, selectedKey, allowMultiple) {
            const existing = userAnswers[qId] ? [...userAnswers[qId]] : [];

            if (allowMultiple) {
                if (existing.includes(selectedKey)) {
                    userAnswers[qId] = existing.filter((k) => k !== selectedKey);
                } else {
                    existing.push(selectedKey);
                    userAnswers[qId] = existing;
                }
            } else {
                userAnswers[qId] = [selectedKey];
            }

            renderQuestion();
        }

        function nextQuestion() {
            if (currentIdx < questions.length - 1) {
                currentIdx++;
                renderQuestion();
            }
        }

        function prevQuestion() {
            if (currentIdx > 0) {
                currentIdx--;
                renderQuestion();
            }
        }

        async function submitExam(isAutoSubmit = false) {
            if (isSubmitting) {
                return;
            }

            const answeredCount = Object.values(userAnswers).filter((v) => Array.isArray(v) && v.length > 0).length;
            if (!isAutoSubmit && answeredCount < questions.length) {
                if (!confirm("You have unanswered questions. Are you sure you want to submit?")) return;
            }

            isSubmitting = true;
            if (timerInterval) {
                clearInterval(timerInterval);
                timerInterval = null;
            }

            document.getElementById('quiz-ui').style.display = 'none';
            document.getElementById('loading').style.display = 'block';
            document.getElementById('loading').innerText = "Grading exam...";

            try {
                const res = await fetch('/api/submit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question_ids: questions.map((q) => q.id),
                        user_answers: userAnswers
                    })
                });
                const result = await res.json();
                showResults(result);
            } catch (err) {
                alert("Error submitting exam.");
                location.reload();
            }
        }

        function showResults(result) {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('result-ui').style.display = 'block';

            document.getElementById('score-text').innerText = `You scored ${result.score} out of ${result.total}`;

            const reviewContainer = document.getElementById('review-container');
            reviewContainer.innerHTML = '';

            result.details.forEach((item, idx) => {
                const isCorrect = item.is_correct;
                const div = document.createElement('div');
                div.className = `review-item ${isCorrect ? 'correct-ans' : 'wrong-ans'}`;

                const badge = `<span class="badge ${isCorrect ? 'bg-green' : 'bg-red'}">${isCorrect ? 'Correct' : 'Incorrect'}</span>`;

                div.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 10px;">${idx + 1}. ${item.question}</div>
                    ${badge}
                    <div class="review-ans-text">
                        <strong>Your Answer:</strong> ${item.user_answer_display}<br>
                        <strong>Correct Answer:</strong> ${item.correct_answer_display}
                    </div>
                `;
                reviewContainer.appendChild(div);
            });
        }
    </script>
</body>
</html>
"""

# --- Endpoints ---

@router.get("/web", response_class=HTMLResponse)
async def serve_web_ui():
    """Serves the frontend Application"""
    return HTMLResponse(content=HTML_CONTENT)

@router.get("/api/exam")
async def get_exam_questions():
    """Generates a list of 50 prioritized questions"""
    return load_and_prioritize_questions()

@router.post("/api/submit")
async def submit_exam(data: SubmitRequest):
    """Grades the exam and returns results"""
    source_dataframes = {}
    source_paths = {}
    for source in QUESTION_SOURCES:
        csv_path = source["path"]
        if os.path.exists(csv_path):
            source_dataframes[source["key"]] = ensure_times_asked_column(pd.read_csv(csv_path))
            source_paths[source["key"]] = csv_path

    if not source_dataframes:
        return {"score": 0, "total": 0, "details": []}
    
    score = 0
    details = []
    asked_questions_count = 0
    seen_ids = set()
    picked_row_ids_by_source: dict[str, set[int]] = {}

    # Grade all questions that were actually shown in this exam.
    for q_id in data.question_ids:
        if q_id in seen_ids:
            continue
        seen_ids.add(q_id)

        try:
            source_key, row_id = parse_question_id(q_id)
            if source_key is None:
                continue

            df = source_dataframes.get(source_key)
            if df is None:
                continue

            row = df.loc[row_id]
            picked_row_ids_by_source.setdefault(source_key, set()).add(row_id)
            options = extract_options(row)
            user_keys = normalize_user_answer(data.user_answers.get(q_id, []))
            correct_keys = sorted(parse_option_keys(row.get("Answer", "")))

            is_correct = user_keys == correct_keys
            if is_correct:
                score += 1

            asked_questions_count += 1
            details.append({
                "question_id": q_id,
                "source": source_key,
                "question": str(row["Question"]).strip(),
                "user_answer_keys": user_keys,
                "correct_answer_keys": correct_keys,
                "user_answer_display": keys_to_display(user_keys, options),
                "correct_answer_display": keys_to_display(correct_keys, options),
                "is_correct": is_correct
            })
        except (KeyError, IndexError, ValueError):
            # Skip invalid IDs safely
            continue

    # Persist "Times Asked" only after a successful submit request.
    for source_key, picked_row_ids in picked_row_ids_by_source.items():
        df = source_dataframes.get(source_key)
        csv_path = source_paths.get(source_key)
        if df is None or not csv_path:
            continue

        valid_row_ids = [row_id for row_id in picked_row_ids if row_id in df.index]
        if not valid_row_ids:
            continue

        df.loc[valid_row_ids, "Times Asked"] = df.loc[valid_row_ids, "Times Asked"] + 1
        try:
            df.to_csv(csv_path, index=False)
        except PermissionError:
            # If the CSV is open in another app (e.g., Excel), keep the exam usable.
            pass

    return {
        "score": score,
        "total": asked_questions_count,
        "details": details
    }

app.include_router(router)

# Instruction on how to run script
if __name__ == "__main__":
    print("Starting Databricks Server...")
    print("Access the UI at: http://127.0.0.1:8002/web")
    uvicorn.run("app:app", host="127.0.0.1", port=8002, reload=True)
