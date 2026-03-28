from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.personality_predictor.charts import build_radar_chart
from src.personality_predictor.config import APP_SUBTITLE, APP_TITLE, MODEL_CONFIGS, QUIZ_DISCLAIMER, TYPE_GROUPS, TYPE_PROFILES
from src.personality_predictor.ml import predict_profile
from src.personality_predictor.quiz import QUESTION_BANK, build_dimension_rows, compose_persona_text, score_answers


def load_css() -> None:
    css_path = Path("assets/styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "stage": "landing",
        "question_index": 0,
        "answers": {},
        "selected_model": "Decision Tree",
        "result": None,
        "prediction": None,
        "persona_text": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_stage(stage: str) -> None:
    st.session_state.stage = stage


def start_quiz() -> None:
    st.session_state.answers = {}
    st.session_state.question_index = 0
    st.session_state.result = None
    st.session_state.prediction = None
    st.session_state.persona_text = ""
    set_stage("quiz")


def restart_quiz() -> None:
    start_quiz()


def choose_answer(letter: str) -> None:
    question = QUESTION_BANK[st.session_state.question_index]
    st.session_state.answers[question.key] = letter
    st.session_state.question_index += 1
    if st.session_state.question_index >= len(QUESTION_BANK):
        finalize_results()


def finalize_results() -> None:
    scored = score_answers(st.session_state.answers)
    persona_text = compose_persona_text(st.session_state.answers, scored)
    try:
        prediction = predict_profile(st.session_state.selected_model, persona_text)
    except FileNotFoundError:
        prediction = {
            "prediction": scored["quiz_type"],
            "confidence": 0.0,
            "top_three": [{"type": scored["quiz_type"], "probability": 100.0}],
            "warning": "Train the model first with python train_model.py to replace the quiz fallback.",
        }

    predicted_type = prediction["prediction"]
    scored["predicted_type"] = predicted_type
    scored["predicted_group"] = TYPE_GROUPS.get(predicted_type, scored["group"])
    scored["predicted_profile"] = TYPE_PROFILES.get(predicted_type, scored["profile"])

    st.session_state.result = scored
    st.session_state.prediction = prediction
    st.session_state.persona_text = persona_text
    set_stage("results")


def render_landing() -> None:
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-copy">
                <span class="eyebrow">MBTI-style personality predictor</span>
                <h1>{APP_TITLE}</h1>
                <p class="hero-text">{APP_SUBTITLE}</p>
                <p class="hero-note">Answer 10 quick questions, generate a profile summary, and let a Decision Tree or KNN model turn that signal into a 16-type prediction.</p>
            </div>
            <div class="hero-panel">
                <div class="mini-card analysts">
                    <span>Analysts</span>
                    <strong>INTJ, INTP, ENTJ, ENTP</strong>
                </div>
                <div class="mini-card diplomats">
                    <span>Diplomats</span>
                    <strong>INFJ, INFP, ENFJ, ENFP</strong>
                </div>
                <div class="mini-card sentinels">
                    <span>Sentinels</span>
                    <strong>ISTJ, ISFJ, ESTJ, ESFJ</strong>
                </div>
                <div class="mini-card explorers">
                    <span>Explorers</span>
                    <strong>ISTP, ISFP, ESTP, ESFP</strong>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9], vertical_alignment="center")
    with left:
        selected = st.radio(
            "Choose a model",
            options=list(MODEL_CONFIGS.keys()),
            index=list(MODEL_CONFIGS.keys()).index(st.session_state.selected_model),
            horizontal=True,
            key="model_selector",
        )
        st.session_state.selected_model = selected
    with right:
        st.markdown(
            """
            <div class="support-copy">
                <strong>Why this setup works</strong>
                <p>The quiz captures your preference pattern. A persona summary built from those answers is then classified by a text model trained on the Kaggle MBTI dataset.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.button("Start the 10-question quiz", type="primary", use_container_width=True, on_click=start_quiz)
    st.caption(QUIZ_DISCLAIMER)


def render_quiz() -> None:
    question = QUESTION_BANK[st.session_state.question_index]
    progress = (st.session_state.question_index + 1) / len(QUESTION_BANK)
    st.markdown(
        f"""
        <div class="progress-shell">
            <div class="progress-meta">
                <span>Question {st.session_state.question_index + 1} of {len(QUESTION_BANK)}</span>
                <span>{int(progress * 100)}%</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{progress * 100:.0f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="question-card">
            <span class="chip">{question.dimension}</span>
            <h2>{question.prompt}</h2>
            <p>Choose the answer that feels more natural most of the time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(2, gap="large")
    for idx, column in enumerate(columns):
        choice = question.choices[idx]
        with column:
            st.markdown(f"<div class='answer-label'><strong>{choice.label}</strong><p>{choice.detail}</p></div>", unsafe_allow_html=True)
            st.button(
                f"Choose {choice.label}",
                key=f"{question.key}_{choice.letter}",
                use_container_width=True,
                on_click=choose_answer,
                args=(choice.letter,),
            )

    if st.session_state.question_index > 0:
        if st.button("Back", use_container_width=False):
            st.session_state.question_index -= 1
            previous_question = QUESTION_BANK[st.session_state.question_index]
            st.session_state.answers.pop(previous_question.key, None)
            st.rerun()


def render_results() -> None:
    result = st.session_state.result
    prediction = st.session_state.prediction
    predicted_type = result["predicted_type"]
    predicted_group = result["predicted_group"]
    predicted_profile = result["predicted_profile"]
    accent = result["accent"]

    st.markdown(
        f"""
        <div class="result-hero group-{predicted_group.lower()}">
            <span class="eyebrow">{predicted_group}</span>
            <h1>{predicted_type}</h1>
            <h3>{predicted_profile['title']}</h3>
            <p>{predicted_profile['summary']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([1.1, 0.9], gap="large")
    with top_left:
        st.markdown(
            f"""
            <div class="result-card">
                <h3>Prediction summary</h3>
                <p><strong>Model:</strong> {st.session_state.selected_model}</p>
                <p><strong>Confidence:</strong> {prediction.get('confidence', 0.0)}%</p>
                <p><strong>Quiz signal:</strong> {result['quiz_type']}</p>
                <p><strong>Persona summary:</strong> {st.session_state.persona_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "warning" in prediction:
            st.warning(prediction["warning"])
    with top_right:
        rows = build_dimension_rows(result)
        chart = build_radar_chart(rows, accent)
        st.pyplot(chart, clear_figure=True)

    st.markdown("<h3 class='section-title'>Dimension readout</h3>", unsafe_allow_html=True)
    for row in build_dimension_rows(result):
        st.markdown(
            f"""
            <div class="dimension-row">
                <div class="dimension-copy">
                    <strong>{row['title']}</strong>
                    <span>{row['left_label']} {row['left_pct']}% / {row['right_label']} {row['right_pct']}%</span>
                </div>
                <div class="dimension-track">
                    <div class="dimension-fill" style="width:{max(row['left_pct'], row['right_pct'])}%; background:{accent};"></div>
                </div>
                <span class="dimension-winner">{row['winner']} leaning</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h3 class='section-title'>Top candidate types</h3>", unsafe_allow_html=True)
    leaderboard = st.columns(3, gap="medium")
    for column, candidate in zip(leaderboard, prediction["top_three"]):
        candidate_profile = TYPE_PROFILES.get(candidate["type"], {"title": "Type profile", "summary": ""})
        with column:
            st.markdown(
                f"""
                <div class="candidate-card">
                    <span>{candidate['probability']}%</span>
                    <strong>{candidate['type']}</strong>
                    <p>{candidate_profile['title']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.button("Retake quiz", use_container_width=True, on_click=restart_quiz)
    st.caption(QUIZ_DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=".", layout="wide")
    load_css()
    init_state()

    if st.session_state.stage == "landing":
        render_landing()
    elif st.session_state.stage == "quiz":
        render_quiz()
    else:
        render_results()


if __name__ == "__main__":
    main()

