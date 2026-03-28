from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.personality_predictor.charts import (
    build_confusion_heatmap,
    build_dimension_balance_chart,
    build_group_donut_chart,
    build_model_comparison_chart,
    build_probability_chart,
    build_radar_chart,
    build_type_distribution_chart,
)
from src.personality_predictor.config import (
    APP_SUBTITLE,
    APP_TAGLINE,
    APP_TITLE,
    GROUP_ACCENTS,
    GROUP_ORDER,
    MODEL_CONFIGS,
    QUIZ_DISCLAIMER,
    TYPE_GROUPS,
    TYPE_PROFILES,
)
from src.personality_predictor.ml import load_metrics, predict_profile, summarize_dataset
from src.personality_predictor.quiz import QUESTION_BANK, build_dimension_rows, compose_persona_text, score_answers


@st.cache_data(show_spinner=False)
def get_metrics() -> dict[str, object]:
    return load_metrics()


@st.cache_data(show_spinner=False)
def get_dataset_summary() -> dict[str, object]:
    metrics = get_metrics()
    if metrics.get("dataset_summary"):
        return metrics["dataset_summary"]
    try:
        return summarize_dataset()
    except Exception:
        return {"rows": 0, "type_counts": {}, "group_counts": {}, "dominant_type": "N/A", "type_count": 0}


def load_css() -> None:
    css_path = Path("assets/styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "stage": "landing",
        "question_index": 0,
        "answers": {},
        "selected_model": "KNN",
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
            "top_six": [{"type": scored["quiz_type"], "probability": 100.0}],
            "group_probabilities": {scored["group"]: 100.0},
            "warning": "Model artifacts are missing. Run python train_model.py to restore trained predictions.",
        }

    predicted_type = prediction["prediction"]
    predicted_group = TYPE_GROUPS.get(predicted_type, scored["group"])
    scored["predicted_type"] = predicted_type
    scored["predicted_group"] = predicted_group
    scored["predicted_profile"] = TYPE_PROFILES.get(predicted_type, scored["profile"])
    scored["type_match"] = predicted_type == scored["quiz_type"]

    st.session_state.result = scored
    st.session_state.prediction = prediction
    st.session_state.persona_text = persona_text
    set_stage("results")


def model_score_text(model_name: str, metrics: dict[str, object]) -> str:
    value = metrics.get("models", {}).get(model_name, {})
    if not value:
        return "Not trained yet"
    return f"{value['accuracy'] * 100:.1f}% accuracy"


def render_stat_pill(label: str, value: str, tone: str = "") -> None:
    extra = f" {tone}" if tone else ""
    st.markdown(
        f"""
        <div class="stat-pill{extra}">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing() -> None:
    metrics = get_metrics()
    dataset_summary = get_dataset_summary()

    st.markdown(
        f"""
        <section class="poster-shell">
            <div class="poster-copy">
                <span class="eyebrow">Streamlit • MBTI • Decision Tree + KNN</span>
                <h1>{APP_TITLE}</h1>
                <p class="hero-text">{APP_SUBTITLE}</p>
                <p class="hero-note">{APP_TAGLINE}</p>
            </div>
            <div class="poster-board">
                <div class="board-card analysts">
                    <span>Analysts</span>
                    <strong>Logic + strategy</strong>
                    <p>INTJ, INTP, ENTJ, ENTP</p>
                </div>
                <div class="board-card diplomats">
                    <span>Diplomats</span>
                    <strong>Empathy + vision</strong>
                    <p>INFJ, INFP, ENFJ, ENFP</p>
                </div>
                <div class="board-card sentinels">
                    <span>Sentinels</span>
                    <strong>Stability + care</strong>
                    <p>ISTJ, ISFJ, ESTJ, ESFJ</p>
                </div>
                <div class="board-card explorers">
                    <span>Explorers</span>
                    <strong>Action + spontaneity</strong>
                    <p>ISTP, ISFP, ESTP, ESFP</p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    pill_columns = st.columns(4)
    with pill_columns[0]:
        render_stat_pill("Dataset rows", f"{dataset_summary.get('rows', 0):,}")
    with pill_columns[1]:
        render_stat_pill("MBTI labels", str(dataset_summary.get("type_count", 16)))
    with pill_columns[2]:
        render_stat_pill("Best model", max(metrics.get("models", {"KNN": {"accuracy": 0}}), key=lambda name: metrics.get("models", {}).get(name, {}).get("accuracy", 0)))
    with pill_columns[3]:
        render_stat_pill("Dominant type", str(dataset_summary.get("dominant_type", "INTP")))

    selector_col, summary_col = st.columns([1.1, 0.9], gap="large")
    with selector_col:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        selected = st.radio(
            "Choose a model for inference",
            options=list(MODEL_CONFIGS.keys()),
            index=list(MODEL_CONFIGS.keys()).index(st.session_state.selected_model),
            horizontal=True,
            key="model_selector",
        )
        st.session_state.selected_model = selected
        st.markdown(
            f"""
            <div class="support-copy">
                <strong>Selected model</strong>
                <p>{selected} is ready to score the persona summary generated from your quiz answers. Current benchmark: {model_score_text(selected, metrics)}.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Start the 10-question quiz", type="primary", use_container_width=True, on_click=start_quiz)
        st.caption(QUIZ_DISCLAIMER)
        st.markdown("</div>", unsafe_allow_html=True)
    with summary_col:
        st.markdown(
            """
            <div class="glass-panel compact">
                <strong class="panel-kicker">What the app does</strong>
                <p class="panel-text">Your answers are converted into MBTI dimension scores, then into a persona paragraph that a text classifier can understand.</p>
                <p class="panel-text">The result screen compares the quiz signal, model prediction, and family-level probabilities so the outcome feels interpretable instead of black-box.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_models = metrics.get("models", {})
    if metric_models:
        st.markdown("<h3 class='section-title'>Model performance snapshot</h3>", unsafe_allow_html=True)
        left, right = st.columns([0.95, 1.05], gap="large")
        with left:
            st.pyplot(build_model_comparison_chart(metric_models), clear_figure=True)
        with right:
            for model_name, model_metrics in metric_models.items():
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <span>{model_name}</span>
                        <strong>{model_metrics['accuracy'] * 100:.1f}% accuracy</strong>
                        <p>Weighted F1: {model_metrics['weighted_avg_f1'] * 100:.1f}% · Train sample: {model_metrics['sample_size']:,} · Fit time: {model_metrics.get('fit_seconds', 0)}s</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if dataset_summary.get("type_counts"):
        st.markdown("<h3 class='section-title'>Dataset view</h3>", unsafe_allow_html=True)
        left, right = st.columns([1.15, 0.85], gap="large")
        with left:
            st.pyplot(build_type_distribution_chart(dataset_summary["type_counts"]), clear_figure=True)
        with right:
            st.pyplot(build_group_donut_chart(dataset_summary["group_counts"]), clear_figure=True)


def render_quiz() -> None:
    question = QUESTION_BANK[st.session_state.question_index]
    progress = (st.session_state.question_index + 1) / len(QUESTION_BANK)
    answered = st.session_state.question_index

    st.markdown(
        f"""
        <div class="quiz-hero">
            <div>
                <span class="eyebrow">Question flow</span>
                <h1>Shape your personality signal</h1>
                <p class="hero-note">Pick the option that feels most natural most of the time. The app converts these choices into four MBTI dimensions and a model-ready persona summary.</p>
            </div>
            <div class="quiz-sidecar">
                <span>{answered} answered</span>
                <strong>{len(QUESTION_BANK) - answered} left</strong>
            </div>
        </div>
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
        <div class="question-card elevated">
            <span class="chip">{question.dimension}</span>
            <h2>{question.prompt}</h2>
            <p>Choose the answer that feels more like your natural default.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(2, gap="large")
    for idx, column in enumerate(columns):
        choice = question.choices[idx]
        with column:
            st.markdown(
                f"""
                <div class="answer-panel">
                    <span class="answer-badge">{choice.letter}</span>
                    <strong>{choice.label}</strong>
                    <p>{choice.detail}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                f"Choose {choice.label}",
                key=f"{question.key}_{choice.letter}",
                use_container_width=True,
                on_click=choose_answer,
                args=(choice.letter,),
            )

    nav_left, nav_right = st.columns([0.22, 0.78])
    with nav_left:
        if st.session_state.question_index > 0 and st.button("Back", use_container_width=True):
            st.session_state.question_index -= 1
            previous_question = QUESTION_BANK[st.session_state.question_index]
            st.session_state.answers.pop(previous_question.key, None)
            st.rerun()
    with nav_right:
        st.markdown(
            f"""
            <div class="helper-strip">
                <span>Selected model: <strong>{st.session_state.selected_model}</strong></span>
                <span>Result cards and charts appear immediately after question 10.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_results() -> None:
    result = st.session_state.result
    prediction = st.session_state.prediction
    metrics = get_metrics()

    predicted_type = result["predicted_type"]
    predicted_group = result["predicted_group"]
    predicted_profile = result["predicted_profile"]
    accent = GROUP_ACCENTS.get(predicted_group, result["accent"])
    rows = build_dimension_rows(result)

    st.markdown(
        f"""
        <div class="result-hero group-{predicted_group.lower()}">
            <div>
                <span class="eyebrow">{predicted_group}</span>
                <h1>{predicted_type}</h1>
                <h3>{predicted_profile['title']}</h3>
                <p>{predicted_profile['summary']}</p>
            </div>
            <div class="result-hero-side">
                <span>Model confidence</span>
                <strong>{prediction.get('confidence', 0.0)}%</strong>
                <p>Quiz signal: {result['quiz_type']} · {'Match' if result['type_match'] else 'Model diverged from quiz signal'}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero_metrics = st.columns(4)
    with hero_metrics[0]:
        render_stat_pill("Predicted family", predicted_group, "accent")
    with hero_metrics[1]:
        render_stat_pill("Quiz signal", result["quiz_type"])
    with hero_metrics[2]:
        render_stat_pill("Confidence", f"{prediction.get('confidence', 0.0)}%")
    with hero_metrics[3]:
        render_stat_pill("Inference model", st.session_state.selected_model)

    overview_tab, viz_tab, eval_tab = st.tabs(["Overview", "Visuals", "Model Insights"])

    with overview_tab:
        left, right = st.columns([1.05, 0.95], gap="large")
        with left:
            st.markdown(
                f"""
                <div class="result-card rich">
                    <h3>Persona summary</h3>
                    <p>{st.session_state.persona_text}</p>
                    <div class="detail-grid">
                        <div><span>Strengths</span><strong>{predicted_profile.get('strengths', 'Not available')}</strong></div>
                        <div><span>Working style</span><strong>{predicted_profile.get('collab', 'Not available')}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "warning" in prediction:
                st.warning(prediction["warning"])
        with right:
            st.pyplot(build_radar_chart(rows, accent), clear_figure=True)

        st.markdown("<h3 class='section-title'>Dimension breakdown</h3>", unsafe_allow_html=True)
        for row in rows:
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

    with viz_tab:
        top_left, top_right = st.columns([1.05, 0.95], gap="large")
        with top_left:
            st.pyplot(build_probability_chart(prediction.get("top_six", prediction["top_three"]), accent), clear_figure=True)
        with top_right:
            st.pyplot(build_group_donut_chart(prediction.get("group_probabilities", {predicted_group: 100.0})), clear_figure=True)

        lower_left, lower_right = st.columns([1.0, 1.0], gap="large")
        with lower_left:
            st.pyplot(build_dimension_balance_chart(rows, accent), clear_figure=True)
        with lower_right:
            leaderboard = prediction.get("top_six", prediction["top_three"])
            st.markdown("<div class='leaderboard-shell'>", unsafe_allow_html=True)
            for candidate in leaderboard:
                candidate_profile = TYPE_PROFILES.get(candidate["type"], {"title": "Type profile"})
                st.markdown(
                    f"""
                    <div class="candidate-card tall">
                        <span>{candidate['probability']}%</span>
                        <strong>{candidate['type']}</strong>
                        <p>{candidate_profile['title']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    with eval_tab:
        model_metrics = metrics.get("models", {}).get(st.session_state.selected_model)
        if not model_metrics:
            st.info("No training metrics found yet. Run python train_model.py to generate evaluation charts.")
        else:
            left, right = st.columns([1.15, 0.85], gap="large")
            with left:
                st.pyplot(
                    build_confusion_heatmap(
                        model_metrics["confusion_matrix"],
                        model_metrics["labels"],
                        f"{st.session_state.selected_model} confusion matrix",
                    ),
                    clear_figure=True,
                )
            with right:
                st.markdown("<div class='glass-panel compact'>", unsafe_allow_html=True)
                st.markdown("**Most confused type pairs**")
                for row in model_metrics.get("top_confusions", [])[:6]:
                    st.markdown(
                        f"""
                        <div class="confusion-row">
                            <span>{row['actual']} → {row['predicted']}</span>
                            <strong>{row['count']}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                weakest = model_metrics.get("class_scores", {}).get("weakest_f1", [])[:4]
                if weakest:
                    st.markdown("<div class='glass-panel compact'>", unsafe_allow_html=True)
                    st.markdown("**Lowest F1 classes**")
                    for row in weakest:
                        st.markdown(
                            f"""
                            <div class="confusion-row">
                                <span>{row['type']}</span>
                                <strong>{row['f1'] * 100:.1f}%</strong>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

    st.button("Retake quiz", use_container_width=True, on_click=restart_quiz)
    st.caption(QUIZ_DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
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
