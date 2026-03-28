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
    MODEL_CONFIGS,
    QUIZ_DISCLAIMER,
    TYPE_GROUPS,
    TYPE_PROFILES,
)
from src.personality_predictor.ml import load_metrics, predict_profile, summarize_dataset
from src.personality_predictor.quiz import QUESTION_BANK, build_dimension_rows, compose_persona_text, score_answers


@st.cache_data(show_spinner=False, ttl=10)
def get_metrics() -> dict[str, object]:
    return load_metrics()


@st.cache_data(show_spinner=False, ttl=120)
def get_dataset_summary() -> dict[str, object]:
    metrics = get_metrics()
    if metrics.get("dataset_summary"):
        return metrics["dataset_summary"]
    try:
        return summarize_dataset()
    except Exception:
        return {"rows": 0, "type_counts": {}, "group_counts": {}, "dominant_type": "N/A", "type_count": 0}


def has_valid_benchmarks(metrics: dict[str, object]) -> bool:
    rows_used = int(metrics.get("rows_used", 0) or 0)
    model_rows = metrics.get("models", {})
    if rows_used < 20000 or not model_rows:
        return False
    for values in model_rows.values():
        if len(values.get("class_scores", {}).get("all", [])) < 8:
            return False
    return True


def load_css() -> None:
    css_path = Path("assets/styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_top_logo() -> None:
    logo_path = Path("assets/cute-logo.svg")
    if not logo_path.exists():
        return
    logo_svg = logo_path.read_text(encoding="utf-8")
    st.markdown(
        f"""
        <div class="top-brand-logo" aria-label="App logo">
            {logo_svg}
            <span>Persona Bloom</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    if not QUESTION_BANK:
        return

    index = int(st.session_state.get("question_index", 0) or 0)
    if index >= len(QUESTION_BANK):
        if st.session_state.get("stage") == "quiz" and st.session_state.get("result") is None:
            finalize_results()
        return

    question = QUESTION_BANK[index]
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
    if not has_valid_benchmarks(metrics):
        return "Preview metrics only"
    return f"{value['accuracy'] * 100:.1f}% accuracy"


def derive_top_confusions(
    confusion_matrix_data: list[list[int]], labels: list[str], limit: int = 6
) -> list[dict[str, object]]:
    """Build the largest off-diagonal confusion pairs directly from the matrix."""
    pairs: list[dict[str, object]] = []
    for actual_idx, row in enumerate(confusion_matrix_data):
        for predicted_idx, count in enumerate(row):
            if actual_idx == predicted_idx or count <= 0:
                continue
            pairs.append(
                {
                    "actual": labels[actual_idx],
                    "predicted": labels[predicted_idx],
                    "count": int(count),
                }
            )
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:limit]


def derive_weakest_f1(class_scores: dict[str, object], limit: int = 4) -> list[dict[str, object]]:
    """Sort class scores by F1 so the list always reflects lowest performers."""
    rows = class_scores.get("all", []) if isinstance(class_scores, dict) else []
    valid_rows = [row for row in rows if isinstance(row, dict) and "f1" in row and "type" in row]
    valid_rows.sort(key=lambda item: float(item.get("f1", 0.0)))
    return valid_rows[:limit]


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
    valid_benchmarks = has_valid_benchmarks(metrics)

    st.markdown(
        f"""
        <section class="poster-shell">
            <div class="poster-copy">
                <span class="eyebrow">Streamlit - MBTI - Decision Tree + KNN</span>
                <h1>{APP_TITLE}</h1>
                <p class="hero-text">{APP_SUBTITLE}</p>
                <p class="hero-note">{APP_TAGLINE}</p>
            </div>
            <div class="poster-board">
                <div class="board-card analysts">
                    <span>🧠 Analysts</span>
                    <strong>Logic + strategy</strong>
                </div>
                <div class="board-card diplomats">
                    <span>❤️ Diplomats</span>
                    <strong>Empathy + vision</strong>
                </div>
                <div class="board-card sentinels">
                    <span>🛡️ Sentinels</span>
                    <strong>Stability + care</strong>
                </div>
                <div class="board-card explorers">
                    <span>⚡ Explorers</span>
                    <strong>Action + spontaneity</strong>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    pill_columns = st.columns(4, gap="small")
    with pill_columns[0]:
        render_stat_pill("Dataset", f"{dataset_summary.get('rows', 0):,}")
    with pill_columns[1]:
        render_stat_pill("Types", str(dataset_summary.get("type_count", 16)))
    with pill_columns[2]:
        best_model = max(metrics.get("models", {"KNN": {"accuracy": 0}}), key=lambda name: metrics.get("models", {}).get(name, {}).get("accuracy", 0))
        render_stat_pill("Best", best_model)
    with pill_columns[3]:
        best_acc = max([m.get('accuracy', 0) for m in metrics.get('models', {}).values()], default=0)
        render_stat_pill("Accuracy", f"{best_acc * 100:.0f}%")

    selector_col, summary_col = st.columns([1.1, 0.9], gap="large")
    with selector_col:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='control-label'><strong>Select a Model</strong></div>", unsafe_allow_html=True)
        selected = st.radio(
            "Choose a model for inference",
            options=list(MODEL_CONFIGS.keys()),
            index=list(MODEL_CONFIGS.keys()).index(st.session_state.selected_model),
            horizontal=True,
            label_visibility="collapsed",
            key="model_selector",
        )
        st.session_state.selected_model = selected
        st.button("🚀 Start Quiz", type="primary", use_container_width=True, on_click=start_quiz)
        st.caption("⏱️ Takes ~2 mins • No data saved")
        st.markdown("</div>", unsafe_allow_html=True)
    with summary_col:
        st.markdown(
            f"""
            <div class="glass-panel compact">
                <strong class="panel-kicker">{selected}</strong>
                <p class="panel-text"><strong>{model_score_text(selected, metrics)}</strong></p>
                <p class="panel-text" style="font-size: 0.85rem; margin-top: 0.8rem;">Sample: {metrics.get('models', {}).get(selected, {}).get('sample_size', 0):,}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    metric_models = metrics.get("models", {})
    if metric_models and valid_benchmarks:
        st.markdown("<h3 class='section-title'>📊 Dataset Overview</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='info-strip'>
                <div><span>Total Records</span><strong>{dataset_summary.get('rows', 0):,}</strong></div>
                <div><span>Types Represented</span><strong>{dataset_summary.get('type_count', 16)}</strong></div>
                <div><span>Most Common</span><strong>{dataset_summary.get('dominant_type', 'N/A')}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        
        chart_col1, chart_col2 = st.columns([1.0, 1.0], gap="large")
        with chart_col1:
            st.markdown("<div class='chart-label'>Model Accuracy Comparison</div>", unsafe_allow_html=True)
            st.pyplot(build_model_comparison_chart(metric_models), clear_figure=True)
        with chart_col2:
            st.markdown("<div class='chart-label'>Response Distribution by Group</div>", unsafe_allow_html=True)
            if dataset_summary.get("group_counts"):
                st.pyplot(build_group_donut_chart(dataset_summary["group_counts"]), clear_figure=True)
    elif metric_models:
        st.info("Complete dataset training unlocks full performance charts.")
    else:
        st.info("Train models to see performance metrics.")


def render_quiz() -> None:
    if not QUESTION_BANK:
        st.warning("Question bank is empty. Add quiz questions to continue.")
        return

    if st.session_state.question_index >= len(QUESTION_BANK):
        if st.session_state.result is None:
            finalize_results()
        return

    question = QUESTION_BANK[st.session_state.question_index]
    progress = (st.session_state.question_index + 1) / len(QUESTION_BANK)
    answered = st.session_state.question_index

    st.markdown(
        f"""
        <div class="quiz-hero">
            <div>
                <span class="eyebrow">Question {st.session_state.question_index + 1} of {len(QUESTION_BANK)}</span>
                <h1>Answer Honestly</h1>
                <p class="hero-note">Choose what feels most natural to you. No right or wrong answers.</p>
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

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="question-card elevated">
            <span class="chip">{question.dimension}</span>
            <h2>{question.prompt}</h2>
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

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    nav_left, nav_right = st.columns([0.22, 0.78])
    with nav_left:
        if st.session_state.question_index > 0 and st.button("← Back", use_container_width=True):
            st.session_state.question_index -= 1
            previous_question = QUESTION_BANK[st.session_state.question_index]
            st.session_state.answers.pop(previous_question.key, None)
            st.rerun()
    with nav_right:
        st.markdown(
            f"""
            <div class="helper-strip">
                <span style="font-size: 0.9rem;">Using <strong>{st.session_state.selected_model}</strong> • {int(progress * 100)}% complete</span>
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
                <p>Quiz signal: {result['quiz_type']} - {'Match' if result['type_match'] else 'Model diverged from quiz signal'}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero_metrics = st.columns(4, gap="small")
    with hero_metrics[0]:
        render_stat_pill("Group", predicted_group, "accent")
    with hero_metrics[1]:
        render_stat_pill("Quiz", result["quiz_type"])
    with hero_metrics[2]:
        render_stat_pill("Confidence", f"{prediction.get('confidence', 0.0)}%")
    with hero_metrics[3]:
        render_stat_pill("Model", st.session_state.selected_model)

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    overview_tab, viz_tab, eval_tab = st.tabs(["Visuals", "Overview", "Insights"])

    with overview_tab:
        st.markdown("<h3 class='section-title'>Your Persona</h3>", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        left, right = st.columns([1.0, 1.0], gap="large")
        with left:
            st.markdown(
                f"""
                <div class="result-card rich">
                    <div class="card-title">Persona Summary</div>
                    <p>{st.session_state.persona_text}</p>
                    <div class="detail-grid">
                        <div><span>Strengths</span><strong>{predicted_profile.get('strengths', 'Not available')}</strong></div>
                        <div><span>Working Style</span><strong>{predicted_profile.get('collab', 'Not available')}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "warning" in prediction:
                st.warning(prediction["warning"])
        with right:
            st.markdown("<div class='radar-heading'>Dimension Radar</div>", unsafe_allow_html=True)
            st.pyplot(build_radar_chart(rows, accent), clear_figure=True)

        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-title'>Dimension Breakdown</h3>", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
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
        st.markdown("<h3 class='section-title'>Prediction Probabilities</h3>", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        top_left, top_right = st.columns([1.0, 1.0], gap="large")
        with top_left:
            st.markdown("<div class='chart-label'>Top Type Predictions</div>", unsafe_allow_html=True)
            st.pyplot(build_probability_chart(prediction.get("top_six", prediction["top_three"]), accent), clear_figure=True)
        with top_right:
            st.markdown("<div class='chart-label'>Group Distribution</div>", unsafe_allow_html=True)
            st.pyplot(build_group_donut_chart(prediction.get("group_probabilities", {predicted_group: 100.0})), clear_figure=True)

        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-title'>Detailed Analysis</h3>", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        lower_left, lower_right = st.columns([1.0, 1.0], gap="large")
        with lower_left:
            st.markdown("<div class='chart-label'>Dimension Balance</div>", unsafe_allow_html=True)
            st.pyplot(build_dimension_balance_chart(rows, accent), clear_figure=True)
        with lower_right:
            st.markdown("<div class='chart-label'>Type Ranking</div>", unsafe_allow_html=True)
            leaderboard = prediction.get("top_six", prediction["top_three"])
            visible_count = min(4, len(leaderboard))
            st.markdown("<div class='leaderboard-shell'>", unsafe_allow_html=True)
            for candidate in leaderboard[:visible_count]:
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
            if len(leaderboard) > visible_count:
                st.caption(f"Showing top {visible_count} of {len(leaderboard)} predictions")

    with eval_tab:
        model_metrics = metrics.get("models", {}).get(st.session_state.selected_model)
        if not model_metrics or not has_valid_benchmarks(metrics):
            st.info("No full training metrics found yet. Run python train_model.py to generate evaluation charts.")
        else:
            labels = model_metrics.get("labels", [])
            confusion_matrix_data = model_metrics.get("confusion_matrix", [])
            top_confusions = derive_top_confusions(confusion_matrix_data, labels)
            weakest = derive_weakest_f1(model_metrics.get("class_scores", {}))

            st.markdown("<h3 class='section-title'>Model Evaluation Metrics</h3>", unsafe_allow_html=True)
            
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            
            chart_left, chart_right = st.columns([1.0, 1.0], gap="large")
            with chart_left:
                st.markdown("<div class='chart-label'>Confusion Matrix</div>", unsafe_allow_html=True)
                st.pyplot(
                    build_confusion_heatmap(
                        confusion_matrix_data,
                        labels,
                        f"{st.session_state.selected_model} confusion matrix",
                    ),
                    clear_figure=True,
                )
            with chart_right:
                st.markdown("<div class='chart-label'>Performance Analysis</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='analysis-panel'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='panel-title'>Most confused type pairs</div>",
                    unsafe_allow_html=True,
                )
                for row in top_confusions:
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
                
                st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
                
                if weakest:
                    st.markdown("<div class='analysis-panel'>", unsafe_allow_html=True)
                    st.markdown(
                        "<div class='panel-title'>Lowest F1 score classes</div>",
                        unsafe_allow_html=True,
                    )
                    for row in weakest:
                        st.markdown(
                            f"""
                            <div class="confusion-row">
                                <span>{row['type']}</span>
                                <strong>{row['f1']:.1%}</strong>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    st.button("Retake quiz", use_container_width=True, on_click=restart_quiz)
    st.caption(QUIZ_DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    load_css()
    init_state()
    render_top_logo()

    if st.session_state.stage == "landing":
        render_landing()
    elif st.session_state.stage == "quiz":
        render_quiz()
    else:
        render_results()


if __name__ == "__main__":
    main()
