from __future__ import annotations

from dataclasses import dataclass

from .config import DIMENSION_META, GROUP_ACCENTS, TYPE_GROUPS, TYPE_PROFILES


@dataclass(frozen=True)
class Choice:
    letter: str
    label: str
    detail: str
    profile_fragment: str


@dataclass(frozen=True)
class Question:
    key: str
    prompt: str
    dimension: str
    weight: int
    choices: tuple[Choice, Choice]


QUESTION_BANK: tuple[Question, ...] = (
    Question(
        key="q1",
        prompt="What recharges you fastest after a busy week?",
        dimension="IE",
        weight=1,
        choices=(
            Choice("I", "Quiet reset", "A solo evening with a book, playlist, or walk feels best.", "I recharge in private, reflective settings."),
            Choice("E", "Social reset", "Meeting friends or talking things through lifts my energy.", "I regain energy by interacting with people and momentum."),
        ),
    ),
    Question(
        key="q2",
        prompt="When a new project starts, what grabs your attention first?",
        dimension="SN",
        weight=1,
        choices=(
            Choice("S", "Concrete facts", "I look for details, examples, and what is already proven.", "I trust concrete details and real examples before abstraction."),
            Choice("N", "Patterns and possibilities", "I zoom out and look for themes, ideas, and future options.", "I naturally spot patterns, themes, and future possibilities."),
        ),
    ),
    Question(
        key="q3",
        prompt="How do you usually make an important decision?",
        dimension="TF",
        weight=2,
        choices=(
            Choice("T", "Logic first", "I compare evidence, tradeoffs, and what makes the most sense.", "I weigh decisions with logic, fairness, and clear criteria."),
            Choice("F", "Values first", "I consider people, harmony, and what feels aligned.", "I weigh decisions through personal values and human impact."),
        ),
    ),
    Question(
        key="q4",
        prompt="Your ideal weekend plan feels...",
        dimension="JP",
        weight=2,
        choices=(
            Choice("J", "Mapped out", "I like knowing the plan, the timing, and the next step.", "I prefer structure, plans, and clear checkpoints."),
            Choice("P", "Open-ended", "I like room to improvise based on energy and curiosity.", "I prefer flexibility, improvisation, and last-minute freedom."),
        ),
    ),
    Question(
        key="q5",
        prompt="In group discussions, your usual style is...",
        dimension="IE",
        weight=1,
        choices=(
            Choice("I", "Observe then speak", "I think internally first and jump in when I am ready.", "I tend to observe, process internally, and then contribute thoughtfully."),
            Choice("E", "Think out loud", "Talking helps me explore ideas in real time.", "I discover ideas by speaking, reacting, and building in the moment."),
        ),
    ),
    Question(
        key="q6",
        prompt="What kind of information feels most trustworthy?",
        dimension="SN",
        weight=1,
        choices=(
            Choice("S", "What is tangible", "I trust evidence I can verify and apply directly.", "I trust practical, tangible evidence and direct observation."),
            Choice("N", "What it implies", "I care about meaning, direction, and the bigger pattern.", "I focus on implications, symbolism, and the bigger picture."),
        ),
    ),
    Question(
        key="q7",
        prompt="If a friend asks for honest feedback, you usually lead with...",
        dimension="TF",
        weight=1,
        choices=(
            Choice("T", "Straight analysis", "I share the clearest fix, even if it sounds blunt.", "I usually start with direct analysis and problem-solving."),
            Choice("F", "Empathy and context", "I frame the truth carefully so it can be received well.", "I usually start with empathy, tone, and emotional context."),
        ),
    ),
    Question(
        key="q8",
        prompt="Which work style sounds more natural?",
        dimension="JP",
        weight=1,
        choices=(
            Choice("J", "Finish one thing at a time", "I feel calm when tasks are sequenced and closed.", "I like orderly progress, closure, and finishing what I start."),
            Choice("P", "Move between options", "I like exploring multiple paths before locking one in.", "I like open loops, experimentation, and adapting as I go."),
        ),
    ),
    Question(
        key="q9",
        prompt="At a new event where you know almost nobody, you usually...",
        dimension="IE",
        weight=1,
        choices=(
            Choice("I", "Settle in quietly", "I prefer a few meaningful interactions over many quick ones.", "I prefer deeper one-to-one interaction over constant social stimulation."),
            Choice("E", "Start mingling", "Meeting new people quickly feels natural and energizing.", "I warm up by engaging quickly and broadly with new people."),
        ),
    ),
    Question(
        key="q10",
        prompt="When solving a tough problem, what excites you more?",
        dimension="SN",
        weight=1,
        choices=(
            Choice("S", "The workable method", "I want a grounded plan that can be tested now.", "I want proven methods and concrete next actions."),
            Choice("N", "The fresh angle", "I want the idea that reframes the whole challenge.", "I enjoy reframing problems with new perspectives and possibilities."),
        ),
    ),
)

DIMENSION_DEFAULTS = {"IE": "I", "SN": "N", "TF": "T", "JP": "J"}


def get_question(question_key: str) -> Question:
    return next(question for question in QUESTION_BANK if question.key == question_key)


def get_choice(question: Question, letter: str) -> Choice | None:
    for choice in question.choices:
        if choice.letter == letter:
            return choice
    return None


def score_answers(answers: dict[str, str]) -> dict[str, object]:
    letter_scores = {letter: 0 for letter in ["I", "E", "S", "N", "T", "F", "J", "P"]}
    dimension_totals = {dimension: 0 for dimension in DIMENSION_META}
    profile_fragments: list[str] = []

    for question in QUESTION_BANK:
        selected_letter = answers.get(question.key)
        if not selected_letter:
            continue
        selected_choice = get_choice(question, selected_letter)
        if selected_choice is None:
            # Ignore stale or invalid letters from session state.
            continue
        dimension_totals[question.dimension] += question.weight
        letter_scores[selected_letter] += question.weight
        profile_fragments.append(selected_choice.profile_fragment)

    dimension_scores: dict[str, dict[str, object]] = {}
    letters: list[str] = []

    for dimension, meta in DIMENSION_META.items():
        left = meta["left"]
        right = meta["right"]
        total = max(dimension_totals[dimension], 1)
        left_score = letter_scores[left]
        right_score = letter_scores[right]
        if left_score == right_score:
            winner = DIMENSION_DEFAULTS[dimension]
        else:
            winner = left if left_score > right_score else right
        confidence = round((max(left_score, right_score) / total) * 100, 1)
        dimension_scores[dimension] = {
            "winner": winner,
            "left_score": left_score,
            "right_score": right_score,
            "left_pct": round((left_score / total) * 100, 1),
            "right_pct": round((right_score / total) * 100, 1),
            "confidence": confidence,
        }
        letters.append(winner)

    quiz_type = "".join(letters)
    group = TYPE_GROUPS[quiz_type]
    accent = GROUP_ACCENTS[group]
    profile = TYPE_PROFILES[quiz_type]

    return {
        "quiz_type": quiz_type,
        "group": group,
        "accent": accent,
        "profile": profile,
        "dimension_scores": dimension_scores,
        "profile_fragments": profile_fragments,
    }


def compose_persona_text(answers: dict[str, str], scored: dict[str, object]) -> str:
    dimension_scores = scored["dimension_scores"]
    fragments = list(scored["profile_fragments"])

    energy = "I prefer reflective, low-noise environments." if dimension_scores["IE"]["winner"] == "I" else "I gain energy by engaging with people and activity."
    focus = "I am drawn to patterns, implications, and future possibilities." if dimension_scores["SN"]["winner"] == "N" else "I ground myself in evidence, practical details, and direct experience."
    decision = "I make decisions with logic and consistency." if dimension_scores["TF"]["winner"] == "T" else "I make decisions with empathy, meaning, and values."
    lifestyle = "I like plans, checkpoints, and closure." if dimension_scores["JP"]["winner"] == "J" else "I like flexibility, experimentation, and keeping options open."

    selected_details = []
    for question in QUESTION_BANK:
        chosen = answers.get(question.key)
        if not chosen:
            continue
        selected_choice = get_choice(question, chosen)
        if selected_choice is None:
            continue
        selected_details.append(selected_choice.detail)

    sentences = [
        " ".join(fragments[:4]),
        energy,
        focus,
        decision,
        lifestyle,
        " ".join(selected_details[:5]),
    ]
    return " ".join(sentence for sentence in sentences if sentence).strip()


def build_dimension_rows(scored: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for dimension, meta in DIMENSION_META.items():
        result = scored["dimension_scores"][dimension]
        rows.append(
            {
                "title": meta["title"],
                "left_label": meta["left_label"],
                "right_label": meta["right_label"],
                "left_pct": result["left_pct"],
                "right_pct": result["right_pct"],
                "winner": result["winner"],
                "confidence": result["confidence"],
            }
        )
    return rows
