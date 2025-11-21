#!/usr/bin/env python3
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import requests  # pip install requests

# ---------------- CONFIG ---------------- #

SCENARIO_CSV_PATH = "scenarios.csv"
LOG_CSV_PATH = "experiment_log.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"  # change to your local model name

# --------------- DATA MODELS --------------- #

@dataclass
class Outcome:
    safety: int
    equity: int
    cost: int
    political: int
    total: int

@dataclass
class Scenario:
    round_num: int
    scenario_id: str
    title: str
    options_text: str
    human_private_info: str
    ai_private_info: str
    outcomes: Dict[str, Outcome]  # key: "A"/"B"/"C"/"D"

# Simple "team memory" for adaptation
@dataclass
class TeamMemory:
    explanation_length: str = "medium"  # "short" | "medium" | "long"
    focus_equity: bool = False
    user_instructions: List[str] = field(default_factory=list)


# --------------- CSV LOADING --------------- #

def parse_outcome_cell(cell: str) -> Outcome:
    """
    Parses strings like:
    "A: safety=2,equity=1,cost=2,political=1,total=6"
    into an Outcome object.
    """
    # Remove "A:" prefix (or B:/C:/D:)
    if ":" in cell:
        cell = cell.split(":", 1)[1].strip()
    parts = cell.split(",")
    values = {}
    for part in parts:
        k, v = part.split("=")
        values[k.strip()] = int(v.strip())
    return Outcome(
        safety=values["safety"],
        equity=values["equity"],
        cost=values["cost"],
        political=values["political"],
        total=values["total"],
    )


def load_scenarios(path: str) -> List[Scenario]:
    scenarios: List[Scenario] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            round_num = int(row["round"])
            scenario_id = row["scenario_id"]
            title = row["scenario_title"]
            options_text = row["options"]
            human_info = row["human_private_info"]
            ai_info = row["ai_private_info"]

            outcomes = {
                "A": parse_outcome_cell(row["option_A_outcome"]),
                "B": parse_outcome_cell(row["option_B_outcome"]),
                "C": parse_outcome_cell(row["option_C_outcome"]),
                "D": parse_outcome_cell(row["option_D_outcome"]),
            }

            scenarios.append(
                Scenario(
                    round_num=round_num,
                    scenario_id=scenario_id,
                    title=title,
                    options_text=options_text,
                    human_private_info=human_info,
                    ai_private_info=ai_info,
                    outcomes=outcomes,
                )
            )
    # sort by round just in case
    scenarios.sort(key=lambda s: s.round_num)
    return scenarios


# --------------- OLLAMA / LLM --------------- #

def generate_ai_reply(
    scenario: Scenario,
    team_memory: TeamMemory,
    chat_history: List[Tuple[str, str]],  # list of (role, text) "participant"/"ai"
    user_message: str,
) -> str:
    """
    Calls a local Ollama model using /api/chat.
    Make sure Ollama is running and the model is pulled.
    """
    system_prompt = (
        "You are an AI policy teammate in a research experiment.\n"
        "You only see the following *technical memo* about this scenario "
        "and NOT the human's stakeholder memo.\n\n"
        f"Technical memo:\n{scenario.ai_private_info}\n\n"
        "Your role:\n"
        "- Collaborate with the human.\n"
        "- Offer reasoning and trade-offs between options A/B/C/D.\n"
        "- Do NOT make the final decision; the human decides.\n"
        "- Be honest that you only see technical data.\n"
    )

    if team_memory.focus_equity:
        system_prompt += (
            "\nThe human has asked you to pay particular attention to equity "
            "and distributional impacts when relevant.\n"
        )

    if team_memory.explanation_length == "short":
        system_prompt += "\nKeep replies concise (2–3 sentences).\n"
    elif team_memory.explanation_length == "long":
        system_prompt += "\nGive more detailed reasoning (4–6 sentences).\n"
    else:
        system_prompt += "\nUse a moderate level of detail (3–4 sentences).\n"

    # Build message history for Ollama
    messages = [{"role": "system", "content": system_prompt}]

    for role, text in chat_history:
        if role == "participant":
            messages.append({"role": "user", "content": text})
        else:
            messages.append({"role": "assistant", "content": text})

    # Add the latest user message
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Ollama /api/chat style
        ai_text = data.get("message", {}).get("content", "").strip()
        if not ai_text:
            ai_text = "[AI did not return any content.]"
        return ai_text
    except Exception as e:
        return f"[Error contacting AI teammate: {e}]"


# --------------- TEAM MEMORY UPDATE --------------- #

def update_team_memory(memory: TeamMemory, instruction: str) -> TeamMemory:
    text = instruction.lower()
    memory.user_instructions.append(instruction)

    if "short" in text or "concise" in text or "brief" in text:
        memory.explanation_length = "short"
    elif "more detail" in text or "longer" in text or "explain more" in text:
        memory.explanation_length = "long"

    if "equity" in text or "fairness" in text or "fair" in text:
        memory.focus_equity = True

    return memory


# --------------- LOGGING --------------- #

def init_log(path: str):
    try:
        # If file does not exist, create with header
        with open(path, "x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "participant_id",
                    "round_num",
                    "scenario_id",
                    "choice",
                    "safety",
                    "equity",
                    "cost",
                    "political",
                    "total",
                    "chat_history_json",
                    "instruction_text",
                ]
            )
    except FileExistsError:
        # Already exists, do nothing
        pass


def log_round(
    participant_id: str,
    scenario: Scenario,
    choice: str,
    outcome: Outcome,
    chat_history: List[Tuple[str, str]],
    instruction_text: str,
    log_path: str = LOG_CSV_PATH,
):
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        participant_id,
        scenario.round_num,
        scenario.scenario_id,
        choice,
        outcome.safety,
        outcome.equity,
        outcome.cost,
        outcome.political,
        outcome.total,
        json.dumps(chat_history, ensure_ascii=False),
        instruction_text,
    ]
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# --------------- EXPERIMENT LOOP --------------- #

def run_experiment():
    print("=== Human–AI Policy Decision Experiment (Terminal Version) ===\n")
    participant_id = input("Enter participant ID (or your name/alias): ").strip()
    if not participant_id:
        participant_id = "anonymous"

    scenarios = load_scenarios(SCENARIO_CSV_PATH)
    team_memory = TeamMemory()

    print("\nInstructions:")
    print("- You will see a policy scenario each round.")
    print("- You get a private memo (stakeholders/politics).")
    print("- Your AI teammate sees a different private memo (technical data).")
    print("- You can chat with the AI, then choose a policy option A/B/C/D.")
    print('- Type "/decide" when you are ready to choose an option.')
    print('- Type "/quit" at any time to exit.\n')
    input("Press Enter to begin...")

    total_score = 0
    init_log(LOG_CSV_PATH)

    for scenario in scenarios:
        print("\n" + "=" * 70)
        print(f"ROUND {scenario.round_num}: {scenario.title}")
        print("=" * 70)
        print("\nYOUR PRIVATE MEMO (Stakeholders/Politics):")
        print(scenario.human_private_info)
        print("\nPolicy Options:")
        print(scenario.options_text)
        print("\nYou can now chat with your AI teammate.")
        print('Type messages and press Enter. Type "/decide" to move to decision.\n')

        chat_history: List[Tuple[str, str]] = []

        # Chat loop
        while True:
            user_msg = input("You: ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ("/quit", "/q"):
                print("\nExiting experiment. Goodbye.")
                sys.exit(0)
            if user_msg.lower() in ("/decide", "/d"):
                break

            chat_history.append(("participant", user_msg))
            ai_reply = generate_ai_reply(scenario, team_memory, chat_history, user_msg)
            chat_history.append(("ai", ai_reply))
            print(f"AI: {ai_reply}\n")

        # Decision
        valid_choices = ["A", "B", "C", "D"]
        choice = ""
        while choice not in valid_choices:
            choice = input("Enter your chosen option (A/B/C/D): ").strip().upper()
            if choice not in valid_choices:
                print("Please enter A, B, C, or D.")

        outcome = scenario.outcomes[choice]
        total_score += outcome.total

        print("\n--- Round Outcome ---")
        print(f"Your choice: {choice}")
        print(f"Safety impact:    {outcome.safety}")
        print(f"Equity impact:    {outcome.equity}")
        print(f"Cost impact:      {outcome.cost}")
        print(f"Political impact: {outcome.political}")
        print(f"Round total:      {outcome.total}")
        print(f"Cumulative total: {total_score}")

        # Adaptation / instruction
        instruction_text = input(
            "\nOptional: What would you like your AI teammate to do differently next round?\n"
            "(Examples: 'be more concise', 'focus more on equity', 'explain more detail')\n"
            "Press Enter to skip: "
        ).strip()

        if instruction_text:
            team_memory = update_team_memory(team_memory, instruction_text)

        # Log everything
        log_round(
            participant_id=participant_id,
            scenario=scenario,
            choice=choice,
            outcome=outcome,
            chat_history=chat_history,
            instruction_text=instruction_text,
        )

        input("\nPress Enter to continue to the next round...")

    print("\n=== Experiment complete ===")
    print(f"Final total score: {total_score}")
    print(f"Data saved to: {LOG_CSV_PATH}")


if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye.")
