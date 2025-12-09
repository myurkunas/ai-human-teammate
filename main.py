#!/usr/bin/env python3
import csv
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import requests  # pip install requests
import streamlit as st

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


# --------------- STREAMLIT UI --------------- #

def init_session_state():
    """Initialize session state variables"""
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = None
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []
    if "current_round" not in st.session_state:
        st.session_state.current_round = 0
    if "team_memory" not in st.session_state:
        st.session_state.team_memory = TeamMemory()
    if "total_score" not in st.session_state:
        st.session_state.total_score = 0
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}  # round_num -> List[Tuple[str, str]]
    if "decisions" not in st.session_state:
        st.session_state.decisions = {}  # round_num -> choice
    if "instructions" not in st.session_state:
        st.session_state.instructions = {}  # round_num -> instruction_text
    if "show_outcome" not in st.session_state:
        st.session_state.show_outcome = {}  # round_num -> bool
    if "experiment_started" not in st.session_state:
        st.session_state.experiment_started = False
    if "experiment_complete" not in st.session_state:
        st.session_state.experiment_complete = False


def render_welcome_page():
    """Render the initial welcome/participant ID page"""
    st.title("🤝 Human–AI Policy Decision Experiment")
    st.markdown("---")
    
    st.markdown("""
    ### Instructions:
    - You will see a policy scenario each round.
    - You get a **private memo** (stakeholders/politics).
    - Your AI teammate sees a **different private memo** (technical data).
    - You can chat with the AI, then choose a policy option A/B/C/D.
    """)
    
    participant_id = st.text_input(
        "Enter participant ID (or your name/alias):",
        value="",
        key="participant_input"
    )
    
    if st.button("Start Experiment", type="primary"):
        if participant_id.strip():
            st.session_state.participant_id = participant_id.strip()
        else:
            st.session_state.participant_id = "anonymous"
        
        # Load scenarios
        try:
            st.session_state.scenarios = load_scenarios(SCENARIO_CSV_PATH)
            init_log(LOG_CSV_PATH)
            st.session_state.experiment_started = True
            st.session_state.current_round = 0
            st.rerun()
        except Exception as e:
            st.error(f"Error loading scenarios: {e}")


def render_chat_interface(scenario: Scenario):
    """Render the chat interface for a scenario"""
    round_key = f"round_{scenario.round_num}"
    
    # Initialize chat history for this round
    if scenario.round_num not in st.session_state.chat_histories:
        st.session_state.chat_histories[scenario.round_num] = []
    
    chat_history = st.session_state.chat_histories[scenario.round_num]
    
    st.subheader("💬 Chat with AI Teammate")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for role, message in chat_history:
            if role == "participant":
                with st.chat_message("user"):
                    st.write(message)
            else:
                with st.chat_message("assistant"):
                    st.write(message)
    
    # Chat input
    user_message = st.chat_input("Type your message here...")
    
    if user_message:
        # Add user message to history
        chat_history.append(("participant", user_message))
        st.session_state.chat_histories[scenario.round_num] = chat_history
        
        # Generate AI reply
        with st.spinner("AI is thinking..."):
            ai_reply = generate_ai_reply(
                scenario,
                st.session_state.team_memory,
                chat_history[:-1],  # Exclude the just-added user message
                user_message
            )
            chat_history.append(("ai", ai_reply))
            st.session_state.chat_histories[scenario.round_num] = chat_history
        
        st.rerun()


def render_decision_interface(scenario: Scenario):
    """Render the decision interface"""
    st.subheader("📋 Make Your Decision")
    
    # Parse options text to show buttons
    options_text = scenario.options_text
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Option A", key=f"option_A_{scenario.round_num}", use_container_width=True):
            # Only update if decision hasn't been made yet
            if scenario.round_num not in st.session_state.decisions:
                st.session_state.decisions[scenario.round_num] = "A"
                # Update total score
                outcome = scenario.outcomes["A"]
                st.session_state.total_score += outcome.total
            st.rerun()
        if st.button("Option C", key=f"option_C_{scenario.round_num}", use_container_width=True):
            # Only update if decision hasn't been made yet
            if scenario.round_num not in st.session_state.decisions:
                st.session_state.decisions[scenario.round_num] = "C"
                # Update total score
                outcome = scenario.outcomes["C"]
                st.session_state.total_score += outcome.total
            st.rerun()
    
    with col2:
        if st.button("Option B", key=f"option_B_{scenario.round_num}", use_container_width=True):
            # Only update if decision hasn't been made yet
            if scenario.round_num not in st.session_state.decisions:
                st.session_state.decisions[scenario.round_num] = "B"
                # Update total score
                outcome = scenario.outcomes["B"]
                st.session_state.total_score += outcome.total
            st.rerun()
        if st.button("Option D", key=f"option_D_{scenario.round_num}", use_container_width=True):
            # Only update if decision hasn't been made yet
            if scenario.round_num not in st.session_state.decisions:
                st.session_state.decisions[scenario.round_num] = "D"
                # Update total score
                outcome = scenario.outcomes["D"]
                st.session_state.total_score += outcome.total
            st.rerun()
    
    # Show options text
    st.info(f"**Options:** {options_text}")


def render_outcome_display(scenario: Scenario, choice: str):
    """Render the outcome for a completed round"""
    outcome = scenario.outcomes[choice]
    
    st.subheader("📊 Round Outcome")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Safety", outcome.safety)
    with col2:
        st.metric("Equity", outcome.equity)
    with col3:
        st.metric("Cost", outcome.cost)
    with col4:
        st.metric("Political", outcome.political)
    
    st.metric("Round Total", outcome.total)
    st.metric("Cumulative Total", st.session_state.total_score)
    
    # Instruction input
    st.subheader("🔄 Adapt AI Teammate (Optional)")
    instruction_text = st.text_input(
        "What would you like your AI teammate to do differently next round?",
        value=st.session_state.instructions.get(scenario.round_num, ""),
        key=f"instruction_{scenario.round_num}",
        placeholder="Examples: 'be more concise', 'focus more on equity', 'explain more detail'"
    )
    
    if instruction_text:
        st.session_state.instructions[scenario.round_num] = instruction_text
        st.session_state.team_memory = update_team_memory(
            st.session_state.team_memory,
            instruction_text
        )
    
    # Continue button
    if st.button("Continue to Next Round", type="primary", key=f"continue_{scenario.round_num}"):
        # Log this round
        log_round(
            participant_id=st.session_state.participant_id,
            scenario=scenario,
            choice=choice,
            outcome=outcome,
            chat_history=st.session_state.chat_histories.get(scenario.round_num, []),
            instruction_text=st.session_state.instructions.get(scenario.round_num, ""),
        )
        
        # Move to next round
        st.session_state.current_round += 1
        if st.session_state.current_round >= len(st.session_state.scenarios):
            st.session_state.experiment_complete = True
        st.rerun()


def render_scenario_page():
    """Render the main scenario page"""
    if not st.session_state.scenarios:
        st.error("No scenarios loaded. Please restart the experiment.")
        return
    
    if st.session_state.current_round >= len(st.session_state.scenarios):
        render_completion_page()
        return
    
    scenario = st.session_state.scenarios[st.session_state.current_round]
    
    # Header
    st.title(f"Round {scenario.round_num}: {scenario.title}")
    
    # Progress bar
    progress = (st.session_state.current_round + 1) / len(st.session_state.scenarios)
    st.progress(progress, text=f"Round {st.session_state.current_round + 1} of {len(st.session_state.scenarios)}")
    
    # Private memo
    with st.expander("📄 Your Private Memo (Stakeholders/Politics)", expanded=True):
        st.write(scenario.human_private_info)
    
    # Check if decision has been made
    if scenario.round_num in st.session_state.decisions:
        choice = st.session_state.decisions[scenario.round_num]
        render_outcome_display(scenario, choice)
    else:
        # Show chat and decision interfaces
        render_chat_interface(scenario)
        st.markdown("---")
        render_decision_interface(scenario)


def render_completion_page():
    """Render the experiment completion page"""
    st.title("🎉 Experiment Complete!")
    st.balloons()
    
    st.metric("Final Total Score", st.session_state.total_score)
    
    st.success(f"Data saved to: {LOG_CSV_PATH}")
    
    if st.button("Start New Experiment"):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="AI-Human Policy Experiment",
        page_icon="🤝",
        layout="wide"
    )
    
    init_session_state()
    
    if not st.session_state.experiment_started:
        render_welcome_page()
    elif st.session_state.experiment_complete:
        render_completion_page()
    else:
        render_scenario_page()


if __name__ == "__main__":
    main()
