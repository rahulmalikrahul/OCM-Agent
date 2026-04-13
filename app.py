# streamlit_app.py
"""
OCM AI Agent — Streamlit single-file app (updated)
This version fixes compatibility with the google.generativeai Python client by using
the chat completions API when available and falling back to REST.

Save this file as streamlit_app.py and deploy to Streamlit Community Cloud via GitHub.
Set GOOGLE_API_KEY in Streamlit secrets or upload a service account JSON in the app UI.

Dependencies:
- See requirements.txt (streamlit, google-generativeai, google-auth, requests, pandas, altair, python-dateutil)

Notes:
- The google.generativeai client exposes different APIs depending on version.
  This app attempts to use chat completions (recommended) and falls back to other shapes.
- The REST fallback is a minimal example and may require adjustment for your Google model and API version.
"""

import os
import json
import time
import tempfile
import textwrap
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except Exception as e:
    raise RuntimeError("streamlit is required. Install with `pip install streamlit`.") from e

# Optional visualization and data libs
try:
    import pandas as pd
except Exception:
    st.warning("pandas not installed. Install with `pip install pandas` for full functionality.")
    import pandas as pd  # will raise if missing

try:
    import altair as alt
except Exception:
    st.warning("altair not installed. Install with `pip install altair` for charts.")
    import altair as alt

# Optional Google Generative AI client
HAS_GENAI = False
GENAI_CLIENT = None
try:
    import google.generativeai as genai  # type: ignore
    HAS_GENAI = True
    GENAI_CLIENT = genai
except Exception:
    HAS_GENAI = False
    GENAI_CLIENT = None

import requests
from dateutil.parser import parse as parse_date

# ---------------------------
# Defaults and configuration
# ---------------------------
DEFAULT_MODEL = "models/text-bison-001"  # or 'chat-bison-001' / 'gemini-1.5'
DEFAULT_REGION = "us-central1"
DEFAULT_PROJECT = "your-gcp-project-id"

AUTONOMOUS_STEP_DELAY = 1.0
DEFAULT_THINKING_BUDGET = 0.5

SYSTEM_PROMPT = """
You are an expert Organization Change Management (OCM) consultant and organizational design advisor.
You are deeply familiar with Prosci ADKAR, Kotter, Bridges, stakeholder mapping, communications planning,
and measurement frameworks. You produce pragmatic, prioritized, and measurable plans tailored to the
business area, industry, organization size, timeline, and risk tolerance provided by the user.
You do not provide legal or HR compliance advice; always recommend consulting legal/HR for compliance-sensitive actions.
"""

FEW_SHOT_EXAMPLES = [
    {
        "scenario": "ERP rollout in manufacturing",
        "input_summary": "Business area: Finance; Industry: Manufacturing; Org size: 5,000; Scope: Global ERP; Timeline: 12 months; Stakeholders: Finance, IT, Ops; Risk tolerance: Medium",
        "output_summary": "Phased ADKAR-based plan with Discovery (stakeholder analysis, readiness), Design (process mapping), Build (pilot), Deploy (region-by-region), Sustain (support, metrics). Communications: executive townhall, manager toolkits, weekly newsletters. KPIs: adoption rate, training completion, support ticket volume."
    },
    {
        "scenario": "CRM modernization in retail",
        "input_summary": "Business area: Sales; Industry: Retail; Org size: 800; Scope: CRM migration; Timeline: 6 months; Stakeholders: Sales, Marketing, Customer Service; Risk tolerance: Low",
        "output_summary": "Focused pilot with sales champions, manager enablement, targeted comms, and measurement via conversion rates and NPS."
    }
]

SAFETY_DISCLAIMER = (
    "This tool provides advisory OCM guidance only. It is not legal, medical, or HR compliance advice. "
    "Consult your organization's legal and HR teams before executing actions that have regulatory, contractual, "
    "or employee relations implications."
)

# ---------------------------
# Google AI helpers (robust client handling)
# ---------------------------

def set_service_account_from_upload(uploaded_file) -> Optional[str]:
    """
    Accept a service account JSON uploaded via Streamlit file_uploader,
    write to a temp file, and set GOOGLE_APPLICATION_CREDENTIALS for the session.
    """
    if uploaded_file is None:
        return None
    try:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        t.write(uploaded_file.read())
        t.flush()
        t.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = t.name
        return t.name
    except Exception as e:
        st.error(f"Failed to write service account file: {e}")
        return None

def get_api_key_from_secrets() -> Optional[str]:
    """
    Read GOOGLE_API_KEY from Streamlit secrets or environment.
    """
    api_key = None
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")  # type: ignore
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    return api_key

def _genai_chat_completion(messages: List[Dict[str, str]], model: str, temperature: float, max_output_tokens: int) -> str:
    """
    Use google.generativeai chat completions if available.
    This function handles different client versions gracefully.
    """
    api_key = get_api_key_from_secrets()
    if not api_key:
        raise RuntimeError("Missing API key for google.generativeai client.")
    try:
        # Configure client
        GENAI_CLIENT.configure(api_key=api_key)
    except Exception:
        # Some client versions don't require configure; ignore
        pass

    # Preferred: chat completions API
    try:
        # Many versions expose chat.completions.create
        if hasattr(GENAI_CLIENT, "chat") and hasattr(GENAI_CLIENT.chat, "completions"):
            resp = GENAI_CLIENT.chat.completions.create(model=model, messages=messages, temperature=temperature, max_output_tokens=max_output_tokens)
            # Response shapes vary: try common fields
            if isinstance(resp, dict):
                # new-style dict
                if "candidates" in resp:
                    return resp["candidates"][0].get("content", "")
                if "choices" in resp and len(resp["choices"]) > 0:
                    return resp["choices"][0].get("message", {}).get("content", "")
            # object-like response
            if hasattr(resp, "candidates"):
                return resp.candidates[0].content
            if hasattr(resp, "choices"):
                choice = resp.choices[0]
                if hasattr(choice, "message"):
                    return choice.message.get("content", "")
                if hasattr(choice, "content"):
                    return choice.content
            return str(resp)
        # Older or alternate client shapes: try a top-level completions.create
        if hasattr(GENAI_CLIENT, "completions") and hasattr(GENAI_CLIENT.completions, "create"):
            resp = GENAI_CLIENT.completions.create(model=model, prompt=messages[-1]["content"], temperature=temperature, max_output_tokens=max_output_tokens)
            if isinstance(resp, dict) and "candidates" in resp:
                return resp["candidates"][0].get("content", "")
            if hasattr(resp, "candidates"):
                return resp.candidates[0].content
            return str(resp)
    except Exception as e:
        # Bubble up to caller to allow fallback
        raise RuntimeError(f"google.generativeai chat completion failed: {e}")

def call_google_genai_text(prompt: str, model: str = DEFAULT_MODEL, max_output_tokens: int = 800, temperature: float = 0.2) -> str:
    """
    Centralized model call. Tries:
      1) google.generativeai chat completions (recommended)
      2) google.generativeai completions (older shapes)
      3) REST fallback to Vertex AI (minimal)
    """
    api_key = get_api_key_from_secrets()

    # 1) Try python client chat completions
    if HAS_GENAI and GENAI_CLIENT:
        try:
            # Build messages: system + user
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": prompt}
            ]
            text = _genai_chat_completion(messages=messages, model=model, temperature=temperature, max_output_tokens=max_output_tokens)
            if text:
                return text
        except Exception as e:
            st.warning(f"google.generativeai client chat attempt failed: {e}. Falling back to other methods.")

        # 2) Try a more generic call if available
        try:
            # Some client versions expose a generate_text or generate API
            if hasattr(GENAI_CLIENT, "generate_text"):
                resp = GENAI_CLIENT.generate_text(model=model, prompt=prompt, max_output_tokens=max_output_tokens, temperature=temperature)
                if isinstance(resp, dict) and "candidates" in resp:
                    return resp["candidates"][0].get("content", "")
                if hasattr(resp, "text"):
                    return resp.text
                return str(resp)
        except Exception as e:
            st.warning(f"google.generativeai generate_text attempt failed: {e}. Falling back to REST.")

    # 3) REST fallback
    return call_vertex_ai_rest(prompt=prompt, model=model, api_key=api_key, max_output_tokens=max_output_tokens, temperature=temperature)

def call_vertex_ai_rest(prompt: str, model: str = DEFAULT_MODEL, api_key: Optional[str] = None, max_output_tokens: int = 800, temperature: float = 0.2) -> str:
    """
    Minimal REST call to Vertex AI text generation endpoint.
    NOTE: This is a simplified example. For production, use official SDKs and proper auth.
    If using an API key, some endpoints expect the API key as a query param; others require OAuth bearer tokens.
    Adjust this function to match your chosen model and API version.
    """
    if not api_key:
        st.error("No Google API key found. Set GOOGLE_API_KEY in Streamlit secrets or environment.")
        return "ERROR: Missing API key."

    project = st.session_state.get("project_id") or DEFAULT_PROJECT
    region = st.session_state.get("region") or DEFAULT_REGION

    # Example endpoint for some Generative Models APIs (subject to change)
    url = f"https://{region}-generativemodels.googleapis.com/v1/models/{model}:generateText"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "prompt": prompt,
        "maxOutputTokens": max_output_tokens,
        "temperature": temperature
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            st.warning(f"Vertex AI REST call returned {resp.status_code}: {resp.text}")
            return f"ERROR: Vertex REST {resp.status_code}: {resp.text}"
        data = resp.json()
        if "candidates" in data:
            return data["candidates"][0].get("content", "")
        if "output" in data:
            out = data["output"]
            if isinstance(out, list) and len(out) > 0:
                # Some endpoints return output[0].content or output[0].text
                return out[0].get("content", out[0].get("text", ""))
        # Fallback: return raw JSON
        return json.dumps(data)
    except Exception as e:
        st.error(f"Vertex REST call failed: {e}")
        return f"ERROR: {e}"

# ---------------------------
# Prompt building and parsing
# ---------------------------

def build_system_prompt(user_inputs: Dict[str, Any]) -> str:
    header = SYSTEM_PROMPT.strip() + "\n\n"
    examples = "\n\n".join(
        f"Scenario: {ex['scenario']}\nInput: {ex['input_summary']}\nOutput Summary: {ex['output_summary']}"
        for ex in FEW_SHOT_EXAMPLES
    )
    user_block = "\n\nUser Inputs:\n" + "\n".join(f"- {k}: {v}" for k, v in user_inputs.items())
    instructions = textwrap.dedent("""
    Instructions:
    1) Produce a high-level OCM strategy summary (3-6 bullet points).
    2) Produce a phased roadmap (Discovery, Design, Build, Deploy, Sustain) with tasks, owners, timelines, and KPIs.
    3) Provide communications artifacts: executive townhall script, manager talking points, FAQ, and 3 email templates tailored to stakeholder groups.
    4) Provide measurement plan: KPIs, targets, data sources, and a simulated baseline and expected progress.
    5) Provide org design recommendations if relevant (roles, RACI, reporting changes).
    6) Provide a prioritized action list for the next 30/90/180 days.
    7) Keep outputs actionable and concise. Use ADKAR language where appropriate.
    8) Add a short list of risks and mitigation actions.
    9) End with a short list of follow-up questions the agent should ask the user to refine the plan.
    """)
    prompt = header + "\n\nExamples:\n" + examples + "\n\n" + user_block + "\n\n" + instructions
    return prompt

def parse_model_plan_to_structured(text: str) -> Dict[str, Any]:
    sections = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    current = "summary"
    buffer = []
    for line in lines:
        lower = line.lower()
        if any(k in lower for k in ["discovery", "design", "build", "deploy", "sustain", "roadmap"]):
            if buffer:
                sections[current] = "\n".join(buffer).strip()
            current = "roadmap"
            buffer = [line]
        elif any(k in lower for k in ["communications", "townhall", "email", "faq", "manager"]):
            if buffer:
                sections[current] = "\n".join(buffer).strip()
            current = "communications"
            buffer = [line]
        elif any(k in lower for k in ["kpi", "measurement", "metrics", "targets"]):
            if buffer:
                sections[current] = "\n".join(buffer).strip()
            current = "measurement"
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        sections[current] = "\n".join(buffer).strip()
    return {"raw": text, "sections": sections}

def generate_ocm_plan(user_inputs: Dict[str, Any], model: str = DEFAULT_MODEL, temperature: float = 0.2) -> Dict[str, Any]:
    prompt = build_system_prompt(user_inputs)
    prompt += "\n\nPlease format the roadmap with clear phase headers and bullet lists."
    # Use the central call function
    text = call_google_genai_text(prompt=prompt, model=model, temperature=temperature)
    structured = parse_model_plan_to_structured(text)
    return {"text": text, "structured": structured, "generated_at": datetime.utcnow().isoformat() + "Z"}

# ---------------------------
# Simulation and visualization helpers
# ---------------------------

def simulate_kpi_progress(kpis: List[Dict[str, Any]], days: int = 180) -> pd.DataFrame:
    start = datetime.utcnow().date()
    dates = [start + timedelta(days=i) for i in range(days + 1)]
    data = {"date": dates}
    for k in kpis:
        name = k.get("name", "KPI")
        baseline = float(k.get("baseline", 0))
        target = float(k.get("target", baseline + 10))
        values = []
        for i in range(len(dates)):
            t = i / max(1, days)
            val = baseline + (target - baseline) * (1 - (1 - t) ** 2)
            values.append(val)
        data[name] = values
    df = pd.DataFrame(data)
    return df

def build_gantt_dataframe(roadmap: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in roadmap:
        try:
            start = parse_date(item.get("start")).date()
            end = parse_date(item.get("end")).date()
        except Exception:
            start = datetime.utcnow().date()
            end = start + timedelta(days=7)
        rows.append({
            "phase": item.get("phase", "Phase"),
            "task": item.get("task", "Task"),
            "start": start,
            "end": end,
            "owner": item.get("owner", "TBD")
        })
    df = pd.DataFrame(rows)
    return df

# ---------------------------
# UI helpers and session state
# ---------------------------

def init_session_state():
    ss = st.session_state
    defaults = {
        "auth_method": "api_key",
        "service_account_path": None,
        "project_id": DEFAULT_PROJECT,
        "region": DEFAULT_REGION,
        "model": DEFAULT_MODEL,
        "temperature": 0.2,
        "thinking_budget": DEFAULT_THINKING_BUDGET,
        "user_inputs": {},
        "plan": None,
        "roadmap": [],
        "kpis": [],
        "activity_log": [],
        "autonomous_running": False,
        "chat_history": [],
        "last_run": None,
        "autonomous_delay": AUTONOMOUS_STEP_DELAY,
        "auto_run_after_generate": False
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

def log_activity(message: str):
    ts = datetime.utcnow().isoformat() + "Z"
    entry = f"{ts} - {message}"
    st.session_state.activity_log.insert(0, entry)
    st.session_state.activity_log = st.session_state.activity_log[:200]

def sidebar_auth_controls():
    st.sidebar.header("Authentication")
    auth_method = st.sidebar.radio("Auth method", options=["api_key", "service_account"], index=0)
    st.session_state.auth_method = auth_method
    if auth_method == "api_key":
        st.sidebar.write("Using API key from Streamlit secrets or environment.")
        api_key = get_api_key_from_secrets()
        if api_key:
            st.sidebar.success("API key detected.")
        else:
            st.sidebar.warning("No API key found. Set GOOGLE_API_KEY in Streamlit secrets or upload a service account JSON.")
    else:
        uploaded = st.sidebar.file_uploader("Upload service account JSON", type=["json"], help="Upload service account JSON to authenticate.")
        if uploaded:
            path = set_service_account_from_upload(uploaded)
            if path:
                st.sidebar.success("Service account uploaded and set for this session.")
                st.session_state.service_account_path = path

    st.sidebar.markdown("---")
    st.sidebar.header("Model & Region")
    st.session_state.model = st.sidebar.text_input("Model name", value=st.session_state.get("model", DEFAULT_MODEL))
    st.session_state.region = st.sidebar.text_input("Region", value=st.session_state.get("region", DEFAULT_REGION))
    st.session_state.project_id = st.sidebar.text_input("Project ID", value=st.session_state.get("project_id", DEFAULT_PROJECT))
    st.sidebar.markdown("---")
    st.sidebar.header("Agent Controls")
    st.session_state.temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, value=float(st.session_state.get("temperature", 0.2)))
    st.session_state.thinking_budget = st.sidebar.slider("Thinking budget", 0.0, 1.0, value=float(st.session_state.get("thinking_budget", DEFAULT_THINKING_BUDGET)))
    st.sidebar.markdown("---")
    st.sidebar.header("Autonomy")
    st.session_state.autonomous_delay = st.sidebar.slider("Autonomous step delay (s)", 0.1, 5.0, value=AUTONOMOUS_STEP_DELAY)
    st.sidebar.checkbox("Auto-run after generate", key="auto_run_after_generate")
    st.sidebar.markdown("---")
    st.sidebar.header("Safety")
    st.sidebar.info(SAFETY_DISCLAIMER)

def input_form():
    st.sidebar.header("Change Context")
    with st.sidebar.form("inputs_form", clear_on_submit=False):
        business_area = st.text_input("Business area (e.g., Finance, HR, Sales)", value="Finance")
        industry = st.text_input("Industry", value="Manufacturing")
        org_size = st.selectbox("Organization size", options=["<50", "50-250", "250-1000", "1000-5000", "5000+"], index=3)
        change_scope = st.text_area("Change scope (short)", value="Global ERP implementation")
        timeline_months = st.number_input("Timeline (months)", min_value=1, max_value=36, value=12)
        stakeholders = st.text_area("Key stakeholders (comma-separated)", value="Finance, IT, Operations, HR")
        risk_tolerance = st.selectbox("Risk tolerance", options=["Low", "Medium", "High"], index=1)
        submit = st.form_submit_button("Save context")
        if submit:
            st.session_state.user_inputs = {
                "business_area": business_area,
                "industry": industry,
                "org_size": org_size,
                "change_scope": change_scope,
                "timeline_months": int(timeline_months),
                "stakeholders": [s.strip() for s in stakeholders.split(",") if s.strip()],
                "risk_tolerance": risk_tolerance
            }
            log_activity("User context saved.")
            st.success("Context saved.")

# ---------------------------
# Agent execution and autonomy
# ---------------------------

def run_agent_once():
    ui = st.session_state.user_inputs
    if not ui:
        st.warning("Please provide context in the sidebar and click Save context.")
        return
    log_activity("Generating OCM plan (single run)...")
    plan = generate_ocm_plan(ui, model=st.session_state.model, temperature=st.session_state.temperature)
    st.session_state.plan = plan
    st.session_state.last_run = datetime.utcnow().isoformat() + "Z"
    kpis = []
    if not kpis:
        kpis = [
            {"name": "Adoption Rate (%)", "baseline": 10, "target": 70},
            {"name": "Training Completion (%)", "baseline": 0, "target": 90},
            {"name": "Support Ticket Volume", "baseline": 100, "target": 30}
        ]
    st.session_state.kpis = kpis
    roadmap = []
    start = datetime.utcnow().date()
    months = st.session_state.user_inputs.get("timeline_months", 6)
    phase_lengths = [max(1, months // 5) for _ in range(5)]
    phases = ["Discovery", "Design", "Build", "Deploy", "Sustain"]
    cur = start
    for i, phase in enumerate(phases):
        length = phase_lengths[i]
        end = cur + timedelta(days=length * 30)
        roadmap.append({
            "phase": phase,
            "task": f"{phase} activities for {st.session_state.user_inputs.get('change_scope')}",
            "start": cur.isoformat(),
            "end": end.isoformat(),
            "owner": "OCM Lead"
        })
        cur = end + timedelta(days=1)
    st.session_state.roadmap = roadmap
    log_activity("OCM plan generated.")
    return plan

def autonomous_run(steps: int = 5):
    st.session_state.autonomous_running = True
    log_activity(f"Autonomous run started for {steps} steps.")
    for step in range(steps):
        if not st.session_state.autonomous_running:
            log_activity("Autonomous run stopped by user.")
            break
        log_activity(f"Autonomous step {step + 1}/{steps} starting.")
        time.sleep(st.session_state.autonomous_delay)
        plan = run_agent_once()
        if plan:
            action = {
                "phase": "Design",
                "task": f"Refinement step {step+1}: stakeholder interviews and champion selection",
                "start": (datetime.utcnow().date() + timedelta(days=step*7)).isoformat(),
                "end": (datetime.utcnow().date() + timedelta(days=step*7 + 14)).isoformat(),
                "owner": "OCM Consultant"
            }
            st.session_state.roadmap.append(action)
            log_activity(f"Autonomous step {step + 1}: added refinement action.")
        st.session_state.chat_history.append({
            "role": "agent",
            "text": f"Completed autonomous step {step + 1}. Generated plan snapshot and added refinement action."
        })
    st.session_state.autonomous_running = False
    log_activity("Autonomous run completed.")

# ---------------------------
# Export helpers
# ---------------------------

def export_as_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, default=str)

def export_as_markdown(plan: Dict[str, Any], user_inputs: Dict[str, Any]) -> str:
    md = []
    md.append(f"# OCM Strategy for {user_inputs.get('business_area')} - {user_inputs.get('change_scope')}")
    md.append(f"**Industry:** {user_inputs.get('industry')}")
    md.append(f"**Organization size:** {user_inputs.get('org_size')}")
    md.append(f"**Timeline (months):** {user_inputs.get('timeline_months')}")
    md.append("\n## Executive Summary\n")
    md.append(plan.get("structured", {}).get("summary", plan.get("text", "")[:500]))
    md.append("\n## Roadmap\n")
    for item in st.session_state.roadmap:
        md.append(f"- **{item.get('phase')}**: {item.get('task')} ({item.get('start')} → {item.get('end')}) Owner: {item.get('owner')}")
    md.append("\n## KPIs\n")
    for k in st.session_state.kpis:
        md.append(f"- **{k.get('name')}**: baseline {k.get('baseline')}, target {k.get('target')}")
    md.append("\n## Communications\n")
    comms = plan.get("structured", {}).get("communications", "")
    md.append(comms or "Communications artifacts generated by the agent.")
    md.append("\n## Activity Log\n")
    for a in st.session_state.activity_log[:50]:
        md.append(f"- {a}")
    return "\n".join(md)

# ---------------------------
# Main app layout
# ---------------------------

def main():
    st.set_page_config(page_title="OCM AI Agent", layout="wide", initial_sidebar_state="expanded")
    init_session_state()
    sidebar_auth_controls()
    input_form()

    st.title("OCM AI Agent — Prosci-savvy Organizational Change Consultant")
    st.caption("Designs OCM strategies, communications, and measurement plans. Interactive and iterative.")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Run (single)"):
            run_agent_once()
        if st.button("Step (autonomous 1)"):
            autonomous_run(steps=1)
    with col2:
        if st.button("Auto-run 5 steps"):
            autonomous_run(steps=5)
        if st.button("Reset"):
            st.session_state.plan = None
            st.session_state.roadmap = []
            st.session_state.kpis = []
            st.session_state.activity_log = []
            st.session_state.chat_history = []
            log_activity("Session reset by user.")
            st.experimental_rerun()
    with col3:
        st.metric("Last run", st.session_state.get("last_run") or "Never")
        st.metric("Activity log", len(st.session_state.activity_log))

    tabs = st.tabs(["Overview", "Roadmap", "Communications", "Measurement", "Chat", "Export"])

    with tabs[0]:
        st.header("Overview")
        st.info(SAFETY_DISCLAIMER)
        if st.session_state.plan:
            st.subheader("Executive Summary")
            st.markdown(st.session_state.plan.get("structured", {}).get("summary", "") or st.session_state.plan.get("text", "")[:1000])
            st.subheader("Key details")
            ui = st.session_state.user_inputs
            st.write(ui)
            st.subheader("Activity Log")
            for a in st.session_state.activity_log[:20]:
                st.write(a)
        else:
            st.write("No plan generated yet. Provide context and click Run.")

    with tabs[1]:
        st.header("Roadmap")
        st.write("Phased roadmap and simulated Gantt chart.")
        if st.session_state.roadmap:
            df = build_gantt_dataframe(st.session_state.roadmap)
            st.dataframe(df)
            try:
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('start:T', title='Start'),
                    x2='end:T',
                    y=alt.Y('phase:N', sort=list(df['phase'].unique())),
                    color=alt.Color('phase:N'),
                    tooltip=['task', 'owner', 'start', 'end']
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render Gantt chart: {e}")
            for item in st.session_state.roadmap:
                with st.expander(f"{item.get('phase')} — {item.get('task')}"):
                    st.write(item)
        else:
            st.write("No roadmap yet. Run the agent to generate a roadmap.")

    with tabs[2]:
        st.header("Communications")
        st.write("Generated communications artifacts tailored to stakeholder groups.")
        if st.session_state.plan:
            comms = st.session_state.plan.get("structured", {}).get("communications", "")
            st.markdown("### Communications (raw)")
            st.code(comms or "No communications section parsed.")
            st.markdown("### Generate specific artifact")
            artifact = st.selectbox("Artifact", options=["Executive townhall script", "Manager talking points", "FAQ", "Email to employees"])
            if st.button("Generate artifact"):
                prompt = f"{SYSTEM_PROMPT}\n\nUser context:\n{json.dumps(st.session_state.user_inputs, indent=2)}\n\nProduce a {artifact} tailored to the stakeholder groups. Keep it concise and actionable."
                text = call_google_genai_text(prompt=prompt, model=st.session_state.model, temperature=st.session_state.temperature)
                st.subheader(artifact)
                st.write(text)
                st.session_state.chat_history.append({"role": "agent", "text": f"Generated {artifact}."})
        else:
            st.write("No plan yet. Run the agent to generate communications.")

    with tabs[3]:
        st.header("Measurement Dashboard")
        st.write("KPIs, simulated progress, and interactive sliders to model effectiveness.")
        if st.session_state.kpis:
            cols = st.columns(len(st.session_state.kpis))
            for i, k in enumerate(st.session_state.kpis):
                with cols[i]:
                    st.metric(k.get("name"), f"{k.get('baseline')}", f"Target: {k.get('target')}")
            days = st.slider("Simulate days", 30, 365, value=180)
            df = simulate_kpi_progress(st.session_state.kpis, days=days)
            st.write("Adjust effectiveness multipliers (0.5 = slower, 1.0 = baseline, 1.5 = faster)")
            multipliers = {}
            for k in st.session_state.kpis:
                key = f"mult_{k['name']}"
                multipliers[k['name']] = st.slider(k['name'], 0.5, 2.0, 1.0, key=key)
            df_plot = df.copy()
            for name in [k['name'] for k in st.session_state.kpis]:
                df_plot[name] = df_plot[name] * multipliers.get(name, 1.0)
            df_melt = df_plot.melt(id_vars=["date"], var_name="kpi", value_name="value")
            chart = alt.Chart(df_melt).mark_line().encode(
                x='date:T',
                y='value:Q',
                color='kpi:N',
                tooltip=['date', 'kpi', 'value']
            ).interactive().properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No KPIs defined. Run the agent to generate KPIs.")

    with tabs[4]:
        st.header("Agent Chat")
        st.write("Ask follow-up questions or request refinements. The agent keeps a chat history.")
        for msg in st.session_state.chat_history[-20:]:
            role = msg.get("role", "agent")
            if role == "user":
                st.markdown(f"**You:** {msg.get('text')}")
            else:
                st.markdown(f"**Agent:** {msg.get('text')}")
        user_msg = st.text_area("Your message", key="chat_input", height=80)
        if st.button("Send message"):
            if not user_msg.strip():
                st.warning("Enter a message.")
            else:
                st.session_state.chat_history.append({"role": "user", "text": user_msg})
                plan_text = st.session_state.plan.get("text", "") if st.session_state.plan else ""
                prompt = f"{SYSTEM_PROMPT}\n\nCurrent plan:\n{plan_text}\n\nUser message:\n{user_msg}\n\nRespond as an OCM consultant. If the user asks to revise the plan, provide a short revised action or question to clarify."
                resp = call_google_genai_text(prompt=prompt, model=st.session_state.model, temperature=st.session_state.temperature)
                st.session_state.chat_history.append({"role": "agent", "text": resp})
                log_activity("Chat message processed by agent.")
                if any(w in user_msg.lower() for w in ["revise", "change", "update", "re-priorit"]):
                    st.session_state.roadmap.append({
                        "phase": "Design",
                        "task": f"User-requested revision: {user_msg[:80]}",
                        "start": datetime.utcnow().date().isoformat(),
                        "end": (datetime.utcnow().date() + timedelta(days=14)).isoformat(),
                        "owner": "OCM Consultant"
                    })
                    log_activity("Applied a user-requested revision to the roadmap (simulated).")
                st.experimental_rerun()

    with tabs[5]:
        st.header("Export")
        st.write("Download or copy the generated strategy as JSON or Markdown.")
        if st.session_state.plan:
            data = {
                "user_inputs": st.session_state.user_inputs,
                "plan": st.session_state.plan,
                "roadmap": st.session_state.roadmap,
                "kpis": st.session_state.kpis,
                "activity_log": st.session_state.activity_log
            }
            json_text = export_as_json(data)
            md_text = export_as_markdown(st.session_state.plan, st.session_state.user_inputs)
            st.download_button("Download JSON", data=json_text, file_name="ocm_plan.json", mime="application/json")
            st.download_button("Download Markdown", data=md_text, file_name="ocm_plan.md", mime="text/markdown")
            st.markdown("### JSON preview")
            st.code(json_text[:1000] + ("...\n(Truncated)" if len(json_text) > 1000 else ""))
            st.markdown("### Markdown preview")
            st.markdown(md_text[:1000] + ("...\n(Truncated)" if len(md_text) > 1000 else ""))
            st.markdown("---")
            st.markdown("**Deployment notes**")
            st.markdown("""
            1. Commit this file and a `requirements.txt` listing dependencies to a GitHub repo.
            2. On Streamlit Community Cloud, create a new app from the repo.
            3. Add `GOOGLE_API_KEY` to App Settings -> Secrets or upload service account JSON at runtime.
            4. Ensure `requirements.txt` includes: streamlit, google-generativeai, google-auth, requests, pandas, altair, python-dateutil
            5. For production, prefer service account authentication and the official Google SDKs.
            """)
        else:
            st.write("No plan to export. Run the agent first.")

    st.markdown("---")
    st.subheader("Prompt templates and tips")
    st.write("Use these templates to ask the agent for focused outputs.")
    st.code("""
System: You are an expert OCM consultant (Prosci ADKAR). Produce a concise plan.
User: Provide a 30/90/180 day prioritized action list for [change_scope] in [industry] for a [org_size] org.
""")

if __name__ == "__main__":
    main()