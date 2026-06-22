import streamlit as st, pandas as pd
from datetime import datetime
from config import COMPANIES, MODEL, INDUSTRIES, EMBEDDING
from collect import build
from agent import analyze, sentiment, get_collection

st.set_page_config(page_title="AI CEO", layout="wide")


@st.cache_data(show_spinner=False)
def get_docs(company, ticker):
    docs = build(company, ticker)        # collect + index; cached per (company, ticker)
    return docs, datetime.now().strftime("%Y-%m-%d %H:%M")


@st.cache_data(show_spinner=False)
def get_analysis(company):
    return analyze(company)              # LLM reasoning; cached per company


# ---------- Sidebar: choose the company ----------
st.sidebar.header("Company")
choice = st.sidebar.selectbox("Preset", list(COMPANIES.keys()) + ["Other…"])
if choice == "Other…":
    company  = st.sidebar.text_input("Company name", "")
    ticker   = st.sidebar.text_input("Stock ticker (optional, e.g. AAPL)", "")
    industry = st.sidebar.text_input("Industry (optional)", "")
else:
    company, ticker = choice, COMPANIES[choice]
    industry = INDUSTRIES.get(choice, "")

if st.sidebar.button("Collect & Analyze", type="primary") and company:
    st.session_state.active = (company, ticker, industry)

if st.sidebar.button("↻ Re-collect (clear cache)"):
    get_docs.clear(); get_analysis.clear()
    st.sidebar.success("Cache cleared — click Collect & Analyze again.")

# ---------- Wait for a selection ----------
active = st.session_state.get("active")
if not active:
    st.title("AI CEO — Strategic Intelligence Agent")
    st.info("Pick a company in the sidebar and click **Collect & Analyze**.")
    st.stop()

company, ticker, industry = active
st.title(f"AI CEO — Strategic Intelligence Agent: {company}")

with st.spinner(f"Collecting live data for {company}…"):
    docs, last_update = get_docs(company, ticker)
col = get_collection(company)
cite = lambda idxs: ", ".join(f"[{i}]" for i in idxs)

#1: Company Overview
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Industry", industry or "—")
c2.metric("Documents", len(docs))
c3.metric("Data sources", len(set(d["source"] for d in docs)))
c4.metric("Reasoning model", MODEL)
c5.metric("Last update", last_update)

with st.spinner("AI CEO agent is researching and reasoning…"):
    a = get_analysis(company)
ev = a["_evidence"]

searches = sum(1 for t in a["_trace"] if t["action"] == "search")
with st.expander(f"Agent reasoning trace — {searches} search step(s)"):
    for t in a["_trace"]:
        if t["action"] == "search":
            st.markdown(f"**Step {t['step']} · search** → `{t['query']}`  ·  {t['evidence']} docs gathered")
        else:
            st.markdown(f"**Step {t['step']} · decide** → **{t['decision']}** — {t['reason']}")

tabs = st.tabs(["Market Intelligence", "Opportunities", "Risks",
                "Sentiment", "Recommendations", "CEO Briefing"])

#2: Market Intelligence
with tabs[0]:
    st.subheader("Emerging trends to monitor")
    for t in a["trends"]:
        st.markdown(f"- {t}")
    st.subheader("Recent news & activity")
    for d in docs[:15]:
        st.markdown(f"**{d['title']}**  \n*{d['source']}* — [link]({d['url']})")

#3: Opportunity Monitor
with tabs[1]:
    for o in a["opportunities"]:
        st.markdown(f"### {o['title']}")
        st.write(f"Impact: **{o['impact']}**  |  Confidence: {o['confidence']}")
        st.caption(f"Evidence: {cite(o['evidence'])}")

#4: Risk Monitor
with tabs[2]:
    for r in a["risks"]:
        st.markdown(f"### {r['title']}")
        st.write(f"Category: {r['category']}  |  Severity: **{r['severity']}**  |  Confidence: {r['confidence']}")
        st.caption(f"Evidence: {cite(r['evidence'])}")

#5: Sentiment Analysis
with tabs[3]:
    df = pd.DataFrame(sentiment(col))
    st.metric("Average news sentiment", round(df["score"].mean(), 3))
    st.bar_chart(df.groupby("source")["score"].mean())
    st.dataframe(df.sort_values("score"), use_container_width=True)

#6: Strategic Recommendations
with tabs[4]:
    for r in a["recommendations"]:
        st.markdown(f"### {r['recommendation']}")
        st.write(f"Priority: **{r['priority']}**  |  Risk level: {r['risk_level']}")
        st.write(f"Expected impact: {r['expected_impact']}")
        st.caption(f"Evidence: {cite(r['evidence'])}")

#7: CEO Briefing
with tabs[5]:
    b = a["briefing"]
    st.subheader("What happened?");                  st.write(b["what_happened"])
    st.subheader("Why does it matter?");             st.write(b["why_it_matters"])
    st.subheader("What should management do next?"); st.write(b["what_to_do_next"])

st.divider()
st.subheader("Evidence index (resolves the [n] citations above)")
st.dataframe(pd.DataFrame(ev), use_container_width=True)