"""
app.py — Streamlit dashboard for the financial workflow pipeline.

Demo-safe by design: every section below reads its own file from outputs/
independently and renders whatever is there. No section here ever calls
Bedrock, imports a phase module, or depends on AWS credentials — the only
code path that touches the pipeline at all is the "Re-run pipeline" button
in the sidebar, which shells out to run.py in a separate process so a
network hang there can never freeze this page.
"""
import json
import subprocess
import sys

import pandas as pd
import streamlit as st

OUTPUTS_DIR = "outputs"
KPIS_PATH = f"{OUTPUTS_DIR}/kpis.json"
CATEGORIZED_PATH = f"{OUTPUTS_DIR}/categorized.json"
SUMMARY_PATH = f"{OUTPUTS_DIR}/summary.txt"
REFLECTION_PATH = f"{OUTPUTS_DIR}/reflection.txt"


def load_json(path):
    """Returns (data, error) — error is None, "missing", or "malformed"."""
    try:
        with open(path, "r") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "missing"
    except json.JSONDecodeError:
        return None, "malformed"


def load_text(path):
    try:
        with open(path, "r") as f:
            return f.read(), None
    except FileNotFoundError:
        return None, "missing"


def load_kpis():
    data, error = load_json(KPIS_PATH)
    if error:
        return None, error
    return data.get("kpis", {}), None


def load_categorized():
    data, error = load_json(CATEGORIZED_PATH)
    if error:
        return None, error
    return data.get("categorized", []), None


def money(x):
    return f"${x:,.2f}"


st.set_page_config(page_title="Financial Workflow Dashboard", layout="wide")
st.title("Financial Workflow Dashboard")
st.caption(f"Reading cached results from `{OUTPUTS_DIR}/` — no live model calls on page load.")

# ---------------------------------------------------------------- summary

summary_text, summary_error = load_text(SUMMARY_PATH)
if summary_error:
    st.info("summary.txt not found yet — run the pipeline to generate it.")
else:
    st.info(summary_text)

# -------------------------------------------------------------------- kpis

kpis, kpis_error = load_kpis()

if kpis_error == "missing":
    st.warning("kpis.json not found yet — run the pipeline to generate it.")
elif kpis_error == "malformed":
    st.warning("kpis.json exists but isn't valid JSON — re-run the pipeline to regenerate it.")
else:
    st.subheader("Total Cash Volume")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Income", money(kpis.get("total_income", 0)))
    col2.metric("Total Spend", money(kpis.get("total_spend", 0)))
    col3.metric("Average Expense", money(kpis.get("average_expense", 0)))
    top_merchants = kpis.get("top_merchants", [])
    with col4:
        st.markdown("**Top Merchants**")
        for m in top_merchants:
            st.markdown(f"- {m}")
    st.divider()

    st.subheader("Net Cash Flow")
    st.metric(
        "Net Cash Flow",
        money(kpis.get("net_cash_flow", 0)),
        delta=kpis.get("status", ""),
    )
    st.divider()

    st.subheader("Spend by Category")
    spend_by_category = kpis.get("spend_by_category", {})
    if spend_by_category:
        cat_df = pd.DataFrame(
            [{"category": cat, "amount": v.get("amount", 0), "pct_of_spend": v.get("pct_of_spend", 0)}
             for cat, v in spend_by_category.items()]
        )
        st.bar_chart(cat_df.set_index("category")["amount"])
        st.dataframe(cat_df, width="stretch", hide_index=True)
    else:
        st.caption("No category breakdown available.")
    st.divider()

    st.subheader("Income Concentration")
    income_concentration = kpis.get("income_concentration", {})
    st.caption(f"Top client: {income_concentration.get('top_client', 'N/A')}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Top Client %", f"{income_concentration.get('top_client_pct', 0)}%")
    col2.metric("Top 3 Clients %", f"{income_concentration.get('top3_clients_pct', 0)}%")
    col3.metric("Income Sources", income_concentration.get("num_income_sources", 0))
    st.divider()

    st.subheader("Fixed vs Discretionary Spend")
    fixed_vs_discretionary = kpis.get("fixed_vs_discretionary", {})
    col1, col2 = st.columns(2)
    col1.metric("Fixed Spend", money(fixed_vs_discretionary.get("fixed_spend", 0)))
    col2.metric("Discretionary Spend", money(fixed_vs_discretionary.get("discretionary_spend", 0)))
    fixed_pct = fixed_vs_discretionary.get("fixed_pct", 0)
    st.progress(min(max(fixed_pct / 100, 0.0), 1.0))
    st.caption(f"{fixed_pct}% of spend is fixed/committed")
    st.divider()

    st.subheader("Daily Burn Rate")
    burn_rate = kpis.get("burn_rate", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Daily Avg Spend", money(burn_rate.get("daily_avg_spend", 0)))
    col2.metric("Monthly Projection", money(burn_rate.get("monthly_projection", 0)))
    col3.metric("Period (days)", burn_rate.get("period_days", 0))
    st.caption(f"Unparseable dates in source data: {burn_rate.get('unparseable_dates', 0)}")
    st.caption(burn_rate.get("runway_note", ""))
    st.divider()

# --------------------------------------------------------------- reflection

st.header("AI Advisor Reflection")
reflection_text, reflection_error = load_text(REFLECTION_PATH)
if reflection_error:
    st.info("reflection.txt not found yet — run the pipeline to generate it.")
else:
    with st.container(border=True):
        st.markdown(reflection_text)

# ------------------------------------------------------------- transactions

st.header("Categorized Transactions")
categorized, categorized_error = load_categorized()
if categorized_error == "missing":
    st.warning("categorized.json not found yet — run the pipeline to generate it.")
elif categorized_error == "malformed":
    st.warning("categorized.json exists but isn't valid JSON — re-run the pipeline to regenerate it.")
elif not categorized:
    st.caption("No transactions to display.")
else:
    tx_df = pd.DataFrame(categorized).sort_values("date", ascending=False)
    st.dataframe(
        tx_df,
        width="stretch",
        hide_index=True,
        column_config={
            "amount": st.column_config.NumberColumn("amount", format="$%.2f"),
        },
    )

# ------------------------------------------------------------ re-run control

with st.sidebar:
    st.header("Pipeline Control")
    st.caption(
        "Re-running calls AWS Bedrock and requires valid AWS credentials. "
        "The dashboard above works from cached files without this."
    )
    if st.button("Re-run pipeline"):
        with st.spinner("Running pipeline..."):
            result = subprocess.run(
                [sys.executable, "run.py"], capture_output=True, text=True
            )
        if result.returncode == 0:
            st.success("Pipeline completed successfully.")
        else:
            st.error(f"Pipeline failed (exit code {result.returncode}). See log below.")
        with st.expander("Pipeline log"):
            st.text(result.stdout)
            if result.stderr:
                st.text(result.stderr)
        if result.returncode == 0:
            st.rerun()
