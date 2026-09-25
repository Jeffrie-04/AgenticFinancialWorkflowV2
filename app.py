"""
app.py — Streamlit dashboard for the financial workflow pipeline.

Multi-business, demo-safe by design: every section reads its own file from
the SELECTED business's outputs/ folder independently and renders whatever
is there. No section here ever calls Bedrock, imports a phase module, or
depends on AWS credentials on page load — the only code paths that touch
the pipeline at all are the "Re-run pipeline" and "Process this CSV"
buttons, which shell out to run.py in a separate process so a network hang
there can never freeze this page, and which only ever run on an explicit
button press.
"""
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

FIXED_BUSINESSES = {
    "Law Firm": "businesses/law_firm",
    "Restaurant": "businesses/restaurant",
    "Landscaper": "businesses/landscaper",
}

REQUIRED_CSV_COLUMNS = {"date", "merchant", "amount"}


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


def load_kpis(path):
    data, error = load_json(path)
    if error:
        return None, error
    return data.get("kpis", {}), None


def load_list(path, key):
    """Returns (list, error) — error is None, "missing", or "malformed" (bad
    JSON, or no list of objects under `key`)."""
    data, error = load_json(path)
    if error:
        return None, error
    items = data.get(key) if isinstance(data, dict) else None
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        return None, "malformed"
    return items, None


def build_transaction_table(transactions, categorized):
    """Every ingested row, with its category joined from categorized.json by
    id. Status and reason are columns, so NEEDS_REVIEW and REJECTED rows are
    shown and flagged rather than hidden."""
    categories = {c.get("id"): c.get("category") for c in categorized}
    df = pd.DataFrame([{
        "status": t.get("status"),
        "date": t.get("date"),
        "merchant": t.get("merchant"),
        "direction": t.get("direction"),
        "amount": t.get("amount"),
        "category": categories.get(t.get("id")),
        "reason": t.get("reason"),
    } for t in transactions])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")  # strings in ingested.json; None if REJECTED
    return df.sort_values("date", ascending=False, na_position="last")


def money(x):
    return f"${x:,.2f}"


st.set_page_config(page_title="Financial Workflow Dashboard", layout="wide")

# ------------------------------------------------------------- business pick

st.session_state.setdefault("uploaded_businesses", {})
all_businesses = {**FIXED_BUSINESSES, **st.session_state.uploaded_businesses}

with st.sidebar:
    st.header("Business")
    selected_label = st.selectbox(
        "Choose a business", options=list(all_businesses.keys()), key="business_selector"
    )

business_dir = all_businesses[selected_label]
OUTPUTS_DIR = os.path.join(business_dir, "outputs")
KPIS_PATH = os.path.join(OUTPUTS_DIR, "kpis.json")
CATEGORIZED_PATH = os.path.join(OUTPUTS_DIR, "categorized.json")
INGESTED_PATH = os.path.join(OUTPUTS_DIR, "ingested.json")
DATA_QUALITY_PATH = os.path.join(OUTPUTS_DIR, "data_quality.json")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "summary.txt")
REFLECTION_PATH = os.path.join(OUTPUTS_DIR, "reflection.txt")

st.title("Financial Workflow Dashboard")
st.caption(f"Showing **{selected_label}** — reading cached results from `{OUTPUTS_DIR}/`. "
           f"No live model calls on page load.")

# ---------------------------------------------------------------- summary

summary_text, summary_error = load_text(SUMMARY_PATH)
if summary_error:
    st.info("summary.txt not found yet — run the pipeline to generate it.")
else:
    st.info(summary_text)

# -------------------------------------------------------------------- kpis

kpis, kpis_error = load_kpis(KPIS_PATH)

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

st.header("Transactions")
transactions, ingested_error = load_list(INGESTED_PATH, "transactions")
if ingested_error == "missing":
    st.warning("ingested.json not found yet — run the pipeline to generate it.")
elif ingested_error == "malformed":
    st.warning("ingested.json isn't in the expected format — re-run the pipeline to regenerate it.")
else:
    categorized, categorized_error = load_list(CATEGORIZED_PATH, "categorized")
    if categorized_error == "missing":
        st.warning("categorized.json not found yet — categories are blank until the pipeline runs.")
    elif categorized_error == "malformed":
        st.warning("categorized.json isn't in the expected format — categories are blank; "
                   "re-run the pipeline to regenerate it.")
    elif not all("id" in c for c in categorized):
        st.warning("categorized.json is in the old format (no ids) — categories are blank; "
                   "re-run the pipeline to regenerate it.")
        categorized_error = "old_format"
    if categorized_error:
        categorized = []

    if not transactions:
        st.caption("No transactions to display.")
    else:
        st.dataframe(
            build_transaction_table(transactions, categorized),
            width="stretch",
            hide_index=True,
            column_config={
                "amount": st.column_config.NumberColumn("amount", format="$%.2f"),
            },
        )
        data_quality, _ = load_json(DATA_QUALITY_PATH)
        counts = data_quality.get("counts") if isinstance(data_quality, dict) else None
        if isinstance(counts, dict):
            st.caption(f"Data quality: {counts.get('ok', 0)} OK · {counts.get('needs_review', 0)} "
                       f"needs review · {counts.get('rejected', 0)} rejected")

# ------------------------------------------------------------ re-run control

with st.sidebar:
    st.divider()
    st.header("Pipeline Control")
    st.caption(
        f"Re-running calls the live model for **{selected_label}** and requires "
        f"valid credentials. The dashboard above works from cached files without this."
    )
    if st.button(f"Re-run pipeline for {selected_label}"):
        with st.spinner("Running pipeline..."):
            result = subprocess.run(
                [sys.executable, "run.py", "--business-dir", business_dir],
                capture_output=True, text=True,
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

# --------------------------------------------------------------- upload flow

with st.sidebar:
    st.divider()
    st.header("Upload Your Own Data")
    st.caption(
        "Runs the full pipeline live on your CSV — calls the model and needs "
        "valid credentials. Only runs when you click Process; never automatic."
    )
    uploaded_file = st.file_uploader("Upload a transactions CSV", type="csv")

    if uploaded_file is not None:
        try:
            preview_df = pd.read_csv(uploaded_file)
            missing_cols = REQUIRED_CSV_COLUMNS - set(preview_df.columns)
        except Exception:
            missing_cols = REQUIRED_CSV_COLUMNS

        if missing_cols:
            st.error(f"CSV is missing required column(s): {', '.join(sorted(missing_cols))}")
        elif st.button("Process this CSV"):
            upload_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            upload_dir = os.path.join("businesses", "_uploads", upload_id)
            os.makedirs(upload_dir, exist_ok=True)
            with open(os.path.join(upload_dir, "transactions.csv"), "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.spinner("Running pipeline on your upload..."):
                result = subprocess.run(
                    [sys.executable, "run.py", "--business-dir", upload_dir],
                    capture_output=True, text=True,
                )
            if result.returncode == 0:
                st.success("Your CSV was processed successfully.")
                new_label = f"Your Upload ({uploaded_file.name})"
                st.session_state.uploaded_businesses[new_label] = upload_dir
                st.session_state["business_selector"] = new_label
            else:
                st.error(f"Processing failed (exit code {result.returncode}). See log below.")
            with st.expander("Pipeline log"):
                st.text(result.stdout)
                if result.stderr:
                    st.text(result.stderr)
            if result.returncode == 0:
                st.rerun()


# Command to run the app:
# ./venv/bin/streamlit run app.py
