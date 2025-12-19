import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import re
import openai 
from openai import OpenAI

#Page Configuration

st.set_page_config(
    page_title='AI Financial Advisor',
    page_icon="🤖",
    layout='wide'
)

# Configure your OpenAI API key

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# App Header
st.title('AI Financial Advisor')
st.subheader('Automated Portfolio Analysis & Asset Allocation')

# Disclaimer
st.warning('''
    **Disclaimer**
    This tool is for educational and information purposes only.
    It does not constitute financial or investment advice.
    Past performance is not indicative of future results.
''')

client_goal_text = st.text_area(
    "Describe your investment goals or preferences:",
    placeholder="E.g., I want moderate growth in tech and green energy over 5 years."
)

st.markdown("---")

# AI Analysis of Client Goals

# Asset Allocation Strategy (AI-Driven)

allocation = {}  # Initialize allocation dictionary

if client_goal_text:
    with st.spinner('Generating recommended portfolio allocation...This may take 15-20 seconds...'):
        response = client.chat.completions.create(
            model='gpt-4.1',
            messages=[
                {"role": "system", "content": "You are a professional AI financial advisor."},
                {"role": "user", "content": f"""
                Based on this client goal, return ONLY a bullet list of asset classes or tickers 
                with percentages that sum to 100%.
                Format strictly as:
                TICKER: XX%
                Investor goal: {client_goal_text}
                """}
            ]
        )
        ai_output = response.choices[0].message.content 
        st.markdown("### AI Suggested Portfolio Allocation")
        st.markdown(ai_output)

        # Robust parsing of AI output into allocation dictionary
        allocation = {}
        lines = ai_output.split('\n')
        for line in ai_output.splitlines():
            line = line.strip("-• ").strip()
            match = re.match(r'^(.+?):\s*([0-9]+(?:\.[0-9]+)?)\s*%$', line)
            if match:
                asset, weight = match.groups()
                allocation[asset.strip().upper()] = float(weight) / 100
            elif line:
                st.warning(f"Could not parse line: {line}")

        total_weight = sum(allocation.values())
        if total_weight>0:
            allocation = {k: v / total_weight for k, v in allocation.items()}

# DISPLAY ALLOCATION
# ===============================
if allocation:
    alloc_df = pd.DataFrame.from_dict(
        allocation, orient="index", columns=["Weight"]
    )
    st.table(alloc_df)

    fig, ax = plt.subplots()
    ax.pie(
        alloc_df["Weight"],
        labels=alloc_df.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)

st.markdown("---")

# ===============================
# MONTE CARLO SIMULATION
# ===============================
if allocation:
    st.header("Portfolio Growth Simulation (Monte Carlo)")

    tickers = list(allocation.keys())
    weights = np.array(list(allocation.values()))

    # Fetch historical data
    data = yf.download(tickers, period="5y")["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Remove tickers with no data
    missing_tickers = [col for col in tickers if col not in data.columns]
    if missing_tickers:
        st.warning(f"Could not fetch data for: {', '.join(missing_tickers)}")
        tickers = [t for t in tickers if t not in missing_tickers]
        weights = np.array([allocation[t] for t in tickers])
        weights = weights / weights.sum()  # normalize

    if data.empty or len(tickers) == 0:
        st.error("No valid tickers available for simulation.")
    else:
        returns = data[tickers].pct_change().dropna()

        # Portfolio statistics
        exp_return = np.dot(returns.mean() * 252, weights)
        volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        col1, col2 = st.columns(2)
        col1.metric("Expected Annual Return", f"{exp_return*100:.2f}%")
        col2.metric("Annual Volatility", f"{volatility*100:.2f}%")

        # Monte Carlo simulation (vectorized)
        n_simulations = 500
        n_days = 252 * 5
        initial_investment = 10_000

        with st.spinner("Running Monte Carlo simulation..."):
            # Generate random indices for bootstrap sampling
            rand_idx = np.random.randint(0, returns.shape[0], size=(n_days, n_simulations))
            sampled_returns = returns.values[rand_idx]  # shape: (n_days, n_simulations, n_assets)

            # Weighted portfolio returns
            # tensordot multiplies weights along assets axis
            portfolio_returns = np.tensordot(sampled_returns, weights, axes=(2, 0))  # shape: (n_days, n_simulations)

            # Portfolio value over time
            simulations = initial_investment * np.cumprod(1 + portfolio_returns, axis=0)

        # Percentiles
        years = np.arange(n_days) / 252
        p5, p50, p95 = np.percentile(simulations, [5, 50, 95], axis=1)

        # Plot
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(years, simulations, color="lightblue", alpha=0.1)
        ax2.plot(years, p50, label="Median", color="blue")
        ax2.plot(years, p5, label="5th Percentile", linestyle="--", color="red")
        ax2.plot(years, p95, label="95th Percentile", linestyle="--", color="green")
        ax2.set_title("Monte Carlo Portfolio Projection")
        ax2.set_xlabel("Years")
        ax2.set_ylabel("Portfolio Value ($)")
        ax2.legend()
        st.pyplot(fig2)

else:
    st.info("Enter your investment goals to generate a portfolio allocation.")
