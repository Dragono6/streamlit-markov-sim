import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import cryptocompare
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, timedelta
from typing import List, Tuple, Dict, Optional
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Markov Chain Stock & Crypto Simulator", page_icon="📈")

# --- Data Loading ---
@st.cache_data(ttl=6 * 60 * 60)
def load_data(tickers: List[str], start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """
    Loads historical price data for tickers from yfinance or cryptocompare.
    """
    combined_data = []
    for ticker in tickers:
        data = None
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data = df[['Close']].copy()
        except Exception:
            st.warning(f"yfinance failed for {ticker}. Trying cryptocompare.")

        if data is None:
            try:
                limit = (end_date - start_date).days
                crypto_data = cryptocompare.get_historical_price_day(ticker, 'USD', limit=limit, toTs=end_date)
                if crypto_data:
                    df = pd.DataFrame(crypto_data)
                    df['time'] = pd.to_datetime(df['time'], unit='s').dt.date
                    df.set_index('time', inplace=True)
                    df.index = pd.to_datetime(df.index)
                    df = df.loc[str(start_date):str(end_date)]
                    data = df[['close']].rename(columns={'close': 'Close'}).copy()
            except Exception as e:
                st.error(f"cryptocompare also failed for {ticker}: {e}")

        if data is not None and not data.empty:
            data['Ticker'] = ticker
            data.index.name = 'Date'
            combined_data.append(data)
        else:
            st.warning(f"Could not load any data for '{ticker}'. Skipping.")
    
    if not combined_data:
        st.error("Failed to load data for all provided tickers.")
        return None

    final_df = pd.concat(combined_data)
    final_df.index = pd.to_datetime(final_df.index)
    return final_df

# --- State Classification & Matrix ---
def classify_states(data: pd.DataFrame, inc_threshold: float, dec_threshold: float) -> pd.DataFrame:
    df = data.copy()
    close_prices = df['Close']
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]
    
    df['Return'] = close_prices.pct_change(fill_method=None) * 100
    df.dropna(inplace=True)

    conditions = [df['Return'] >= inc_threshold, df['Return'] <= -dec_threshold]
    choices = ['Increase', 'Decrease']
    df['State'] = np.select(conditions, choices, default='Stable')
    return df

def build_matrix(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    states = ['Increase', 'Stable', 'Decrease']
    data['Next_State'] = data['State'].shift(-1)
    data.dropna(inplace=True)
    
    transition_counts = pd.crosstab(data['State'], data['Next_State'])
    transition_counts = transition_counts.reindex(index=states, columns=states, fill_value=0)
    
    row_sums = transition_counts.sum(axis=1)
    row_sums[row_sums == 0] = 1 # Avoid division by zero
    
    transition_probs = transition_counts.div(row_sums, axis=0)
    return transition_counts, transition_probs.fillna(0)

# --- Simulation ---
def get_most_likely_path(last_price: float, last_state: str, trans_matrix: pd.DataFrame, n_days: int, volatility: Dict[str, Tuple[float, float]]) -> List[float]:
    """
    Calculates the single most likely path based on the transition matrix.
    """
    path = [last_price]
    current_state = last_state
    for _ in range(n_days):
        # Deterministically choose the next state with the highest probability
        next_state = trans_matrix.loc[current_state].idxmax()
        
        # Use the average expected return for that state
        min_ret, max_ret = volatility[next_state]
        ret = (min_ret + max_ret) / 2 / 100
        
        path.append(path[-1] * (1 + ret))
        current_state = next_state
    return path

def simulate_paths(last_price: float, last_state: str, trans_matrix: pd.DataFrame, n_days: int, n_runs: int, volatility: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    states = trans_matrix.index.to_list()
    all_paths = []
    for _ in range(n_runs):
        path = [last_price]
        current_state = last_state
        for _ in range(n_days):
            next_state = np.random.choice(states, p=trans_matrix.loc[current_state])
            min_ret, max_ret = volatility[next_state]
            ret = np.random.uniform(min_ret, max_ret) / 100
            path.append(path[-1] * (1 + ret))
            current_state = next_state
        all_paths.append(path)
    
    sim_df = pd.DataFrame(all_paths).T
    sim_df.columns = [f"Run_{i+1}" for i in range(n_runs)]
    sim_df.index.name = "Day"
    return sim_df

# --- Plotting ---
def plot_historical_data(state_data: pd.DataFrame):
    state_data_reset = state_data.reset_index().copy()

    # Robustly handle price data to ensure it's a Series
    close_prices = state_data['Close']
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    min_price, max_price = close_prices.min(), close_prices.max()
    padding = (max_price - min_price) * 0.1
    domain = [min_price - padding, max_price + padding]

    if pd.isna(domain[0]) or pd.isna(domain[1]):
        st.warning("Could not plot historical data due to invalid price range.")
        return

    color_map = {'Increase': 'rgba(0, 255, 0, 0.2)', 'Stable': 'rgba(255, 255, 0, 0.2)', 'Decrease': 'rgba(255, 0, 0, 0.2)'}
    state_data_reset['color'] = state_data_reset['State'].map(color_map)

    base = alt.Chart(state_data_reset).encode(x='Date:T')
    bands = base.mark_rect().encode(
        y=alt.Y('min:Q', title="Price").scale(domain=domain),
        y2='max:Q',
        color=alt.Color('color:N', scale=None)
    ).transform_calculate(min=str(domain[0]), max=str(domain[1]))
    
    line = base.mark_line().encode(y=alt.Y('Close:Q', scale=alt.Scale(domain=domain)), tooltip=['Date:T', 'Close:Q', 'Return:Q', 'State:N'])
    chart = (bands + line).properties(title="Historical Price with State Bands", height=300).interactive()
    st.altair_chart(chart, use_container_width=True)

def plot_simulation_paths(sim_df: pd.DataFrame, most_likely_path: Optional[pd.Series] = None):
    sim_df_melted = sim_df.melt(var_name='Run', value_name='Price', ignore_index=False).reset_index()
    line = alt.Chart(sim_df_melted).mark_line(opacity=0.1).encode(
        x='Day:Q',
        y=alt.Y('Price:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('Run:N', legend=None)
    )
    
    stats_df = sim_df.agg(['mean', 'std'], axis=1)
    stats_df['ci_upper'] = stats_df['mean'] + 1.96 * stats_df['std']
    stats_df['ci_lower'] = stats_df['mean'] - 1.96 * stats_df['std']
    stats_df.reset_index(inplace=True)
    
    mean_line = alt.Chart(stats_df).mark_line(color='red', size=2).encode(x='Day:Q', y='mean:Q')
    confidence_area = alt.Chart(stats_df).mark_area(opacity=0.3, color='red').encode(x='Day:Q', y='ci_upper:Q', y2='ci_lower:Q')
    
    chart = (line + confidence_area + mean_line)

    if most_likely_path is not None:
        most_likely_df = most_likely_path.reset_index()
        most_likely_df.columns = ['Day', 'Price']
        
        most_likely_line = alt.Chart(most_likely_df).mark_line(
            color='yellow', 
            size=3, 
            strokeDash=[5, 5],
            tooltip="Most Likely Path"
        ).encode(
            x='Day:Q',
            y='Price:Q'
        )
        chart += most_likely_line

    chart = chart.properties(
        title="Simulated Price Paths with 95% Confidence Interval", 
        height=400
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def main():
    st.title("📈 Markov Chain Stock & Crypto Simulator")

    # --- Sidebar Inputs ---
    st.sidebar.header("1. Input Data")
    tickers_input = st.sidebar.text_input("Enter Ticker(s) (comma-separated)", "TSLA")
    end_date, start_date = date.today(), st.sidebar.date_input("Start Date", date.today() - timedelta(days=2*365))
    end_date_input = st.sidebar.date_input("End Date", end_date)

    if st.sidebar.button("Load Data", key="load_data"):
        with st.spinner("Loading data..."):
            tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
            raw_data = load_data(tickers, start_date, end_date_input)
            if raw_data is not None:
                st.session_state.raw_data = raw_data
                st.session_state.tickers = tickers
                st.session_state.selected_ticker = tickers[0]
                st.success("Data loaded successfully!")

    if 'raw_data' in st.session_state:
        all_tickers = st.session_state.tickers
        st.session_state.selected_ticker = st.sidebar.selectbox("Select Ticker for Analysis", options=all_tickers, index=all_tickers.index(st.session_state.get('selected_ticker', all_tickers[0])))
        data = st.session_state.raw_data[st.session_state.raw_data['Ticker'] == st.session_state.selected_ticker].copy()
        data.drop(columns=['Ticker'], inplace=True)
        st.header(f"Analysis for: {st.session_state.selected_ticker}")

        # --- State Classification ---
        st.sidebar.header("2. State Classification")
        close_prices = data['Close']
        if isinstance(close_prices, pd.DataFrame): close_prices = close_prices.iloc[:, 0]
        returns = close_prices.pct_change(fill_method=None).dropna() * 100
        
        auto_inc = returns[returns > 0].quantile(0.75) if not returns[returns > 0].empty else 1.0
        auto_dec = abs(returns[returns < 0].quantile(0.25)) if not returns[returns < 0].empty else 1.0

        if st.sidebar.checkbox("Auto-set from History", value=True, key="auto_thresh"):
            inc_thresh, dec_thresh = auto_inc, auto_dec
            st.sidebar.markdown(f"Auto Increase: `{inc_thresh:.2f}%` | Auto Decrease: `{dec_thresh:.2f}%`")
        else:
            inc_thresh = st.sidebar.slider("Increase Threshold", 0.1, 5.0, 1.0, 0.1, key="inc_slider", format="%.1f%%")
            dec_thresh = st.sidebar.slider("Decrease Threshold", 0.1, 5.0, 1.0, 0.1, key="dec_slider", format="%.1f%%")
        
        state_data = classify_states(data, inc_thresh, dec_thresh)
        counts_matrix, probs_matrix = build_matrix(state_data.copy())
        
        st.markdown("---")
        st.header("📈 Historical Performance")
        plot_historical_data(state_data)

        with st.expander("🧮 View Transition Matrix Details"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Counts"); st.dataframe(counts_matrix)
                st.subheader("Probabilities"); st.dataframe(probs_matrix.style.format("{:.2%}"))
            with col2:
                fig, ax = plt.subplots(); sns.heatmap(probs_matrix, annot=True, fmt=".2%", cmap="viridis", ax=ax); st.pyplot(fig)

        # --- Simulation ---
        st.sidebar.header("3. Monte Carlo Simulation")
        n_days = st.sidebar.slider("Days to Project", 10, 250, 90)
        n_runs = st.sidebar.slider("Number of Runs", 1, 500, 100)
        
        st.sidebar.subheader("Volatility per State (% move)")
        if st.sidebar.checkbox("Auto-set from History", value=True, key="auto_vol"):
            vol_stats = state_data.groupby('State')['Return'].agg(['min', 'max'])
            
            # Rebuild volatility dict robustly, providing defaults for missing states
            volatility = {}
            for state in ['Increase', 'Stable', 'Decrease']:
                if state in vol_stats.index:
                    volatility[state] = (vol_stats.loc[state, 'min'], vol_stats.loc[state, 'max'])
                else:
                    # Provide sensible defaults if a state never occurred
                    if state == 'Increase':
                        volatility[state] = (inc_thresh, inc_thresh + 1.0)
                    elif state == 'Stable':
                        volatility[state] = (-dec_thresh, inc_thresh)
                    else: # Decrease
                        volatility[state] = (-dec_thresh - 1.0, -dec_thresh)

            st.sidebar.dataframe(vol_stats.style.format("{:.2f}"))
        else:
            volatility = {
                'Increase': (st.sidebar.number_input("Inc Min", value=0.5), st.sidebar.number_input("Inc Max", value=3.0)),
                'Stable': (st.sidebar.number_input("Stab Min", value=-0.5), st.sidebar.number_input("Stab Max", value=0.5)),
                'Decrease': (st.sidebar.number_input("Dec Min", value=-3.0), st.sidebar.number_input("Dec Max", value=-0.5))
            }

        if st.sidebar.button("Run Simulation", key="run_sim"):
            # Robustly get last price and state to ensure they are scalar values
            last_price = state_data['Close'].iloc[-1]
            if isinstance(last_price, pd.Series):
                last_price = last_price.iloc[0]
            last_price = float(last_price)

            last_state = state_data['State'].iloc[-1]
            if isinstance(last_state, pd.Series):
                last_state = last_state.iloc[0]

            with st.spinner("Running simulation..."):
                st.session_state.sim_results = simulate_paths(last_price, last_state, probs_matrix, n_days, n_runs, volatility)
                st.session_state.most_likely_path = get_most_likely_path(last_price, last_state, probs_matrix, n_days, volatility)
                st.success("Simulation complete!")

        if 'sim_results' in st.session_state:
            st.markdown("---"); st.header("🔮 Simulation Results")
            most_likely_series = pd.Series(st.session_state.most_likely_path) if 'most_likely_path' in st.session_state else None
            plot_simulation_paths(st.session_state.sim_results, most_likely_path=most_likely_series)

            st.header("📊 Key Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Steady-State Probabilities")
                try:
                    eigenvals, eigenvects = np.linalg.eig(probs_matrix.T.to_numpy())
                    steady_state_vector = np.real(eigenvects[:, np.isclose(eigenvals, 1)][:, 0])
                    steady_df = pd.DataFrame(steady_state_vector / steady_state_vector.sum(), index=probs_matrix.index, columns=['Probability'])
                    st.dataframe(steady_df.style.format("{:.2%}"))
                except Exception as e:
                    st.warning(f"Could not calculate steady-state: {e}")
                
                csv = st.session_state.sim_results.to_csv().encode('utf-8')
                st.download_button("Download Simulation Data (CSV)", csv, f"{st.session_state.selected_ticker}_sim.csv", "text/csv")
            
            with col2:
                final_prices = st.session_state.sim_results.iloc[-1]
                st.subheader(f"Expected Price Range (Day {n_days})")
                st.json({"Min": final_prices.min(), "Mean": final_prices.mean(), "Max": final_prices.max()})

                st.subheader("Final Price Distribution")
                fig, ax = plt.subplots(); sns.histplot(final_prices, kde=True, ax=ax, bins=50)
                ax.axvline(final_prices.mean(), color='r', linestyle='--', label=f"Mean: ${final_prices.mean():.2f}"); ax.legend()
                st.pyplot(fig)
                
                buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
                st.download_button("Download Chart (PNG)", buf, f"{st.session_state.selected_ticker}_dist.png", "image/png")

    # --- Footer ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset All"):
        st.session_state.clear(); st.rerun()
    st.sidebar.info("Disclaimer: Not financial advice. For educational use only.")

if __name__ == "__main__":
    main() 
