# Markov Chain Stock & Crypto Simulator: Build Instructions

This document provides a step-by-step guide to building the interactive Streamlit application from scratch. Following these instructions will ensure a clean, robust, and error-free final product.

## Project Goal

Create an interactive Streamlit app that pulls live price data for any equity or crypto ticker, estimates a 3-state Markov chain from recent history, and lets the user visually explore future price paths with fully adjustable sliders.

## Step 1: Project Setup

1.  **Create `requirements.txt`**: This file lists all the Python dependencies needed for the project.
2.  **Create `markov_stock_sim.py`**: This will be the single Python file containing all the application code.

## Step 2: Initial App Layout (`markov_stock_sim.py`)

1.  **Import Libraries**: Import all necessary modules (`streamlit`, `pandas`, `numpy`, `yfinance`, `cryptocompare`, `altair`, etc.) and include type hints.
2.  **Page Configuration**: Set the page layout to "wide" and define a title and icon using `st.set_page_config()`.
3.  **Main Function**: Define a `main()` function to house the application logic and call it inside an `if __name__ == "__main__:"` block.
4.  **UI Scaffolding**:
    *   Add the main title: `st.title("📈 Markov Chain Stock & Crypto Simulator")`.
    *   Set up the sidebar with `st.sidebar.header("1. Input Data")`.
    *   Add input widgets: a text input for tickers, two date inputs for the range, and a "Load Data" button.
    *   Add a "Reset All" button and a disclaimer to the bottom of the sidebar.

## Step 3: Data Loading & Selection

1.  **`load_data` Function**:
    *   Create a function `load_data(tickers, start_date, end_date)` decorated with `@st.cache_data(ttl=6*60*60)` to cache results.
    *   Inside, loop through tickers. Use a `try/except` block to attempt downloading data with `yfinance`.
    *   If `yfinance` fails, fall back to `cryptocompare` in another `try/except` block.
    *   Standardize the data from both sources into a DataFrame with 'Date' as the index and a 'Close' column.
    *   Use `.copy()` when creating DataFrame slices to prevent `SettingWithCopyWarning`.
    *   Return a single combined DataFrame or `None` on failure.
2.  **Integrate into `main()`**:
    *   When the "Load Data" button is clicked, call `load_data()` and store the result in `st.session_state`.
    *   Add a spinner to show progress while loading.
    *   Add a `st.sidebar.selectbox` to allow the user to choose which of the loaded tickers to analyze.

## Step 4: State Classification and Transition Matrix

1.  **`classify_states` Function**:
    *   Create a function `classify_states(data, inc_threshold, dec_threshold)`.
    *   Calculate daily returns using `.pct_change()`. Ensure you are operating on a `pd.Series` (e.g., `data['Close'].iloc[:, 0]` if it's a DataFrame) to prevent errors.
    *   Use `np.select` to classify returns into 'Increase', 'Stable', or 'Decrease' states.
2.  **`build_matrix` Function**:
    *   Create a function `build_matrix(data)`.
    *   Create a 'Next_State' column by shifting the 'State' column.
    *   Use `pd.crosstab` to create a transition counts matrix.
    *   Normalize the counts to get the transition probability matrix.
3.  **Integrate into `main()`**:
    *   Add a sidebar section "2. State Classification" with sliders for the thresholds.
    *   Add a checkbox "Auto-set from History" that, when checked, calculates thresholds based on the data's return distribution (e.g., quartiles).
    *   Call `classify_states` and `build_matrix`.
    *   Display the resulting matrices and a `seaborn` heatmap inside a `st.expander`.

## Step 5: Historical Data Visualization

1.  **`plot_historical_data` Function**:
    *   Create a function `plot_historical_data(state_data)`.
    *   Use `altair` to build the chart.
    *   **Crucially**, to add detail, calculate the y-axis domain (`min_price`, `max_price`) from the data and add some padding. Pass this domain to `alt.Scale(domain=[...])` for the y-axis encoding. This will "zoom in" on the price action.
    *   Use `mark_rect` for the background color bands representing states, ensuring their y-span matches the chart's new dynamic domain.
    *   Overlay a `mark_line` for the price.
2.  **Integrate into `main()`**: Call this function after state classification.

## Step 6: Monte Carlo Simulation

1.  **`simulate_paths` Function**:
    *   Create a function `simulate_paths(...)` that takes the last price, last state, transition matrix, and simulation parameters as input.
    *   Loop for `n_runs`. In an inner loop for `n_days`, predict the next state using `np.random.choice` with the transition probabilities.
    *   Generate a random return for that state based on the volatility parameters.
    *   Calculate and append the next price.
    *   Return a DataFrame of all simulated paths.
2.  **Integrate into `main()`**:
    *   Add a sidebar section "3. Monte Carlo Simulation" with sliders for projection days and number of runs.
    *   Add a subheader for "Volatility per State" and provide number inputs for min/max moves.
    *   Add a checkbox "Auto-set from History" that, when checked, uses the historical min/max return observed for each state as the volatility.
    *   Add a "Run Simulation" button that calls `simulate_paths()` and stores the results in `st.session_state`.

## Step 7: Simulation Output & Visualization

1.  **`plot_simulation_paths` Function**:
    *   Create a function `plot_simulation_paths(sim_df)`.
    *   Use `altair` to plot all simulation runs with low opacity.
    *   Calculate the mean and 95% confidence interval across all runs for each day.
    *   Overlay the mean as a distinct line and the confidence interval as a shaded `mark_area`.
2.  **Statistics Panel**: In `main()`, after a simulation is run:
    *   Calculate and display steady-state probabilities from the transition matrix.
    *   Show the expected price range (min, mean, max) on the final day.
    *   Create and display a histogram of the final day's price distribution.
3.  **Download Buttons**:
    *   Add a `st.download_button` for the full simulation data as a CSV.
    *   Add another `st.download_button` for the final distribution chart as a PNG.

## Step 8: Running the Application

1.  Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the Streamlit app. Use the `--browser.gatherUsageStats=false` flag to avoid any startup prompts.
    ```bash
    streamlit run markov_stock_sim.py --browser.gatherUsageStats=false
    ``` 