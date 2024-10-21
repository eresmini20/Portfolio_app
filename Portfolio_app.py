import streamlit as st
import pandas as pd
import numpy as np
from stqdm import stqdm
import matplotlib.pyplot as plt


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import scipy.optimize as sco
import time

# # Load the returns data
# df = pd.read_excel(
#     r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_2ème\1er_semestre\Quantitative Risk and Asset Management 2\Projet_PortfolioOptimization\Data\DS_RI_T_USD_M.xlsx",
#     header=None,
# )

# # Transpose the DataFrame
# df = df.T

# # Set the second row (index 1) as the column headers
# df.columns = df.iloc[0]
# column_names = df.iloc[1].values
# print(column_names)

# # Remove the first two rows as they are now redundant
# df = df.drop([0, 1])

# # Rename the first column to 'Date' and set it as the index
# df = df.rename(columns={df.columns[0]: "Date"}).set_index("Date")

# # Convert all entries to floats for uniformity and handling
# df = df.astype(float)

# # Initialize a set to keep track of dropped stocks
# dropped_stocks = set()

# # 1. Remove stocks with initial zero prices
# initial_zeros = df.iloc[0] == 0
# dropped_stocks.update(df.columns[initial_zeros])
# print(f"Initial zero : {df.columns[initial_zeros]}")
# df = df.loc[:, ~initial_zeros]

# # 2. Remove stocks that ever drop to zero
# ever_zeros = (df == 0).any()
# dropped_stocks.update(df.columns[ever_zeros])
# print(f"Ever zero : {df.columns[ever_zeros]}")
# df = df.loc[:, ~ever_zeros]

# # 3. Remove stocks that do not recover after dropping to zero
# max_prior = df.cummax()
# recovered = ((df / max_prior.shift()) > 0.1).any()
# non_recovered = df.columns[~recovered]
# dropped_stocks.update(non_recovered)
# print(f"Non recovered : {non_recovered}")
# df = df.loc[:, recovered]

# # # Filter based on sector information
# # static_file = pd.read_excel(
# #     r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_2ème\1er_semestre\Quantitative Risk and Asset Management 2\Projet_PortfolioOptimization\Data\Static.xlsx"
# # )
# # sectors = ["Energy", "Materials", "Utilities", "Industrials"]
# # companies = static_file[static_file["GICSSectorName"].isin(sectors)]
# # isin_list = companies["ISIN"].tolist()

# # # Identify stocks that are not in the highly polluting sectors
# # non_polluting_stocks = set(df.columns) - set(isin_list)
# # dropped_stocks.update(non_polluting_stocks)

# # df = df[df.columns.intersection(isin_list)]


# # # Reset column names to the original names after modifications
# # df.columns = column_names[
# #     1 : len(df.columns) + 1
# # ]  # Skip the first name since it corresponds to the Date column

# # Proceed with any further data processing, such as calculating returns
# monthly_returns = df.pct_change()
# monthly_returns = monthly_returns.drop(monthly_returns.index[0])

# # Handling NaN and infinite values
# monthly_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
# monthly_returns.interpolate(method="linear", axis=0, inplace=True)
# monthly_returns.fillna(method="ffill", axis=0, inplace=True)
# monthly_returns.fillna(method="bfill", axis=0, inplace=True)

# # Display results
# print("Remaining NaN values in monthly returns:", monthly_returns.isnull().sum().sum())
# df.to_csv("Cleaned_df.csv", index=True)
# monthly_returns.to_csv("Cleaned_df_returns.csv", index=True)


data = pd.read_csv("Cleaned_df.csv", index_col="Date")
static_data = pd.read_excel(
    r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_2ème\1er_semestre\Quantitative Risk and Asset Management 2\Projet_PortfolioOptimization\Data\Static.xlsx"
)
print(data.head())
# data = data[
#     [
#         "AN8068571086",
#         "ARALUA010258",
#         "ARP125991090",
#         "ARSIDE010029",
#         "AT00000VIE62",
#         "AT0000652011",
#         "AT0000743059",
#         "AT0000746409",
#         "AT0000831706",
#         "AT0000908504",
#         "ZAE000109815",
#         "ZAE000117321",
#         "ZAE000134961",
#         "ZAE000170049",
#         "ZAE000179420",
#         "ZAE000191342",
#         "ZAE000255915",
#         "ZAE000298253",
#         "ZAE000302618",
#         "ZAE000322095",
#     ]
# ]
assets = data.columns.tolist()


# Risk aversion quiz using a form
def risk_aversion_quiz():
    st.header("Risk Aversion Quiz")
    with st.form(key="quiz_form"):
        score = 0
        # Question 1
        q1 = st.radio(
            "How would you describe your investment experience?",
            ("No experience", "Some experience", "Experienced", "Very experienced"),
            key="q1",
        )
        # Question 2
        q2 = st.radio(
            "Which statement best describes your attitude towards investment risk?",
            (
                "Prefer minimal risk",
                "Accept some risk",
                "Comfortable with significant risk",
                "Seek maximum returns with high risk",
            ),
            key="q2",
        )
        # Question 3
        q3 = st.radio(
            "How long is your investment horizon?",
            ("Less than 1 year", "1-3 years", "3-5 years", "More than 5 years"),
            key="q3",
        )
        # Question 4
        q4 = st.radio(
            "If your investment dropped 20% in value, how would you react?",
            (
                "Sell immediately",
                "Consider selling",
                "Hold expecting rebound",
                "Buy more",
            ),
            key="q4",
        )
        # Question 5
        q5 = st.radio(
            "Which portfolio would you prefer?",
            (
                "Low return, low risk",
                "Moderate return, moderate risk",
                "High return, high risk",
                "Very high return, very high risk",
            ),
            key="q5",
        )

        submit_quiz = st.form_submit_button("Submit Quiz")

    if submit_quiz:
        # Scoring
        score += {
            "No experience": 1,
            "Some experience": 2,
            "Experienced": 3,
            "Very experienced": 4,
        }[q1]
        score += {
            "Prefer minimal risk": 1,
            "Accept some risk": 2,
            "Comfortable with significant risk": 3,
            "Seek maximum returns with high risk": 4,
        }[q2]
        score += {
            "Less than 1 year": 1,
            "1-3 years": 2,
            "3-5 years": 3,
            "More than 5 years": 4,
        }[q3]
        score += {
            "Sell immediately": 1,
            "Consider selling": 2,
            "Hold expecting rebound": 3,
            "Buy more": 4,
        }[q4]
        score += {
            "Low return, low risk": 1,
            "Moderate return, moderate risk": 2,
            "High return, high risk": 3,
            "Very high return, very high risk": 4,
        }[q5]

        # Calculate risk aversion
        risk_aversion = (25 - score) / 10  # Higher score indicates lower risk aversion
        st.write("Your risk aversion score is:", score)
        st.write("Estimated risk aversion coefficient:", risk_aversion)
        st.session_state["risk_aversion"] = risk_aversion
    else:
        st.stop()


# Initialize risk aversion
if "risk_aversion" not in st.session_state:
    risk_aversion_quiz()
else:
    risk_aversion = st.session_state["risk_aversion"]

# Initialize session state variables
if "optimization_run" not in st.session_state:
    st.session_state["optimization_run"] = False
if "weights" not in st.session_state:
    st.session_state["weights"] = None
if "mean_returns" not in st.session_state:
    st.session_state["mean_returns"] = None
if "cov_matrix" not in st.session_state:
    st.session_state["cov_matrix"] = None
if "previous_params" not in st.session_state:
    st.session_state["previous_params"] = None


# Constraints
st.header("Constraints Selection")
long_only = st.checkbox("Long only", value=True)
use_sentiment = st.checkbox("Use sentiment data?")
sectors_filter = st.checkbox("Sectors filter")
country_filter = st.checkbox("Country filter")
carbon_footprint = st.checkbox("Carbon footprint")
min_weight_constraint = st.checkbox("Minimum weight constraint")
max_weight_constraint = st.checkbox("Maximum weight constraint")
leverage_limit = st.checkbox("Leverage limit")

# Risk-Free Asset Inclusion
st.header("Risk-Free Asset Inclusion")
include_risk_free_asset = st.checkbox(
    "Include a Risk-Free Asset in the Optimization?", value=True
)

if include_risk_free_asset:
    risk_free_rate = st.number_input(
        "Enter the risk-free rate (e.g., 0.01 for 1%)",
        value=0.01,
        min_value=0.0,
        max_value=1.0,
    )
else:
    risk_free_rate = 0.0


# Additional inputs
if sectors_filter:
    sectors = static_data["GICSSectorName"].unique().tolist()
    selected_sectors = st.multiselect("Select sectors to include", sectors)
else:
    selected_sectors = None


if country_filter:
    countries = static_data["Country"].unique().tolist()
    selected_countries = st.multiselect("Select countries to include", countries)
else:
    selected_countries = None

if min_weight_constraint:
    min_weight_value = (
        st.number_input("Minimum weight (%)", min_value=0.0, max_value=100.0, value=0.0)
        / 100
    )
else:
    min_weight_value = 0.0
if max_weight_constraint:
    max_weight_value = (
        st.number_input(
            "Maximum weight (%)", min_value=0.0, max_value=100.0, value=100.0
        )
        / 100
    )
else:
    max_weight_value = 1.0

if leverage_limit:
    leverage_limit_value = st.number_input("Leverage limit", min_value=0.0, value=1.0)
else:
    leverage_limit_value = 1.0


# Function to get current parameters
def get_current_params():
    params = {
        "long_only": long_only,
        "use_sentiment": use_sentiment,
        "sectors_filter": sectors_filter,
        "selected_sectors": (
            tuple(sorted(selected_sectors)) if selected_sectors else None
        ),
        "country_filter": country_filter,
        "selected_countries": (
            tuple(sorted(selected_countries)) if selected_countries else None
        ),
        "carbon_footprint": carbon_footprint,
        "min_weight_constraint": min_weight_constraint,
        "min_weight_value": min_weight_value,
        "max_weight_constraint": max_weight_constraint,
        "max_weight_value": max_weight_value,
        "leverage_limit": leverage_limit,
        "leverage_limit_value": leverage_limit_value,
        "include_risk_free_asset": include_risk_free_asset,
        "risk_free_rate": risk_free_rate,
        # Include risk_aversion if it can change
        "risk_aversion": risk_aversion,
    }
    return params


# Get current parameters
current_params = get_current_params()
previous_params = st.session_state.get("previous_params", None)

# Compare current and previous parameters
if previous_params is not None and current_params != previous_params:
    st.session_state["optimization_run"] = False

# Update previous parameters
st.session_state["previous_params"] = current_params

# Output the total number of stocks before filtering
st.write(f"Total number of stocks before filtering: {data.shape[1]}")


# Filtering based on sectors and countries using ISIN numbers
def filter_stocks(data, sectors=None, countries=None):
    all_isins = data.columns.tolist()

    if sectors is not None:
        companies_sector = static_data[static_data["GICSSectorName"].isin(sectors)]
        sector_isins = companies_sector["ISIN"].tolist()
        all_isins = list(set(all_isins).intersection(set(sector_isins)))
        st.write(f"Total number of stocks after sector filtering: {len(all_isins)}")

    if countries is not None:
        companies_country = static_data[static_data["Country"].isin(countries)]
        country_isins = companies_country["ISIN"].tolist()
        all_isins = list(set(all_isins).intersection(set(country_isins)))
        st.write(f"Total number of stocks after country filtering: {len(all_isins)}")

    data_filtered = data[all_isins]
    return data_filtered


# Apply filtering
data = filter_stocks(data, sectors=selected_sectors, countries=selected_countries)

# Assets list after filtering
assets = data.columns.tolist()


# Efficient Frontier Calculation
def calculate_efficient_frontier(
    mean_returns,
    cov_matrix,
    risk_free_rate,
    include_risk_free_asset,
    long_only,
    leverage_limit_value,
    min_weight_value,
    max_weight_value,
):
    target_returns = np.linspace(mean_returns.min() * 12, mean_returns.max() * 12, 20)
    frontier_volatility = []
    frontier_returns = []
    frontier_weights = []

    num_assets = len(mean_returns)

    for target_return in stqdm(target_returns, desc="Computing the frontier... "):
        # Constraints for the optimization
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Sum of weights equals 1
            {
                "type": "eq",
                "fun": lambda x: np.sum(x * mean_returns * 12)
                - target_return,  # Target return constraint
            },
        ]

        # Leverage limit constraint
        if leverage_limit:
            constraints = [
                {
                    "type": "ineq",
                    "fun": lambda x: leverage_limit_value - np.sum(x),
                },
                {
                    "type": "eq",
                    "fun": lambda x: np.sum(x * mean_returns * 12)
                    - target_return,  # Target return constraint
                },
            ]

        # Bounds
        if long_only:
            # Apply minimum and maximum weight constraints
            bounds = tuple(
                (max(min_weight_value, 0.0), min(max_weight_value, 1.0))
                for _ in range(num_assets)
            )
        else:
            # Allow short selling within the leverage limit
            bounds = tuple(
                (-leverage_limit_value, leverage_limit_value) for _ in range(num_assets)
            )

        # Optimization
        result = minimize(
            lambda x: np.sqrt(
                np.dot(x.T, np.dot(cov_matrix * 12, x))
            ),  # Minimize volatility
            num_assets * [1.0 / num_assets],  # Initial guess
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            frontier_volatility.append(result.fun)
            frontier_returns.append(target_return)
            frontier_weights.append(result.x)
        else:
            # Handle optimization failure
            st.warning(f"Optimization failed for target return {target_return:.2%}")
            pass

    return frontier_volatility, frontier_returns, frontier_weights


# Efficient Frontier Plotting Function
def plot_efficient_frontier(
    mean_returns,
    cov_matrix,
    risk_free_rate,
    include_risk_free_asset,
    weights_optimal,
    long_only,
    leverage_limit_value,
    min_weight_value,
    max_weight_value,
):
    # Calculate the efficient frontier with updated constraints
    frontier_volatility, frontier_returns, frontier_weights = (
        calculate_efficient_frontier(
            mean_returns,
            cov_matrix,
            risk_free_rate,
            include_risk_free_asset,
            long_only,
            leverage_limit_value,
            min_weight_value,
            max_weight_value,
        )
    )

    # Generate random portfolios
    st.info("Generating random portfolios to display inefficient portfolios...")
    num_portfolios = 1000
    results = np.zeros((3, num_portfolios))

    np.random.seed(42)  # For reproducibility

    for i in stqdm(range(num_portfolios), desc="Generating random weights... "):
        # Generate random weights
        weights = np.random.dirichlet(np.ones(len(mean_returns)))
        weights /= np.sum(weights)

        # Apply weight bounds
        if long_only:
            weights = np.clip(
                weights, max(min_weight_value, 0.0), min(max_weight_value, 1.0)
            )
        else:
            weights = np.clip(weights, -leverage_limit_value, leverage_limit_value)

        # Check if weights satisfy leverage limit
        if leverage_limit:
            if np.sum(np.abs(weights)) > leverage_limit_value:
                continue  # Skip weights that exceed leverage limit

        # Calculate portfolio performance
        portfolio_return = np.sum(mean_returns * weights) * 12  # Annualized return
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix * 12, weights))
        )  # Annualized volatility

        # Sharpe Ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio

    # Remove zeros (portfolios that were skipped)
    mask = results[0, :] != 0
    results = results[:, mask]

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(
        results[0],
        results[1],
        c=results[2],
        cmap="viridis",
        s=2,
        alpha=0.4,
        label="Random Portfolios",
    )
    plt.colorbar(label="Sharpe Ratio")
    plt.plot(
        frontier_volatility,
        frontier_returns,
        "r--",
        linewidth=3,
        label="Efficient Frontier",
    )

    if include_risk_free_asset:
        # # Calculate the tangency portfolio
        # def neg_sharpe_ratio(weights):
        #     portfolio_return = np.sum(mean_returns * weights) * 12
        #     portfolio_volatility = np.sqrt(
        #         np.dot(weights.T, np.dot(cov_matrix * 12, weights))
        #     )
        #     sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        #     return -sharpe_ratio

        # # Constraints for tangency portfolio
        # constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        # if leverage_limit:
        #     constraints = [
        #         {
        #             "type": "ineq",
        #             "fun": lambda x: leverage_limit_value - np.sum(np.abs(x)),
        #         }
        #     ]

        # # Bounds
        # if long_only:
        #     bounds = tuple(
        #         (max(min_weight_value, 0.0), min(max_weight_value, 1.0))
        #         for _ in range(num_assets)
        #     )
        # else:
        #     bounds = tuple(
        #         (-leverage_limit_value, leverage_limit_value) for _ in range(num_assets)
        #     )

        # result = minimize(
        #     neg_sharpe_ratio,
        #     num_assets * [1.0 / num_assets],
        #     method="SLSQP",
        #     bounds=bounds,
        #     constraints=constraints,
        # )

        # if result.success:
        tangency_weights = weights_optimal
        tangency_return = np.sum(mean_returns * tangency_weights) * 12
        tangency_volatility = np.sqrt(
            np.dot(tangency_weights.T, np.dot(cov_matrix * 12, tangency_weights))
        )

        # Plot the Capital Market Line
        cml_x = [0, tangency_volatility]
        cml_y = [risk_free_rate, tangency_return]
        plt.plot(
            cml_x, cml_y, color="green", linestyle="--", label="Capital Market Line"
        )

        # Highlight the tangency portfolio
        plt.scatter(
            tangency_volatility,
            tangency_return,
            marker="*",
            color="red",
            s=500,
            label="Tangency Portfolio",
        )
        # else:
        #     st.warning("Failed to compute the tangency portfolio.")
    else:
        # Highlight the optimal portfolio
        portfolio_return = np.sum(mean_returns * weights_optimal) * 12
        portfolio_volatility = np.sqrt(
            np.dot(weights_optimal.T, np.dot(cov_matrix * 12, weights_optimal))
        )
        plt.scatter(
            portfolio_volatility,
            portfolio_return,
            marker="*",
            color="red",
            s=500,
            label="Optimal Portfolio",
        )

    plt.title("Efficient Frontier with Random Portfolios")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Expected Returns")
    plt.legend()
    st.pyplot(plt)


# Optimization function
def optimize_portfolio(
    data,
    long_only,
    min_weight,
    max_weight,
    leverage_limit_value,
    risk_free_rate,
    include_risk_free_asset,
    risk_aversion,
):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    initial_weights = num_assets * [
        1.0 / num_assets,
    ]

    # Constraints
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    if leverage_limit:
        constraints = [
            {"type": "ineq", "fun": lambda x: leverage_limit_value - np.sum(x)}
        ]

    # Bounds
    if long_only:
        bounds = tuple(
            (max(min_weight, 0.0), min(max_weight, 1.0)) for _ in range(num_assets)
        )
    else:
        bounds = tuple((-1.0, 1.0) for _ in range(num_assets))

    # Objective functions
    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(mean_returns * weights) * 12
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix * 12, weights))
        )
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    def negative_utility(weights):
        portfolio_return = np.sum(mean_returns * weights) * 12
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix * 12, weights))
        )
        utility = portfolio_return - 0.5 * risk_aversion * (portfolio_volatility**2)
        return -utility

    # Choose the appropriate objective function
    if include_risk_free_asset:
        objective_function = neg_sharpe_ratio
    else:
        objective_function = negative_utility

    # Progress bar
    progress_bar = st.progress(0)
    iteration_container = st.empty()

    max_iterations = (
        1000  # Set maximum number of iterations for estimation if taking too long
    )

    iteration_counter = {"n_iter": 0}

    # Callback function to update progress
    def callbackF(xk):
        iteration_counter["n_iter"] += 1
        progress = iteration_counter["n_iter"] / max_iterations
        progress_bar.progress(min(progress, 1.0))
        iteration_container.text(f"Iteration: {iteration_counter['n_iter']}")

    # Estimated time indicator
    st.info("Estimated time to complete optimization: depends on data and constraints.")

    with st.spinner("Optimization in progress..."):
        start_time = time.time()
        result = minimize(
            objective_function,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iterations},
            callback=callbackF,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

    progress_bar.empty()
    iteration_container.empty()

    st.success(f"Optimization completed in {elapsed_time:.2f} seconds")
    return result, mean_returns, cov_matrix


# Run the functions

if st.button("Run Optimization"):
    tangency_result, tangency_mean_returns, tangency_cov_matrix = optimize_portfolio(
        data,
        long_only,
        min_weight_value,
        max_weight_value,
        leverage_limit_value,
        risk_free_rate,
        include_risk_free_asset,
        risk_aversion,
    )
    weights = pd.Series(tangency_result.x, index=assets)
    st.session_state["optimization_run"] = True
    st.session_state["weights"] = weights
    st.session_state["mean_returns"] = tangency_mean_returns
    st.session_state["cov_matrix"] = tangency_cov_matrix

    # Display optimization results
    st.subheader("Optimized Portfolio Weights:")
    st.write(weights.apply(lambda x: f"{x:.2%}"))

    # Calculate portfolio performance
    portfolio_return = np.sum(tangency_mean_returns * weights) * 12  # Annualized return
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(tangency_cov_matrix * 12, weights))
    )  # Annualized volatility

    if include_risk_free_asset:
        # Calculate Sharpe Ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        # Calculate allocation between risk-free asset and tangency portfolio
        allocation_tangency = (portfolio_return - risk_free_rate) / (
            risk_aversion * (portfolio_volatility**2)
        )
        # allocation_tangency = min(max(allocation_tangency, 0), 1)
        allocation_risk_free = max(1 - allocation_tangency, 0)

        st.subheader("Portfolio Performance with Risk-Free Asset:")
        st.write(f"Expected Annual Return: {portfolio_return:.2%}")
        st.write(f"Annual Volatility: {portfolio_volatility:.2%}")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        st.write(f"Invest {allocation_tangency * 100:.2f}% in the tangency portfolio.")
        st.write(f"Invest {allocation_risk_free * 100:.2f}% in the risk-free asset.")

    else:
        # Calculate Sharpe Ratio
        sharpe_ratio = portfolio_return / portfolio_volatility

        st.subheader("Portfolio Performance without Risk-Free Asset:")
        st.write(f"Expected Annual Return: {portfolio_return:.2%}")
        st.write(f"Annual Volatility: {portfolio_volatility:.2%}")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Show the allocation
        allocation_df = pd.DataFrame({"ISIN": assets, "Weight": weights})
        st.write("Optimal Portfolio Allocation:")
        st.write(allocation_df)
    st.write(f"Sum of the weights: {np.sum(weights)}")
else:
    st.write('Click "Run Optimization" to compute the optimized portfolio.')

# Run the efficient frontier
if st.session_state["optimization_run"]:
    if st.button("Show Efficient Frontier"):
        # Retrieve necessary variables from session state
        weights = st.session_state["weights"]

        returns = data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(mean_returns)
        if "weights" in locals():
            weights_optimal = weights.values
        else:
            weights_optimal = None

        plot_efficient_frontier(
            mean_returns,
            cov_matrix,
            risk_free_rate,
            include_risk_free_asset,
            weights_optimal,
            long_only,
            leverage_limit_value,
            min_weight_value,
            max_weight_value,
        )
    else:
        st.write('Click "Show Efficient Frontier" to display the graph.')
else:
    st.write("Run the optimization first to display the efficient frontier.")

# # Efficient Frontier Plotting Function
# def plot_efficient_frontier(
#     mean_returns,
#     cov_matrix,
#     risk_free_rate,
#     include_risk_free_asset,
#     weights_optimal,
#     long_only,
#     bounds,
# ):
#     frontier_volatility, frontier_returns, frontier_weights = (
#         calculate_efficient_frontier(
#             mean_returns,
#             cov_matrix,
#             risk_free_rate,
#             include_risk_free_asset,
#             long_only,
#             bounds,
#         )
#     )

#     plt.figure(figsize=(10, 7))
#     plt.plot(frontier_volatility, frontier_returns, "b--", label="Efficient Frontier")

#     if include_risk_free_asset:
#         # Calculate the tangency portfolio
#         def neg_sharpe_ratio(weights):
#             portfolio_return = np.sum(mean_returns * weights) * 12
#             portfolio_volatility = np.sqrt(
#                 np.dot(weights.T, np.dot(cov_matrix * 12, weights))
#             )
#             sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
#             return -sharpe_ratio

#         constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
#         result = minimize(
#             neg_sharpe_ratio,
#             num_assets * [1.0 / num_assets],
#             method="SLSQP",
#             bounds=bounds,
#             constraints=constraints,
#         )

#         if result.success:
#             tangency_weights = result.x
#             tangency_return = np.sum(mean_returns * tangency_weights) * 12
#             tangency_volatility = np.sqrt(
#                 np.dot(tangency_weights.T, np.dot(cov_matrix * 12, tangency_weights))
#             )

#             # Plot the Capital Market Line
#             cml_x = [0, tangency_volatility]
#             cml_y = [risk_free_rate, tangency_return]
#             plt.plot(cml_x, cml_y, color="red", label="Capital Market Line")

#             # Highlight the tangency portfolio
#             plt.scatter(
#                 tangency_volatility,
#                 tangency_return,
#                 marker="*",
#                 color="red",
#                 s=500,
#                 label="Tangency Portfolio",
#             )
#         else:
#             st.warning("Failed to compute the tangency portfolio.")
#     else:
#         # Highlight the optimal portfolio
#         portfolio_return = np.sum(mean_returns * weights_optimal) * 12
#         portfolio_volatility = np.sqrt(
#             np.dot(weights_optimal.T, np.dot(cov_matrix * 12, weights_optimal))
#         )
#         plt.scatter(
#             portfolio_volatility,
#             portfolio_return,
#             marker="*",
#             color="red",
#             s=500,
#             label="Optimal Portfolio",
#         )

#     plt.title("Efficient Frontier")
#     plt.xlabel("Annualized Volatility")
#     plt.ylabel("Annualized Expected Returns")
#     plt.legend()
#     st.pyplot(plt)


# if st.button("Show Efficient Frontier"):
#     returns = data.pct_change().dropna()
#     mean_returns = returns.mean()
#     cov_matrix = returns.cov()
#     num_assets = len(mean_returns)
#     if "weights" in locals():
#         weights_optimal = weights.values
#     else:
#         weights_optimal = None

#     # Prepare bounds for efficient frontier calculation
#     if long_only:
#         bounds = tuple(
#             (max(min_weight_value, 0.0), min(max_weight_value, 1.0))
#             for _ in range(num_assets)
#         )
#     else:
#         bounds = tuple((-1.0, 1.0) for _ in range(num_assets))

#     plot_efficient_frontier(
#         mean_returns,
#         cov_matrix,
#         risk_free_rate,
#         include_risk_free_asset,
#         weights_optimal,
#         long_only,
#         bounds,
#     )
# else:
#     st.write('Click "Show Efficient Frontier" to display the graph.')


# # Efficient Frontier using statistical generation
# def plot_efficient_frontier(
#     mean_returns, cov_matrix, risk_free_rate, include_risk_free_asset, weights_optimal
# ):
#     num_portfolios = 5000
#     results = np.zeros((3, num_portfolios))
#     for i in range(num_portfolios):
#         weights = np.random.dirichlet(np.ones(len(mean_returns)))
#         portfolio_return = np.sum(mean_returns * weights) * 12
#         portfolio_volatility = np.sqrt(
#             np.dot(weights.T, np.dot(cov_matrix * 12, weights))
#         )
#         sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
#         results[0, i] = portfolio_volatility
#         results[1, i] = portfolio_return
#         results[2, i] = sharpe_ratio

#     max_sharpe_idx = np.argmax(results[2])
#     sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]

#     plt.figure(figsize=(10, 7))
#     plt.scatter(
#         results[0],
#         results[1],
#         c=results[2],
#         cmap="viridis",
#         marker="o",
#         s=10,
#         alpha=0.3,
#     )
#     plt.colorbar(label="Sharpe Ratio")
#     if include_risk_free_asset:
#         # Plot the Capital Market Line
#         cml_x = [0, sdp]
#         cml_y = [risk_free_rate, rp]
#         plt.plot(cml_x, cml_y, color="red", label="Capital Market Line")

#         # Highlight the tangency portfolio
#         plt.scatter(sdp, rp, marker="*", color="red", s=500, label="Tangency Portfolio")
#     else:
#         # Highlight the optimal portfolio
#         portfolio_return = np.sum(mean_returns * weights_optimal) * 12
#         portfolio_volatility = np.sqrt(
#             np.dot(weights_optimal.T, np.dot(cov_matrix * 12, weights_optimal))
#         )
#         plt.scatter(
#             portfolio_volatility,
#             portfolio_return,
#             marker="*",
#             color="red",
#             s=500,
#             label="Optimal Portfolio",
#         )

#     plt.title("Efficient Frontier")
#     plt.xlabel("Volatility (Std. Deviation)")
#     plt.ylabel("Expected Returns")
#     plt.legend()
#     st.pyplot(plt)


# if st.button("Show Efficient Frontier"):
#     returns = data.pct_change().dropna()
#     mean_returns = returns.mean()
#     cov_matrix = returns.cov()
#     if "weights" in locals():
#         weights_optimal = weights.values
#     else:
#         weights_optimal = None
#     plot_efficient_frontier(
#         mean_returns,
#         cov_matrix,
#         risk_free_rate,
#         include_risk_free_asset,
#         weights_optimal,
#     )
# else:
#     st.write('Click "Show Efficient Frontier" to display the graph.')

# if __name__ == "__main__":
#     weights = optimize_portfolio(data, True, 0, 1, 1, 0.01)
#     print(weights)
