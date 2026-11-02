
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model and dataset
model = joblib.load("models/demand_model.pkl")
df = pd.read_csv("datasets/vegetable_sales.csv")

df["date"] = pd.to_datetime(df["date"])

# Page configuration
st.set_page_config(
    page_title="Seasonal Produce Demand Forecaster",
    page_icon="🥕",
    layout="wide"
)

# Main heading
st.title("🥕 Seasonal Produce Demand Forecaster")

st.write(
    "Predict vegetable demand and make smarter "
    "purchasing decisions using Machine Learning."
)

st.info("This application currently uses synthetic sales data for demonstration.")

# Sidebar
st.sidebar.header("Forecast Settings")

vegetable = st.sidebar.selectbox(
    "Select Vegetable",
    ["Tomato", "Onion", "Potato", "Carrot", "Beans"]
)

price = st.sidebar.number_input(
    "Selling Price per Kg (₹)",
    min_value=1.0,
    value=30.0,
    step=1.0
)

forecast_date = st.sidebar.date_input(
    "Forecast Date",
    value=pd.Timestamp.today().date()
)

stock = st.sidebar.number_input(
    "Available Stock (Kg)",
    min_value=0.0,
    value=10.0,
    step=1.0
)

safety_stock = st.sidebar.number_input(
    "Safety Stock (Kg)",
    min_value=0.0,
    value=5.0,
    step=1.0
)


# Prediction button
if st.sidebar.button("Predict Demand", type="primary"):

    day = forecast_date.weekday()
    month = forecast_date.month
    weekend = 1 if day >= 5 else 0

    # Get historical records before forecast date
    history = df[
        (df["vegetable"] == vegetable) &
        (df["date"].dt.date < forecast_date)
    ].sort_values("date")

    if len(history) == 0:
        st.error("No historical sales data available for this date.")
    else:

        # Calculate historical sales features
        previous_day_sales = history.iloc[-1][
            "quantity_sold_kg"
        ]

        previous_7_day_average = history.tail(7)[
            "quantity_sold_kg"
        ].mean()

        previous_30_day_average = history.tail(30)[
            "quantity_sold_kg"
        ].mean()

        # Prepare all 8 model inputs
        input_data = pd.DataFrame([{
            "vegetable": vegetable,
            "price_per_kg": price,
            "day_of_week": day,
            "month": month,
            "is_weekend": weekend,
            "previous_day_sales": previous_day_sales,
            "previous_7_day_average": previous_7_day_average,
            "previous_30_day_average": previous_30_day_average
        }])

        # Predict demand
        prediction = model.predict(input_data)[0]

        prediction = max(0, prediction)

        # Calculate purchase recommendation
        purchase = max(
            0,
            prediction + safety_stock - stock
        )

        st.subheader("Demand Forecast Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Demand",
            f"{prediction:.2f} Kg"
        )

        col2.metric(
            "Available Stock",
            f"{stock:.2f} Kg"
        )

        col3.metric(
            "Suggested Purchase",
            f"{purchase:.2f} Kg"
        )

        st.success(
            f"Recommended purchase quantity: {purchase:.2f} Kg"
        )

        st.caption(
            "Forecast uses the trained Random Forest model "
            "and historical sales information."
        )

        with st.expander("View historical features used"):
            st.write(
                "Previous day sales:",
                round(previous_day_sales, 2), "Kg"
            )
            st.write(
                "Previous 7-day average:",
                round(previous_7_day_average, 2), "Kg"
            )
            st.write(
                "Previous 30-day average:",
                round(previous_30_day_average, 2), "Kg"
            )



# Sales analytics
st.divider()

st.header("📊 Vegetable Sales Analytics")

st.subheader("Average Daily Sales by Vegetable")

average_sales = (
    df.groupby("vegetable")["quantity_sold_kg"]
    .mean()
    .sort_values(ascending=False)
)

# Dark-themed bar chart
fig, ax = plt.subplots(figsize=(10, 5))

fig.patch.set_facecolor("#0E1117")
ax.set_facecolor("#0E1117")

average_sales.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    "Average Daily Sales by Vegetable",
    color="white"
)

ax.set_xlabel("Vegetable", color="white")
ax.set_ylabel("Quantity Sold (Kg)", color="white")

ax.tick_params(colors="white")

for spine in ax.spines.values():
    spine.set_color("#555555")

ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Historical sales trend
st.subheader("📈 Historical Sales Trend")

selected_vegetable = st.selectbox(
    "Choose Vegetable",
    df["vegetable"].unique()
)

trend = df[
    df["vegetable"] == selected_vegetable
].groupby("date")["quantity_sold_kg"].sum()

st.line_chart(trend)

# Footer
st.divider()

st.caption(
    "Seasonal Produce Demand Forecaster | "
    "Machine Learning Mini Project"
)
