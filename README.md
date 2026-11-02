
# Seasonal Produce Demand Forecaster

## Project Overview

The Seasonal Produce Demand Forecaster is a Machine Learning
application that predicts the expected sales quantity of
fruits and vegetables.

It helps small vegetable shops make better purchasing
decisions by considering historical sales, selling prices,
and seasonal patterns.

## Features

- Demand prediction for five vegetables.
- Historical sales analysis.
- Previous-day sales feature.
- Previous 7-day average sales.
- Previous 30-day average sales.
- Random Forest Regression.
- Purchase quantity recommendation.
- Interactive Streamlit dashboard.
- Sales analytics and visualizations.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- Streamlit
- Matplotlib

## Machine Learning Evaluation

| Metric | Random Forest |
|---|---:|
| MAE | 3.52 |
| RMSE | 4.41 |
| R2 Score | 0.8388 |

The model was evaluated using a synthetic dataset.

## How to Run

1. Clone or copy the project.
2. Install the required libraries:

   pip install -r requirements.txt

3. Generate the dataset:

   python generate_dataset.py

4. Train the model:

   python train_model.py

5. Run the application:

   streamlit run app.py

## Dataset

The project currently uses synthetically generated
daily sales records for:

- Tomato
- Onion
- Potato
- Carrot
- Beans

The dataset is intended for demonstration and testing.

## Project Team

Member 1: ML Model Development

Member 2: Dashboard Development and Integration
