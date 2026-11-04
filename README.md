# CodeAlpha Task 3: Car Price Prediction with Machine Learning

## Project Overview
This project focuses on predicting the selling price of used cars using Machine Learning techniques. The main objective is to develop a regression model that estimates car prices based on various car-related features.

## Dataset
The project uses the Car Price dataset (`car data.csv`).

The dataset contains information about used cars and their selling prices.

**Target Variable:** `Selling_Price`

## Technologies and Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Machine Learning Algorithm
**Random Forest Regressor**

Random Forest Regression is used to predict car selling prices based on the available input features.

## Project Workflow
1. Dataset Loading
2. Data Cleaning and Preprocessing
3. Feature Selection
4. Handling Missing Values
5. Encoding Categorical Variables
6. Splitting Data into Training and Testing Sets
7. Model Training
8. Car Price Prediction
9. Model Evaluation
10. Data Visualization and Feature Importance Analysis

## Model Evaluation Metrics
The model performance is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared Score (R²)

## Project Structure

```text
CodeAlpha_CarPricePrediction/
│
├── car data.csv
├── car_price_prediction.py
├── README.md
│
└── outputs/
    ├── cleaned_car_data.csv
    ├── predictions.csv
    ├── actual_vs_predicted.png
    ├── feature_importance.csv
    ├── feature_importance.png
    └── model_evaluation.txt
```

## Output Files
- **cleaned_car_data.csv:** Cleaned dataset.
- **predictions.csv:** Actual and predicted car prices.
- **actual_vs_predicted.png:** Visualization comparing actual and predicted prices.
- **feature_importance.csv:** Feature importance values.
- **feature_importance.png:** Feature importance visualization.
- **model_evaluation.txt:** Model evaluation results.

## How to Run the Project

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Run the Python script:

```bash
python car_price_prediction.py
```

## Internship Details
**Organization:** CodeAlpha

**Domain:** Data Science

**Task:** Task 3 – Car Price Prediction with Machine Learning

## Conclusion
This project demonstrates the application of Machine Learning regression techniques to predict used car selling prices. It includes data preprocessing, model training, prediction, evaluation, and visualization.
