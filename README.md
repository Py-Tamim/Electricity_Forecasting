# Bangladesh Electricity Forecasting & Demand Prediction

## Project Overview

This project focuses on forecasting electricity demand, supply, generation, and potential load shedding in Bangladesh using Machine Learning and Deep Learning techniques.

The project consists of two independent forecasting systems:

**1. Electricity Demand Prediction**

Predicts daily electricity demand, supply, and potential load shedding across nine divisions of Bangladesh.

**2. Electricity Generation Forecasting**

Predicts hourly electricity generation and analyzes the contribution of different energy sources, including gas, liquid fuel, coal, hydro, solar, wind, and electricity imports from neighboring countries.

Both systems allow users to enter a specific date or date range and retrieve historical information or future predictions directly through the terminal.

The project also compares different forecasting approaches, including Baseline, XGBoost, and LSTM, to evaluate their performance and understand which models are more suitable for electricity forecasting.

## Project Objectives

- Forecast electricity demand across different divisions of Bangladesh.
- Predict hourly electricity generation.
- Estimate electricity supply and potential load shedding.
- Analyze electricity generation from different energy sources.
- Compare Baseline, XGBoost, and LSTM forecasting models.
- Evaluate model performance using MAE, RMSE, and MAPE.
- Analyze forecasting accuracy across different days, seasons, and hours.
- Compare model training time and prediction performance.
- Visualize historical data and forecasting results through graphs.
- Develop a flexible forecasting system that supports specific dates and date ranges.

## Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing and analysis |
| NumPy | Numerical calculations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning utilities and evaluation |
| XGBoost | Gradient boosting regression |
| PyTorch | LSTM deep learning implementation |
| Pathlib | File and directory management |

## Project Structure

```text
Bangladesh-Electricity-Forecasting/
│
├── electricity_demand_prediction.py
├── electricity_forecasting.py
│
├── Bangladesh_Electricity_Dataset_2019_2026.csv
├── bangladesh_power_generation_2019_2026_processed.csv
│
├── result.png
│
├── requirements.txt
└── README.md
```

# Project 1: Electricity Demand Prediction

## Overview

The Electricity Demand Prediction system forecasts daily electricity demand and supply across nine divisions of Bangladesh.

It also estimates potential load shedding by calculating the difference between predicted electricity demand and predicted supply.

This system helps analyze regional electricity requirements and understand possible electricity shortages.

## Covered Divisions

The system covers the following nine divisions:

1. Barishal
2. Chattogram
3. Cumilla
4. Dhaka
5. Khulna
6. Mymensingh
7. Rajshahi
8. Rangpur
9. Sylhet

## Dataset Information

**Dataset:** `Bangladesh_Electricity_Dataset_2019_2026.csv`

The dataset contains daily electricity-related information, including:

- Date and year.
- Month and day of the week.
- Public holiday information.
- Maximum temperature in Dhaka.
- Electricity demand by division.
- Electricity supply by division.
- Load-shedding information.

## Data Preprocessing

The following preprocessing techniques are applied:

- Convert date columns into datetime format.
- Identify Fridays and Saturdays as weekends.
- Identify public holidays.
- Replace invalid zero temperature readings with missing values.
- Fill missing numerical values using the median of the same year and month.
- Use the overall median as a backup for remaining missing values.
- Fill missing categorical values using the most frequent value.

These steps help prepare the historical electricity data for model training and forecasting.

## Machine Learning and Deep Learning Models

Three different forecasting approaches are implemented.

### 1. Baseline Model

The Baseline model uses historical average electricity demand and supply based on the month and holiday category.

It provides a reference point for evaluating the performance of more advanced machine learning and deep learning models.

### 2. XGBoost Regressor

XGBoost is used to predict electricity demand and supply based on different input features.

The model considers information such as:

- Calendar information.
- Division or area information.
- Temperature-related features.
- Historical electricity patterns.

XGBoost is used as a regression model to learn relationships between the input features and electricity demand and supply.

### 3. LSTM (Long Short-Term Memory)

LSTM is a type of recurrent neural network designed to learn patterns from sequential data.

In this project, LSTM uses historical daily information to identify time-dependent patterns and generate electricity demand and supply forecasts.

**Model Configuration:**

| Parameter | Value |
|---|---|
| Sequence Length | 14 days |
| Hidden Layer Size | 48 |
| Optimizer | Adam |
| Learning Rate | 0.003 |
| Training Epochs | 30 |

## Prediction Output

The system can generate electricity demand, supply, and estimated load-shedding predictions for individual divisions.

**Example: 18 December 2026**

| Division | Demand (MW) | Supply (MW) | Load Shedding (MW) |
|---|---:|---:|---:|
| Barishal | 399 | 357 | 42 |
| Chattogram | 1,281 | 1,186 | 95 |
| Cumilla | 1,234 | 1,079 | 154 |
| Dhaka | 4,668 | 4,273 | 395 |
| Khulna | 1,570 | 1,399 | 171 |
| Mymensingh | 1,121 | 933 | 188 |
| Rajshahi | 1,277 | 1,160 | 116 |
| Rangpur | 929 | 826 | 103 |
| Sylhet | 548 | 473 | 75 |

*These are example outputs from the forecasting system.*

## Visualization

The demand prediction system generates graphical representations of:

- Electricity demand.
- Electricity supply.
- Estimated load shedding.
- Daily forecasting trends.
- Multi-day forecasting trends.

For future dates, the forecast period is highlighted in grey on multi-day graphs to distinguish predicted values from historical information.

---

# Project 2: Electricity Generation Forecasting

## Overview

The Electricity Generation Forecasting system predicts Bangladesh's electricity generation on an hourly basis.

It forecasts total electricity generation and analyzes the contribution of different energy sources and electricity imports.

The objective is to understand hourly generation patterns and estimate how different energy sources contribute to the overall electricity supply.

## Dataset Information

**Dataset:** `bangladesh_power_generation_2019_2026_processed.csv`

The dataset contains hourly electricity generation information, including total generation and individual generation sources.

## Prediction Features

The model predicts 11 electricity-related parameters.

| Parameter | Description |
|---|---|
| Generation (MW) | Total electricity generation |
| Gas | Electricity generated from gas |
| Liquid Fuel | Electricity generated from liquid fuel |
| Coal | Electricity generated from coal |
| Hydro | Hydroelectric generation |
| Solar | Solar electricity generation |
| Wind | Wind electricity generation |
| Bheramara | Electricity imported through Bheramara HVDC |
| Tripura | Electricity imported from Tripura |
| Adani | Electricity imported from Adani |
| Nepal | Electricity imported from Nepal |

## Data Preprocessing

The following preprocessing techniques are applied:

- Combine date and time columns into a single timestamp.
- Group multiple readings within the same hour and calculate their average.
- Identify and remove unusually high readings.
- Remove invalid total generation readings below 2,000 MW.
- Create a continuous hourly timeline.
- Fill missing values using the median of the same year, month, and hour.
- Use the overall median as a backup for remaining missing values.

These operations help maintain a consistent hourly dataset for forecasting.

## Machine Learning and Deep Learning Models

Three forecasting approaches are implemented.

### 1. Baseline Model

The Baseline model uses historical average generation values based on:

- Month.
- Holiday status.
- Hour of the day.

It provides a reference for comparing the performance of XGBoost and LSTM.

### 2. XGBoost Regressor

XGBoost is used to forecast multiple electricity generation parameters using calendar-based features.

The input features include:

- Hour.
- Weekday.
- Month.
- Day of the year.
- Holiday information.
- Day number.

The implementation uses `MultiOutputRegressor` to predict multiple electricity generation and import parameters.

### 3. LSTM (Long Short-Term Memory)

LSTM is used to learn sequential hourly patterns from historical electricity generation data.

It processes previous hourly information to identify time-dependent generation patterns and forecast future electricity generation values.

**Model Configuration:**

| Parameter | Value |
|---|---|
| Sequence Length | 24 hours |
| Hidden Layer Size | 48 |
| Optimizer | Adam |
| Learning Rate | 0.003 |
| Training Epochs | 20 |

## Prediction Output

The system generates hourly electricity generation predictions for a selected date and time range.

**Example: 18 December 2026**

| Time | Generation (MW) | Gas | Liquid Fuel | Coal | Solar |
|---|---:|---:|---:|---:|---:|
| 10 AM | 8,773 | 3,646 | 402 | 4,158 | 543 |
| 11 AM | 9,115 | 3,673 | 417 | 4,153 | 596 |
| 12 PM | 9,502 | 3,800 | 684 | 4,166 | 626 |
| 1 PM | 9,890 | 3,783 | 1,090 | 4,247 | 593 |
| 2 PM | 9,247 | 3,495 | 1,037 | 4,178 | 510 |
| 3 PM | 9,081 | 3,499 | 991 | 4,160 | 373 |
| 4 PM | 8,853 | 3,513 | 964 | 4,136 | 173 |
| 5 PM | 9,021 | 3,504 | 1,149 | 4,163 | 41 |

*The table displays selected columns from the example terminal output.*

## Visualization

The generation forecasting system produces two graphical comparisons.

**1. Bangladesh Electricity Generation**

Displays the final predicted or historical electricity generation results.

**2. Current Result vs Previous Year's Actual Generation**

Compares the current results with actual electricity generation recorded at the same date and time one year earlier, when matching historical data is available.

These visualizations help identify hourly generation patterns and compare forecasting results with historical performance.

---

# Model Evaluation and Comparison

Both forecasting systems support model comparison to evaluate the performance of different approaches.

The following models are available:

- Baseline
- XGBoost
- LSTM

## Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| Training Time | Time required to train each model |

These metrics help measure prediction errors and compare the performance of different forecasting approaches.

## Electricity Demand Evaluation

The demand prediction system evaluates model performance across different conditions:

- All days.
- Working days.
- Weekends and holidays.
- Hot months (April–August).
- Cool months (January–March).

It also calculates MAE improvement against the Baseline model.

## Electricity Generation Evaluation

The generation forecasting system evaluates model performance across:

- All hours.
- Working days.
- Weekends and holidays.
- Night hours (12 AM–5 AM).
- Evening peak hours (6 PM–10 PM).

It also compares the MAE of individual electricity generation and import parameters.

## Train-Test Split

Both forecasting systems use a chronological train-test split ending on:

**31 December 2025**

- Training data: Up to 31 December 2025.
- Testing data: After 31 December 2025.

A chronological split is important for time-series forecasting because it preserves the original order of observations and helps evaluate how models perform when predicting future periods.

---

# Installation and Usage

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Bangladesh-Electricity-Forecasting.git
```

## 2. Navigate to the Project Folder

```bash
cd Bangladesh-Electricity-Forecasting
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install the required libraries manually:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

For LSTM support:

```bash
pip install torch
```

**Note:** PyTorch is optional when using Baseline and XGBoost. Both scripts import PyTorch only when the LSTM model is selected.

## 4. Run Electricity Demand Prediction

```bash
python electricity_demand_prediction.py
```

Enter a date or date range when prompted.

**Example: Single Date**

```text
18/12/2026
```

**Example: Date Range**

```text
18/11/2026 to 17/01/2027
```

Enter a division name or press Enter to forecast all divisions.

To compare the available models:

```bash
python electricity_demand_prediction.py compare
```

## 5. Run Electricity Generation Forecasting

```bash
python electricity_forecasting.py
```

Enter a date or date range:

```text
18/12/2026
```

Enter a time range:

```text
10 am to 5 pm
```

Press Enter at the time prompt to forecast all 24 hours.

To compare the available models:

```bash
python electricity_forecasting.py compare
```

---

# Changing the Forecasting Model

Both Python scripts contain a `MODEL` variable that allows users to select the forecasting approach.

```python
MODEL = "XGBoost"
```

Available options:

```python
MODEL = "Baseline"
MODEL = "XGBoost"
MODEL = "LSTM"
```

Change the variable to select the model you want to use.

---

# Important Notes

- Historical dates return available recorded data.
- Dates beyond the available dataset generate model-based predictions.
- Forecasting results are estimates, not official electricity supply or generation announcements.
- Model accuracy depends on data quality, historical patterns, and the availability of relevant features.
- Future weather conditions, unexpected demand changes, fuel shortages, and power plant outages are not directly known to the forecasting models.
- Model comparison results may vary depending on the selected forecasting period and available training data.

# AI Assistance & Acknowledgement

During the planning and development of this project, I used AI tools as a supporting resource for research, learning, and exploring different technical approaches.

AI assistance was used for:

- Exploring and developing the initial project idea.
- Finding suitable CSV datasets related to Bangladesh's electricity demand, supply, and generation.
- Understanding the available dataset structures and identifying relevant features for forecasting.
- Researching different machine learning and deep learning models suitable for electricity forecasting.
- Comparing Baseline, XGBoost, and LSTM to understand their advantages and suitability for the two forecasting tasks.
- Understanding data preprocessing techniques, time-series forecasting concepts, and model evaluation methods.
- Exploring possible improvements to the forecasting workflow.

AI was used as a learning and research assistant throughout the project.

# Future Improvements
Some possible improvements for this project include:

- Integrating real-time electricity generation data.
- Including additional weather-related features.
- Improving long-term forecasting accuracy.
- Experimenting with advanced time-series forecasting models.
- Developing an interactive dashboard using Streamlit.
- Implementing automated model retraining.
- Improving forecasting performance during peak electricity demand periods.
- Incorporating additional external factors such as fuel prices, power plant availability, and seasonal demand patterns.
---

*This project demonstrates the practical application of machine learning, deep learning, time-series forecasting, data preprocessing, model evaluation, and data visualization in Bangladesh's electricity sector.*
