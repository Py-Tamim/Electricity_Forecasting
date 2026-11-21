# Bangladesh Electricity Forecasting & Demand Prediction

## Project Overview

This project focuses on analyzing and forecasting electricity demand, supply, generation, and load shedding in Bangladesh using machine learning and deep learning techniques.

It contains two independent forecasting systems:

**1. Electricity Demand Prediction**

Predicts daily electricity demand, supply, and load shedding across nine divisions of Bangladesh.

**2. Electricity Generation Forecasting**

Predicts hourly electricity generation and its different energy sources, including gas, liquid fuel, coal, hydro, solar, wind, and electricity imports from neighboring countries.

Both systems allow users to enter a specific date or date range and retrieve historical or future predictions through the terminal.

## Project Objectives

- Forecast electricity demand across different divisions of Bangladesh.
- Predict electricity generation on an hourly basis.
- Estimate electricity supply and potential load shedding.
- Analyze different electricity generation sources.
- Compare Baseline, XGBoost, and LSTM models.
- Evaluate forecasting performance using MAE, RMSE, and MAPE.
- Analyze model performance across different days, seasons, and hours.
- Visualize forecasting results through graphical comparisons.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing and analysis |
| NumPy | Numerical calculations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning utilities |
| XGBoost | Gradient boosting regression |
| PyTorch | LSTM deep learning model |
| Pathlib | File path management |

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
├── README.md
└── requirements.txt
```

## Project 1: Electricity Demand Prediction

### Overview

This system predicts electricity demand and supply across nine divisions of Bangladesh.

It also estimates potential load shedding by calculating the difference between predicted demand and predicted supply.

### Covered Divisions

- Barishal
- Chattogram
- Cumilla
- Dhaka
- Khulna
- Mymensingh
- Rajshahi
- Rangpur
- Sylhet

### Dataset

**File:** `Bangladesh_Electricity_Dataset_2019_2026.csv`

The dataset contains daily electricity-related information, including:

- Date and year
- Month and day of the week
- Holiday information
- Maximum temperature in Dhaka
- Electricity demand by division
- Electricity supply by division
- Load-shedding information

### Data Preprocessing

The following preprocessing techniques are applied:

- Convert dates into datetime format.
- Identify Fridays and Saturdays as weekends.
- Identify public holidays.
- Replace invalid zero temperature readings with missing values.
- Fill missing numerical values using the median of the same year and month.
- Use the overall median as a backup for remaining missing values.
- Fill missing categorical values using the most frequent value.

### Machine Learning Models

Three models are implemented:

**1. Baseline Model**

Uses historical average electricity demand and supply based on the month and holiday category.

**2. XGBoost Regressor**

Uses calendar information, area information, and temperature-related features to predict electricity demand and supply.

**3. LSTM (Long Short-Term Memory)**

Uses sequential daily information to learn patterns over time.

- Sequence length: 14 days
- Hidden layer size: 48
- Optimizer: Adam
- Learning rate: 0.003
- Training epochs: 30

### Prediction Output

Example: 18 December 2026

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

### Visualization

The demand prediction system generates graphs showing:

- Electricity demand
- Electricity supply
- Estimated load shedding
- Daily and multi-day forecasting trends

For future dates, the forecast period is highlighted in grey on multi-day graphs.

## Project 2: Electricity Generation Forecasting

### Overview

This system forecasts Bangladesh's electricity generation on an hourly basis.

It predicts total generation as well as the contribution of different energy sources and electricity imports.

### Dataset

**File:** `bangladesh_power_generation_2019_2026_processed.csv`

The dataset contains hourly electricity generation information.

### Prediction Features

The model predicts 11 different electricity-related values:

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

### Data Preprocessing

The following preprocessing techniques are applied:

- Combine date and time into a single timestamp.
- Group multiple readings within the same hour and calculate their average.
- Identify and remove unusually high readings.
- Remove invalid total generation readings below 2,000 MW.
- Create a continuous hourly timeline.
- Fill missing values using the median of the same year, month, and hour.
- Use the overall median as a backup for remaining missing values.

### Machine Learning Models

**1. Baseline Model**

Uses historical average generation values based on month, holiday status, and hour of the day.

**2. XGBoost**

Uses calendar-based features such as:

- Hour
- Weekday
- Month
- Day of the year
- Holiday information
- Day number

The model uses MultiOutputRegressor to predict multiple electricity generation parameters.

**3. LSTM (Long Short-Term Memory)**

Uses sequential hourly calendar features to learn time-based patterns.

- Sequence length: 24 hours
- Hidden layer size: 48
- Optimizer: Adam
- Learning rate: 0.003
- Training epochs: 20

### Prediction Output

Example: 18 December 2026

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

*The table shows selected columns from the example terminal output.*

### Visualization

The generation forecasting system produces two graphs:

1. **Bangladesh Electricity Generation:** Displays the final predicted or historical generation results.
2. **Current Result vs Previous Year's Actual Generation:** Compares the current results with actual generation recorded at the same date and time one year earlier, when matching historical data is available.

## Model Evaluation and Comparison

Both forecasting systems support model comparison.

The available models are:

- Baseline
- XGBoost
- LSTM

### Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| Training Time | Time required to train each model |

### Electricity Demand Evaluation

The demand prediction system compares model performance across:

- All days
- Working days
- Weekends and holidays
- Hot months (April–August)
- Cool months (January–March)

It also calculates the MAE gain against the Baseline model.

### Electricity Generation Evaluation

The generation forecasting system evaluates performance across:

- All hours
- Working days
- Weekends and holidays
- Night hours (12 AM–5 AM)
- Evening peak hours (6 PM–10 PM)

It also compares the MAE of individual generation and import parameters.

### Train-Test Split

Both systems use a chronological train-test split ending on:

**31 December 2025**

- Training data: Up to 31 December 2025.
- Testing data: After 31 December 2025.

This approach preserves the time order of the observations when evaluating forecasting performance.

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Bangladesh-Electricity-Forecasting.git
```

### 2. Navigate to the Project Folder

```bash
cd Bangladesh-Electricity-Forecasting
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

For LSTM support:

```bash
pip install torch
```

**Note:** PyTorch is optional for Baseline and XGBoost models. Both scripts import PyTorch only when the LSTM model is selected.

### 4. Run Electricity Demand Prediction

```bash
python electricity_demand_prediction.py
```

Enter a date or date range when prompted.

Example:

```text
18/12/2026
```

Or:

```text
18/11/2026 to 17/01/2027
```

Enter a division name or press Enter to forecast all divisions.

To compare the models:

```bash
python electricity_demand_prediction.py compare
```

### 5. Run Electricity Generation Forecasting

```bash
python electricity_forecasting.py
```

Enter a date or date range:

```text
18/12/2026
```

Enter a time:

```text
10 am to 5 pm
```

Press Enter at the time prompt to forecast all 24 hours.

To compare the models:

```bash
python electricity_forecasting.py compare
```

## Changing the Forecasting Model

Both Python scripts contain a `MODEL` variable.

```python
MODEL = "XGBoost"
```

Available options:

```python
MODEL = "Baseline"
MODEL = "XGBoost"
MODEL = "LSTM"
```

Change this variable to select the model you want to use.

## Important Notes

- Historical dates return available recorded data.
- Dates beyond the available dataset generate model-based predictions.
- Forecasting results are estimates, not official electricity supply or generation announcements.
- Model accuracy depends on data quality, historical patterns, and the availability of relevant features.
- Future weather, unexpected demand changes, fuel shortages, and power plant outages are not directly known to the forecasting models.

## Future Improvements

- Integrate real-time electricity generation data.
- Include more weather-related features.
- Improve long-term forecasting accuracy.
- Experiment with advanced time-series models.
- Develop an interactive dashboard using Streamlit.
- Add automated model retraining.
- Improve forecasting for peak electricity demand periods.

*##This project demonstrates the practical application of machine learning, deep learning, time-series forecasting, data preprocessing, model evaluation, and data visualization in Bangladesh's electricity sector.*
