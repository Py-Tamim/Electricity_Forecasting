Electricity Forecasting – Bangladesh
Two console programs that forecast Bangladesh's electricity data from 2019–2026 history. Enter a date (or a date range) and get a table plus graphs. Dates inside the dataset return the recorded values, and dates after the data ends are forecast by a machine-learning model.
Script	What it forecasts	Data file	Resolution
`electricity_demand_prediction.py`	Demand, supply and load shedding for each area (Barishal, Chattogram, Cumilla, Dhaka, Khulna, Mymensingh, Rajshahi, Rangpur, Sylhet)	`Bangladesh_Electricity_Dataset_2019_2026.csv`	Daily
`electricity_forecasting.py`	Total generation and each source: gas, liquid fuel, coal, hydro, solar, wind, India (Bheramara, Tripura, Adani) and Nepal import	`bangladesh_power_generation_2019_2026_processed.csv`	Hourly
Models
Each script includes three models and can compare them:
Baseline – average value for the same month, holiday/weekend flag (and hour, in the hourly script).
XGBoost (default) – gradient-boosted trees on calendar features (month, day of year, weekday, holiday, temperature for the daily script, hour for the hourly script).
LSTM (PyTorch) – a recurrent network that looks back 14 days (daily) or 24 hours (hourly).
Pick the model for searches with the `MODEL` variable at the top of each file. Run `python <script>.py compare` to train all three on data up to 2025-12-31 and compare MAE, RMSE, MAPE and training time on the later period.
Data cleaning
Fridays and Saturdays are treated as weekends, and fixed public holidays are flagged.
A temperature of 0 °C for Dhaka is treated as a missing reading.
Impossible values are removed (for example, extreme generation spikes and generation below 2,000 MW).
Missing values are filled with the median of the same year, month (and hour), then the overall median.
Hourly script: readings within the same hour (such as 7:30 pm peak readings) are averaged.
Results
Both examples below are forecasts for 18-12-2026 (Friday, weekend).
1. Area-wise demand, supply and load shedding
![Area-wise forecast for 18-12-2026](images/area_demand_forecast.png)

Area	              Demand (MW)	    Supply (MW)	    Load shedding (MW)
Barishal	             399	          357	                 42
Chattogram	          1281	         1186	                 95
Cumilla	              1234	         1079	                 154
Dhaka	                4668	         4273	                 395
Khulna	              1570	         1399	                 171
Mymensingh	          1121	         933	                 188
Rajshahi	            1277	         1160	                 116
Rangpur	              929	           826	                 103
Sylhet	              548	           473	                 75
All areas	            13,027	       11,686	              ≈1,339
Takeaways:
Dhaka is by far the largest area, at about 36% of total forecast demand (4,668 of 13,027 MW), and also has the largest load shedding (395 MW).
Mymensingh has the highest load shedding relative to its demand (188 of 1,121 MW, about 17%).
Supply is capped at demand in the model, so load shedding is the gap between the two.
3. Hourly generation, 10 AM – 5 PM
![Hourly generation forecast](images/hourly_generation_forecast.png)
Time	 Generation	 Gas	Liquid 	Coal	Hydro	Solar	Wind	Bheramara	Tripura	Adani	Nepal
10 AM	   8773	    3646	402	    4158	  40	 543	  0	      514	     57	   938	 0
11 AM	   9115	    3673	417	    4153	  41	 596	  0	      531	     59	   939	 0
12 PM	   9502	    3800	684	    4166	  42	 626	  0	      530	     64	   974	 0
1 PM	   9890	    3783	1090	  4247	  42	 593	  0	      530	     66	   1011	 0
2 PM	   9247	    3495	1037	4178	   42	   510	  0	      527	     65	   1004	 0
3 PM	   9081	    3499	991	  4160	   42	   373	  0	      527	     61	   995	 0
4 PM	   8853	    3513	964	  4136	   42	   173	  0	      523	     59	   995	 0
5 PM	   9021	    3504	1149	4163	   42	   41	    0	      547	     60	   1027	 0
All values are in MW.
Takeaways:
Generation peaks at 9,890 MW at 1 PM and is lowest at 8,773 MW at 10 AM.
Solar peaks around midday (626 MW at 12 PM) and fades to 41 MW by 5 PM. Liquid fuel rises in the afternoon as solar drops.
Coal (about 4,150 MW) and gas (about 3,500–3,800 MW) provide most of the generation.
4. Forecast vs. the previous year's actual generation
![Forecast vs previous year](images/generation_vs_previous_year.png)
For ranges of more than 3 rows, the hourly script also plots the actual generation at the same date and time one year earlier. In this example the forecast follows the same shape as last year (rise to a 1 PM peak, then a decline), but it is roughly 500–1,000 MW lower at every hour. Last year's values are read from the graph, so they are approximate.
Installation
Python 3.15 is used for this project.
```bash
python -m pip install numpy pandas matplotlib scikit-learn xgboost torch
```
`torch` is imported only when the LSTM is used, so Baseline and XGBoost work without it.
Usage
Put the CSV files in the same folder as the scripts, then run:
```bash
python electricity_demand_prediction.py     # daily, by area
python electricity_forecasting.py           # hourly, by source
```
Example input (daily script):
```
Enter a date or a timeline (e.g. 18/12/2026 or 18/11/2026 to 17/01/2027): 18/12/2026
Area name (press Enter for all areas):
```
Example input (hourly script):
```
Enter a date or a timeline (e.g. 18/12/2026 or 18/11/2026 to 17/01/2027): 18/12/2026
Enter the time (e.g. 10 am or 10 am to 1 pm, press Enter for all 24 hours): 10 am to 5 pm
```
Rules: dates are `dd/mm/yyyy`, a range can span at most 366 days, and dates before the start of the dataset are rejected.
To compare the models:
```bash
python electricity_demand_prediction.py compare
python electricity_forecasting.py compare
```
Project structure
```
.
├── electricity_demand_prediction.py
├── electricity_forecasting.py
├── Bangladesh_Electricity_Dataset_2019_2026.csv
├── bangladesh_power_generation_2019_2026_processed.csv
├── images/
│   ├── area_demand_forecast.png
│   ├── hourly_generation_forecast.png
│   └── generation_vs_previous_year.png
└── README.md
```
Limitations
Forecasts use only calendar information (and monthly typical temperature in the daily script), not real weather, fuel availability or plant outages, so they show typical patterns rather than exact values.
Holidays that move each year (such as Eid) are not in the fixed holiday list.
Forecasts far beyond the end of the data become less reliable.
