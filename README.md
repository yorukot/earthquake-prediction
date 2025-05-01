# Earthquake Prediction Project

## 📊 Data Sources
- **CWA (Central Weather Administration, Taiwan):**  
  [Earthquake Data (1980/1/1 - 2025/5/1)](https://scweb.cwa.gov.tw/zh-tw/earthquake/data/)
- **USGS (United States Geological Survey):**  
  [USGS Earthquake Map & Data](https://earthquake.usgs.gov/earthquakes/map/?extent=20.43216,-241.58936&extent=26.70636,-236.37634&range=search&sort=largest&timeZone=utc&search=%7B%22name%22:%22Search%20Results%22,%22params%22:%7B%22starttime%22:%221980-04-24%2000:00:00%22,%22endtime%22:%222025-05-01%2023:59:59%22,%22maxlatitude%22:25.524,%22minlatitude%22:21.659,%22maxlongitude%22:-237.766,%22minlongitude%22:-240.205,%22minmagnitude%22:2.5,%22orderby%22:%22magnitude%22%7D%7D)

## 🛠️ Data Preprocessing
- All `magType` values are converted to **Mw** (Moment Magnitude) using empirical formulas.
- Data is cleaned and merged from both CWA and USGS sources.
- Only relevant columns are kept for modeling (e.g., time, latitude, longitude, depth, mag).

## 🚀 Features
- Supports both traditional ML (XGBoost) and deep learning (LSTM–CNN) models for earthquake magnitude prediction.
- Training and evaluation scripts automatically save results and code snapshots for reproducibility.

---

> **Note:**
> This project is for research and educational purposes. Earthquake prediction is a complex and unsolved scientific challenge.