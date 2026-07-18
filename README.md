# Seoul Bike Rental Demand Prediction and Analytics

> **GitHub Repository:** Add your public repository link here before submission.

## 1. Project Description

This project develops an end-to-end data analytics and machine learning system that predicts the number of bicycles likely to be rented during a specific operating hour in Seoul. The project is designed to support bike-sharing operators in planning bicycle availability, redistribution, staffing, and maintenance before demand occurs.

The prediction target is **Rented Bike Count**, which represents the number of bicycles rented within one hourly record. The final dashboard allows users to enter a historical date, time, temperature, humidity, solar radiation, and rainfall. The system automatically identifies the corresponding season in Seoul and predicts expected hourly rental demand.

## 2. Dataset Overview

The project uses the **Seoul Bike Sharing Demand** dataset from the UCI Machine Learning Repository.

The dataset contains:

- **8,760 hourly observations**
- **13 predictive features**
- Hourly rented-bike counts
- Date and time information
- Weather measurements
- Seasonal information
- Holiday status
- Service-functioning status

The prediction task is a **regression problem** because the target is a numerical bike-rental count.


## 3. Data Preparation

The notebook performs the following preparation steps:

- Loads the CSV file using a compatible encoding.
- Checks the dataset shape, column types, missing values, and unique values.
- Removes exact duplicate records.
- Converts the `Date` column into datetime format.
- Renames columns into clear Python-friendly names.
- Reviews numerical outliers using the Interquartile Range method.
- Retains genuine high-demand and extreme-weather observations because they contain useful operational information.
- Filters the dataset to include only records where `Functioning Day = Yes`.

Filtering non-functioning records is a row-level data-preparation decision. Feature selection is performed separately to identify which columns should be used by the final prediction model.

## 4. Final Predictor Inputs

The deployed predictor uses the following inputs:

- Prediction date
- Prediction time
- Temperature
- Humidity
- Solar radiation
- Rainfall

The season is detected automatically from the selected date.

The model also uses automatically calculated features:

- Season
- Peak-hour status
- Comfort index
- Hour sine
- Hour cosine

## 5. Machine Learning Models

The project compares the following regression models:

### Ridge Regression

Ridge Regression provides a simple linear baseline and helps show whether the relationships can be explained using a linear model.

### Random Forest Regressor

Random Forest combines many decision trees and can learn nonlinear relationships between time, weather, season, and demand.

### Extra Trees Regressor

Extra Trees is another ensemble model that introduces greater randomness when creating trees. It can perform strongly on complex tabular data.

### Tuned Extra Trees

GridSearchCV is used to test different Extra Trees parameter combinations using cross-validation.

The models are evaluated using the same 80% training and 20% testing split for a fair comparison.

## 6. Setup Instructions

### Requirements

Install Python 3.10 or a newer compatible version.

Recommended tools:

- Python
- Jupyter Notebook or JupyterLab
- Visual Studio Code or Anaconda
- Web browser
- Git and GitHub

### Step 1: Download or Clone the Repository

Using Git:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd WheelWise_Predictor
```

Alternatively, download the repository as a ZIP file and extract it.

### Step 2: Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install the Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Jupyter Notebook

```bash
jupyter notebook WheelWise.ipynb
```

Run the notebook cells from top to bottom.

Running the notebook performs the full analysis and recreates:

- Trained model
- Model comparison results
- Feature-importance results

### Step 5: Run the Streamlit Dashboard

```bash
streamlit run app.py
```

A browser window should open automatically. The default local address is usually:

```bash
http://localhost:8501
```

## 6. Required Python Libraries

The main libraries are:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- streamlit
- plotly
- jupyter

The exact versions or minimum versions are listed in `requirements.txt`.

## 7. How to Use the Predictor

1. Open the **Bike Predictor** page.
2. Choose a date within the dataset period.
3. Select a prediction time.
4. Enter temperature and humidity.
5. Enter solar-radiation and rainfall conditions.
6. Click **Predict Bike Demand**.
7. Review:
   - Predicted rental count
   - Demand category
   - Selected period
   - Demand gauge

The output should be treated as decision support rather than a guaranteed future value.

## 8. Data Source and Citation

### Official Dataset Page

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

### Dataset Citation

UCI Machine Learning Repository. (2020). *Seoul Bike Sharing Demand* [Dataset]. https://doi.org/10.24432/C5F62R

### Licence

The dataset is distributed under the **Creative Commons Attribution 4.0 International licence (CC BY 4.0)**. The dataset may be shared and adapted as long as appropriate credit is provided.

Licence information:

https://creativecommons.org/licenses/by/4.0/
