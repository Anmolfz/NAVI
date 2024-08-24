from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import warnings
import functools
import zipfile
import io

warnings.filterwarnings('ignore')

app = Flask(__name__)


# Custom caching decorator for in-memory caching of data and model
def cache_data(func):
    cached_data = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if func not in cached_data:
            cached_data[func] = func(*args, **kwargs)
        return cached_data[func]

    return wrapper


@cache_data
def load_data():
    # Read 'New_test.csv'
    df = pd.read_csv("New_test.csv")
    # Open the ZIP file and read 'merged_county_data.csv'
    with zipfile.ZipFile("merged_county_data.zip", 'r') as z:
        with z.open("merged_county_data.csv") as f:
            plans_df = pd.read_csv(f)
    df['Income'] = df['Income'].replace('[\\$,]', '', regex=True).astype(float)
    return df, plans_df


@cache_data
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model


def suggest_plan(age, income, devices, county_name, df, plans_df):
    features = ['Age', 'Income', 'Number_of_Devices']
    target = 'Current_Internet_Speed'

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Number_of_Devices': [devices]
    })

    prediction = model.predict(input_data)[0]
    filtered_df = plans_df[plans_df['county_name'].str.contains(county_name, case=False, na=False)]

    if filtered_df.empty:
        return f"No data found for the specified county: '{county_name}'."

    preds_per_tree = np.array([tree.predict(input_data)[0] for tree in model.estimators_])
    mean_pred = np.mean(preds_per_tree)
    std_dev = np.std(preds_per_tree)

    confidence_score = 100.0 if std_dev == 0 else 100.0 - (std_dev / mean_pred) * 100
    confidence_score = max(0, confidence_score)

    range_min = prediction - 500.0
    range_max = prediction + 500.0
    matches = filtered_df[(filtered_df['max_advertised_download_speed'] >= range_min) &
                          (filtered_df['max_advertised_download_speed'] <= range_max)]

    if not matches.empty:
        unique_matches = matches.drop_duplicates(subset='max_advertised_download_speed')
        sorted_unique_matches = unique_matches.iloc[
            (unique_matches['max_advertised_download_speed'] - prediction).abs().argsort()]

        top_n = 3
        top_unique_matches = sorted_unique_matches.head(top_n)

        if len(top_unique_matches) < top_n:
            remaining_matches = unique_matches[~unique_matches.isin(top_unique_matches)]
            top_unique_matches = pd.concat(
                [top_unique_matches, remaining_matches.head(top_n - len(top_unique_matches))])
        top_unique_matches = top_unique_matches.sort_values(by='max_advertised_download_speed', ascending=True)

        results = [f"<div class='plan-box'><h3>Plan: {row['brand_name']}</h3>"
                   f"<p>Download Speed: {row['max_advertised_download_speed']:.2f} Mbps</p>"
                   f"<p>Upload Speed: {row['max_advertised_upload_speed']:.2f} Mbps</p></div>"
                   for _, row in top_unique_matches.iterrows()]
        return f"Suggested Internet Speed: {prediction:.2f} Mbps<br>Confidence Score: {confidence_score:.2f}%<br>" + "".join(
            results)
    else:
        return "No unique exact matches found."


@app.route('/', methods=['GET', 'POST'])
def index():
    # Load the data
    df, plans_df = load_data()
    counties = plans_df['county_name'].unique()

    result = None
    selected_county = None
    age = ""
    income = ""
    devices = ""

    if request.method == 'POST':
        # Extract the form data
        selected_county = request.form['county_name']
        age = request.form['age']
        income = request.form['income']
        devices = request.form['devices']

        # Generate the plan suggestion
        result = suggest_plan(int(age), float(income), int(devices), selected_county, df, plans_df)

    # Render the template with the form data and the result (if any)
    return render_template('index.html', counties=counties, selected_county=selected_county, age=age, income=income,
                           devices=devices, result=result)


if __name__ == "__main__":
    app.run(debug=True)
