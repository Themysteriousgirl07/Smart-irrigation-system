import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

print("Loading data...")
soil = pd.read_csv('data/sm_Karnataka_2018.csv')
rain = pd.read_csv('data/ka_dist_rainfall_2025.csv')
print(f"Soil rows: {len(soil)}, Rain rows: {len(rain)}")

print("cleaning data...")
soil['Date'] = pd.to_datetime(soil['Date'])
soil['month'] = soil['Date'].dt.month
soil['day_of_year'] = soil['Date'].dt.dayofyear
soil['moisture_pct'] = soil['Aggregate Soilmoisture Percentage (at 15cm)']
soil['moisture_vol'] = soil['Volume Soilmoisture percentage (at 15cm)']


rain['District_key'] = rain['District'].str.upper().str.strip()
soil['District_key'] = soil['DistrictName'].str.upper().str.strip()
 
name_map = {
    'CHIKMAGALUR': 'CHIKKAMAGALURU',
    'CHAMRAJNAGAR': 'CHAMARAJANAGARA',
    'BELLARY': 'BALLARI',
    'BELGAUM': 'BELAGAVI',
    'BIJAPUR': 'VIJAYAPURA',
    'SHIMOGA': 'SHIVAMOGGA',
    'TUMKUR': 'TUMAKURU',
    'KOPPAL': 'KOPPALA',
    'GULBARGA': 'KALABURAGI',
    'BANGALORE': 'BENGALURU URBAN',
    'BANGALORE RURAL': 'BENGALURU RURAL',
    'MYSORE': 'MYSURU',
}
soil['District_key'] = soil['District_key'].replace(name_map)

merged = soil.merge(
    rain[['District_key', 'Annual Normal', 'Annual Actual',
          'Annual Departure (%)', 'SWM Normal', 'SWM Actual']],
    on='District_key', how='left'
).dropna(subset=['Annual Actual'])
 
print(f"Merged: {len(merged)} rows across {merged['District_key'].nunique()} districts")

print("Step 3: Adding weather features...")
np.random.seed(42)
n = len(merged)
 
merged['temp_max'] = np.random.uniform(25, 38, n)
merged['humidity'] = np.random.uniform(40, 90, n)
merged['et0'] = np.random.uniform(2, 8, n)
merged['solar_radiation'] = np.random.uniform(10, 25, n)
merged['wind_speed'] = np.random.uniform(5, 20, n)
merged['live_rain'] = np.random.uniform(0, 15, n)

def estimate_moisture(baseline,et0, live_rain, humidity):
    humidity_factor = 1 - (humidity / 200)
    net_change = live_rain -(et0 * humidity_factor)
    estimated = baseline + net_change
    return float(np.clip(estimated, 0,100))
merged['estimated_moisture'] = merged.apply(
    lambda r: estimate_moisture(
        r['moisture_pct'], r['et0'], r['live_rain'], r['humidity']
    ), axis=1
)


print("Step 4: building  Training dataset...")

CROP_NEEDS = { 
    'paddy':8, 'ragi':4, 'maize':5,
    'sugarcane':10, 'groundnut':4, 'cotton':6
}

THRESHOLDS = {
    'paddy': 50,
    'ragi': 40,
    'maize': 45,
    'sugarcane': 65,
    'groundnut': 35,
    'cotton': 45
}

rows = []
for _, row in merged.iterrows():
    for crop, daily_need in CROP_NEEDS.items():
        threshold = THRESHOLDS[crop]
        rain_dep = row['Annual Departure (%)']
        irrigate = int(row['estimated_moisture'] < threshold and rain_dep < 20)
        rows.append({
            'estimated_moisture': row['estimated_moisture'],
            'baseline_moisture': row['moisture_pct'],
            'temp_max': row['temp_max'],
            'humidity': row['humidity'],
            'et0': row['et0'],
            'solar_radiation': row['solar_radiation'],
            'wind_speed': row['wind_speed'],
            'live_rain': row['live_rain'],
            'month': row['month'],
            'day_of_year': row['day_of_year'],
            'annual_normal': row['Annual Normal'],
            'rain_departure': rain_dep,
            'swm_actual': row['SWM Actual'],
            'crop_need_mm': daily_need,
            'min_moisture_threshold': threshold,
            'irrigate': irrigate
        })

df = pd.DataFrame(rows)
print(f"Total samples: {len(df)}")
print(f"Irrigate=1: {df['irrigate'].sum()} | Skip=0: {(df['irrigate']==0).sum()}")

print("Step 5: Training Random Forest...")

# IMPORTANT: This order must match exactly in app.py
FEATURES = [
    'estimated_moisture',
    'baseline_moisture',
    'temp_max',
    'humidity',
    'et0',
    'solar_radiation',
    'wind_speed',
    'live_rain',
    'month',
    'day_of_year',
    'annual_normal',
    'rain_departure',
    'swm_actual',
    'crop_need_mm',
    'min_moisture_threshold'
]

X = df[FEATURES]
y = df['irrigate']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────
# STEP 8 — Evaluate
# ─────────────────────────────────────────
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"\nModel Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, preds, target_names=['Skip', 'Irrigate']))

print("Feature Importance:")
importances = pd.Series(
    model.feature_importances_, index=FEATURES
).sort_values(ascending=False)
print(importances.round(3).to_string())

# ─────────────────────────────────────────
# STEP 9 — Save
# ─────────────────────────────────────────
pickle.dump(model, open('irrigation_model.pkl', 'wb'))
print("\n✅ irrigation_model.pkl saved!")
print("Now run: python app.py")

