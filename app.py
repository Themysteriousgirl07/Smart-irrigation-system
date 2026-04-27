from flask import Flask, render_template, request
import pickle, requests, numpy as np, pandas as pd
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────
# Load ML model
# ─────────────────────────────────────────
model = pickle.load(open('irrigation_model.pkl', 'rb'))

# ─────────────────────────────────────────
# Load district soil moisture baselines from CSV
# ─────────────────────────────────────────
soil_df = pd.read_csv('data/sm_Karnataka_2018.csv')
soil_df['District_key'] = soil_df['DistrictName'].str.upper().str.strip()
name_map = {
    'CHIKMAGALUR': 'CHIKKAMAGALURU', 'CHAMRAJNAGAR': 'CHAMARAJANAGARA',
    'BELLARY': 'BALLARI', 'BELGAUM': 'BELAGAVI', 'BIJAPUR': 'VIJAYAPURA',
    'SHIMOGA': 'SHIVAMOGGA', 'TUMKUR': 'TUMAKURU', 'KOPPAL': 'KOPPALA',
    'GULBARGA': 'KALABURAGI', 'BANGALORE': 'BENGALURU URBAN',
    'BANGALORE RURAL': 'BENGALURU RURAL', 'MYSORE': 'MYSURU',
}
soil_df['District_key'] = soil_df['District_key'].replace(name_map)
SOIL_BASELINE = soil_df.groupby('District_key')['Aggregate Soilmoisture Percentage (at 15cm)'].mean().to_dict()

# ─────────────────────────────────────────
# Load rainfall data
# ─────────────────────────────────────────
rain_df = pd.read_csv('data/ka_dist_rainfall_2025.csv')
rain_df['District_key'] = rain_df['District'].str.upper().str.strip()
RAIN_DATA = rain_df.set_index('District_key')[
    ['Annual Normal', 'Annual Actual', 'Annual Departure (%)', 'SWM Normal', 'SWM Actual']
].to_dict('index')

# ─────────────────────────────────────────
# District coordinates for weather API
# ─────────────────────────────────────────
DISTRICT_COORDS = {
    'BAGALKOTE': (16.18, 75.70), 'BALLARI': (15.14, 76.92),
    'BELAGAVI': (15.85, 74.50), 'BENGALURU URBAN': (12.97, 77.59),
    'BENGALURU RURAL': (13.21, 77.51), 'BIDAR': (17.91, 77.52),
    'CHAMARAJANAGARA': (11.92, 76.94), 'CHIKKABALLAPURA': (13.43, 77.73),
    'CHIKKAMAGALURU': (13.32, 75.77), 'CHITRADURGA': (14.23, 76.40),
    'DAKSHINA KANNADA': (12.87, 75.02), 'DAVANAGERE': (14.46, 75.92),
    'DHARWAD': (15.45, 75.01), 'GADAG': (15.42, 75.62),
    'HASSAN': (13.00, 76.10), 'HAVERI': (14.79, 75.40),
    'KALABURAGI': (17.33, 76.82), 'KODAGU': (12.42, 75.74),
    'KOLAR': (13.13, 78.13), 'KOPPALA': (15.35, 76.15),
    'MANDYA': (12.52, 76.90), 'MYSURU': (12.29, 76.64),
    'RAICHUR': (16.21, 77.36), 'RAMANAGARA': (12.72, 77.28),
    'SHIVAMOGGA': (13.93, 75.57), 'TUMAKURU': (13.34, 77.10),
    'UDUPI': (13.34, 74.75), 'UTTARA KANNADA': (14.79, 74.68),
    'VIJAYANAGAR': (15.17, 76.38), 'VIJAYAPURA': (16.83, 75.72),
    'YADGIR': (16.77, 77.13),
}

# ─────────────────────────────────────────
# Crop data
# ─────────────────────────────────────────
CROP_NEEDS = {
    'paddy': 8, 'ragi': 4, 'maize': 5,
    'sugarcane': 10, 'groundnut': 4, 'jowar': 3, 'cotton': 6
}
THRESHOLDS = {
    'paddy': 60, 'ragi': 40, 'maize': 45,
    'sugarcane': 65, 'groundnut': 40, 'jowar': 35, 'cotton': 45
}
CROP_TIPS = {
    'paddy': 'Paddy needs standing water during transplanting. Keep fields flooded 2-5cm during vegetative stage.',
    'ragi': 'Ragi is drought-tolerant. Irrigate only at critical stages: germination, tillering, grain-filling.',
    'maize': 'Maize is sensitive to drought at tasseling. Ensure moisture during silking stage.',
    'sugarcane': 'Sugarcane is a heavy water user. Irrigate every 7-10 days during dry periods.',
    'groundnut': 'Irrigate at flowering and pegging stages. Avoid waterlogging.',
    'jowar': 'Jowar is very drought tolerant. Irrigate only if soil is extremely dry.',
    'cotton': 'Cotton needs water at squaring and boll development. Avoid over-irrigation.',
}


# ─────────────────────────────────────────
# Estimate current soil moisture using real time weather
#
# Formula: current moisture = baseline + rain - evaporation
# ET0 = how much water the soil loses to heat, wind, and sun
# High humidity slows down evaporation
# ─────────────────────────────────────────
def estimate_moisture(baseline, et0, live_rain, humidity):
    humidity_factor = 1 - (humidity / 200)
    net_change = live_rain - (et0 * humidity_factor)
    estimated = baseline + net_change
    return float(np.clip(estimated, 0, 100))


# ─────────────────────────────────────────
# Fetch live weather from Open-Meteo API
# Free, no API key needed
# Returns: temp, humidity, ET0, solar radiation, wind, rainfall
# ─────────────────────────────────────────
def get_live_weather(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "precipitation_sum",
                "temperature_2m_max",
                "temperature_2m_min",
                "et0_fao_evapotranspiration",
                "windspeed_10m_max",
                "shortwave_radiation_sum"
            ],
            "hourly": ["relativehumidity_2m"],
            "forecast_days": 7,
            "timezone": "Asia/Kolkata"
        }
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        daily = data['daily']

        # Average humidity from hourly readings
        avg_humidity = sum(data['hourly']['relativehumidity_2m']) / len(data['hourly']['relativehumidity_2m'])

        return {
            'dates': daily['time'],
            'rain': daily['precipitation_sum'],
            'temp_max': daily['temperature_2m_max'],
            'et0': daily['et0_fao_evapotranspiration'],
            'solar': daily['shortwave_radiation_sum'],
            'wind': daily['windspeed_10m_max'],
            'humidity': avg_humidity,
            'rain_today': daily['precipitation_sum'][0],
            'rain_next3': sum(daily['precipitation_sum'][:3]),
            'rain_next7': sum(daily['precipitation_sum']),
            'temp_today': daily['temperature_2m_max'][0],
            'et0_today': daily['et0_fao_evapotranspiration'][0],
            'solar_today': daily['shortwave_radiation_sum'][0],
            'wind_today': daily['windspeed_10m_max'][0],
        }
    except Exception:
        # Fallback values if API is unreachable
        return {
            'dates': [], 'rain': [], 'temp_max': [], 'et0': [],
            'solar': [], 'wind': [], 'humidity': 60,
            'rain_today': 0, 'rain_next3': 0, 'rain_next7': 0,
            'temp_today': 30, 'et0_today': 5,
            'solar_today': 18, 'wind_today': 10,
        }


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route('/')
def home():
    districts = sorted(DISTRICT_COORDS.keys())
    crops = list(CROP_NEEDS.keys())
    return render_template('index.html', districts=districts, crops=crops)


@app.route('/predict', methods=['POST'])
def predict():
    district = request.form['district'].upper()
    crop = request.form['crop'].lower()

    # Coordinates for this district
    lat, lon = DISTRICT_COORDS.get(district, (15.0, 75.7))

    # Get live weather from Open-Meteo
    weather = get_live_weather(lat, lon)

    # Get historical data from CSVs
    baseline_moisture = SOIL_BASELINE.get(district, 35.0)
    rain_info = RAIN_DATA.get(district, {})
    annual_normal = rain_info.get('Annual Normal', 800)
    rain_departure = rain_info.get('Annual Departure (%)', 0)
    swm_actual = rain_info.get('SWM Actual', 400)

    # Current date info
    now = datetime.now()
    month = now.month
    day_of_year = now.timetuple().tm_yday

    # Crop specific values
    crop_need = CROP_NEEDS[crop]
    min_threshold = THRESHOLDS[crop]

    # Estimate current soil moisture using live weather
    estimated_moisture = estimate_moisture(
        baseline_moisture,
        weather['et0_today'],
        weather['rain_today'],
        weather['humidity']
    )

    # Build feature array - MUST match order in train_model.py
    features = np.array([[
        estimated_moisture,        # real time estimated soil moisture
        baseline_moisture,         # historical baseline from CSV
        weather['temp_today'],     # live temperature
        weather['humidity'],       # live humidity
        weather['et0_today'],      # live evapotranspiration
        weather['solar_today'],    # live solar radiation
        weather['wind_today'],     # live wind speed
        weather['rain_today'],     # live rainfall today
        month,                     # current month
        day_of_year,               # day of year
        annual_normal,             # district normal rainfall
        rain_departure,            # surplus or deficit this year
        swm_actual,                # monsoon rainfall
        crop_need,                 # this crop's daily water need
        min_threshold              # this crop's minimum moisture need
    ]])

    # ML Model prediction
    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0]) * 100

    # Override: heavy rain coming = skip regardless
    override_msg = None
    if weather['rain_next3'] > 25:
        prediction = 0
        override_msg = f"Heavy rain expected ({round(weather['rain_next3'])}mm in 3 days). Skip irrigation."
    elif weather['rain_next3'] > 10 and prediction == 1:
        override_msg = f"Light rain coming ({round(weather['rain_next3'])}mm). Reduce irrigation amount."

    # How much to irrigate (subtract rain credit)
    net_rain_credit = weather['rain_next3'] / 3
    irrigate_amount = max(0, round(crop_need - net_rain_credit))

    result = {
        'district': district.title(),
        'crop': crop.title(),
        'decision': 'IRRIGATE' if prediction == 1 else 'SKIP',
        'confidence': round(confidence, 1),
        'irrigate_mm': irrigate_amount,

        # Moisture
        'baseline_moisture': round(baseline_moisture, 1),
        'estimated_moisture': round(estimated_moisture, 1),

        # Live weather
        'temp_today': round(weather['temp_today'], 1),
        'humidity': round(weather['humidity'], 1),
        'et0_today': round(weather['et0_today'], 1),
        'solar_today': round(weather['solar_today'], 1),
        'wind_today': round(weather['wind_today'], 1),
        'rain_today': round(weather['rain_today'], 1),
        'rain_next3': round(weather['rain_next3'], 1),
        'rain_next7': round(weather['rain_next7'], 1),

        # Chart data
        'weather_dates': weather['dates'],
        'weather_rain': weather['rain'],
        'weather_temp': weather['temp_max'],

        # Rainfall context
        'annual_normal': annual_normal,
        'annual_actual': rain_info.get('Annual Actual', 'N/A'),
        'rain_departure': rain_departure,

        # Messages
        'override_msg': override_msg,
        'crop_tip': CROP_TIPS[crop],
    }

    return render_template('result.html', r=result , estimated_moisture=estimated_moisture)


if __name__ == '__main__':
    app.run(debug=True)
