import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Seattle & Washington House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")

# =========================
# ORIGINAL CITY VALUES
# =========================
city_options = [
    "Seattle",
    "Renton",
    "Bellevue",
    "Redmond",
    "Kirkland",
    "Issaquah",
    "Kent",
    "Auburn",
    "Sammamish",
    "Federal Way",
    "Shoreline",
    "Woodinville",
    "Maple Valley",
    "Mercer Island",
    "Burien",
    "Snoqualmie",
    "Kenmore",
    "Des Moines",
    "North Bend",
    "Covington",
    "Duvall",
    "Lake Forest Park",
    "Bothell",
    "Newcastle",
    "Tukwila",
    "SeaTac",
    "Vashon",
    "Enumclaw",
    "Carnation",
    "Normandy Park",
    "Clyde Hill",
    "Fall City",
    "Medina",
    "Black Diamond",
    "Ravensdale",
    "Pacific",
    "Algona",
    "Yarrow Point",
    "Skykomish",
    "Milton",
    "Preston",
    "Inglewood-Finn Hill",
    "Snoqualmie Pass",
    "Beaux Arts Village"
]

# =========================
# ORIGINAL STATEZIP VALUES
# =========================
statezip_options = [
    "WA 98103",
    "WA 98052",
    "WA 98117",
    "WA 98115",
    "WA 98006",
    "WA 98004",
    "WA 98033",
    "WA 98074",
    "WA 98053",
    "WA 98027",
    "WA 98125",
    "WA 98133",
    "WA 98059",
    "WA 98029",
    "WA 98155",
    "WA 98118",
    "WA 98034",
    "WA 98008",
    "WA 98042",
    "WA 98038",
    "WA 98040",
    "WA 98058",
    "WA 98072",
    "WA 98011",
    "WA 98146",
    "WA 98075",
    "WA 98065",
    "WA 98092",
    "WA 98116",
    "WA 98107",
    "WA 98105",
    "WA 98109",
    "WA 98007",
    "WA 98144",
    "WA 98023",
    "WA 98045",
    "WA 98119",
    "WA 98028",
    "WA 98014",
    "WA 98070",
    "WA 98198",
    "WA 98010",
    "WA 98166",
    "WA 98019",
    "WA 98056",
    "WA 98001",
    "WA 98047",
    "WA 98288",
    "WA 98050",
    "WA 98354",
    "WA 98068"
]

# =========================
# MANUAL ENCODING MAPS
# =========================
city_encoding = {city: idx for idx, city in enumerate(city_options)}
statezip_encoding = {
    state: idx for idx, state in enumerate(statezip_options)
}

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📘 Feature Descriptions")

st.sidebar.markdown("""
### 🏠 Input Features

#### 1. sqft_living
Total living area of the house in square feet.

#### 2. sqft_above
Area above ground level.

#### 3. bathrooms
Number of bathrooms.

#### 4. sqft_lot
Total land area of the property.

#### 5. bedrooms
Number of bedrooms.

#### 6. city
City where the house is located.

#### 7. statezip
ZIP code area.

#### 8. house_age
Current house age.

#### 9. sqft_per_bedroom
Living area divided by bedrooms.
""")

# =========================
# MAIN TITLE
# =========================
st.title("🏠 House Price Prediction App")

st.write("Enter house details to predict the price.")

# =========================
# INPUTS
# =========================
col1, col2 = st.columns(2)

with col1:

    sqft_living = st.number_input(
        "sqft_living",
        min_value=100,
        value=2000
    )

    sqft_above = st.number_input(
        "sqft_above",
        min_value=100,
        value=1500
    )

    bathrooms = st.number_input(
        "bathrooms",
        min_value=1.0,
        value=2.0
    )

    sqft_lot = st.number_input(
        "sqft_lot",
        min_value=500,
        value=4000
    )

    bedrooms = st.number_input(
        "bedrooms",
        min_value=1,
        value=3
    )

with col2:

    city = st.selectbox(
        "Select City",
        city_options
    )

    statezip = st.selectbox(
        "Select StateZIP",
        statezip_options
    )

    house_age = st.number_input(
        "house_age",
        min_value=0,
        value=10
    )

    sqft_per_bedroom = st.number_input(
        "sqft_per_bedroom",
        min_value=1.0,
        value=500.0
    )

# =========================
# ENCODING
# =========================
city_encoded = city_encoding[city]
statezip_encoded = statezip_encoding[statezip]

# =========================
# PREDICTION
# =========================
if st.button("Predict House Price"):

    input_data = np.array([[
        sqft_living,
        sqft_above,
        bathrooms,
        sqft_lot,
        bedrooms,
        city_encoded,
        statezip_encoded,
        house_age,
        sqft_per_bedroom
    ]])

    prediction = model.predict(input_data)

    st.success(
        f"🏡 Predicted House Price: ${prediction[0]:,.2f}"
    )

    # Show Input Summary
    st.subheader("📋 Input Summary")

    summary_df = pd.DataFrame({
        "Feature": [
            "sqft_living",
            "sqft_above",
            "bathrooms",
            "sqft_lot",
            "bedrooms",
            "city",
            "statezip",
            "house_age",
            "sqft_per_bedroom"
        ],
        "Value": [
            sqft_living,
            sqft_above,
            bathrooms,
            sqft_lot,
            bedrooms,
            city,
            statezip,
            house_age,
            sqft_per_bedroom
        ]
    })

    st.dataframe(summary_df)