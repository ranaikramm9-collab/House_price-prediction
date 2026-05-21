from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")

# =========================
# CITY OPTIONS
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
# STATEZIP OPTIONS
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
# ENCODING MAPS
# =========================
city_encoding = {
    city: idx for idx, city in enumerate(city_options)
}

statezip_encoding = {
    state: idx for idx, state in enumerate(statezip_options)
}

# =========================
# REQUEST MODEL
# =========================
class HouseData(BaseModel):

    sqft_living: float
    sqft_above: float
    bathrooms: float
    sqft_lot: float
    bedrooms: int

    city: str
    statezip: str

    house_age: float
    sqft_per_bedroom: float

# =========================
# HOME ROUTE
# =========================
@app.get("/")
def home():

    return {
        "message": "House Price Prediction API Running"
    }

# =========================
# PREDICTION ROUTE
# =========================
@app.post("/predict")
def predict(data: HouseData):

    # =========================
    # ENCODING
    # =========================
    city_encoded = city_encoding.get(data.city)

    statezip_encoded = statezip_encoding.get(data.statezip)

    # =========================
    # UNKNOWN VALUE CHECK
    # =========================
    if city_encoded is None:
        return {
            "error": "Invalid city name"
        }

    if statezip_encoded is None:
        return {
            "error": "Invalid statezip"
        }

    # =========================
    # CREATE INPUT ARRAY
    # =========================
    input_data = np.array([[
        data.sqft_living,
        data.sqft_above,
        data.bathrooms,
        data.sqft_lot,
        data.bedrooms,
        city_encoded,
        statezip_encoded,
        data.house_age,
        data.sqft_per_bedroom
    ]])

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(input_data)

    # =========================
    # RETURN RESPONSE
    # =========================
    return {
        "predicted_price": float(prediction[0]),
        "city_encoded": city_encoded,
        "statezip_encoded": statezip_encoded
    }