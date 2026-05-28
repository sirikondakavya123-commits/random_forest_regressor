# IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import pickle

# PAGE CONFIG

st.set_page_config(

    page_title="Housing Price Prediction",

    page_icon="🏠",

    layout="wide"

)

# LOAD MODEL

model = pickle.load(

    open(
        "models/random_forest_regressor.pkl",
        "rb"
    )

)

# TITLE

st.markdown(

    """
    <h1 style='text-align:center; color:#1E3A8A;'>
    🏠 Housing Price Prediction
    </h1>
    """,

    unsafe_allow_html=True

)

st.write(
    "Predict Housing Prices using Random Forest Regressor"
)

# SIDEBAR

st.sidebar.header("Enter House Details")

longitude = st.sidebar.number_input(
    "Longitude",
    value=-122.23
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=37.88
)

housing_median_age = st.sidebar.number_input(
    "Housing Median Age",
    value=20
)

total_rooms = st.sidebar.number_input(
    "Total Rooms",
    value=1000
)

total_bedrooms = st.sidebar.number_input(
    "Total Bedrooms",
    value=200
)

population = st.sidebar.number_input(
    "Population",
    value=800
)

households = st.sidebar.number_input(
    "Households",
    value=300
)

median_income = st.sidebar.number_input(
    "Median Income",
    value=3.5
)

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    [0, 1, 2, 3, 4]
)

# INPUT DATAFRAME

input_data = pd.DataFrame({

    "longitude": [longitude],

    "latitude": [latitude],

    "housing_median_age": [housing_median_age],

    "total_rooms": [total_rooms],

    "total_bedrooms": [total_bedrooms],

    "population": [population],

    "households": [households],

    "median_income": [median_income],

    "ocean_proximity": [ocean_proximity]

})

# SHOW INPUT

st.subheader("Input Data")

st.dataframe(input_data)

# PREDICTION

if st.button("Predict House Price"):

    prediction = model.predict(
        input_data
    )

    st.success(
        f"Predicted House Price: ${prediction[0]:,.2f}"
    )

# FOOTER

st.markdown("---")

st.markdown(

    """
    <center>
    Developed using Streamlit and Random Forest Regressor
    </center>
    """,

    unsafe_allow_html=True

)
