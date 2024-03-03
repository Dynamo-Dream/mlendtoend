import streamlit as st
import pandas as pd
from src.pipeline.train_pipeline import CustomData, PredictPipeline
from pathlib import Path

pkl_path = Path(__file__).parents[1]
st.set_page_config(layout="wide")

def main():
    st.title("Predict Student Scores")
    
    st.write("Enter the details of the student:")
    
    gender = st.selectbox("Gender", ["male", "female"])
    race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Education Level", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)
    
    if st.button("Predict"):
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        st.write("Predicted Score:")
        st.write(results[0])

if __name__ == "__main__":
    main()
