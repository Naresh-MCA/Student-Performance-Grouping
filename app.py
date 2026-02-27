import pickle
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('student_performance_model.keras')

with open('label_class.pkl', 'rb') as f:
    label_class = pickle.load(f)

with open('label_gender.pkl', 'rb') as f:
    label_gender = pickle.load(f)

with open('label_test.pkl', 'rb') as f:
    label_test = pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Student Performance Grouping App')

# Input fields
gender = st.selectbox("Select Gender", label_gender.classes_)
parental_level_of_education = st.selectbox("Select Parental Level of Education", one_hot_encoder.categories_[0])
test_preparation_course = st.selectbox("Select Test Preparation Course", label_test.classes_)
math_score = st.number_input("Enter Math Score", min_value=0, max_value=100, value=0)
reading_score = st.number_input("Enter Reading Score", min_value=0, max_value=100, value=0)
writing_score = st.number_input("Enter Writing Score", min_value=0, max_value=100, value=0)
average = st.number_input('Enter Average Marks', min_value=0, max_value=100, value=0)

if st.button("Predict"):

    # Encode gender and test preparation course
    gender_encoded = label_gender.transform([gender])[0]
    test_encoded = label_test.transform([test_preparation_course])[0]

    # One-hot encode parental level of education
    one_hot_encoded = one_hot_encoder.transform([[parental_level_of_education]])
    columns = one_hot_encoder.get_feature_names_out(['parental level of education'])  # ✅ fixed
    encoded_df = pd.DataFrame(one_hot_encoded, columns=columns)

    # Build input dataframe
    input_df = pd.DataFrame({
        'gender': [gender_encoded],
        'test preparation course': [test_encoded],
        'math score': [math_score],
        'reading score': [reading_score],
        'writing score': [writing_score],
        'average': [average]
    })

    # Concatenate and scale
    final_df = pd.concat([input_df, encoded_df], axis=1)
    scaled_df = scaler.transform(final_df)

    # Predict
    prediction = model.predict(scaled_df)
    predicted_class = np.argmax(prediction)
    final_label = label_class.inverse_transform([predicted_class])

    st.success(f"Predicted Class: {final_label[0]}")
    st.write("Prediction Probabilities:", prediction)