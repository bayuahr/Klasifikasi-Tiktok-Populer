import streamlit as st
import pickle
import pandas as pd  # Import pandas for DataFrame support

# Load the classifier model
@st.cache_resource
def load_model():
    with open('rf_classifier_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title of the app
st.title("Klasifikasi Video Populer Tiktok")

# Input fields for metrics
likes = st.number_input("Jumlah Likes", min_value=0)
shares = st.number_input("Jumlah Shares", min_value=0)
comments = st.number_input("Jumlah Comments", min_value=0)
plays = st.number_input("Jumlah Plays", min_value=0)

# Hashtag checkboxes
st.subheader("Select Hashtags")
hashtags_selected = []
if st.checkbox("For You (#foryou)"):
    hashtags_selected.append("#foryou")
if st.checkbox("FY (#fy)"):
    hashtags_selected.append("#fy")
if st.checkbox("FYP (#fyp)"):
    hashtags_selected.append("#fyp")
if st.checkbox("Viral (#viral)"):
    hashtags_selected.append("#viral")
if st.checkbox("For You Page (#foryoupage)"):
    hashtags_selected.append("#foryoupage")

# Encode hashtags into binary features
hashtags_binary = [
    int("#foryou" in hashtags_selected),
    int("#fy" in hashtags_selected),
    int("#fyp" in hashtags_selected),
    int("#viral" in hashtags_selected),
    int("#foryoupage" in hashtags_selected),
]

# Feature names (should match the names used during model training)
feature_names = [
    "Likes", "Shares", "Comments", "Plays",
    "foryou", "fy", "fyp", "viral", "foryoupage"
]

# Display entered data
if st.button("Submit"):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame([[
        likes, shares, comments, plays
    ] + hashtags_binary], columns=feature_names)

    # Predict using the loaded model
    prediction = model.predict(input_data)
    st.subheader("Prediction Result")
    st.write("This video is predicted to be:", prediction[0])
