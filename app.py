import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

''' Function Definitions '''
def extract_hashtags(text):
    if pd.isna(text):
        return []
    hashtags = [tag.strip('#') for tag in text.split() if tag.startswith('#')]
    return hashtags

def popularity_score(row):
    likes = row['Likes']
    shares = row['Shares']
    comments = row['Comments']
    plays = row['Plays']
    score = (likes * 0.4) + (shares * 0.3) + (comments * 0.2) + (plays * 0.1)
    return score

def is_popular(row):
    score = row['popularity_score']
    return "Popular" if score > 250000 else "Not Popular"

''' Load and Preprocess Data '''
df = pd.read_excel("datasetTiktok.xlsx")
df['hashtags_list'] = df['Caption'].apply(extract_hashtags)
df_with_hashtags = df[df['hashtags_list'].map(len) > 0]
df_with_hashtags['popularity_score'] = df_with_hashtags.apply(popularity_score, axis=1)
df_with_hashtags['is_popular'] = df_with_hashtags.apply(is_popular, axis=1)

# Add features for hashtags
hashtag_features = ['foryou', 'fy', 'fyp', 'viral', 'foryoupage']
for feature in hashtag_features:
    df_with_hashtags[feature] = df_with_hashtags['hashtags_list'].apply(
        lambda hashtags: int(any(feature in tag.lower() for tag in hashtags))
    )

# Prepare data for training
X = df_with_hashtags[['Likes', 'Shares', 'Comments', 'Plays'] + hashtag_features]
y = df_with_hashtags['is_popular']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))

''' Streamlit App '''
st.title("Tiktok Video Popularity Predictor")

# Login Section
st.subheader("Login")
username = st.text_input("Username")
password = st.text_input("Password", type="password")

# Dummy credentials for login
VALID_USERNAME = "admin"
VALID_PASSWORD = "password123"

if st.button("Login"):
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        st.success("Login successful!")

        # Sidebar Navigation Menu
        menu = st.sidebar.radio(
            "Navigation",
            ["Dashboard", "Klasifikasi", "Dataset", "Evaluation"]
        )

        # Dashboard Page
        if menu == "Dashboard":
            st.header("Dashboard")
            st.write("Welcome to the Tiktok Video Popularity Predictor Dashboard!")
            st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
            st.metric("Total Records", len(df_with_hashtags))

        # Klasifikasi Page
        elif menu == "Klasifikasi":
            st.header("Klasifikasi Video Populer")
            likes = st.number_input("Jumlah Likes", min_value=0)
            shares = st.number_input("Jumlah Shares", min_value=0)
            comments = st.number_input("Jumlah Comments", min_value=0)
            plays = st.number_input("Jumlah Plays", min_value=0)

            st.subheader("Select Hashtags")
            hashtags_binary = [
                st.checkbox(f"#{feature.capitalize()}", key=feature) for feature in hashtag_features
            ]

            if st.button("Submit"):
                input_data = pd.DataFrame([[
                    likes, shares, comments, plays
                ] + hashtags_binary], columns=['Likes', 'Shares', 'Comments', 'Plays'] + hashtag_features)

                prediction = rf_classifier.predict(input_data)
                st.subheader("Prediction Result")
                st.write(f"This video is predicted to be: **{prediction[0]}**")

        # Dataset Page
        elif menu == "Dataset":
            st.header("Dataset")
            st.write("Here is a preview of the dataset:")
            st.dataframe(df_with_hashtags.head(10))
            st.download_button(
                "Download Dataset",
                df_with_hashtags.to_csv(index=False).encode('utf-8'),
                "dataset.csv",
                "text/csv",
                key="download-dataset"
            )

        # Evaluation Page
        elif menu == "Evaluation":
            st.header("Evaluation")
            st.write("### Model Performance")
            st.write(f"**Accuracy**: {accuracy * 100:.2f}%")
            st.write("### Classification Report")
            st.text(classification_report(y_test, rf_classifier.predict(X_test)))
    else:
        st.error("Invalid username or password. Please try again.")
