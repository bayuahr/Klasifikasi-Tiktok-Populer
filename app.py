import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

# Function Definitions
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

# Load and Preprocess Data
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

st.set_page_config(layout="wide")
# Initialize session state for login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Show login page if not authenticated
if not st.session_state.authenticated:
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Dummy credentials for login
    VALID_USERNAME = "admin"
    VALID_PASSWORD = "password123"

    login_button = st.button("Login")

    if login_button:
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.authenticated = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password. Please try again.")

else:
    # Sidebar Navigation Menu
    menu = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Klasifikasi", "Dataset", "Evaluation"]
    )

    # --- Dashboard Page ---
    if menu == "Dashboard":
        st.title("📊 TikTok Video Dashboard")

        with st.container():
            # Metrics
            total_videos = len(df_with_hashtags)
            popular_videos = len(df_with_hashtags[df_with_hashtags['is_popular'] == "Popular"])
            not_popular_videos = len(df_with_hashtags[df_with_hashtags['is_popular'] == "Not Popular"])
            avg_likes = df_with_hashtags['Likes'].mean()
            avg_shares = df_with_hashtags['Shares'].mean()
            avg_comments = df_with_hashtags['Comments'].mean()

            col1, col2, col3 , col4, col5= st.columns([1, 1, 1,1,1]) 
            with col1:
                st.markdown(
                    f"""
                    <div style="background-color:#4CAF50; padding:20px; border-radius:10px; text-align:center; color:white;">
                        <h3>🎥</h3>
                        <h5>Total Videos</h5>
                        <h5>{total_videos}</h5>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown(
                    f"""
                    <div style="background-color:#FF9800; padding:20px; border-radius:10px; text-align:center; color:white;">
                        <h3>⭐</h3>
                        <h5>Popular Videos</h5>
                        <h5>{popular_videos}</h5>
                    </div>
                    """, unsafe_allow_html=True)

            with col3:
                st.markdown(
                    f"""
                    <div style="background-color:#F44336; padding:20px; border-radius:10px; text-align:center; color:white;">
                        <h3>😞</h3>
                        <h5>Not Popular Videos</h5>
                        <h5>{not_popular_videos}</h5>
                    </div>
                    """, unsafe_allow_html=True)
            with col4:
                st.markdown(
                    f"""
                    <div style="background-color:#03A9F4; padding:20px; border-radius:10px; text-align:center; color:white;">
                        <h3>💖</h3>
                        <h5>Average Likes</h5>
                        <h5>{avg_likes:.2f}</h5>
                    </div>
                    """, unsafe_allow_html=True)

            with col5:
                st.markdown(
                    f"""
                    <div style="background-color:#9C27B0; padding:20px; border-radius:10px; text-align:center; color:white;">
                        <h3>🔁</h3>
                        <h5>Average Shares</h5>
                        <h5>{avg_shares:.2f}</h5>
                    </div>
                    """, unsafe_allow_html=True)


        st.divider()

        col1, col2 = st.columns(2)

        with col1:
                fig, ax = plt.subplots(figsize=(6, 4.4))
                sns.countplot(data=df_with_hashtags, x='is_popular', palette="coolwarm", ax=ax)
                ax.set_title("Popular vs. Not Popular Videos")
                st.pyplot(fig)

        with col2:
                hashtag_sums = df_with_hashtags[hashtag_features].sum().reset_index()
                hashtag_sums.columns = ['Hashtag', 'Frequency']
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.barplot(data=hashtag_sums, x='Frequency', y='Hashtag', palette="Blues_r", ax=ax)
                ax.set_title("Hashtag Frequency")
                st.pyplot(fig)


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
    elif menu == "Evaluation":
        st.header("Evaluation")
        st.write("### Model Performance")
        # Calculate precision, recall, f1-score, and support for each class
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test, rf_classifier.predict(X_test), average=None)
        # Display overall metrics (macro average)
        st.write(f"- Accuracy: {accuracy * 100:.2f}%")
        st.write(f"- Precision: {precision.mean()* 100:.2f}%")
        st.write(f"- Recall: {recall.mean()* 100:.2f}%")
        st.write(f"- F1-Score: {f1_score.mean()* 100:.2f}%")
