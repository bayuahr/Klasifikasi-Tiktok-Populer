import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

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

    if st.button("Login"):
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

    # Dashboard Page
    if menu == "Dashboard":
        st.header("Dashboard")

        import seaborn as sns
        import matplotlib.pyplot as plt

        # Set Seaborn theme for styling
        sns.set_theme(style="whitegrid")

        # 1. Distribution of Popular vs. Not Popular Videos
        st.subheader("1. Distribution of Popular vs. Not Popular Videos")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df_with_hashtags, x='is_popular', palette="coolwarm", ax=ax1)
        ax1.set_title("Popular vs. Not Popular Videos", fontsize=16)
        ax1.set_xlabel("Popularity", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        st.pyplot(fig1)

        # 2. Average Popularity Metrics
        st.subheader("2. Average Popularity Metrics")
        avg_metrics = df_with_hashtags.groupby('is_popular')[['Likes', 'Shares', 'Comments', 'Plays']].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Melt the data for easier plotting with Seaborn
        avg_metrics_melted = avg_metrics.melt(id_vars='is_popular', var_name='Metric', value_name='Average')
        bar_plot = sns.barplot(data=avg_metrics_melted, x='Metric', y='Average', hue='is_popular', palette="viridis", ax=ax2)

        # Annotate bars with values
        for container in bar_plot.containers:
            ax2.bar_label(container, fmt="%.0f", label_type='edge', fontsize=10, padding=3)

        # Customize plot
        ax2.set_title("Average Likes, Shares, Comments, and Plays by Popularity", fontsize=16)
        ax2.set_xlabel("Metric", fontsize=12)
        ax2.set_ylabel("Average Value", fontsize=12)
        ax2.legend(title="Popularity", loc="upper right")

        # Display the plot
        st.pyplot(fig2)


        # 3. Hashtag Analysis
        st.subheader("3. Hashtag Analysis")
        hashtag_sums = df_with_hashtags[hashtag_features].sum().reset_index()
        hashtag_sums.columns = ['Hashtag', 'Frequency']
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=hashtag_sums, x='Frequency', y='Hashtag', palette="Blues_r", ax=ax3)
        ax3.set_title("Frequency of Hashtags Used", fontsize=16)
        ax3.set_xlabel("Frequency", fontsize=12)
        ax3.set_ylabel("Hashtag", fontsize=12)
        st.pyplot(fig3)

        # 4. Correlation Heatmap
        st.subheader("4. Correlation Heatmap")
        corr = df_with_hashtags[['Likes', 'Shares', 'Comments', 'Plays'] + hashtag_features].corr()
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4, cbar_kws={'label': 'Correlation'})
        ax4.set_title("Feature Correlation Heatmap", fontsize=16)
        st.pyplot(fig4)

        # Dataset summary
        st.subheader("Dataset Summary")
        st.write("Overview of the dataset:")
        st.dataframe(df_with_hashtags.describe())



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