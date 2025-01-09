import pandas as pd
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import streamlit as st
from nltk.corpus import stopwords
import string

# Download stopwords for text processing
nltk.download('stopwords')

# 1. Load and Merge CSV Files
def load_and_merge_data():
    folder_path = os.path.join(os.getcwd(), 'Datasets')
    #folder_path = r'C:\Users\Win10\Downloads\DataSets'
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the directory.")
        return None

    dataframes = []
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(file_path, encoding='latin1')
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        return merged_df
    else:
        print("No DataFrames to merge.")
        return None

# 2. Data Exploration
def explore_data(df):
    print("Data Overview:")
    print(df.head())
    print(df.info())
    print(df.describe())

# 3. Data Cleaning
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Dynamically select relevant columns (assumes some basic industry-related columns)
    relevant_columns = ['NIC Name', 'India/States', 'Main Workers - Total -  Persons']
    available_columns = [col for col in relevant_columns if col in df.columns]
    
    df = df[available_columns]
    
    print("\nMissing Values after Cleaning:")
    print(df.isnull().sum())
    return df

# 4. Text Preprocessing (dynamic handling of stopwords)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Lowercase the text
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# 5. Feature Engineering using TF-IDF
def feature_engineering(df):
    df['cleaned_industry'] = df['NIC Name'].apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_industry'])
    print("TF-IDF Features Shape:", X.shape)
    return X

# 6. Clustering (Unsupervised Learning - KMeans)
def perform_clustering(X, df, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    print("\nCluster Labels:")
    print(df[['NIC Name', 'cluster']].head())
    return df

# 7. Dimensionality Reduction (PCA)
def reduce_dimensions(X):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())
    print("PCA Reduced Features Shape:", X_reduced.shape)
    return X_reduced

# 8. Streamlit Visualization
def visualize_data(df):
    fig = px.bar(df, x="NIC Name", y="Main Workers - Total -  Persons", color="India/States",
                 title="Workers Population by Industry and Geography", labels={"Main Workers - Total -  Persons": "Number of Workers"})
    st.plotly_chart(fig)

# Main function to execute the entire pipeline and build Streamlit dashboard
def main():
    # Step 1: Load and Merge CSV data
    df = load_and_merge_data()

    if df is not None:
        # Step 2: Data Exploration
        explore_data(df)

        # Step 3: Data Cleaning
        df_cleaned = clean_data(df)

        # Step 4: Feature Engineering using TF-IDF
        X = feature_engineering(df_cleaned)

        # Step 5: Perform Clustering
        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=4)  # Dynamic clustering
        df_clustered = perform_clustering(X, df_cleaned, num_clusters=num_clusters)

        # Step 6: Reduce Dimensions using PCA
        X_reduced = reduce_dimensions(X)

        # Step 7: Create Streamlit app for visualization
        st.title('Business Industry Analysis Dashboard')
        visualize_data(df_clustered)

if __name__ == '__main__':
    main()
