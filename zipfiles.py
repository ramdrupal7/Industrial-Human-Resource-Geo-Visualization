import zipfile
import os
import pandas as pd
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

# 1. Load and Merge Data from ZIP file (Modified for Streamlit file uploader)
def extract_zip(uploaded_file):
    # Create a temporary directory to extract the ZIP file
    temp_dir = "temp_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Navigate into the 'DataSets' folder
    datasets_folder = os.path.join(temp_dir, 'DataSets')
    if not os.path.exists(datasets_folder):
        st.write("DataSets folder not found in the ZIP file.")
        return []

    # Get all CSV files inside the 'DataSets' folder
    csv_files = [os.path.join(datasets_folder, file) for file in os.listdir(datasets_folder) if file.endswith('.csv')]
    
    return csv_files

# Load and merge CSV files extracted from ZIP
def load_and_merge_data():
    # Use Streamlit file uploader for ZIP files
    uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")

    if not uploaded_file:
        st.write("No ZIP file uploaded.")
        return None

    # Extract the ZIP and get the list of CSV files
    csv_files = extract_zip(uploaded_file)
    if not csv_files:
        st.write("No CSV files found inside the 'DataSets' folder.")
        return None

    st.write(f"Extracted {len(csv_files)} CSV files.")

    # Load all CSV files into DataFrames and merge them
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding='latin1')
            dataframes.append(df)
        except Exception as e:
            st.write(f"Error reading {file}: {e}")

    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        st.write("Files merged successfully!")
        return merged_df
    else:
        st.write("No DataFrames to merge.")
        return None

# 2. Data Exploration
def explore_data(df):
    st.write("### Data Overview:")
    st.write(df.head())
    st.write(df.info())
    st.write(df.describe())

# 3. Data Cleaning
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Dynamically select relevant columns (assumes some basic industry-related columns)
    relevant_columns = ['NIC Name', 'India/States', 'Main Workers - Total -  Persons']
    available_columns = [col for col in relevant_columns if col in df.columns]
    
    df = df[available_columns]
    
    st.write("\nMissing Values after Cleaning:")
    st.write(df.isnull().sum())
    return df

# 4. Text Preprocessing
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
    st.write("TF-IDF Features Shape:", X.shape)
    return X

# 6. Clustering
def perform_clustering(X, df, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    st.write("\nCluster Labels:")
    st.write(df[['NIC Name', 'cluster']].head())
    return df

# 7. Dimensionality Reduction (PCA)
def reduce_dimensions(X):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())
    st.write("PCA Reduced Features Shape:", X_reduced.shape)
    return X_reduced

# 8. Streamlit Visualization - State-wise and Overall
def visualize_data_dynamic(df):
    # Dropdown for selecting a specific state
    states = df['India/States'].unique()
    selected_state = st.selectbox("Select a State for Drill-Down Analysis", ["All States"] + list(states))

    # Filter data based on the selected state
    if selected_state != "All States":
        state_data = df[df['India/States'] == selected_state]
        st.write(f"### Data for {selected_state}")
    else:
        state_data = df
        st.write("### Overall Data for All States")

    # Bar Chart Visualization
    fig = px.bar(state_data, x="NIC Name", y="Main Workers - Total -  Persons", color="India/States",
                 title=f"Workers Population by Industry ({selected_state})",
                 labels={"Main Workers - Total -  Persons": "Number of Workers"})
    st.plotly_chart(fig)

# 9. Streamlit PCA Visualization - State-wise and Overall
def visualize_pca_dynamic(X_reduced, df):
    # Dropdown for selecting a specific state
    states = df['India/States'].unique()
    selected_state = st.selectbox("Select a State for PCA Analysis", ["All States"] + list(states))

    # Filter data based on the selected state
    if selected_state != "All States":
        state_data = df[df['India/States'] == selected_state]
        X_reduced = X_reduced[df['India/States'] == selected_state]
        st.write(f"### PCA Analysis for {selected_state}")
    else:
        state_data = df
        st.write("### Overall PCA Analysis for All States")

    # PCA Visualization
    pca_df = pd.DataFrame(X_reduced, columns=["PC1", "PC2"])
    pca_df['NIC Name'] = state_data['NIC Name']
    pca_df['India/States'] = state_data['India/States']

    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="India/States",
                         title=f"PCA Visualization ({selected_state})")
    st.plotly_chart(fig_pca)

# Main function
def main():
    # Step 1: Load and Merge CSV data
    df = load_and_merge_data()

    if df is not None:
        # Step 2: Data Exploration
        st.write("### Data Exploration")
        explore_data(df)

        # Step 3: Data Cleaning
        st.write("### Data Cleaning")
        df_cleaned = clean_data(df)

        # Step 4: Feature Engineering using TF-IDF
        st.write("### Feature Engineering")
        X = feature_engineering(df_cleaned)

        # Step 5: Perform Clustering
        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=4)  # Dynamic clustering
        df_clustered = perform_clustering(X, df_cleaned, num_clusters=num_clusters)

        # Step 6: Reduce Dimensions using PCA
        X_reduced = reduce_dimensions(X)

        # Step 7: Create Streamlit app for visualization
        st.title('Business Industry Analysis Dashboard')
        st.write("### Industry Cluster Visualization")
        visualize_data_dynamic(df_clustered)
        visualize_pca_dynamic(X_reduced, df_clustered)

if __name__ == '__main__':
    main()
