import numpy as np
import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import streamlit as st

# Define the extract_zip function
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

# Streamlit file uploader for ZIP files
uploaded_file = st.file_uploader("Upload a ZIP file containing datasets", type="zip")
if uploaded_file is not None:
    csv_files = extract_zip(uploaded_file)
    if csv_files:
        # Load all extracted CSV files and combine them into a single DataFrame
        dataframes = [pd.read_csv(file, encoding='latin1') for file in csv_files]
        df = pd.concat(dataframes, ignore_index=True)

        # Save combined data to a CSV
        df.to_csv("all_combined.csv", index=False)

        # Data Cleaning
        columns_to_clean = ['State Code', 'District Code', 'Division', 'Group', 'Class']
        for col in columns_to_clean:
            df[col] = df[col].astype(str).str.replace('`', '').astype(int)

        # Remove special characters from column names
        df.columns = [''.join(c for c in col if c.isalnum() or c in ('_', '.')) for col in df.columns]

        print(df.columns)

        # Feature Engineering
        df['TotalWorkers'] = df['MainWorkersTotalPersons'] + df['MarginalWorkersTotalPersons']
        df['MaleFemaleRatio'] = df['MainWorkersTotalMales'] / df['MainWorkersTotalFemales'].replace(0, 1)

        # NLP Analysis for Industries
        industry_column = 'NICName'  # Assuming this column contains industry info
        tfidf = TfidfVectorizer(stop_words='english')
        industry_tfidf = tfidf.fit_transform(df[industry_column].fillna(''))

        # Cluster industries
        kmeans = KMeans(n_clusters=5, random_state=42)
        df['IndustryCluster'] = kmeans.fit_predict(industry_tfidf)

        # Visualization: Worker Distribution by Industries
        fig1 = px.bar(df, x='IndustryCluster', y='TotalWorkers', color='IndustryCluster',
                      title="Worker Population Distribution Across Industries")
        st.plotly_chart(fig1)

        # Data Cleaning for Numeric Columns and Correlation Matrix
        # Filter out non-numeric columns for correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])

        # Check for missing values and handle them
        numeric_df = numeric_df.fillna(0)  # or you can use numeric_df.dropna() if you want to drop rows with NaN values

        # Compute and plot the correlation matrix
        fig2, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig2)

        # Interactive Choropleth Map (Total Workers)
        fig3 = go.Figure(data=go.Choropleth(
            geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/indiana-counties.geojson",  # Change this to match your geography
            featureidkey="properties.name",  # Adjust based on your geojson file
            locations=df['StateCode'],  # Ensure this matches your dataset's state codes
            z=df['TotalWorkers'],   # Replace with appropriate numeric data
            colorscale='Reds',
            marker_line_color='peachpuff',
            colorbar=dict(title="Total Workers")
        ))

        fig3.update_geos(
            visible=False,
            projection=dict(type='albers usa'),  # Adjust for your country
            lonaxis={'range': [-180, 180]},
            lataxis={'range': [-90, 90]}
        )
        fig3.update_layout(title="Total Workers Across States", height=550, width=550)
        st.plotly_chart(fig3)

        # Word Cloud for Industries
        wordcloud = WordCloud(width=800, height=400).generate(' '.join(df['NICName'].fillna('')))
        st.image(wordcloud.to_array(), use_container_width=True)

        # Additional Cluster Visualization with PCA for better understanding of clustering
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(industry_tfidf.toarray())
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]

        fig_pca = px.scatter(df, x='PCA1', y='PCA2', color='IndustryCluster', title="PCA Visualization of Industry Clusters")
        st.plotly_chart(fig_pca)

        # Create a scatter plot for Male-Female Ratio vs Total Workers
        fig_male_female = px.scatter(df, x='MaleFemaleRatio', y='TotalWorkers', color='IndustryCluster',
                                      title="Male-Female Ratio vs Total Workers")
        st.plotly_chart(fig_male_female)

        # Show the cleaned dataset
        st.write("Cleaned Dataset:")
        st.write(df.head())
