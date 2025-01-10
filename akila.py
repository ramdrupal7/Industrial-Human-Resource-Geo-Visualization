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
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locationmode='geojson-id',
            locations=df['StateCode'],  # Ensure this matches your dataset
            z=df['TotalWorkers'],   # Replace with appropriate numeric data
            colorscale='Reds',
            marker_line_color='peachpuff',
            colorbar=dict(title="Total Workers")
        ))
        fig3.update_geos(
            visible=False,
            projection=dict(type='conic conformal', parallels=[12.472944444, 35.172805555556], rotation={'lat': 24, 'lon': 80}),
            lonaxis={'range': [68, 98]},
            lataxis={'range': [6, 38]}
        )
        fig3.update_layout(title="Total Workers Across States", height=550, width=550)
        st.plotly_chart(fig3)

        # New Choropleth Map (Active Cases)
        fig4 = go.Figure(data=go.Choropleth(
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locationmode='geojson-id',
            locations=df['StateCode'],  # Ensure this matches your dataset
            z=df['active_cases'],  # Ensure this column exists in your dataset
            autocolorscale=False,
            colorscale='Reds',
            marker_line_color='peachpuff',
            colorbar=dict(
                title={'text': "Active Cases"},
                thickness=15,
                len=0.35,
                bgcolor='rgba(255,255,255,0.6)',
                tick0=0,
                dtick=20000,
                xanchor='left',
                x=0.01,
                yanchor='bottom',
                y=0.05
            )
        ))
        fig4.update_geos(
            visible=False,
            projection=dict(
                type='conic conformal',
                parallels=[12.472944444, 35.172805555556],
                rotation={'lat': 24, 'lon': 80}
            ),
            lonaxis={'range': [68, 98]},
            lataxis={'range': [6, 38]}
        )
        fig4.update_layout(
            title=dict(
                text="Active COVID-19 Cases in India by State as of July 17, 2020",
                xanchor='center',
                x=0.5,
                yref='paper',
                yanchor='bottom',
                y=1,
                pad={'b': 10}
            ),
            margin={'r': 0, 't': 30, 'l': 0, 'b': 0},
            height=550,
            width=550
        )
        st.plotly_chart(fig4)

        # WordCloud for Industry Terms
        wordcloud = WordCloud(background_color='white').generate(' '.join(df[industry_column].dropna()))
        fig5, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig5)
    else:
        st.write("No valid CSV files found in the ZIP file.")
