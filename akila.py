import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import streamlit as st

# Set the directory containing the datasets
directory = "D:\\workspace\\datascience\\Streamlite\\DataSets"
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

# Load all CSV files and combine them into a single DataFrame
dataframes = [pd.read_csv(file, encoding='latin1') for file in files]
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

# Visualization: Correlation Matrix
fig2, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig2)

# Interactive Choropleth Map
fig3 = go.Figure(data=go.Choropleth(
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locationmode='geojson-id',
    locations=df['State'],  # Ensure this matches your dataset
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

# WordCloud for Industry Terms
wordcloud = WordCloud(background_color='white').generate(' '.join(df[industry_column].dropna()))
fig4, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig4)
