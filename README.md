# Business Industry Analysis Dashboard

This is a Streamlit-based Business Industry Analysis Dashboard that processes and visualizes human resource data from various CSV files. The app leverages machine learning techniques, such as TF-IDF vectorization and KMeans clustering, to analyze and cluster industries, as well as dimensionality reduction using PCA for visualization. The dashboard also includes interactive data visualization using Plotly and a Streamlit interface.

## Features

- **Data Merging**: Combines multiple CSV files containing industry, geography, and workers' population data.
- **Data Cleaning**: Cleans and preprocesses the data by removing duplicates and handling missing values.
- **Feature Engineering**: Uses TF-IDF vectorization to convert industry names into numerical features for clustering.
- **Clustering**: Applies KMeans clustering to group industries based on similarities in their text data.
- **Dimensionality Reduction**: Reduces the dimensionality of feature data using PCA for visualization.
- **Visualization**: Interactive bar charts of workers' population across industries and geographical locations using Plotly.

## Requirements

- Python 3.x
- Streamlit
- pandas
- scikit-learn
- nltk
- plotly

To install the necessary libraries, you can run:

```bash
pip install -r requirements.txt
