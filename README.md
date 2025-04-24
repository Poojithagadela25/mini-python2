# K-Drama Data Analysis

This project explores trends and patterns in a dataset of Korean dramas (K-Dramas), using Python for data cleaning, visualization, and preliminary analysis. The focus is on understanding how variables like rating, number of episodes, and release year interact.

#  Project Overview

K-Dramas have seen a massive surge in global popularity. This analysis aims to:

Understand distribution and trends in ratings.

Explore relationships between episode count, release year, and audience rating.

Provide visual insights into the dataset for better understanding and further research.

#  Dataset

Filename: kdrama_DATASET.csv

Contents: The dataset includes details such as:

Title

Genre

Number of Episodes

Rating

Year of Release

Other categorical and numerical features

# Tools & Libraries

Python 3

pandas – Data manipulation

numpy – Numerical operations

matplotlib – Plotting library

seaborn – Statistical data visualization

# How to Run

Clone the repository or download the notebook.

Place kdrama_DATASET.csv in the same directory.

Run the main.ipynb Jupyter Notebook in any Python environment (e.g., Jupyter Lab, VS Code, Google Colab).

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn

# Analysis Performed
Data Cleaning:

Trimmed column names.

Dropped rows with missing values in key fields (Rating, Number of Episodes, Year of release).

Descriptive Statistics:

Summary of numeric features to assess central tendencies and variation.

Correlation Analysis:

A heatmap was generated to visualize the correlations between numerical features, such as Rating, Number of Episodes, and Year of release.

# Visualizations:

Box Plot: Shows how ratings vary with the number of episodes.

Scatter Plot: Illustrates rating trends over the years, factoring in episode count.

Heatmap: Correlation matrix showing relationships between key numeric features

# Future Work
Sentiment analysis on user reviews.

Genre-based rating comparison.

Predictive modeling for K-Drama ratings.

------##-------



