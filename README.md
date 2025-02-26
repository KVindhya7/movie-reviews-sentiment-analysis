# movie-reviews-sentiment-analysis
# Sentiment Analysis on Movie Reviews

This project demonstrates Sentiment Analysis on movie reviews using Machine Learning, specifically with a Logistic Regression model. The project includes data preprocessing, model training, evaluation, and deployment as a web application.

## Table of Contents
- [Introduction](#introduction)
- [Sentiment Analysis](#sentiment-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Building the Logistic Regression Model](#building-the-logistic-regression-model)
- [Model Training & Testing](#model-training--testing)
- [Word Cloud Visualization](#word-cloud-visualization)
- [Model Evaluation using Confusion Matrix](#model-evaluation-using-confusion-matrix)
- [Deploying the Model as a Web Application](#deploying-the-model-as-a-web-application)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)

## Introduction
Sentiment analysis is a technique used to determine the emotional tone of a text, typically classified as positive, negative, or neutral. This project applies sentiment analysis to movie reviews.

## Sentiment Analysis
The main goal of sentiment analysis is to extract and classify sentiments from text data. Here, we classify movie reviews as either **positive** or **negative** based on their content.

## Data Preprocessing
To prepare the text data for analysis, the following preprocessing steps are applied:
- Removing special characters and punctuation
- Converting text to lowercase
- Removing stopwords
- Tokenization and stemming/lemmatization
- Converting text into numerical form using **TF-IDF Vectorization**

## Building the Logistic Regression Model
A **Logistic Regression** model is used for classification. The model is trained using labeled movie reviews and is optimized for accuracy.

## Model Training & Testing
- The dataset is split into training and testing sets.
- The Logistic Regression model is trained on the training set.
- Performance is evaluated using accuracy, precision, recall, and F1-score.

## Word Cloud Visualization
A **Word Cloud** is generated to visualize the most frequently occurring words in positive and negative reviews using Python's `wordcloud` library.

## Model Evaluation using Confusion Matrix
A **Confusion Matrix** is used to evaluate model performance by analyzing true positives, false positives, true negatives, and false negatives.

## Deploying the Model as a Web Application
The trained sentiment analysis model is deployed as a web application using **Flask**.

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/sentiment-analysis-movie-reviews.git
   cd sentiment-analysis-movie-reviews
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   python app.py
   ```

## Usage
- Run the web application and enter a movie review.
- The model predicts whether the review is positive or negative.

## Technologies Used
- Python.
- Scikit-learn.
- Flask.
- Pandas, NumPy.
- WordCloud.
- Matplotlib, Seaborn.





