# EECS 731 - Project 6 (Anomaly Detection)

## Project Objectives
1.  Set up a data science project structure in a new git repository in your GitHub account
2.  Download the benchmark data set from
https://www.kaggle.com/boltzmannbrain/nab  or
https://github.com/numenta/NAB/tree/master/data
3.  Load one of the data sets into panda data frames
4.  Formulate one or two ideas on how feature engineering would help the data set to establish additional value using exploratory data analysis
5.  Build one or more anomaly detection models to determine the anomalies using the other columns as features
6.  Document your process and results
7.  Commit your notebook, source code, visualizations and other supporting files to the git repository in GitHub

## Summary
The purpose of this project was to build anomaly detection models for a chosen dataset. We used a dataset that tracked the Tweet quantity about the Facebook (FB) ticker symbol over time. We used our original feature to generate 5 additional features. These features were designed to give each data point value "context" with respect to the other data points' values and value averages. After extensive feature engineering, we chose to test 3 models on our engineered dataset. Since anomaly detection is naturally unsupervised, we simply had to compare the outlier/anomaly percentage for each of the models used. We had to tweak parameters to reach reasonable percentages with the models. Given more time, we would explore the use of a GAT (Generative Adversarial Network). See the full report [here.](../notebooks/ticker_volume_FB.md)
