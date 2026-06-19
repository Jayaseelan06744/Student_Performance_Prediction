# Student Performance Prediction using Machine Learning

## Project Overview

This project predicts student exam scores using Machine Learning techniques. The objective is to analyze various academic, personal, and environmental factors that influence student performance and build predictive models to estimate exam scores.

## Dataset

Dataset: Student Performance Factors Dataset (Kaggle)

The dataset contains information about students such as study habits, attendance, family background, learning environment, and exam scores.

## Features Used

* Hours Studied
* Attendance
* Parental Involvement
* Access to Resources
* Extracurricular Activities
* Sleep Hours
* Previous Scores
* Motivation Level
* Internet Access
* Tutoring Sessions
* Family Income
* Teacher Quality
* School Type
* Peer Influence
* Physical Activity
* Learning Disabilities
* Parental Education Level
* Distance from Home
* Gender

### Target Variable

* Exam_Score

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Missing Value Handling
4. Exploratory Data Analysis (EDA)
5. Data Preprocessing
6. Label Encoding
7. Feature Selection
8. Train-Test Split
9. Model Training
10. Model Evaluation
11. Model Comparison
12. Model Saving

## Machine Learning Models Used

### Linear Regression

* MAE: 1.02
* RMSE: 2.10
* R² Score: 0.69

### Decision Tree Regressor

* MAE: 1.73
* RMSE: 3.30
* R² Score: 0.23

### Random Forest Regressor

* MAE: 1.20
* RMSE: 2.29
* R² Score: 0.63

## Best Model

Linear Regression achieved the best performance with an R² Score of 0.69 and was selected as the final model.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Pickle

## Project Structure

Student_Performance_Prediction/

├── dataset.csv

├── main.py

├── model.pkl

├── requirements.txt

├── README.md

└── graphs/

    ├── attendance_vs_score.png

    ├── hours_vs_score.png

    └── correlation_heatmap.png

## Future Improvements

* Flask Web Application
* Streamlit Dashboard
* Hyperparameter Tuning
* Model Deployment
* Feature Engineering
* Cross Validation

## Author

Jayaseelan

Computer Science Engineering Student
