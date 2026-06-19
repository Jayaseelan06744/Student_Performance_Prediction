
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("dataset.csv")

print("\nFIRST 5 ROWS")
print(df.head())
print("\nDATASET SHAPE")
print(df.shape)
print("\nDATASET INFO")
print(df.info())
print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())

df["Teacher_Quality"] = df["Teacher_Quality"].fillna(
    df["Teacher_Quality"].mode()[0]
)
df["Parental_Education_Level"] = df[
    "Parental_Education_Level"
].fillna(
    df["Parental_Education_Level"].mode()[0]
)
df["Distance_from_Home"] = df[
    "Distance_from_Home"
].fillna(
    df["Distance_from_Home"].mode()[0]
)
print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

df = df.drop_duplicates()
plt.figure(figsize=(8,5))

plt.scatter(
    df["Attendance"],
    df["Exam_Score"]
)
plt.xlabel("Attendance")
plt.ylabel("Exam Score")
plt.title("Attendance vs Exam Score")

plt.savefig(
    "graphs/attendance_vs_score.png"
)
plt.close()

plt.figure(figsize=(8,5))
plt.scatter(
    df["Hours_Studied"],
    df["Exam_Score"]
)
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")
plt.savefig(
    "graphs/hours_vs_score.png"
)
plt.close()
plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("graphs/Correlation Heatmap")

plt.savefig(
    "correlation_heatmap.png"
)

plt.close()

encoder = LabelEncoder()
categorical_columns = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"

]
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
print("\nDATASET AFTER ENCODING")
print(df.head())
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]
print("\nFEATURES SHAPE")
print(X.shape)
print("\nTARGET SHAPE")
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(
X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTRAINING RECORDS")
print(len(X_train))
print("\nTESTING RECORDS")
print(len(X_test))

model = LinearRegression()
model.fit(X_train, y_train)
print("\nMODEL TRAINED SUCCESSFULLY")
predictions = model.predict(X_test)
print("\nFIRST 10 PREDICTIONS")
print(predictions[:10])
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nACTUAL VS PREDICTED")
print(comparison.head(10))
mae = mean_absolute_error(
    y_test,
    predictions
)
mse = mean_squared_error(
    y_test,
    predictions
)
rmse = np.sqrt(mse)
r2 = r2_score(
    y_test,
    predictions
)
print("\nMODEL EVALUATION")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 2))



dt_model = DecisionTreeRegressor(
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

dt_predictions = dt_model.predict(
    X_test
)

dt_mae = mean_absolute_error(
    y_test,
    dt_predictions
)

dt_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        dt_predictions
    )
)

dt_r2 = r2_score(
    y_test,
    dt_predictions
)

print("\nDECISION TREE RESULTS")

print("MAE :", round(dt_mae, 2))
print("RMSE:", round(dt_rmse, 2))
print("R2  :", round(dt_r2, 2))
# ==========================================
# RANDOM FOREST REGRESSOR
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=20,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\nRANDOM FOREST RESULTS")

print("MAE :", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R2  :", round(rf_r2, 2))
comparison_table = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        round(mae, 2),
        round(dt_mae, 2),
        round(rf_mae, 2)
    ],

    "RMSE": [
        round(rmse, 2),
        round(dt_rmse, 2),
        round(rf_rmse, 2)
    ],

    "R2 Score": [
        round(r2, 2),
        round(dt_r2, 2),
        round(rf_r2, 2)
    ]
})

print("\nMODEL COMPARISON")
print(comparison_table)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nFEATURE IMPORTANCE")
importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance.head(10))
import pickle
pickle.dump(
    model,
    open("model.pkl", "wb")
)
print("\nMODEL SAVED AS model.pkl")

sample_student = pd.DataFrame(
    [[
        25,   # Hours_Studied
        90,   # Attendance
        1,    # Parental_Involvement
        1,    # Access_to_Resources
        1,    # Extracurricular_Activities
        8,    # Sleep_Hours
        80,   # Previous_Scores
        2,    # Motivation_Level
        1,    # Internet_Access
        5,    # Tutoring_Sessions
        1,    # Family_Income
        2,    # Teacher_Quality
        1,    # School_Type
        2,    # Peer_Influence
        3,    # Physical_Activity
        0,    # Learning_Disabilities
        2,    # Parental_Education_Level
        1,    # Distance_from_Home
        1     # Gender
    ]],
    columns=X.columns
)
predicted_score = model.predict(sample_student)
print("\nNEW STUDENT PREDICTION")
print(
    "Predicted Exam Score:",
    round(predicted_score[0], 2)
)