# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: Import the required libraries
(numpy, pandas, sklearn).

Step 3: Load the employee dataset containing attributes such as:

Years of experience

Education level

Job role

Performance rating

Salary (target variable)

Step 4: Separate the dataset into:

Independent variables (X) – employee features

Dependent variable (Y) – salary

Step 5: Perform data preprocessing:

Handle missing values

Encode categorical variables (if any)

Step 6: Split the dataset into:

Training dataset

Testing dataset

Step 7: Initialize the Decision Tree Regressor model with suitable parameters
(e.g., criterion = “squared_error”).

Step 8: Train the Decision Tree Regressor using the training dataset.

Step 9: Use the trained model to predict employee salaries for the test dataset.

Step 10: Evaluate the model using regression metrics such as:

Mean Squared Error (MSE)

R² score

Step 11: Display the predicted salary values.

Step 12: Stop the program.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: 
RegisterNumber:  
*/
```
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant',
                 'Manager', 'Country Manager', 'Region Manager',
                 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Split features and target
# ------------------------------
X = df[['Level']]     # Feature (Level)
y = df['Salary']      # Target (Salary)

# ------------------------------
# Step 3: Create Decision Tree Regressor
# ------------------------------
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

# ------------------------------
# Step 4: Predict salary for the dataset or new levels
# ------------------------------
y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)

# Example: predict salary for a new employee at level 6.5
level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")

# ------------------------------
# Step 5: Visualize the results (High-resolution curve)
# ------------------------------
X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01)  # High-resolution for smoother curve
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="1808" height="208" alt="image" src="https://github.com/user-attachments/assets/80624402-9542-4e28-b3c3-f1db7c416701" />
<img width="714" height="576" alt="image" src="https://github.com/user-attachments/assets/708533ec-172b-4ee0-9938-f8f5d7259877" />






## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
