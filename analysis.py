import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the student performance dataset
data = pd.read_csv("student_performance.csv")

# Define the dependent variable
y = data["Academic_Performance"]

# Define the explanatory variables
predictors = [
    "Study_Time",
    "Attendance",
    "Previous_Performance",
    "Learning_Resources",
]

X = data[predictors]

# Add the intercept term (beta_0)
X = sm.add_constant(X)

# Fit Equation (1) using multiple linear regression
model = sm.OLS(y, X).fit()

# Display the regression results
print("MULTIPLE LINEAR REGRESSION RESULTS")
print("=" * 50)
print(model.summary())

# Display confidence intervals for the coefficients
print("\n95% CONFIDENCE INTERVALS")
print("=" * 50)
print(model.conf_int())

# Display model statistics
print("\nMODEL STATISTICS")
print("=" * 50)
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
print(f"F-statistic: {model.fvalue:.4f}")
print(f"F-test p-value: {model.f_pvalue:.6f}")

# Calculate variance inflation factors (VIF)
print("\nVARIANCE INFLATION FACTORS")
print("=" * 50)

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print(vif_data.to_string(index=False))