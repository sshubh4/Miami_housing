import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

Load data
df = pd.read_csv("miami-housing.csv")
df = df.drop(columns=["PARCELNO"])
df = df.dropna()

Define features and target
X = df.drop("SALE_PRC", axis=1)
y = df["SALE_PRC"]

Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

--- LINEAR REGRESSION ---
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

--- DECISION TREE REGRESSION ---
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

--- RANDOM FOREST REGRESSION ---
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

--- EVALUATE MODELS ---
def evaluate(name, y_true, y_pred):
    print(f"\n{name} Results")
    print("R² Score:", r2_score(y_true, y_pred))
    print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))

evaluate("Linear Regression", y_test, pred_lr)
evaluate("Decision Tree", y_test, pred_dt)
evaluate("Random Forest", y_test, pred_rf)

--- VISUALIZATION: Actual vs Predicted (Random Forest) ---
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=pred_rf)
plt.title("Actual vs Predicted Sale Price (Random Forest)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True)
plt.show()

Presentation
importances = rf.featureimportances
feature_names = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()