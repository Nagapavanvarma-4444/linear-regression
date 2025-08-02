import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'square_feet': [1100, 1350, 1600, 1800, 2000, 2250, 2500, 2750, 3000, 3200],
    'bedrooms':     [2, 3, 3, 3, 4, 4, 4, 5, 5, 6],
    'bathrooms':    [1, 2, 2, 2, 3, 3, 3, 4, 4, 5],
    'price':        [30, 42, 50, 55, 65, 72, 80, 95, 105, 120]
}
df = pd.DataFrame(data)
X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
new_house = [[2400, 4, 3]]
predicted_price = model.predict(new_house)
print("\nPredicted Price (in lakhs ₹) for new house:", predicted_price[0])
