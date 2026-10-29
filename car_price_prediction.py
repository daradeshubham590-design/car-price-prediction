import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create Car Dataset
data = {
    "Year": [2018, 2019, 2020, 2021, 2022, 2017, 2023, 2020, 2019, 2022],
    "Kilometers_Driven": [50000, 40000, 30000, 25000, 15000, 70000, 10000, 35000, 45000, 20000],
    "Fuel_Type": [1, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    "Price_Lakhs": [5.0, 6.0, 7.5, 8.0, 10.0, 4.0, 12.0, 7.0, 6.5, 9.0]
}

df = pd.DataFrame(data)

print("Car Dataset:")
print(df)

# Select Features and Target
X = df[["Year", "Kilometers_Driven", "Fuel_Type"]]
y = df["Price_Lakhs"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Machine Learning Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Model
predictions = model.predict(X_test)

print("\nModel Training Completed!")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2), "lakhs")
print("R2 Score:", round(r2_score(y_test, predictions), 2))

# Predict Price of a New Car
new_car = pd.DataFrame({
    "Year": [2022],
    "Kilometers_Driven": [20000],
    "Fuel_Type": [1]
})

predicted_price = model.predict(new_car)

print("\nCar Year: 2022")
print("Kilometers Driven: 20000")
print("Fuel Type: Petrol")
print("Predicted Car Price:", round(predicted_price[0], 2), "Lakhs")
