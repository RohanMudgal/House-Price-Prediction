# House-Price-Prediction
Use a dataset that includes information about housing prices and features like square footage, number of bedrooms, etc. to train a model that can predict the price of a new house


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv('Kc_Price.csv')

# Drop the 'id' and 'date' columns as they are not relevant for prediction
data.drop(columns=['id', 'date'], inplace=True)

# Separate the target variable and features
X = data.drop(columns=['price'])
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the performance of the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# Assuming you have a new dataset named X_new, you can scale and make predictions for it
# Scale the new dataset using the same scaler used for training data
X_new_scaled = scaler.transform(X_new)

# Predict target values for the new dataset
y_pred_new = model.predict(X_new_scaled)

# Print the predicted house prices for X_new
print("Predicted House Prices for X_new:", y_pred_new)
