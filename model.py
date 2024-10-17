# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create sample data (e.g., house sizes and prices)
# x represents the independent variable (house size), y represents the dependent variable (price)
x = np.array([500, 600, 700, 800, 900, 1000]).reshape(-1, 1)  # Reshape for a single feature
y = np.array([150000, 180000, 210000, 240000, 270000, 300000])

# Step 2: Create a LinearRegression model and fit it to the data
model = LinearRegression()
model.fit(x, y)

# Step 3: Predict values (optional)
y_pred = model.predict(x)

# Step 4: Print the slope (m) and intercept (c)
print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (c): {model.intercept_}")

# Step 5: Plot the results
plt.scatter(x, y, color='blue')   # Plot the data points
plt.plot(x, y_pred, color='red')  # Plot the regression line
plt.title("House Size vs Price")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price (USD)")
plt.show()
