import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([500, 600, 700, 800, 900, 1000]).reshape(-1, 1)  # Reshape for a single feature
y = np.array([150000, 180000, 210000, 240000, 270000, 300000])

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (c): {model.intercept_}")

plt.scatter(x, y, color='blue')   
plt.plot(x, y_pred, color='red')
plt.title("House Size vs Price")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price (USD)")
plt.show()
