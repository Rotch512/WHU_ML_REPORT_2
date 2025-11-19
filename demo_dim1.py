# Copyright Rotch 2025
# Licence(GPL)
# Author: Rotch
# Demo of 1-D Linear Regression implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data from CSV
def get_data(file_name, x_col, y_col):
  df = pd.read_csv(file_name)
  x = df.iloc[:, x_col].values.reshape(-1, 1)
  y = df.iloc[:, y_col].values.reshape(-1, 1)
  return x, y

# Create and train linear model
def get_linear_model(x, y):
  model = LinearRegression()
  model.fit(x, y)
  return model

# Visualize
def show_linear_lines(x1, y1, model1, x2, y2, model2,
  title1, xlabel1, ylabel1, title2, xlabel2, ylabel2):
  plt.figure(figsize=(12, 5))

  # First subplot
  plt.subplot(1, 2, 1)
  plt.title(title1)
  plt.xlabel(xlabel1)
  plt.ylabel(ylabel1)
  plt.grid(True)
  plt.plot(x1, y1, "k.")
  x_line1 = np.linspace(min(x1), max(x1), 100).reshape(-1, 1)
  y_line1 = model1.predict(x_line1)
  plt.plot(x_line1, y_line1, "g-")

  # Second subplot
  plt.subplot(1, 2, 2)
  plt.title(title2)
  plt.xlabel(xlabel2)
  plt.ylabel(ylabel2)
  plt.grid(True)
  plt.plot(x2, y2, "k.")
  x_line2 = np.linspace(min(x2), max(x2), 100).reshape(-1, 1)
  y_line2 = model2.predict(x_line2)
  plt.plot(x_line2, y_line2, "g-")

  plt.tight_layout()
  plt.show()

# Two demo implementations
def main():
  # House Price Prediction
  PATH_PRICE = "./Linear/data/implementation/price_info.csv"
  x_price, y_price = get_data(PATH_PRICE, 1, 2)
  model_price = get_linear_model(x_price, y_price)
  predicted_price = model_price.predict(np.array([[700]]))[0]
  print("Predicted price for 700 square feet:", predicted_price)

  # Vehicle MPG Prediction
  PATH_MPG = "./Linear/data/implementation/auto_mpg.csv"
  x_mpg, y_mpg = get_data(PATH_MPG, 4, 0)
  model_mpg = get_linear_model(x_mpg, y_mpg)
  predicted_mpg = model_mpg.predict(np.array([[3000]]))[0]
  print("Predicted MPG for 3000 weight:", predicted_mpg)

  # Show plots
  show_linear_lines(
    x_price, y_price, model_price, x_mpg, y_mpg, model_mpg,
    title1="House Price Prediction", xlabel1="Square Feet", ylabel1="Price",
    title2="Vehicle MPG Prediction", xlabel2="Weight", ylabel2="MPG"
  )

if __name__ == "__main__":
  main()
