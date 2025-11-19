# Copyright Rotch 2025
# Licence(GPL)
# Author: Rotch
# Demo of Linear Regression using Advertising dataset (without statsmodels)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

PATH = "./Linear/data/advertisement/Advertising.csv"

def main():
  # Load dataset
  data = pd.read_csv(PATH, index_col=0)

  # Visualize relationships
  _, axs = plt.subplots(1, 3, sharey=True, figsize=(15,5))
  axs[0].scatter(data.TV, data.sales)
  axs[0].set_title("TV vs Sales")
  axs[1].scatter(data.radio, data.sales)
  axs[1].set_title("Radio vs Sales")
  axs[2].scatter(data.newspaper, data.sales)
  axs[2].set_title("Newspaper vs Sales")
  plt.show()

  # Simple linear regression using TV as predictor
  X_tv = data[["TV"]].values
  y = data.sales.values
  lm_tv = LinearRegression()
  lm_tv.fit(X_tv, y)
  print("Intercept:", lm_tv.intercept_)
  print("Coefficient:", lm_tv.coef_)

  # plot least squares line
  X_new = np.array([[data.TV.min()], [data.TV.max()]])
  y_pred = lm_tv.predict(X_new)
  plt.scatter(data.TV, data.sales)
  plt.plot(X_new, y_pred, c="red", linewidth=2)
  plt.show()

  # Multiple linear regression using TV, radio, and newspaper
  feature_cols = ["TV", "radio", "newspaper"]
  X_multi = data[feature_cols].values
  lm_multi = LinearRegression()
  lm_multi.fit(X_multi, y)
  print("Intercept:", lm_multi.intercept_)
  print("Coefficients:", lm_multi.coef_)
  for each in zip(feature_cols, lm_multi.coef_):
      print(each)

  # predict for a new observation
  X_new_obs = np.array([[100, 25, 25]])
  print(lm_multi.predict(X_new_obs))

  # calculate R-squared
  print(lm_multi.score(X_multi, y))

  # handling categorical variable with two categories
  np.random.seed(12345)
  nums = np.random.rand(len(data))
  mask_large = nums > 0.5
  data["Size"] = "small"
  data.loc[mask_large, "Size"] = "large"
  data["IsLarge"] = data.Size.map({"small":0, "large":1})

  feature_cols = ["TV", "radio", "newspaper", "IsLarge"]
  X_cat = data[feature_cols].values
  lm_cat = LinearRegression()
  lm_cat.fit(X_cat, y)
  for each in zip(feature_cols, lm_cat.coef_):
      print(each)

  # handling categorical variable with more than two categories
  np.random.seed(123456)
  nums = np.random.rand(len(data))
  mask_suburban = (nums > 0.33) & (nums < 0.66)
  mask_urban = nums > 0.66
  data["Area"] = "rural"
  data.loc[mask_suburban, "Area"] = "suburban"
  data.loc[mask_urban, "Area"] = "urban"
  area_dummies = pd.get_dummies(data.Area, prefix="Area").iloc[:, 1:]
  data = pd.concat([data, area_dummies], axis=1)
  print(data.head())

if __name__ == "__main__":
  main()
