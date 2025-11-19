# Copyright Rotch 2025
# Licence(GPL)
# Author: Rotch
# Demo of Linear Regression on Boston Housing dataset

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from scipy.stats import probplot
import matplotlib.pyplot as plt

def main():
  # Dataset
  boston = fetch_openml(name="boston", version=1, as_frame=True)
  x = boston.data.to_numpy().astype(np.float64)
  y = boston.target.to_numpy().astype(np.float64)

  # Linear Regression model
  lr = LinearRegression()

  # Fit the model and make predictions
  lr.fit(x, y)
  predictions = lr.predict(x)

  # Figures
  _, axs = plt.subplots(1, 3, figsize=(21, 5))

  # 1. Residual Histogram
  axs[0].hist(y - predictions, bins=40, color="b", alpha=0.5, label="Residuals Linear")
  axs[0].set_title("Histogram of Residuals")
  axs[0].legend()

  # 2. Q-Q Plot
  probplot(y - predictions, plot=axs[1])
  axs[1].set_title("Q-Q plot of Residuals")

  # 3. Bootstrap Confidence Interval for Coefficient
  n_bootstraps = 1000
  len_boston = len(y)
  subsample_size = int(0.5 * len_boston)
  subsample = lambda: np.random.choice(np.arange(len_boston), size=subsample_size)
  coefs = np.ones(n_bootstraps)

  for i in range(n_bootstraps):
    subsample_idx = subsample()
    subsample_x = x[subsample_idx]
    subsample_y = y[subsample_idx]
    lr.fit(subsample_x, subsample_y)
    coefs[i] = lr.coef_[0]

  axs[2].hist(coefs, bins=50, color="b", alpha=0.5)
  axs[2].set_title("Histogram of the lr.coef_[0]")

  plt.tight_layout()
  plt.show()

  # Performance Metrics
  print("MSE:", mean_squared_error(y, predictions))
  print("MAD:", mean_absolute_error(y, predictions))
  print("Residual mean:", np.mean(y - predictions))

  # 95% confidence interval for coefficient
  ci = np.percentile(coefs, [2.5, 97.5])
  print("95% confidence interval for coef_[0]:", ci)

if __name__ == "__main__":
  main()
