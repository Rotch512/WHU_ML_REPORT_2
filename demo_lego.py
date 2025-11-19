# Copyright Rotch 2025
# Licence(GPL)
# Author: Rotch
# Demo of Linear Regression on Lego price dataset

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import Ridge

def load_dataset(file_path):
  num_features = len(open(file_path).readline().split("\t")) - 1
  x_data = []
  y_data = []
  with open(file_path) as f:
    for line in f:
      parts = line.strip().split("\t")
      x_data.append([float(parts[i]) for i in range(num_features)])
      y_data.append(float(parts[-1]))
  return np.array(x_data), np.array(y_data)

def standardize(x_data, y_data):
  x_copy = x_data.copy()
  y_copy = y_data.copy()
  y_mean = np.mean(y_copy, axis=0)
  y_copy -= y_mean
  x_mean = np.mean(x_copy, axis=0)
  x_var = np.var(x_copy, axis=0)
  x_copy = (x_copy - x_mean) / x_var
  return x_copy, y_copy, x_mean, x_var, y_mean

def rss_error(y_true, y_pred):
  return np.sum((y_true - y_pred) ** 2)

def standard_regression(x_data, y_data):
  x_aug = np.hstack([np.ones((x_data.shape[0], 1)), x_data])
  xtx = x_aug.T @ x_aug
  if np.linalg.det(xtx) == 0.0:
    print("Matrix is singular, cannot invert")
    return
  ws = np.linalg.inv(xtx) @ (x_aug.T @ y_data.reshape(-1,1))
  return ws

def ridge_regression(x_data, y_data, lam=0.2):
  x_aug = np.hstack([np.ones((x_data.shape[0], 1)), x_data])
  xtx = x_aug.T @ x_aug
  denom = xtx + np.eye(xtx.shape[0]) * lam
  if np.linalg.det(denom) == 0.0:
    print("Matrix is singular, cannot invert")
    return
  ws = np.linalg.inv(denom) @ (x_aug.T @ y_data.reshape(-1,1))
  return ws

def ridge_test(x_data, y_data):
  x_std, y_std, _, _, _ = standardize(x_data, y_data)
  num_tests = 30
  w_mat = np.zeros((num_tests, x_std.shape[1] + 1))
  for i in range(num_tests):
    lam = np.exp(i - 10)
    ws = ridge_regression(x_std, y_std, lam)
    w_mat[i, :] = ws.flatten()
  return w_mat

def cross_validation(x_train, y_train, num_val=10):
  m = len(y_train)
  index_list = list(range(m))
  error_mat = np.zeros((num_val, 30))
  for i in range(num_val):
    random.shuffle(index_list)
    split = int(m * 0.9)
    train_idx, test_idx = index_list[:split], index_list[split:]
    x_tr, y_tr = x_train[train_idx], y_train[train_idx]
    x_te, y_te = x_train[test_idx], y_train[test_idx]
    w_mat = ridge_test(x_tr, y_tr)
    mean_train = np.mean(x_tr, axis=0)
    var_train = np.var(x_tr, axis=0)
    x_te_std = (x_te - mean_train) / var_train
    x_te_aug = np.hstack([np.ones((x_te_std.shape[0],1)), x_te_std])
    for k in range(30):
      y_pred = x_te_aug @ w_mat[k,:].reshape(-1,1)
      error_mat[i, k] = rss_error(y_te.reshape(-1,1), y_pred)
  mean_errors = np.mean(error_mat, axis=0)
  best_idx = np.argmin(mean_errors)
  best_weights = w_mat[best_idx,:]
  mean_x = np.mean(x_train, axis=0)
  var_x = np.var(x_train, axis=0)
  unreg = best_weights / np.hstack([1, var_x])
  print("Best ridge regression weights (unstandardized):", unreg)

def use_sklearn(x_data, y_data):
  reg = Ridge(alpha=0.5)
  reg.fit(x_data, y_data)
  print("Intercept: %f, Coefficients: %s" % (reg.intercept_, reg.coef_))

if __name__ == "__main__":
  lego_x, lego_y = load_dataset("./Linear/data/lego/lego.txt")
  ws_std = standard_regression(lego_x, lego_y)
  print("Standard regression coefficients:", ws_std.flatten())
  ridge_w_mat = ridge_test(lego_x, lego_y)
  print("Ridge regression coefficient matrix:\n", ridge_w_mat)
  cross_validation(lego_x, lego_y)
  use_sklearn(lego_x, lego_y)
