# Copyright Rotch 2025
# Licence(GPL)
# Author: Rotch
# Demo of normal n-D Linear Regression implementation

import numpy as np
import matplotlib.pyplot as plt

PATH = "./Linear/data/implementation/ex0.txt"

# Load dataset from file
def load_dataset(file_name):
  num_feat = len(open(file_name).readline().split("\t")) - 1
  x_arr = []
  y_arr = []
  fr = open(file_name)
  for line in fr.readlines():
    cur_line = line.strip().split("\t")
    line_arr = [float(cur_line[i]) for i in range(num_feat)]
    x_arr.append(line_arr)
    y_arr.append(float(cur_line[-1]))
  return np.array(x_arr), np.array(y_arr).reshape(-1, 1)

# Standard Linear Regression
def standard_regression(x_mat, y_mat):
  x_tx = x_mat.T.dot(x_mat)
  if np.linalg.det(x_tx) == 0.0:
    raise ValueError("Matrix is singular")
  ws = np.linalg.inv(x_tx).dot(x_mat.T).dot(y_mat)
  return ws

# Visualize standard linear regression
def show_linear_regression(x_mat, y_mat, ws):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(x_mat[:, 1], y_mat.flatten(), s=20, c="green", alpha=0.5)
  x_copy = np.copy(x_mat)
  x_copy = x_copy[x_copy[:, 1].argsort()]
  y_hat = x_copy.dot(ws)
  ax.plot(x_copy[:, 1], y_hat, c="red")
  plt.title("Standard Linear Regression")
  plt.xlabel("x1")
  plt.ylabel("y")
  plt.show()

# Locally Weighted Linear Regression
def lwlr(test_point, x_mat, y_mat, k=1.0):
  m = x_mat.shape[0]
  weights = np.eye(m)
  for i in range(m):
    diff_mat = test_point - x_mat[i, :]
    weights[i, i] = np.exp(-(diff_mat.dot(diff_mat.T)) / (2.0 * k * k))
  x_tx = x_mat.T.dot(weights).dot(x_mat)
  if np.linalg.det(x_tx) == 0.0:
    raise ValueError("Matrix is singular")
  ws = np.linalg.inv(x_tx).dot(x_mat.T).dot(weights).dot(y_mat)
  return test_point.dot(ws)

# Test LWLR
def lwlr_test(test_mat, x_mat, y_mat, k=1.0):
  m = test_mat.shape[0]
  y_hat = np.zeros(m)
  for i in range(m):
    y_hat[i] = lwlr(test_mat[i], x_mat, y_mat, k).item()
  return y_hat

# Plot LWLR results
def plot_lwlr(x_mat, y_mat):
  y_hat_1 = lwlr_test(x_mat, x_mat, y_mat, 1.0)
  y_hat_2 = lwlr_test(x_mat, x_mat, y_mat, 0.01)
  y_hat_3 = lwlr_test(x_mat, x_mat, y_mat, 0.003)
  srt_ind = x_mat[:, 1].argsort()
  x_sort = x_mat[srt_ind]
  fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
  axs[0].plot(x_sort[:, 1], y_hat_1[srt_ind], c="red")
  axs[1].plot(x_sort[:, 1], y_hat_2[srt_ind], c="red")
  axs[2].plot(x_sort[:, 1], y_hat_3[srt_ind], c="red")
  for ax in axs:
    ax.scatter(x_mat[:, 1], y_mat.flatten(), s=20, c="green", alpha=0.5)
  axs[0].set_title("LWLR k=1.0")
  axs[1].set_title("LWLR k=0.01")
  axs[2].set_title("LWLR k=0.003")
  plt.xlabel("X")
  plt.show()

if __name__ == "__main__":
  x_mat, y_mat = load_dataset(PATH)
  ws = standard_regression(x_mat, y_mat)
  show_linear_regression(x_mat, y_mat, ws)
  plot_lwlr(x_mat, y_mat)
