# Copyright Rotch 2025
# Licence(GPL)
# Author: Rotch
# Demo of Logistic Regression on Horse Colic dataset

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

# Paths of datasets
TRAIN_PATH = "./Linear/data/lr_horse/horseColicTraining.txt"
TEST_PATH = "./Linear/data/lr_horse/horseColicTest.txt"

# Stochastic Gradient Ascent
def stoc_grad_ascent(data_matrix, class_labels, num_iter=150):
  m, n = np.shape(data_matrix)
  weights = np.ones(n)

  # Iteration
  for j in range(num_iter):
    data_index = list(range(m))

    # Update weights
    for i in range(m):
      alpha = 4 / (1.0 + j + i) + 0.01
      rand_index = int(np.random.uniform(0, len(data_index)))
      h = expit(np.sum(data_matrix[rand_index] * weights))
      error = class_labels[rand_index] - h
      weights = weights + alpha * error * data_matrix[rand_index]
      del(data_index[rand_index])

  return weights

# Classify vector
def classify_vector(in_x, weights):
  prob = expit(np.sum(in_x * weights))
  return 1.0 if prob > 0.5 else 0.0

# Logistic Regression using Stochastic Gradient Ascent
def colic_test():
  fr_train = open(TRAIN_PATH)
  fr_test = open(TEST_PATH)

  # Load training data
  training_set = []
  training_labels = []
  for line in fr_train.readlines():
    curr_line = line.strip().split("\t")
    line_arr = [float(curr_line[i]) for i in range(len(curr_line)-1)]
    training_set.append(line_arr)
    training_labels.append(float(curr_line[-1]))

  # Train weights
  train_weights = stoc_grad_ascent(np.array(training_set), training_labels, 500)

  # Test error rate
  error_count = 0
  num_test_vec = 0.0
  for line in fr_test.readlines():
    num_test_vec += 1.0
    curr_line = line.strip().split("\t")
    line_arr = [float(curr_line[i]) for i in range(len(curr_line)-1)]
    if int(classify_vector(np.array(line_arr), train_weights)) != int(curr_line[-1]):
      error_count += 1
  error_rate = (float(error_count) / num_test_vec) * 100
  print("Test error rate: %.2f%%" % error_rate)

# Logistic Regression using sklearn
def colic_sklearn():
  fr_train = open(TRAIN_PATH)
  fr_test = open(TEST_PATH)

  # Load training data
  training_set = []
  training_labels = []
  test_set = []
  test_labels = []
  for line in fr_train.readlines():
    curr_line = line.strip().split("\t")
    line_arr = [float(curr_line[i]) for i in range(len(curr_line)-1)]
    training_set.append(line_arr)
    training_labels.append(float(curr_line[-1]))
  for line in fr_test.readlines():
    curr_line = line.strip().split("\t")
    line_arr = [float(curr_line[i]) for i in range(len(curr_line)-1)]
    test_set.append(line_arr)
    test_labels.append(float(curr_line[-1]))

  # Train and test using sklearn Logistic Regression
  clf = LogisticRegression(max_iter=5000, solver="liblinear")
  clf.fit(training_set, training_labels)
  test_accuracy = clf.score(test_set, test_labels) * 100
  print("Accuracy: %f%%" % test_accuracy)

if __name__ == "__main__":
  print("Stochastic Gradient Ascent:")
  colic_test()
  print("Sklearn Logistic Regression:")
  colic_sklearn()
