# Copyright Rotch 2025
# Licence(GPL)
# Author: Rotch
# Demo of Abalone age prediction

import numpy as np

PATH = "./Linear/data/abalone/abalone.txt"

def loadDataSet(fileName):
  numFeat = len(open(fileName).readline().split("\t")) - 1
  dataMat = []
  labelMat = []
  fr = open(fileName)
  for line in fr.readlines():
    lineArr = line.strip().split("\t")
    dataMat.append([float(num) for num in lineArr[0:numFeat]])
    labelMat.append(float(lineArr[-1]))
  return np.array(dataMat), np.array(labelMat)

def standRegres(xArr, yArr):
  xMat = np.array(xArr)
  yMat = np.array(yArr).reshape(-1, 1)
  xTx = xMat.T @ xMat
  if np.linalg.det(xTx) == 0.0:
    raise ValueError("This matrix is singular, cannot do inverse")
  ws = np.linalg.inv(xTx) @ xMat.T @ yMat
  return ws

def lwlr(testPoint, xArr, yArr, k=1.0):
  xMat = np.array(xArr)
  yMat = np.array(yArr).reshape(-1, 1)
  m = xMat.shape[0]
  weights = np.eye(m)
  for i in range(m):
    diffMat = testPoint - xMat[i]
    weights[i, i] = np.exp(-(diffMat @ diffMat.T) / (2 * k**2))
  xTx = xMat.T @ (weights @ xMat)
  if np.linalg.det(xTx) == 0.0:
    xTx = np.linalg.pinv(xTx)
  ws = np.linalg.pinv(xTx) @ xMat.T @ (weights @ yMat)
  return testPoint @ ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
  testMat = np.array(testArr)
  yHat = np.zeros(testMat.shape[0])
  for i in range(testMat.shape[0]):
    yHat[i] = lwlr(testMat[i], xArr, yArr, k).item()
  return yHat

def rssError(yArr, yHatArr):
  return ((yArr - yHatArr)**2).sum()

if __name__ == "__main__":
  abX, abY = loadDataSet(PATH)
  yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
  yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
  yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
  print("Training error using locally weighted linear regression:")
  print("Kernel=0.1:", rssError(abY[0:99], yHat01))
  print("Kernel=1:", rssError(abY[0:99], yHat1))
  print("Kernel=10:", rssError(abY[0:99], yHat10))

  yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
  yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
  yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
  print("Testing error on new data using locally weighted linear regression:")
  print("Kernel=0.1:", rssError(abY[100:199], yHat01))
  print("Kernel=1:", rssError(abY[100:199], yHat1))
  print("Kernel=10:", rssError(abY[100:199], yHat10))

  ws = standRegres(abX[0:99], abY[0:99])
  yHat = abX[100:199] @ ws
  print("Error using standard linear regression:", rssError(abY[100:199], yHat.flatten()))
