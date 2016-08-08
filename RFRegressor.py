import sklearn.ensemble as sk
from sklearn.cross_validation import cross_val_score
import WQmodule as wq
from scipy.stats.stats import pearsonr
import cython
import numpy as np
import os
import math as math
from math import log
mpath = os.path.abspath(os.curdir)
directory = mpath + '/DATA/RFTrain/'

def ReadInput(fname):
	file = open(fname, 'r')
	f = file.readlines()
	inp = []
	for i in xrange(len(f)):
		day = f[i].split(';')
		k = 0
		l = []
		for j in xrange(len(day) - 1):
			l.append(float(day[j]))
		inp.append(l)
	inp = np.array(inp)
	inp = np.transpose(inp)
	return inp



def ReadOutput(fname):
	file = open(fname, 'r')
	f = file.readlines()
	inp = []
	k = 0
	day = f[0].split(';')
	for k in xrange(len(day) - 1):
		inp.append(float(day[k]))
	return np.array(inp)

res = []
corr = 0
k = 0
res = np.array(res)
X = ReadInput(directory + 'insample1.csv')
Y = ReadOutput(directory + 'output1.csv')
Y_2  = ReadOutput(directory + 'returns1.csv')
print pearsonr(Y, Y_2)
print len(Y), len(Y_2)
np.savetxt('checkdif.csv',np.sign(Y) - np.sign(Y_2), delimiter = ';')
equity_1 = []
equity_2 = []
equity_1.append(0)
equity_2.append(0)
for i in xrange(len(Y)):
	equity_1.append(equity_1[i]+Y[i])
	equity_2.append(equity_2[i]+Y_2[i])
wq.PlotSignal(equity_1, 'eq1.pdf')
wq.PlotSignal(equity_2, 'eq2.pdf')
# X = np.transpose(X)
# for i in xrange(len(X)):
# 	for j in xrange(i+1, len(X) - 1):
# 		corr = corr + pearsonr(X[i], X[j])[0]
# 		k = k + 1
# 		print pearsonr(X[i], X[j]), i, j
# print corr/k
# X = np.transpose(X)
# MYRFR = sk.ExtraTreesRegressor(n_estimators = 20)
# MYRFR = MYRFR.fit(X, Y)
# a = MYRFR.feature_importances_
# sc = 0
# for i in xrange(2, 1540):
#  	X = ReadInput(directory + 'insample' + str(i) + '.csv')
#  	Y = ReadOutput(directory + 'output' + str(i) + '.csv')
#  	MYRFR2 = sk.ExtraTreesRegressor(n_estimators = 20)
#  	MYRFR2 = MYRFR.fit(X, Y)
#  	print MYRFR2.score(X, Y)
#  	a = a + MYRFR2.feature_importances_
#  	print i
# print a/1539

# X2 = ReadInput(directory + 'insample5.csv')
# Y2 = ReadOutput(directory + 'output5.csv')
# MYRFR2 = sk.RandomForestRegressor(n_estimators = 30)
# MYRFR2 = MYRFR2.fit(X2, Y2)
# print MYRFR2.score(X2, Y2)
# # print MYRFR.score(X2, Y2)
# scores = cross_val_score(MYRFR2, X2, Y2)
# scores.mean()
# print scores

