import WQmodule as wq
import cython
import numpy as np
import os
import math as math
from math import log
mpath = os.path.abspath(os.curdir);
directory = mpath + '/DATA/USTOP2000final/';
director2 = mpath + '/DATA/RFTrain/';
timeperiod = ['2010-08-01', '2016-07-20']
ustop2000 = wq.DATA(timeperiod, directory)
close = np.transpose(np.array(ustop2000.get_data('close')))
high = np.transpose(np.array(ustop2000.get_data('high')))
low  = np.transpose(np.array(ustop2000.get_data('low')))
op = np.transpose(np.array(ustop2000.get_data('open')))
volume = np.transpose(np.array(ustop2000.get_data('volume')))

def SignsSignals(list_of_signal, begin):
	signs = []
	for i in xrange(len(list_of_signal)-1):
		sign = []
		for j in xrange(begin, len(list_of_signal[i])):
			sign.append(log(list_of_signal[i][j]/list_of_signal[i+1][j]))
		signs.append(sign)
		#print i
	return np.array(signs)

def cut(array , begin, end2 = 0):
	listik = []
	end  = len(array) - 1
	if end2 != 0:
		end = end2
	for i in xrange(begin, end):
		listik.append(array[i])
	return listik

def savecsv2d(array, fname):
	file = open(fname, 'wb')
	for i in xrange(len(array)):
		string = str(array[i][0]) + ';'
		for j in xrange(1, len(array[i]) - 1):
			string = string + str(array[i][j]) + ';'
		string = string + '\n'
		file.write(string)
	file.close()

def savecsv1d(array, fname):
	file = open(fname, 'wb')
	string = str(array[0]) + ';'
	for j in xrange(1, len(array)):
		string = string + str(array[j]) + ';'
	string = string + '\n'
	file.write(string)
	file.close()

for i in xrange(len(close)):
	insample = []
	emas = []
	emas_2 = []
	vwaps = []
	price = wq.DB12(close[i, :], 11, filters = 'low')
	price = wq.DB12(price, 11, filters = 'low')
	returns_1 = wq.returns(price, 1, 43)
	returns_2 = wq.returns(close[i, :], 1, 65)

	#insample.append(wq.RB(close[i, :], op[i, :], 64))
	#insample.append(wq.US(close[i, :], op[i, :], high[i, :], 64))
	#insample.append(wq.LS(close[i, :], op[i, :], low[i, :], 64))
	#insample.append(wq.HL(high[i, :], low[i, :], 64))
	insample.append(wq.DB4(close[i, :], 64, filters = 'high'))
	insample.append(wq.DB6(close[i, :], 64, filters = 'high'))
	insample.append(wq.DB12(close[i, :], 64, filters = 'high'))
	insample.append(wq.returns(close[i, :], 1, 64))
	insample.append(wq.returns(close[i, :], 2, 64))
	insample.append(wq.returns(close[i, :], 4, 64))
	#insample.append(wq.returns(close[i, :], 8, 64))
	#insample.append(wq.stddev(wq.returns(close[i, :], 1, 1), 8, 63))
	insample.append(wq.stddev(wq.returns(close[i, :], 1, 1), 16, 63))
	# insample.append(wq.stddev(wq.returns(close[i, :], 1, 1), 32, 63))
	# insample.append(wq.stddev(wq.returns(close[i, :], 1, 1), 60, 63))
	# insample.append(wq.stddev(high[i, :] - low[i, :], 8, 64))
	# insample.append(wq.stddev(high[i, :] - low[i, :], 16, 64))
	# insample.append(wq.stddev(high[i, :] - low[i, :], 32, 64))
	# insample.append(wq.stddev(high[i, :] - low[i, :], 64, 64))
	# insample.append(wq.meanreversion(close[i ,:], 8, 64))
	# insample.append(wq.meanreversion(close[i ,:], 16, 64))
	# insample.append(wq.meanreversion(close[i ,:], 32, 64))
	# insample.append(wq.meanreversion(close[i ,:], 64, 64))
	# insample.append(wq.meanreversion(high[i, :] - low[i, :], 8, 64))
	# insample.append(wq.meanreversion(high[i, :] - low[i, :], 16, 64))
	# insample.append(wq.meanreversion(high[i, :] - low[i, :], 32, 64))
	# insample.append(wq.meanreversion(high[i, :] - low[i, :], 64, 64))

	#insample.append(wq.returns(close[i, :], 16, 64))
	#insample.append(wq.returns(close[i, :], 32, 64))
	#insample.append(wq.returns(close[i, :], 64, 64))
	emas.append(close[i, :])
	emas.append(wq.EMA(close[i, :], 2))
	emas.append(wq.EMA(close[i, :], 4))
	emas.append(wq.EMA(close[i, :], 8))
	emas.append(wq.EMA(close[i, :], 16))
	#emas.append(wq.EMA(close[i, :], 32))
	#emas.append(wq.EMA(close[i, :], 64))

	emas_2.append(cut(close[i, :], 0))
	emas_2.append(cut(wq.EMA(close[i, :], 2), 0))
	emas_2.append(cut(wq.EMA(close[i, :], 4), 0))
	#emas_2.append(cut(wq.EMA(close[i, :], 8), 0))
	#emas_2.append(cut(wq.EMA(close[i, :], 16),0))
	#emas_2.append(cut(wq.EMA(close[i, :], 32), 0))
	#emas_2.append(cut(wq.EMA(close[i, :], 64), 0))

	vwaps.append(cut(close[i, :], 64, end2 = len(close[i, :])))
	vwaps.append(wq.vwap(close[i, :], volume[i, :], 2, 64))
	vwaps.append(wq.vwap(close[i, :], volume[i, :], 4, 64))
	vwaps.append(wq.vwap(close[i, :], volume[i, :], 8, 64))
	#vwaps.append(wq.vwap(close[i, :], volume[i, :], 16, 64))
	#vwaps.append(wq.vwap(close[i, :], volume[i, :], 32, 64))
	#vwaps.append(wq.vwap(close[i, :], volume[i, :], 64, 64))
	emas = SignsSignals(emas, 64)
	emas_2 = SignsSignals(emas_2, 63)
	vwaps = SignsSignals(vwaps, 0)
	for j in xrange(len(emas)):
		insample.append(emas[j])

	for j in xrange(len(emas_2)):
		insample.append(emas_2[j])

	for j in xrange(len(vwaps)):
		insample.append(vwaps[j])
	insample = np.array(insample)
	savecsv2d(insample ,director2 + 'insample' + str(i) + '.csv')
	savecsv1d(returns_1, director2 + 'output' + str(i) + '.csv')
	savecsv1d(returns_2, director2 + 'returns' + str(i) + '.csv')
	# print len(insample)
	# for k in xrange(len(insample)):
	# 	print len(insample[k])
	print i





