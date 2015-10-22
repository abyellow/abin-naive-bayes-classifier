from numpy import *

label_pred = loadtxt('data/test.label_predict')
label = loadtxt('data/test.label')

label_num = size(label)
accu = 0

for i in range(label_num):
	if label[i]==label_pred[i]:
		accu += 1

accu = accu/double(label_num)
print 'Accu:', accu, '%'
