from numpy import *
#print label[0:1000]
#print label[0:1000]
renormal = .8*10**3
prob = double(loadtxt('train.prob'))*renormal
prob_cls = loadtxt('train.prob_class')
mean = int(loadtxt('train.mean'))
testdata = loadtxt('test.data')
testlabel = loadtxt('test.label')
#testmap = loadtxt('test.map')

data_num = size(testdata[:,0])
doc_num = size(testlabel)
cls_num = 20 #size(testmap)

testmean  = int(max(testdata[:,1]))
ocur_sum = zeros((doc_num, testmean))
label_pred = zeros(doc_num)
doc_prob = ones((doc_num,cls_num))
j=1

for i in range(data_num): 
	if testdata[i,0] > j:
		j+=1
	ocur_sum[j-1,testdata[i,1]-1] += testdata[i,2] 
#print sum(ocur_sum)- sum(testdata[:,2])

for m in range(doc_num):
	if m % 100==0:
		print m
	for k in range(cls_num):
		doc_prob[m,k] = prod(prob[k,:]**ocur_sum[m,0:mean])


for u in range(cls_num):
	doc_prob[:,u] = doc_prob[:,u]*prob_cls[u]


for l in range(doc_num):
	#findmax = argmax(doc_prob[l,:])
	label_pred[l] = argmax(doc_prob[l,:]) + 1

#print label_pred[0:500]

savetxt('test.label_predict', label_pred)
