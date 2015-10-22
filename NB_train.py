from numpy import *

data = loadtxt('data/train.data')
label = loadtxt('data/train.label')
#map = loadtxt('data/train.map')

data_num = size(data[:,0])
doc_num = size(label)
cls_num = 20 #size(map)
delta = .1
mean  = max(data[:,1]) 

prob = zeros((cls_num, mean))
ocur_sum = zeros((cls_num, mean))
label_count = zeros(cls_num)
data_count = zeros(cls_num+1)
prob_class = zeros(cls_num)
label_acu = zeros(cls_num)
p=1
b=1

# calculate document number in the same class
for i in range(doc_num):
	for j in range(1,cls_num+1):
		if label[i] == j:
			label_count[j-1] += 1
			break
prob_class = label_count/double(sum(label_count))
savetxt('data/train.prob_class',prob_class)

#accumulate the number 
for n in range(cls_num-1):
	label_acu[n+1] = label_count[n] + label_acu[n]


#calculate the starting index in test.data list for each class
for m in range(data_num):
	if data[m,0] > label_acu[p] :
		data_count[p] = m+1
		p+=1
		if p == cls_num:
			break	

#run through the list to sum up word-occurrence in different word-id for each class
for a in range(data_num):
	if a == data_count[b]:
		b+=1
	ocur_sum[b-1,data[a,1]-1] += data[a,2] 


#calculate probability p(w|C) base on ocur_time and the equation in Prob.1.
for r in range(cls_num):
	prob[r,:]= (1-delta)*ocur_sum[r,:]/double(sum(ocur_sum[r,:])) + delta/double(mean)

	
#save probability
savetxt('data/train.prob',prob)
savetxt('data/train.mean',[mean])





