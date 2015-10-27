#Naive Bayes Classifier Project
#Editor: Bin H.

from numpy import *

class NBClr:

	"""
	Naive Bayes Classifier Class
	"""
	def __init__(self,data,label,cls_num = 20, delta = .1):

		self.data = data
		self.label = label
		self.data_num = size(data[:,0])
		self.doc_num = size(label)
		self.cls_num = cls_num #size(mapself.)
		self.delta = delta
		self.mean  = max(data[:,1]) 

	def clc_prob(self):
	
		data = self.data
		data_num = self.data_num	
		cls_num = self.cls_num
		mean = self.mean

		print  'a.calculate document number in the same class'
		prob_class = zeros(cls_num)
		label_count = zeros(cls_num)
		for i in range(self.doc_num):
			for j in range(1,cls_num+1):
				if self.label[i] == j:
					label_count[j-1] += 1
					break
		prob_class = label_count/double(sum(label_count))
		savetxt('data/train.prob_class',prob_class)

		print 'b.accumulate the number '
		label_acu = zeros(cls_num)
		for n in range(cls_num-1):
			label_acu[n+1] = label_count[n] + label_acu[n]

		print 'c.calculate the starting index in test.data list for each class'
		data_count = zeros(cls_num+1)
		p = 1
		for m in range(data_num):
			if data[m,0] > label_acu[p] :
				data_count[p] = m+1
				p+=1
				if p == cls_num:
					break	

		print 'd.run through the list to sum up word-occurrence in different word-id for each class'
		ocur_sum = zeros((cls_num, mean))
		b = 1
		for a in range(data_num):
			if a == data_count[b]:
				b+=1
			ocur_sum[b-1,data[a,1]-1] += data[a,2] 


		print 'e.calculate probability p(w|C) base on ocur_time and the equation'
		prob = zeros((cls_num, mean))
		for r in range(cls_num):
			prob[r,:]= (1-self.delta)*ocur_sum[r,:]/double(sum(ocur_sum[r,:])) + self.delta/double(mean)

			
		print 'f.save probability'
		savetxt('data/train.prob',prob)
		savetxt('data/train.mean',[mean])
		
		print 'g.done!!'
		return prob_class, prob, mean

	def pred(self, testdata, testlabel, prob_cls, prob, mean):
			
		print 'start to predict your data set!'	
		
		renormal = .8*10**3
		prob = double(prob)*renormal

		data_num = size(testdata[:,0])
		doc_num = size(testlabel)
		cls_num = self.cls_num	
		mean = int(mean)
		testmean  = int(max(testdata[:,1]))
		ocur_sum = zeros((doc_num, testmean))
		label_pred = zeros(doc_num)
		doc_prob = ones((doc_num,cls_num))
		
		j=1
		for i in range(data_num): 
			if testdata[i,0] > j:
				j+=1
			ocur_sum[j-1,testdata[i,1]-1] += testdata[i,2] 

		for m in xrange(doc_num):
			if m % 100==0:
				print m
			for k in xrange(cls_num):
				doc_prob[m,k] = prod(prob[k,:]**ocur_sum[m,0:mean])

		for u in range(cls_num):
			doc_prob[:,u] = doc_prob[:,u]*prob_cls[u]

		for l in range(doc_num):
			#findmax = argmax(doc_prob[l,:])
			label_pred[l] = argmax(doc_prob[l,:]) + 1

		savetxt('data/test.label_predict', label_pred)
		print 'save and done'
		return label_pred

	def accu(self,label_pred, label):
			
		label_num = size(label)
		accu = 0

		for i in range(label_num):
			if label[i]==label_pred[i]:
				accu += 1

		accu = accu/double(label_num)
		print 'Accuracy:', accu*100, '%'
		return accu


if __name__ == '__main__':

	print 'load data and label'
	train_data = loadtxt('data/train.data')
	train_label = loadtxt('data/train.label')
	#map = loadtxt('data/train.map')
	test_data = loadtxt('data/test.data')
	test_label = loadtxt('data/test.label')

	nb = NBClr(train_data,train_label)
	prob_cls, prob, mean = nb.clc_prob()
	label_pred = nb.pred(test_data, test_label, prob_cls, prob, mean)
	accu = nb.accu(label_pred, test_label)

