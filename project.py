import tensorflow as tf
import numpy as np
import random
import downloadData
from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
 
amazonFile="amazon_cells_labelled.txt"
imdbFile="imdb_labelled.txt"
yelpFile="yelp_labelled.txt"


trainX, trainY = downloadData.getData()


vectorizer = CountVectorizer()
vectorizer.fit_transform(trainX).todense() 
#print( vectorizer.vocabulary_ )
print(len(vectorizer.vocabulary_))

vectorizedTrainX = []
arrayX = []
for x in trainX:
	transformed = []
	for word in x.split():
		num = vectorizer.vocabulary_.get(word)
		add = num if num!=None else 0
		transformed.append(add)
	
	while len(transformed) < 71:
		transformed.append(0)

	vectorizedTrainX.append(transformed)

print(trainX[1])
print(vectorizedTrainX[1])
print(trainY[1])
trainX = vectorizedTrainX
print( vectorizer.vocabulary_.get("great") )
print( vectorizer.vocabulary_.get("phone") )
print (len(max(trainX, key=len)))

## max length is 71 https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.pad.html

# Great phone!.
# r
#   (0, 2023)	1
#   (0, 3322)


#npArr = np.array(np.zeros([len(trainX),71]))
npArr = np.array(trainX)
print(npArr)

# for i in range(len(trainX)):
# 	for j in range(len(trainX[i])):
# 		npArr[i][j] = trainX[j]
# print(npArr[1])
# print(npArr.shape)






