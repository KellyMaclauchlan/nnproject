import tensorflow as tf
import numpy as np
import random
from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
 
amazonFile="amazon_cells_labelled.txt"
imdbFile="imdb_labelled.txt"
yelpFile="yelp_labelled.txt"

# read in files
def readData(fileName):
	with open(fileName) as f:
		content = f.readlines()
	allText=content
	content = [x.strip() for x in content]
	X=[]
	Y=[]
	for x in content:
		data=x.split("\t")
		X.append(data[0])
		Y.append(data[1])
	return X, Y, allText

# read in amazon file
amazonX, amazonY, aall = readData(amazonFile)
# test data
print(amazonX[500])
print(amazonY[500])
print(len(amazonX))
print(len(amazonY))

# read in imdb file
imdbX, imdbY, iall = readData(imdbFile)
# test data
print(imdbX[500])
print(imdbY[500])
print(len(imdbX))
print(len(imdbY))

#read in yelp file
yelpX, yelpY, yall = readData(yelpFile)
# test data
print(yelpX[500])
print(yelpY[500])
print(len(yelpX))
print(len(yelpY))

#print(yall)

vectorizer = CountVectorizer()
print( vectorizer.fit_transform(yall).todense() )
print( vectorizer.vocabulary_ )
print(len(vectorizer.vocabulary_))
# print(X[0])


trainX = amazonX + imdbX + yelpX
trainY = amazonY + imdbY + yelpY

shuffle = list(zip(trainX,trainY))
random.shuffle(shuffle)
trainX , trainY = zip(*shuffle)



