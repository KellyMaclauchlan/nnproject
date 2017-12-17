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
		transformed.append(vectorizer.vocabulary_.get(word))
	vectorizedTrainX.append(transformed)

print(trainX[1])
print(vectorizedTrainX[1])
print(trainY[1])
trainX = vectorizedTrainX
print( vectorizer.vocabulary_.get("great") )
print( vectorizer.vocabulary_.get("phone") )

# Great phone!.
# r
#   (0, 2023)	1
#   (0, 3322)