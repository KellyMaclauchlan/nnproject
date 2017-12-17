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
print( vectorizer.fit_transform(trainX).todense() )
print( vectorizer.vocabulary_ )
print(len(vectorizer.vocabulary_))

vectorizedTrainX = vectorizer.transform(trainX)
print(trainX[1])
print(vectorizedTrainX[1])
print(trainY[1])
trainX = vectorizedTrainX





