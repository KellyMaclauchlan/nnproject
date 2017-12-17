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
print(trainX[0])



