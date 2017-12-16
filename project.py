import tensorflow as tf
import numpy as np


amazonFile="amazon_cells_labelled.txt"
imdbFile="imdb_labelled.txt"
yelpFile="yelp_labelled.txt"

# read in files
def readData(fileName):
	with open(fileName) as f:
		content = f.readlines()
	content = [x.strip() for x in content]
	X=[]
	Y=[]
	for x in content:
		data=x.split("\t")
		X.append(data[0])
		Y.append(data[1])
	return X, Y

# read in amazon file
amazonX, amazonY = readData(amazonFile)
# test data
print(amazonX[500])
print(amazonY[500])
print(len(amazonX))
print(len(amazonY))

# read in imdb file
imdbX, imdbY = readData(imdbFile)
# test data
print(imdbX[500])
print(imdbY[500])
print(len(imdbX))
print(len(imdbY))

#read in yelp file
yelpX, yelpY = readData(yelpFile)
# test data
print(yelpX[500])
print(yelpY[500])
print(len(yelpX))
print(len(yelpY))