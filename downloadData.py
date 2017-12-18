from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
import random
 
amazonFile="amazon_cells_labelled.txt"
imdbFile="imdb_labelled.txt"
yelpFile="yelp_labelled.txt"
def getData():
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

	def makeArray(x): return [1,0] if x=='0' else [0,1]
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
	# print(X[0])


	trainX = amazonX + imdbX + yelpX
	trainY = amazonY + imdbY + yelpY
	trainY = list(map(makeArray,trainY))


	shuffle = list(zip(trainX,trainY))
	random.shuffle(shuffle)
	trainX , trainY = zip(*shuffle)
	return trainX,trainY
