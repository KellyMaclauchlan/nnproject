from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
import random
from sklearn.feature_extraction.text import CountVectorizer
amazonFile="amazon_cells_labelled.txt"
imdbFile="imdb_labelled.txt"
yelpFile="yelp_labelled.txt"
# lengthSize determins the length for padding
def getData(lengthSize=71):
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
	#change y values to array
	def makeArray(x): return [1,0] if x=='0' else [0,1]
	# read in amazon file
	amazonX, amazonY, aall = readData(amazonFile)

	# read in imdb file
	imdbX, imdbY, iall = readData(imdbFile)


	#read in yelp file
	yelpX, yelpY, yall = readData(yelpFile)

	#combine into one input
	trainX = amazonX + imdbX + yelpX
	trainY = amazonY + imdbY + yelpY
	trainY = list(map(makeArray,trainY))

	# mix the data together
	shuffle = list(zip(trainX,trainY))
	random.shuffle(shuffle)
	trX , trY = zip(*shuffle)
	#turn the input sentence into an array mapped to the wanted size
	vectorizer = CountVectorizer()
	vectorizer.fit_transform(trX).todense() 

	vectorizedtrX = []
	arrayX = []
	for x in trX:
		transformed = []
		for word in x.split():
			num = vectorizer.vocabulary_.get(word)
			add = num if num!=None else 0
			transformed.append(add)
		
		while len(transformed) < lengthSize:
			transformed.append(0)

		vectorizedtrX.append(transformed)

	trX = vectorizedtrX


	return trX, trY
