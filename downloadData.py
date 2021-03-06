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
	print(len(vectorizer.vocabulary_))
	vectorizedtrX = []
	arrayX = []
	#create an array of the dictionary size where each value is the number of times that word appears (bag-of-words model)
	for x in trX:
		transformed=[]
		for w in vectorizer.vocabulary_.keys():
			transformed.append(x.count(w))
		
		vectorizedtrX.append(transformed)

	# if we want to have the input as a sentence turned into an array of numbers for each word
	# 	transformed = []
	# 	for word in x.split():
	# 		num = vectorizer.vocabulary_.get(word)
	# 		add = num if num!=None else 0
	# 		transformed.append(add)
		
	# 	while len(transformed) < 71:
	# 		transformed.append(0)

	# 	vectorizedtrX.append(transformed)

	#set the X values to the dictionary transformed input
	trX = vectorizedtrX

	return trX, trY,len(vectorizer.vocabulary_)
