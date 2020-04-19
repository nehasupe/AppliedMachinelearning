#!/usr/local/bin/python3
# CS B551 Fall 2019, Assignment #4
#
# Your names and user ids:
#
# Kelly Wheeler (kellwhee), Neha Supe (nehasupe), Bobby Rathore(brathore)
# 
#
import sys
import math
import numpy as np
from queue import PriorityQueue
import time
import pickle
from nn_layers import Dense, Activation, Dropout
from nn_model import start
from nn_process_images import load_numpy_from_pickle
from nn_utils import accuracy_score

def save_model_to_txt(obj, file_name, model):
    with open(file_name, "w") as f:
        f.write(
            " ".join(
                [
                    "{0}.{1} = {2}".format(model, key, value)
                    for key, value in obj.__dict__.items()
                ]
            )
        )


def save_model_to_pickle(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_model_from_pickle(file_name):
    file = open(file_name, "rb")
    model_object = pickle.load(file)
    return model_object

splits = []

def read_data(fname):
	train_data = []
	file = open(fname, 'r')
	for line in file:
		data = [x for x in line.split()]
		train_data.append(data)
	return train_data

def knn_classifier_train(train):
	# write to a model output file
	# # pop the first column which is filename.jpg
	for data in train:
		data.pop(0)
	return train

def knn_classifier_test(train, test):
	num_neighbors = 9
	# from test remove the first column which is the name of the feature
	test_id = [data[0] for data in test]
	# Create target test labels list
	test_labels = [int(data[1]) for data in test]

	# Create a list of all test samples
	for data in test:
		data.pop(0)
		data.pop(0)
	test = [[int(d) for d in data] for data in test]
	# Create target train labels list
	train_labels = [int(data[0]) for data in train]
	# Remove classes from the train set
	for data in train:
		data.pop(0)

	train = [[int(d) for d in data] for data in train]
	
	data = np.array(train)
	test_sample = np.array(test)

	# Calculating Euclidean distances
	# https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
	euclidean_distance1 = -2 * np.dot(test_sample, data.T) 
	euclidean_distance2 = np.sum(data**2,    axis=1) 
	euclidean_distance3 =  np.sum(test_sample**2, axis=1)[:, np.newaxis]
	euclidean_distance = euclidean_distance1 + euclidean_distance2 + euclidean_distance3
	euclidean_distance = np.power(euclidean_distance, 0.5)
	# get each row from the euclidean distance, extract k neighbors, count how many classes, put max in the output\
	output = [0] * len(test_labels)
	for i in range(len(test_labels)):
		distlabel = PriorityQueue()
		knn = {0: 0, 90:0, 180: 0, 270: 0}
		for j in range(len(train_labels)):
			distlabel.put((euclidean_distance[i][j], train_labels[j]))
		for k in range(num_neighbors):
			(dist, label)= distlabel.get()
			knn[label] +=1
		values = list(knn.values())
		keys = list(knn.keys())

		output[i] = keys[values.index(max(values))]
	
	# Calculating accuracy
	summ = 0
	for i in range(len(test_labels)):
		if test_labels[i] == output[i]:
			summ = summ + 1

	print("Accuracy: ", summ/len(test_labels))
	
	outputs = [0] * len(test_id)
	for i in range(len(test_id)):
		outputs[i] = (test_id[i], output[i])

	return outputs


#######################################################

def haveSplitAlready(ind):
    for itm in splits:
        if itm == ind:
            return 1
    return 0


def naieveEntropy(desiredOrientation, images):
    entropyValuesDict = {}

    #192 color pixels on board
    #range(2, 194) from 2 to 193
    imageCount = len(images)
    for ind in range(2, 194):
        #if you have not split on this point caluclate entropy
        if haveSplitAlready(ind) == 0:
            #0-255 is pixel value range
            entroForIndex = 0
            for pixNum in range(0, 25):
                bottomVal = pixNum * 10
                topVal = bottomVal + 9
            for pixVal in range(0, 256):
                pixValCount = 0
                desiredOrientationCount = 0
                for img in images:
                    colorVal = int(img[ind])
                    if colorVal >= bottomVal:
                        if colorVal <= topVal:
                            #it is within 10 of the same value
                            pixValCount += 1
                            if img[1] == desiredOrientation:
                                desiredOrientationCount += 1
                occurenceAvg = pixValCount / imageCount
                accuracy = desiredOrientationCount / pixValCount
                entropyValue = accuracy * -1
                if accuracy == 0:
                    entropyValue = 0
                else:
                    entropyValue = entropyValue * math.log(accuracy, 2)

                entropyValue = entropyValue * occurenceAvg
                entroForIndex += entropyValue

            #add dictionary input for index with its entropy
            entropyValuesDict[ind] = entroForIndex
    return entropyValuesDict


def rowEntropy(desiredOrientation, images):
    entropyValuesDict = {}
    #now 24 pixel color values since rows are averaged
    imageCount = len(images)
    for ind in range(2, 26):
        if haveSplitAlready(ind) == 0:
            #0-255 pixel range, do them in 10 pt increments
            entroForIndex = 0
            for pixNum in range(0, 25):
                bottomVal = pixNum * 10
                topVal = bottomVal + 9
                pixValCount = 0
                desiredOrientationCount = 0
                for img in images:
                    colorVal = int(img[ind])
                    if colorVal >= bottomVal:
                        if colorVal <= topVal:
                            pixValCount += 1
                            if img[1] == desiredOrientation:
                                    desiredOrientationCount += 1
                occurenceAvg = pixValCount / imageCount
                if desiredOrientationCount == 0:
                        entropyValue = 0
                else:
                        accuracy = desiredOrientationCount / pixValCount
                        entropyValue = accuracy * -1
                        entropyValue = entropyValue * math.log(accuracy, 2)

                entropyValue = entropyValue * occurenceAvg
                entroForIndex += entropyValue
            entropyValuesDict[ind] = entroForIndex
    return entropyValuesDict

#split the images based on given item
def split(ind, images):
    splits.append(ind)
    sortedImages = []
    for num in range(0, 25):
        sortedImages.append([])
    for img in images:
        pixVal = int(img[ind])
        for num in range(0, 25):
            bottomVal = num * 10
            topVal = bottomVal + 9
            if pixVal >= bottomVal:
                if pixVal <= topVal:
                    sortedImages[num].append(img)
                    break
    return sortedImages


#return index of the best entropy value
def bestAccuracy(entropyValuesDict):
    bestKey = -1
    bestValue = 2
    for key, value in entropyValuesDict.items():
        if value < bestValue:
            bestValue = value
            bestKey = key
    return bestKey


#recurs on children
def splitter(node, depth, maxDepth):
        index = node[0]
        value = node[1]
        imageCollections = node[2]
        if depth >= maxDepth:
                #terminate
                node[2] = {-1}
                return
        if value != 0:#not leaf, keep recuring
                #process all children
                for images in imageCollections:
                        newNode = getSplit(images)
                        splitter(newNode, depth+1, maxDepth)
        #terminate
        node[2] = {-1}

#used the following website for help, also used for predictions
#https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
#creates the decision tree
def buildTree(images, max_depth):
        root = getSplit(images)
        splitter(root, 1, max_depth)
        return root


#splits the data based on the best option
def getSplit(images):
        entropyValuesDict = rowEntropy(0, images)
        bestKey = bestAccuracy(entropyValuesDict)
        bestValue = entropyValuesDict[bestKey]
        sortedImages = split(bestKey, images)
        rv = []
        #index, value, listGroups
        rv.append(bestKey)
        rv.append(bestValue)
        rv.append(sortedImages)
        return rv


def naieve(images):
    #calculate entropy
    entropyValuesDict = rowEntropy(0, images)

    #split on best value
    bestKey = bestAccuracy(entropyValuesDict)
    bestValue = entropyValuesDict[bestKey]
    sortedImages = split(bestKey, images)
    
    sortedAllImages = []
    if bestValue != 0:#not leaf node keep recuring
        for imageCollection in sortedImages:
            sortCollection = naieve(imageCollection)
            sortedAllImages.append(sortCollection)
        return sortedAllImages
    return sortedImages


#averages a single row and returns r g and b values
def avgRow(image, start, end):
    r = 0
    g = 0
    b = 0
    base = 8
    #turn 0 for r, 1 for g, 2 for b
    turn = 0
    for ind in range(start, end+1):
        if turn == 0:#r
            r += int(image[ind])
            turn = 1
        elif turn == 1:#g
            g += int(image[ind])
            turn = 2
    else:#g
            b += int(image[ind])
            turn = 0
    r = r / base
    g = g / base
    b = b / base
    rv = [r, g, b]
    return rv


#merges ls2 into ls1
def combine(ls1, ls2):
        for itm in ls2:
            ls1.append(itm)
        return ls1


#takes a full image and averages the pixel values for each row
def avgRows(image):
    avg = []
    avg.append(image[0])
    avg.append(image[1])

    avg = combine(avg, avgRow(image, 2, 25))
    avg = combine(avg, avgRow(image, 26, 49))
    avg = combine(avg, avgRow(image, 50, 73))
    avg = combine(avg, avgRow(image, 74, 97))
    avg = combine(avg, avgRow(image, 98, 121))
    avg = combine(avg, avgRow(image, 122, 145))
    avg = combine(avg, avgRow(image, 146, 169))
    avg = combine(avg, avgRow(image, 170, 193))

    return avg

#averages the rows in every image
def avgAllImages(images):
    newImages = []
    for image in images:
        newImages.append(avgRows(image))
    return newImages

#determines if you have hit a leaf or not
def isBranch(node):
        if -1 in node[2]:
                return 1
        else:
                return 0


#predicts the orientation of an image with the given decision tree
def predict(tree, image):
        index = tree[0]
        value = tree[1]
        imageTrees = tree[2]
        for num in range(0, 25):
                bottomVal = num * 10
                topVal = bottomVal + 9
                if value >= bottomVal:
                        if value <= topVal:
                                if isBranch(imageTrees[num]) == 0:#imageTrees[num] is instance
                                        return predict(imageTrees[num], image)
                                return imageTrees[num]


#makes predicitons for all the given images
def makePredictions(tree, testImages, maxDepth):
        predictions = []
        for image in testImages:
                predictions.append(predict(tree, image))
        return predictions

#######################################################


if __name__ == "__main__":
    if(len(sys.argv) != 5):
        raise(Exception("Error: expected 5 arguments"))
    if sys.argv[4] == 'nearest':
    	if sys.argv[1] == 'train':
    		train_data = read_data(sys.argv[2])
    		start_time1 = time.time()
    		model_data = knn_classifier_train(train_data)
    		time1 = time.time() - start_time1
    		print("Training time:", time1)
    		# write model data to file
    		with open(sys.argv[3], "w") as model_file:
   				for data in model_data:
   					model_file.write(" ".join([str(i) for i in data]))
   					model_file.write("\n")
    	elif sys.argv[1] == 'test':
    		test_data = read_data(sys.argv[2])
    		model_data = read_data(sys.argv[3])
    		start_time1 = time.time()
    		output_data = knn_classifier_test(model_data, test_data)
    		time1 = time.time() - start_time1
    		print("Testing time:", time1)
    		# store the output to a text file
    		with open('output.txt', "w") as output_file:
   				for data in output_data:
   					output_file.write(" ".join([str(i) for i in data]))
   					output_file.write("\n")
    	else:
    		raise(Exception("Wrong parameter 2: Should be train or test"))
    if sys.argv[4] == 'tree':
    	if sys.argv[1] == 'train':
                train_data = read_data(sys.argv[2])
                newImages = avgAllImages(train_data)
                #tree = naieve(newImages)
                tree = buildTree(newImages, 20)
#                with open(sys.argv[3], "w") as model_file:
#                        for data in tree:
#                                model_file.write(" ".join([str(i) for i in data]))
#                                model_file.write("\n")
    	elif sys.argv[1] == 'test':
                test_data = read_data(sys.argv[2])
                test_data = avgAllImages(test_data)
                model_data = read_data(sys.argv[3])
                #tree = makePredictions(model_data, test_data, 10)
#                with open('output.txt', "w") as output_file:
#                        for data in tree:
#                                output_file.write(" ".join([str(i) for i in data]))
#                                output_file.write("\n")
                
    	else:
    		raise(Exception("Wrong parameter 2: Should be train or test"))
    if sys.argv[4] == 'nnet' or sys.argv[4] == 'best':
    	if sys.argv[1] == 'train':
            trained_neural_network = start(
                layers=[
                    Dense(300),
                    Activation("relu"),
                    Dropout(0.2),
                    Dense(300),
                    Activation("relu"),
                    Dropout(0.2),
                    Dense(4),
                    Activation("softmax"),
                ],
                epochs=35,
                learning_rate=0.001,
                rho=0.9,
                number_of_folds=5,
            )

            save_model_to_pickle(obj=trained_neural_network, file_name="nnet.pkl")
            save_model_to_txt(
                obj=trained_neural_network,
                file_name="nnet_model.txt",
                model="neural_network",
            )
    	elif sys.argv[1] == 'test':
    		# load and test
            trained_neural_network = load_model_from_pickle(file_name="nnet.pkl")
            X_test = load_numpy_from_pickle(file_name="X_test.pkl")
            y_test = load_numpy_from_pickle(file_name="y_test.pkl")

            # Get predictions
            y_pred = trained_neural_network.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            print(
                "\n–– Test accuracy: {0:.2f}%\n–– Error: {1:.2f}%\n\n**************************".format(
                    100.0 * score, 100.0 * (1.0 - score)
                )
            )
    	else:
    		raise(Exception("Wrong parameter 2: Should be train or test"))
