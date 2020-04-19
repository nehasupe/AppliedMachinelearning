# a4

please run this command before running (These files dataset files and model files used for neural net)

bash get_pickles.sh

To run the code: 
train
> python3 orient.py train train-data.txt nearest_model.txt nearest

> python3 orient.py test test-data.txt nearest_model.txt nearest



## Nearest

Steps:

Train: This algorithm is called a lazy learner algorithm as at the training time it doesn’t learn anything. It doesn’t have a training stage. So, for the training stage, we simply pass the training data to the model file.

Test: 

For each test sample in the test set:
> calculate the euclidean distance of test sample to each training sample

> Select k minimum distances and get their corresponding class labels

> Class label which appears the maximum number of times is the output class for this test sample


After studying this algorithm and I noted that it is an easy algorithm to understand but involves too many calculations at the test time. So to minimize the run time, I initially started out with Numpy arrays and using numpy methods to sort the arrays and extract k neighbors. But I was not able to implement the algorithm correctly. It resulted in giving me a low accuracy. Hence, I decided to use the data structure that I am comfortable with- Priority queue. The Euclidean distances are calculated using numpy arrays, for each test sample its corresponding distances are put in a priority queue along with class labels. From the priority queue, we get K values and class labels and set the output as the class which is found maximum in K neighbors. for k = 25 and 41, we have the highest accuracy but also it takes the most time so we have considered k = 9 for our final model. 
KNN would give 100% accuracy on the train set as it has memorized the model. If we go on increasing the size of the training set, the accuracy will increase as it sees more samples and can find samples that are at a lesser euclidean distance. 

Correctly classified: 
test/10008707066.jpg 0
test/10099910984.jpg 0
test/10107730656.jpg 180
 
Wrongly classified:
test/10196604813.jpg 0
test/10351347465.jpg 180
test/10352491496.jpg 180



| K  | Accuracy | Time |
| ------------- | ------------- | -------------- |
| 1  |  0.672322375397667 | 144.77188324928284 |
| 5  |   0.6914103923647932 |  150.0043842792511 |
| 9  |   0.7073170731707317 | 138.08184432983398 |
| 15 |  0.7020148462354189 | 229.2675096988678 |
| 21 |   0.7083775185577943 | 398.4095296859741 |
| 25 |  0.7104984093319194 | 411.6706552505493 |
| 41 |  0.7104984093319194 | 414.9000241756439 |
| 51 |   0.7083775185577943 |  402.6303000450134 |
| 75 |   0.7051961823966065 | 307.22897934913635 |
| 99 |   0.7020148462354189 | 204.8821964263916 |
| 135 | 0.7094379639448568 | 258.5539526939392 |
| 151 | 0.704135737009544 | 196.48592877388 |
| 199 |  0.711558854718982 | 256.99634623527527 |


## Tree
Train:
To train the data I implemented a greedy approach initially using the 'naieve' function.  To do this I had to:
1)calculate the entropy
2)find the best spot to split on
3)recur on the split until a solution is found
To calculate the entropy I initially analyzed each pixel with surmounted to 192 values.  I went through every image and for each pixel I categorized it into 25 different ranges (0-255 is the pixel value so about increments of 10) calculating using the entropy formula.  The assumption was that having the orientation be 0 was the goal (true state) and everything else was not (false state).  I was not sure how to make it differently because every example had the result be either true or false (you get a sunburn or you dont, you play tennis or you dont, etc.). Looking at every individual pixel was way too complicated and time consuming so my next solution was to average the r, g, and b pixel values for each row, this means I only had to deal with 25 different points instead of 192, other than that the entropy was calculated the same way.
I really struggled to understand how to document this data to apply to a testing set, and referenced a website (https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/) for some help.  So that I would not lose my work I created a new function called 'buildTree' instead of changing 'naieve'.  I was still not able to understand it fully so, although I think I am iterating correctly it is not properly documented.

Test:
For testing I still referenced the above website.  The goal was to go through every given image and iterate the tree we created based on various row color values until you got to a leaf and therefore a solution for each image.  Again, I was confused on how to properly document the tree so this did not fully work.

Problems:
My biggest problem I think was finding good examples to help clarify my solution.  I was able to find a lot of examples but all of them had only two results (true/false) where this had 4 (all orientations).  I really struggled on how to apply this information to the problem.  The other issue was documentation.  I did not know how to really document the tree to use as a prediction model.  Examples showed visual representations of trees but it was hard to find anything about how to acutally document it.  All of the notes talked about algorithms for creating the tree but nothing more.  Im sure documenting and applying the tree was a relatively simple solution that I just was unable to make sense of because there was little documentation on it.
Because I was unable to get this section working there are no tables analyzing accuracies.

## Neural Network
#### Please note that after [consulting with David](https://drive.google.com/file/d/19_FEnIYULNgX4wjnQWTzkmY-pXHR1TB-/view?usp=sharing) and forwarding the case to entire 551 staff, the code for this model and it's markdown report will be the same for both Bobby-Scott-Sharad's repository and this repository, since this model was worked on by Bobby.
To build the neural network from scratch, we used the model Bobby had built for his 556 assignment that required him to build a model similar [to this](https://keras.io/examples/mnist_mlp/) that included Dense, Activation and Dropout layers with RMSProp optimization. The primary challenge was to figure out a way to preprocess the training images, since we had only 10000 images to work with. Even though those 10000 images could be rotated in 3 other normal degrees to get 30000 more, for the model it'd still be counted as a bit of overfitting because those are still the same images, with similar pixel densities, which a neural net can easily identify. We thought of scraping more images from flickr using the flickr api to get more training data but decided to move into another direction. Instead of going for the given 8x8 images, we went for the original 75x75 images that could be resized to 32x32 or 28x28 to get essentially more features for the model to play with. For that we wrote a separate script that goes through all images in train folder, rotates each to all 3 normal degrees (90, 180, 270) and return a flattened numpy array of the images. Given we used the pillow (PIL) package for image processing, some images (~16-18) were unable to be processed/reshaped to desired dimensions, so we ignored them. We got out with a numpy ndarray of shape: (40000, 2352) if 28x28x3 sized images or (40000, 3072) if 32x32x3 sized images. This array had a uint8 dtype and we had to normalize the values by dividing by 255.0 after converting the values to float32. We chose the labels as follows: 
```python
# Key is degree, value denotes its chosen label.
train_mapping = {
    "0": 0,
    "90": 1,
    "180": 2,
    "270": 3
}
```
This was while setting up training data. While setting up for testing data, we found out that the way we rotated and labeled our images was different than the way the actual test images were labeled. That is, our 90 degrees was actually 270 degrees in the actual rotated testing data and vice-versa. So, for creating test data, we had to exchange the values of 90 and 270 to get:
```python
test_mapping = {
    "0": 0,
    "90": 3,
    "180": 2,
    "270": 1
}
```
This was the data pre-processing and augmentation step. Before starting training, we added 5 fold cross validation to the model using a little helper function from sklearn. It simply splits the training dataset into 5 random folds: (32000 random training samples + 8000 random validation samples) x 5. We added 2 dense layers with ReLU as the activation function and the final dense layer with 4 nodes (one for each orientation degree) and softmax for classification. Choosing the learning rate for RMSProp was tricky since we didn't want to overshoot with a larger value, so we went with 0.00005 to be safe. Training is done in 1024 mini batches, both forward and backward pass. Iterating through batches was done through a custom k fold class. Each layer (Dense, Activation and Dropout) has its own forward pass and backward pass function. We went with categorical crossentropy as the loss function since it seems to be the standard for classification problems. Upon training, validation was done with accuracy as a metric. The training and validation loss and accuracy is shown for each epoch, given verbose is set to True. Once the training is complete, the train() function returns the best model (fold) to evaluate on the test data. Average training and validation accuracies at the end of training each fold is around 80-85%. Average test accuracies range from 75-79%.

### Results
(BEST)
Dense Layers: 2 <br/>
Neurons: 300, 300 <br/>
Dropout values: 0.2, 0.2 <br/>
Epochs: 35 <br/>

| Fold | Training accuracy | Validation accuracy | Testing accuracy | Running time |
|:----:|:-----------------:|:-------------------:|:----------------:|:------------:|
|   1  |       81.09%      |        76.11%       |      75.80%      |  839 seconds |
|   2  |       84.27%      |        77.18%       |      76.10%      |  820 seconds |
|   3  |       80.03%      |        74.76%       |      74.99%      |  821 seconds |
|   4  |       81.88%      |        75.02%       |      73.35%      |  829 seconds |
|   5  |       86.14%      |        78.97%       |      78.20%      |  833 seconds |

Dense Layers: 3 <br/>
Neurons: 256, 256, 128 <br/>
Dropout values: 0.2, 0.2, 0.2 <br/>
Epochs: 35 <br/>

| Fold | Training accuracy | Validation accuracy | Testing accuracy | Running time |
|:----:|:-----------------:|:-------------------:|:----------------:|:------------:|
|   1  |       78.44%      |        72.74%       |      70.00%      | 1021 seconds |
|   2  |       80.03%      |        74.52%       |      72.51%      | 1105 seconds |
|   3  |       79.32%      |        74.90%       |      71.12%      | 1066 seconds |
|   4  |       81.11%      |        77.41%       |      74.36%      | 1132 seconds |
|   5  |       79.81%      |        73.15%       |      69.45%      | 1040 seconds |

## best
Our best performing model here is the neural net. It takes some time to train, but testing is quick and gives better accuracy so we would recommend that to our client.

Parameters: 

Dense Layers: 2

Neurons: 300, 300

Dropout values: 0.2, 0.2

Epochs: 35


