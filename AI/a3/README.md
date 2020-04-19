# a3
### Part 1: Part-of-speech tagging 

For POS problem, we had to implement 3 models-  Simplified model, HMM and complicated model 

In our train function, we are calculating the prior probabilities for each POS, initial distribution probabilities for the starting word of the sentence, emission probabilities for word and POS, transition probabilities of transitioning from part of speech of one word to part of speech of next word 

##### Simplified model: 
The simplified model is the Naïve Bayes model. For this model, we are calculating posterior probability for each word in the sentence by multiplying emission probability of that word and every part of speech to prior probability of that speech. Then we are finding for which POS we have the maximum posterior probability and appending it to the output list. If the word is not seen before in the train set, then noun is assigned for that word 

##### Hmm_viterbi: 
For our Viterbi algorithm implementation, for the first word we are calculating probabilities by multiplying initial probability of the POS to emission probability of the word and POS. The POS for which we have maximum value is selected and appended to the output list. For rest of the words in the sentence, we are iterating through all POS for each word. We use the values calculated for previous word and multiply them with transition probabilities of the POS and the current POS for which we are iterating. We select the maximum of this value and multiply it with emission probability of that word and current POS. After iterating through all POS for every word, we take maximum of the values and find the corresponding POS and append it to the output list 

##### MCMC:
For this we are to implement Gibbs Sampling.  For this we generate thousands of samples checking which occured most often.  A first guess is created (everything is a Noun) and then I throw away the first 1000 samples as they are very inaccurate in the beginning.  When creating the samples I go through each unobserved variable and sample a value for it given that the other variables are set, using variable elimination.  Then we combine the samples that are kept.


### PART 2: Code Breaking
The problem is code breaking using the Metropolis-Hastings algorithm.  I had to read the problem a few times and take note of certain things said to make sure I understood everything properly.  For example, I read it a few times and had to look at some of the code before I realized that both Replacement and Rearrangement were applied, I thought it was just one or the other.  The algorithm steps we were given were very helpful.  With it I was able to get started fairly easily and have a good grasp on most of what was needed for the problem.  From there I referenced notes and online articles to fill in the blank.

To decode the document well it follows the provided Metropolis-Hastings algorithm.  I first have to make an assumption so a good default seemed to be to assume that everything is in the correct spot (a:a and index 0 -> index 0).  From there I go to a recursive function that loops around 900 times.  It modifies a new guess by randomly switching two letters and two indexes.  I then try using the current guess and new guess to decrypt the document and calculate the probability of each of them being english text.  Based on which has a higher probablility I modify what my main probability is and loop.

I was confused on whether n was supposed to be set at 4 or not.  In the descriptio it said that we should know what n was but not the mapping function, but I could not find where n was defined, in all the examples I found on it in the description and code comments they used 4 so I came to the conclusion that that is what it should be.
I really struggled on the probabilities and just knowing what to do for that.  I attempted to go to office hours early on but the one I could make it to before break got canceled last minute.  Luckily the professor had office hours right before the assignment deadline and I was able to get on with him and clarify it a bit.  He gave a brief overview of how it worked and referenced that something similar happend in part 1.  With this information I tried to immitate the train method for part 1 translated for what we need.  I do not think it is fully correct but I am still confused on the subject and out of time to explore it further.
I know that it should be iterated many times but I got an error for looping even 1000 times so I decided to limit it to just 900 so the code would compile.


### PART 3: Spam classification 
To run the code on server: python3 spam.py train test output-file

The given problem is a document classification problem where an e-mail is classified as spam or notspam. This problem is implemented using a bag of words model and Naïve Bayes classifier. Essentially, each document is split into words and the frequency of the word occurring in the dataset is recorded for each class. For representation purposes, for each document a vector of size of bag of words can be created which will have entries 1 and 0, 1 indicating that the word in the document is present. 

First, we imported the training and testing files in our program and stored them in training and testing dataframes.  

For coding purposes, we have used a dictionary to store our Bag of Words model. The words are the keys and the values are the frequency count of that word in the dataset. Our NaiveBayes() class has two functions- fit and predict. The fit function performs training. That is it calculates prior probabities of both classes using the formula: 

P ( Spam ) = Number of documents classified as Spam/  Total number of documents in the dataset 

P(notspam) = Number of documents classified as notspam/ Total number of documents in the dataset 

The function creates the Bag of Words model. We use a dictionary to store all the words occuring in the dataset. Then we try to calculate the likelihood probabilities.  

P(word| spam) =  Probability of word given that the document is spam = number of times word appears in class spam/ Total number of words for spam 

P(word| notspam) =  Probability of word given that the document is notspam = number of times word appears in class notspam/ Total number of words for notspam 

The predict function loops through the test dataset. For each class, posterior probabilities are calculated. For each document, the document is split into tokens. For each word, likelihood probabilities are multiplied. Posterior probability for each class is calculated by multiplying prior probabilities with the likelihoods. Initially, we were getting inf values for probabilities. So, we switched to log calculation for posterior probability. Also, we did not know what to do with words which do not occur in the training set but occur in the testing set. It resulted in zero probabilities. We considered dropping those words or assigning small values. But we researched a bit and found this blog: https://medium.com/@theflyingmantis/text-classification-in-nlp-naive-bayes-a606bf419f8c which talks about text classification using Naïve Bayes and Bag of words model. The author has mentioned on using Laplace smoothing. So the smoothing works in such a way that while calculating likelihood probabilities, 1 is added to the number of times the word appears in a class which is the numerator of the equation P(word| spam). To balance that we also add length of our vocabulary to the denominator of the equation. In that way even if the word never appeared in the dataset or for that class, we wont get zero probabilities. Finally, we have posterior probabilites as: 

P(cj|wi) = P ( wi | cj ) = [ count( wi, cj ) + 1 ] / [ Σw∈V( count ( w, cj ) ) + |V| ] 

From: https://medium.com/@theflyingmantis/text-classification-in-nlp-naive-bayes-a606bf419f8c 

Accuracy achieved by the model is: 87.66%

This assignment was submitted on December 1
The following changes were made because it failed to run on Linux server

Added line-
#!/usr/local/bin/python3

line 98 in spam.py  gave an encoding error on the server-

F = open(sys.argv[1]+"/notspam/"+f, 'r')
was updated to
F = open(sys.argv[1]+"/notspam/"+f, 'r', encoding="Latin-1")
