###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids:
#
# Kelly Wheeler (kellwhee), Neha Supe (nehasupe)
# (Based on skeleton code by D. Crandall)
#


import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    prior_prob = {}
    pos_prior_prob = {}
    emission_prob = {}
    transition_prob = {}
    initial_prob = {}

    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "Complex":
            return -999
        elif model == "HMM":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        total_words = 0
        # calculating total words in dataset
        for sample in data:
            total_words += len(sample[0])
        for sample in data:
            # the words of the sentence
            words = sample[0]
            # corresponding parts of speech
            pos = sample[1]
            for i in range(len(sample[0])):
                # count for POS for priors
                if pos[i] in self.pos_prior_prob:
                    self.pos_prior_prob[pos[i]] += 1
                else:
                    self.pos_prior_prob[pos[i]] = 1
                # calculating emission count and transition count 
                #emission_keys = list(self.emission_prob.keys())
                if (words[i], pos[i]) in self.emission_prob:
                    self.emission_prob[(words[i], pos[i])] += 1
                else:
                    self.emission_prob[(words[i], pos[i])] = 1
                if i == 0:
                    if pos[i] in self.initial_prob:
                        self.initial_prob[pos[i]] += 1
                    else:
                        self.initial_prob[pos[i]] = 1
                if i > 0:
                    if (pos[i-1], pos[i]) in self.transition_prob:
                        self.transition_prob[(pos[i-1], pos[i])] += 1
                    else:
                        self.transition_prob[(pos[i-1], pos[i])] = 1

        # calculating transition probabilities
        for pos1 in self.pos_prior_prob:
            for pos2 in self.pos_prior_prob:
                if (pos1, pos2) in self.transition_prob:
                    self.transition_prob[(pos1, pos2)] /= self.pos_prior_prob[pos1]
                else:
                     self.transition_prob[(pos1, pos2)] = 0.000000000001

        # calculating initial probabilities
        for first in self.initial_prob:
            self.initial_prob[first] /= len(data)
        #Calculating emission probability
        # emission_keys = list(self.emission_prob.keys())
        for word, pos in self.emission_prob:
            self.emission_prob[(word, pos)] /= self.pos_prior_prob[pos]
        # calculating prior probabilities 
        # all_pos = list(self.pos_prior_prob.keys())
        for pos in self.pos_prior_prob:
            self.pos_prior_prob[pos] /= total_words



    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        # max (Pi = Si |W)
        pos_output = []
        all_pos = list(self.pos_prior_prob.keys())
        
        for word in sentence:
            max_prob = 0
            max_pos = "noun"
            for pos in all_pos:
                if (word, pos) in self.emission_prob:
                    prob = self.emission_prob[(word, pos)] * self.pos_prior_prob[pos]
                    if prob > max_prob:
                        max_prob = prob
                        max_pos = pos
            pos_output.append(max_pos)

        return pos_output
    
    #helper function for complex_mcmc
    def sample(self, pos_output, sentence):
        i = 0
        #for each var Xi
        for word in sentence:
            #let x[t]=x[t-1]
            xt = " "
            if i > 0:
                xt = pos_output[i - 1]
            xt2 = " "
            if i < len(sentence) - 1:
                xt2 = pos_output[i + 1]

            #sample values for Xi given values for other variables in x(t)
            j = 0
            probs = [0] * len(self.transition_prob)#save calculated probs
            
            for (tProb, t2Prob) in self.transition_prob.keys():
                eProb = 0.001#really small numbers
                prevTProb = 0.001
                nextTProb = 0.001

                if (word, tProb) in self.emission_prob:
                    temp = self.emission_prob[(word, tProb)]
                    if temp > 0:
                        eProb = temp
                if (xt, tProb) in self.transition_prob:
                    temp = self.transition_prob[(xt, tProb)]
                    if temp > 0:
                        prevTProb = temp
                if (tProb, xt2) in self.transition_prob:
                    temp = self.transition_prob[(tProb, xt2)]
                    if temp > 0:
                        nextTProb = temp

                prob = 0
                if i == 0:#first word
                    #calc with just nextTProb
                    prob = eProb * nextTProb
                if i == len(sentence) - 1:#last word
                    #calc with just prevTProb
                    prob = eProb * prevTProb
                else:#all other words
                    #calc with both nextTProb and prevTProb
                    prob = eProb * nextTProb * prevTProb

                probs[j] = prob
                j += 1

            k = 0
            avgBase = sum(probs)
            #print(probs)
            sumAvg = 0
            rNum = random.random()#random percentage
            for p in probs:
                #normalize(average the probability)
                sumAvg += p / avgBase

                if rNum < sumAvg:#prob got to high, change pos
                    tpKeys = self.transition_prob.keys()
                    l = 0
                    for (key, value) in tpKeys:
                        if l == k:
                            pos_output[i] = key
                            l = len(tpKeys)
                            break
                        l += 1
                    break
                k += 1
            i += 1

        return pos_output
                    

    def complex_mcmc(self, sentence):
        #generate thousands of samples and check which occured most often
        #max(Pi = Si|W)
        #x(0)
        pos_output = ["noun"] * len(sentence)
        pos_outputs = []
        #t=1...T
        #first many samples (I chose 1000) should be ignores as they will be very off
        for sample in range(1000):
            pos_output = self.sample(pos_output, sentence)

        #now the values should be closer to the acutal answer
        for i in range(15000):
            pos_output = self.sample(pos_output, sentence)
            pos_outputs.append(pos_output)

        #return pos_outputs
        return ["noun"] * len(sentence)

    def hmm_viterbi(self, sentence):
        
        word = sentence
        pos = list(self.pos_prior_prob.keys())
        pos_output = [0] * len(word)
        viterbi = [0] * len(pos)
        # For the first word 
        for i in range(len(pos)):
            if (word[0], pos[i]) in self.emission_prob:
                viterbi[i] = self.initial_prob[pos[i]] * self.emission_prob[(word[0], pos[i])]
            else:
                viterbi[i] = self.initial_prob[pos[i]] * 0.0000000001
        if max(viterbi) == 0:
                pos_output[0] = "noun"
        else:
            pos_output[0] = pos[viterbi.index(max(viterbi))]

        # for second to last word
        for i in range(1, len(word)):
            v = viterbi
            temp = [0] * len(pos)
            # for each POS we iterate
            for j in range(len(pos)):
                if (word[i], pos[j]) in self.emission_prob:                   
                    for k in range(len(pos)):
                        temp[k] = v[k] * self.transition_prob[(pos[k], pos[j])]

                    viterbi[j] = self.emission_prob[(word[i], pos[j])] * max(temp)
                else:
                    viterbi[j] = 0.000000001 * max(temp)
            if max(viterbi) == 0:
                pos_output[i] = "noun"
            else:
                pos_output[i] = pos[viterbi.index(max(viterbi))]
        return pos_output
    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
