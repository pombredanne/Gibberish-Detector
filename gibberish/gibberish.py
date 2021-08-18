#!/usr/bin/python
#
# 12Jun2017 Petr Janata - added srcfile and outfile 
# 17Jun2107 Petr Janata - expanded set of accepted characters to include digits and hyphen

import math
import pickle
import os

accepted_chars = 'abcdefghijklmnopqrstuvwxyz0123456789- '

pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

if 'GIBBERISH_DATA_DIR' in os.environ:
    data_dir  = os.environ['GIBBERISH_DATA_DIR'] + '/'
else:
    data_dir =  os.path.dirname(os.path.abspath(__file__)) + '/data/'

model_path = data_dir + '/gib_model.pki'


class Gibberish(object):
    def __init__(self):
        self.train_if_necessary()

    def train_if_necessary(self):
        if not os.path.isfile(model_path):
            self.train()
        else:
            self.load_persisted_model()

    def persist_model(self):
        with open(model_path, 'wb') as f:
            pickle.dump(vars(self), f)

    def load_persisted_model(self):
        with open(model_path, 'rb') as f:
            persisted_model = pickle.load(f)
            for key, value in persisted_model.items():
                setattr(self, key, value)

    def normalize(self, line):
        """ Return only the subset of chars from accepted_chars.
        This helps keep the  model relatively small by ignoring punctuation, 
        infrequenty symbols, etc. """
        return [c.lower() for c in line if c.lower() in accepted_chars]

    def ngram(self, n, l):
        """ Return all n grams from l after normalizing """
        filtered = self.normalize(l)
        for start in range(0, len(filtered) - n + 1):
            yield ''.join(filtered[start:start + n])

    def avg_transition_prob(self, l, log_prob_mat):
        """ Return the average transition prob from l through log_prob_mat. """
        log_prob = 0.0
        transition_ct = 0
        for a, b in self.ngram(2, l):
            log_prob += log_prob_mat[pos[a]][pos[b]]
            transition_ct += 1
        # The exponentiation translates from log probs to probs.
        return math.exp(log_prob / (transition_ct or 1))

    def train(self, bigfile=data_dir + 'big.txt', goodfile=data_dir + 'good.txt',
              badfile=data_dir + 'bad.txt'):
        """ Write a simple model as a pickle file """
        k = len(accepted_chars)
        # Assume we have seen 10 of each character pair.  This acts as a kind of
        # prior or smoothing factor.  This way, if we see a character transition
        # live that we've never observed in the past, we won't assume the entire
        # string has 0 probability.
        counts = [[10 for i in range(k)] for i in range(k)]

        # Count transitions from big text file, taken
        # from http://norvig.com/spell-correct.html
        for line in open(bigfile):
            for a, b in self.ngram(2, line):
                counts[pos[a]][pos[b]] += 1

        # Normalize the counts so that they become log probabilities.
        # We use log probabilities rather than straight probabilities to avoid
        # numeric underflow issues with long texts.
        # This contains a justification:
        # http://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/
        for i, row in enumerate(counts):
            s = float(sum(row))
            for j in range(len(row)):
                row[j] = math.log(row[j] / s)

        # Find the probability of generating a few arbitrarily choosen good and
        # bad phrases.
        good_probs = [self.avg_transition_prob(l, counts) for l in open(goodfile)]
        bad_probs = [self.avg_transition_prob(l, counts) for l in open(badfile)]

        # Assert that we actually are capable of detecting the junk.
        assert min(good_probs) > max(bad_probs)

        # And pick a threshold halfway between the worst good and best bad inputs.
        thresh = (min(good_probs) + max(bad_probs)) / 2
        self.mat = counts
        self.thresh = thresh
        self.persist_model()

    def detect_gibberish(self, text):

        text = ''.join(self.normalize(text))

        return self.avg_transition_prob(text, self.mat) < self.thresh

    def percent_gibberish(self, text):
        text = ''.join(self.normalize(text))
        text = text.strip()
        words = text.split(' ')
        if len(words) == 0:
            return 0

        gibberish_count = 0
        for word in words:
            if self.detect_gibberish(word):
                gibberish_count += 1

        return float(gibberish_count) / float(len(words))

    def gibberish_pct(self, text):
        text = ''.join(self.normalize(text))
        return self.avg_transition_prob(text, self.mat)
