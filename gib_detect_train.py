#!/usr/bin/python

import math
import json

accepted_chars = 'abcdefghijklmnopqrstuvwxyz '

pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

def normalize(line):
    """ Return only the subset of chars from accepted_chars.
    This helps keep the  model relatively small by ignoring punctuation, 
    infrequenty symbols, etc. """
    return [c.lower() for c in line if c.lower() in accepted_chars]

def ngram(n, l):
    """ Return all n grams from l after normalizing """
    for start in range(0, len(l) - n + 1):
        yield ''.join(l[start:start + n])

def avg(li):
    return (sum(li)/len(li))

def train():
    """ Write a simple model as a pickle file """
    k = len(accepted_chars)
    # Assume we have seen 10 of each character pair.  This acts as a kind of
    # prior or smoothing factor.  This way, if we see a character transition
    # live that we've never observed in the past, we won't assume the entire
    # string has 0 probability.
    counts = [[10 for i in range(k)] for i in range(k)]

    # Count transitions from big text file, taken 
    # from http://norvig.com/spell-correct.html
    for line in open('big.txt'):
        line = normalize(line)
        if len(line) <= 2:
            for a, b in ngram(2, line):
                counts[pos[a]][pos[b]] += 1
        else:
            for a, b, c in ngram(3, line):
                counts[pos[a]][pos[b]] += 1
                counts[pos[b]][pos[c]] += 1

    # Normalize the counts so that they become log probabilities.  
    # We use log probabilities rather than straight probabilities to avoid
    # numeric underflow issues with long texts.
    # This contains a justification:
    # http://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/
    for i, row in enumerate(counts):
        s = float(sum(row))
        for j in range(len(row)):
            row[j] = math.log(row[j] / s)

    with open('counts.txt', 'w') as f:
        for row in counts:
            for i in range(len(row)):
                f.write(str(row[i]) + ',') if i < len(row) - 1 else f.write(str(row[i]))
            f.write('\n')
        f.close()

    # Find the probability of generating a few arbitrarily choosen good and
    # bad phrases.
    good_probs = [avg_transition_prob(l, counts) for l in open('good.txt')]
    bad_probs = [avg_transition_prob(l, counts) for l in open('bad.txt')]
    # print(good_probs)
    # print(bad_probs)
    # print(avg(good_probs), avg(bad_probs))
    # Assert that we actually are capable of detecting the junk.
    # assert min(good_probs) > max(bad_probs)

    # And pick a threshold halfway between the worst good and best bad inputs.
    thresh = (avg(good_probs) + avg(bad_probs)) / 2
    json.dump({'thresh': thresh}, open('thresh.json', 'w'))

def avg_transition_prob(l, log_prob_mat):
    """ Return the average transition prob from l through log_prob_mat. """
    log_prob = 0.0
    transition_ct = 0
    l = normalize(l)
    if len(l) <= 2:
        for a, b in ngram(2, l):
            log_prob += log_prob_mat[pos[a]][pos[b]]
            transition_ct += 1
    else:
        for a, b, c in ngram(3, l):
            log_prob += log_prob_mat[pos[a]][pos[b]]
            log_prob += log_prob_mat[pos[b]][pos[c]]
            transition_ct += 2
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))

if __name__ == '__main__':
    train()
