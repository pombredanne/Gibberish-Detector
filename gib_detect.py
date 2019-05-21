#!/usr/bin/python

import json
import gib_detect_train

# Matrix
mat_data = open('counts.txt', 'r+')
model_mat = [[float(count) for count in row.split(',')] for row in mat_data]

# Threshold
thresh_data = json.load(open('thresh.json', 'r'))
threshold = thresh_data['thresh']

# Default Non-gibberish Words
f = open("default.txt", "r+")
default_words = f.read().splitlines()

while True:
    l = input()

    if l.lower() in default_words:
        print(True)
    else:
        # print(gib_detect_train.avg_transition_prob(l, model_mat))
        # print(threshold)
        print(gib_detect_train.avg_transition_prob(l, model_mat) > threshold)
    print()