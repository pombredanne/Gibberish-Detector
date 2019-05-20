#!/usr/bin/python

import pickle
import gib_detect_train

model_data = pickle.load(open('gib_model.pki', 'rb'))
f = open("default.txt", "r+")
default_words = f.read().splitlines()

while True:
    l = input()
    model_mat = model_data['mat']
    threshold = model_data['thresh']

    if l.lower() in default_words:
        print(True)
    else:
        # print(gib_detect_train.avg_transition_prob(l, model_mat))
        # print(threshold)
        print(gib_detect_train.avg_transition_prob(l, model_mat) > threshold)
