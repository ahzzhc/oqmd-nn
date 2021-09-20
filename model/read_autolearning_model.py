#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Read automatic machine learning model

import deepdish as dd

# ****.h5:Model derived from automatic machine learning model
dataset = "****.h5"
def load_h5(file_path):
    mean_val = dd.io.load(file_path)
    print(mean_val)
load_h5(dataset)

