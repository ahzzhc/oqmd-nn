#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Build training data based on composition and pressure

from qmpy import *
import numpy as np
import pandas as pd

# Select pure alloy material
elts = Element.objects.filter(symbol__in=element_groups['simple-metals'])
out_elts = Element.objects.exclude(symbol__in=element_groups['simple-metals'])
models = Calculation.objects.filter(path__contains='icsd')
models = models.filter(converged=True, label__in=['static', 'standard'])
models = models.exclude(composition__element_set=out_elts)


data_stress = pd.read_csv('MyDatasetOutput.csv', header=0,usecols = ["o.sxx","o.syy","o.szz",
                                                           "o.sxy","o.syz","o.szx"])
data_stress = np.array(data_stress)
data_element = pd.read_csv('metallic_element.csv', header=0,usecols = ["elements"])
data_element = np.array(data_element)
data = models.values_list('composition_id','output__volume_pa')
X1=[] # fully elements
y1=[] # volume
for c,v in data:
    if v != None :
        X1.append(get_composition_descriptors(c))
        y1.append(v)

# Remove excess non-metallic elements feature
for index in range(len(X1)):
    dict=X1[index]
    for key,value in dict.items():
        if key not in data_element:
            del dict[key]
X2=[] # metallic element
for i in range(len(X1)):
    dict_value = X1[i]
    X2.append(dict_value.values())

# save data
X3=np.concatenate((data_stress,X2),axis=1)
np.savez('volume.npz', X=X3, y=y1)