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
data_id = pd.read_csv('MyDatasetOutput.csv', header=0,usecols = ["id"])
data_id = np.array(data_id)
data = models.values_list('id','composition_id',"formationenergy")
y1=[] # formationenergy data
X1=[] # stress data
X2=[] # composition data
for id,c,f in data:
    for i in range(len(data_id)):
        if id == data_id[i] and f != None:
            y1.append(FormationEnergy.objects.get(id = f).delta_e)
            X2.append(get_composition_descriptors(c))
            X1.append(data_stress[i])

# Remove excess non-metallic elements feature
for index in range(len(X1)):
    dict=X2[index]
    for key,value in dict.items():
        if key not in data_element:
            del dict[key]
X3=[] # metallic element
for i in range(len(X1)):
    dict_value = X2[i]
    X3.append(dict_value.values())

# save data
X4=np.concatenate((X1,X3),axis=1)
np.savez('formationenergy.npz', X=X4, y=y1)