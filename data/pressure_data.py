#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Read pressure and elements

from qmpy import *
import csv

# Select pure alloy materials
elts = Element.objects.filter(symbol__in=element_groups['simple-metals'])
out_elts = Element.objects.exclude(symbol__in=element_groups['simple-metals'])
models = Calculation.objects.filter(path__contains='icsd')
models = models.filter(converged=True, label__in=['static', 'standard'])
models = models.exclude(composition__element_set=out_elts)

# Export the element and its pressure
f = open('MyDatasetOutput.csv','wb')
csv_writer = csv.writer(f)
csv_writer.writerow(["id","composition_id","o.sxx",
                     "o.syy","o.szz","o.sxy","o.syz","o.szx"])
for m in models:
    csv_writer.writerow([m.id,m.composition_id,m.output.stresses[0],
                         m.output.stresses[1], m.output.stresses[2],
                         m.output.stresses[3], m.output.stresses[4],
                         m.output.stresses[5],
                         ])
f.close()
exit()