# IPS and Splice dataset
## Parameters
4 parameters: dataset, budget, modeltype, time

dataset: 'Splice' (default), 'IPS'

budget: 5(default)

modeltype: 'Normal'(default), 'adversarial'

time: 60(default)



## Template

python OMPGSmax.py --dataset Splice --modeltype adversarial

python FSGSmax.py --dataset Splice --modeltype adversarial --time 60 --budget 5

