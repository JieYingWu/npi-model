This repository is a Python port of the model proposed by report 13 from the MRC Centre for Global Infectious Disease Analysis, Imperial College London. The origin code in R is available at: https://github.com/ImperialCollegeLondon/covid19model 

## Dependencies
* pystan (this requires Cython compiler - https://pystan.readthedocs.io/en/latest/installation_beginner.html)

## To Run
For US data, call `python scripts/base.py data US` in the base directory and it will save the summary over runs in the results folder.
Replace US with europe to reproduce results from Imperial College. 
