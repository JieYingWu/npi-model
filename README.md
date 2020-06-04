This repository is a framework to model the effect of non-pharmaceutical interventions on the reproductive rate of COVID-19 in the US on a county level. We cluster counties and based on features known to affect infectious disease spread and fit a model to the clusters. The model is based on a Python port of the model proposed by report 13 from the MRC Centre for Global Infectious Disease Analysis, Imperial College London. The original code is available at: https://github.com/ImperialCollegeLondon/covid19model. Below is an overview of our model.

![](https://github.com/JieYingWu/npi-model/blob/master/visualizations/model.png)


We provide...
* Support to run model on the US state and county level 
* Fit clusters of relevant features and model the clusters and supercounties 
* Code to visualize the model fit with confidence intervals
* Plotting tools of how US interventions have affected modeled Rt


The figures below show our cluster on US counties and how their reproductive rate changes over time. 
![](https://github.com/JieYingWu/npi-model/blob/master/visualizations/us_clustering_final.PNG)
![](https://github.com/JieYingWu/npi-model/blob/master/visualizations/transit.png)


Words of warning: We have noticed that the time series of the reported cases and deaths are of inconsistent quality; for example, our cumulative time series of cases and death counts are not monotonically increasing (so, checks have been implemented for the same). For days where the case or death count is negative, we offer either interpolation or filtering out negative values or countie

Additionally, fits to the US-county level data is less certain than the European Countries considered by the original model as there are fewer cases in most counties than any of the European countries. We also provide state-level analysis for more similar comparison. 


## To Run
For US data, call `python scripts/main.py` in the base directory and it will save the summary over runs in the results folder.
Use flag `--supercounties` to group counties with too few cases to model by themselves with similar features
Use flag `--cluster {0-4}` to model only counties or supercounties that fall into a cluster



## How to plot:
- To create "Relation of Reproductive Rate over Transit score, Density, Median Income" plots navigate to scripts folder and run  `python3 plot_rt_over_density.py`
- To create the validation plots (figure 7) run `python3 scripts/ValidationResult.py --results-path results/national_no_supercounty_no_validation results/national_no_supercounty_validation `

## Dependencies
* pystan (this requires Cython compiler - https://pystan.readthedocs.io/en/latest/installation_beginner.html)
Pystan is only partially supported on Windows: https://pystan.readthedocs.io/en/latest/windows.html **(Tip: Use Anaconda!!)**

## Citation

If you find our dataset or code useful, please consider citing our [paper](https://arxiv.org/abs/2004.00756):
```latex
@article {wuChanges2020,
	author = {Wu, Jie Ying and Killeen, Benjamin D and Nikutta, Philipp and Thies, Mareike and Zapaishchykova, Anna and Chakraborty, Shreya and Unberath, Mathias},
	title = {Changes in Reproductive Rate of SARS-CoV-2 Due to Non-pharmaceutical Interventions in 1,417 U.S. Counties},
	elocation-id = {2020.05.31.20118687},
	year = {2020},
	doi = {10.1101/2020.05.31.20118687},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2020/06/02/2020.05.31.20118687},
	eprint = {https://www.medrxiv.org/content/early/2020/06/02/2020.05.31.20118687.full.pdf},
	journal = {medRxiv}
}
```

## Acknowledgements
This framework was constructed by a group of students led by Mathias Unberath at Johns Hopkins University. Special thanks goes to Jie Ying Wu, Benjamin Killeen, Philipp Nikutta, Mareike Thies, Anna Zapaishchykova and Shreya Chakraborty.
