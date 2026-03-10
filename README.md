# PLIAR, a PLI-aligment resource

This repository provides code and data for the paper
<center>
<b><i>
Evaluation Beyond Goodness of Fit: Quantifying Biophysical Alignment of AI Models for Kinase‑Centric Drug Discovery
</i></b>
</center>


Note that model training and inference was conducted using [the kinodata3D codebase](https://github.com/volkamerlab/kinodata-3D-affinity-prediction).
This repository only contains code to evaluate Protein-Ligand Interaction (PLI) alignment of already trained models whose predictions are provided in the required format.

## Installation
### Execution Environment
We recommend using uv to install and run our code. To install uv, please follow the instructions at https://docs.astral.sh/uv/.

Installing the package and its dependencies can then be done by running
```bash
uv sync
```
### Data Acquisition
Running our alignment scripts requires reference explanation data. 
You can either [download this data here](https://zenodo.org/records/17488593) and extract it manually or run
```
uv run obtain_data.py https://zenodo.org/records/17488593 .
```
for an automated version.

## Usage
### Model evaluation prerequisites
To be able to evaluate PLI alignment of any model you need to provide two CSV files with the following schemata
1. **Predictions for clean structures**

| Column Name       | Type    | Description                                                                        |
|-------------------|---------|------------------------------------------------------------------------------------|
| `activity_id`     | `int`   | ChEMBL activity ID reported in kinodata3D (unique ID for a kinase-ligand complex)  |
| `predicted_value` | `float` | Corresponding, continuous model prediction, e.g. a class logit or regressor output |

2. **Predictions for masked structures**

| Column Name            | Type    | Description                                                               |
|------------------------|---------|---------------------------------------------------------------------------|
| `activity_id`          | `int`   | Same as above                                                             | 
| `predicted_value`      | `float` | Same type of model prediction as above but for the masked input structure | 
| `masked_residue_index` | `int`   | KLIFS index of the residue that was masked in the input structure         |

See the `data/example` folder for example files.

### Running PLI alignment evaluation
Given such data, computing PLI alignment ranking metrics is as simple as running the following command
```bash
uv run run_pliar.py data/example/clean.csv data/example/masked.csv
```
If you want to evaluate your own models just replace the prediction files with your own.

The evaluation method can also be called programmatically, see e.g. [run_pliar.ipynb](notebooks/run_pliar.ipynb) for an example usage in a Jupyter notebook.

This script will produce csv files with the attribution rankings and auroc metrics.

### Reproducing paper results
To reproduce paper results you will need to download additional data from Zenodo: 
This includes processed PLIP interaction data as well as model predictions for the models evaluated in the paper.
Either download and extract the data manually or run
```
uv run obtain_data.py https://zenodo.org/records/17488593 .
```
for an automated version.


## Citing this work
If you use this code in your research, please cite the following paper
```
TODO insert bibtex here
```
