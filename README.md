# PLIAR, a PLI-aligment resource

This repository provides code and data for the paper
<center>
<b><i>
Evaluation Beyond Goodness of Fit: Quantifying Biophysical Alignment of AI Models for Kinase‑Centric Drug Discovery
</i></b>
</center>

## Installation
We recommend using uv to install and run this code. To install uv, please follow the instructions at thttps://docs.astral.sh/uv/.

Installing the package and its dependencies can then be done by running
```bash
uv sync
```

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

For the remained of this section we will refer to these files as `clean.csv` and `masked.csv` respectively.

See the `data/predictions` folder for example files ;)

### Running PLI alignment evaluation
Given such data, computing PLI alignment ranking metrics is as simple as running the following command
```bash
uv run compute_alignment.py data/predictions/clean.csv data/predictions/masked.csv
```
If you want to evaluate your own models just replace the prediction files with your own.


## Citing this work
If you use this code in your research, please cite the following paper
```
TODO insert bibtex here
```
