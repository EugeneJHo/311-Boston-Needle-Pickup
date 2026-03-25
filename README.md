# Boston 311 Needle Pickup Risk (Midterm Group Project)

This repository contains a midterm project analyzing **Boston 311 “Needle Pickup” service calls** (from Analyze Boston) as a proxy for opioid/drug-use risk and street-level harm. The goal is to build an ML-driven workflow that helps policy stakeholders **predict where/when risk is highest** to support **proactive resource allocation**.

## Decision context + research question

Needle pickups are a public health and city-operations concern. If we can anticipate high-need periods/areas, city teams can better target cleanup crews, outreach, and prevention resources.

**Research question:** *Can high-risk locations or periods be predicted in advance to support proactive resource allocation?*

## Approach (high level)

The main modeling workflow uses a **Random Forest classifier** to predict whether a **neighborhood-month** is **high-risk** for needle pickup activity.

- **Unit of analysis**: neighborhood × month (aggregated from request-level 311 data)
- **Target**: `high_risk` (binary), defined as neighborhood-months with needle pickup counts at/above the **75th percentile**
- **Evaluation**: time-based train/test split to mimic forward-looking deployment
- **Operating goal**: prioritize **high recall** (minimize false negatives) because missing a high-risk area is more costly than over-flagging

## Repository contents

- **Notebooks**
  - `data_processing.ipynb`: data loading + cleaning + building intermediate/master datasets
  - `eda.ipynb`: exploratory data analysis (EDA) to understand distributions, trends, and features
  - `modeling.ipynb`: feature engineering + Random Forest classification + threshold tuning + risk mapping
- **Data (CSV)**
  - `2018.csv`, `2019.csv`, `2020.csv`, `2021.csv`, `2022.csv`, `2024.csv`, `2025.csv`: raw-ish yearly extracts
  - `master_311.csv`: combined dataset
  - `master_311_clean.csv`: cleaned dataset used by `modeling.ipynb`
- **Docs**
  - `midterm_final_report.docx`: final written report
  - `Midterm Group Project Guidelines and Rubric.pdf`: assignment spec
  - `Proposal of Ideas/`: proposal drafts
  - `Technology/`: documentation/links

## Setup

### Requirements

- Python 3.10+ recommended
- Jupyter (Notebook or JupyterLab)

### Create an environment + install packages

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter contextily
```

Notes:
- `modeling.ipynb` imports `contextily` for basemap tiles; install it if you plan to run the map section.

## How to run

Open Jupyter from the repo root:

```bash
jupyter lab
```

Suggested notebook order:

1. `data_processing.ipynb` (build/clean master datasets)
2. `eda.ipynb` (explore patterns and sanity-check features)
3. `modeling.ipynb` (train classifier, tune threshold, and generate risk outputs/plots)

## Team

- Eugene
- Matthew
- Jonah

