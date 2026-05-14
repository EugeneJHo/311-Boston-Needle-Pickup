# Boston 311 Needle Pickup Risk

This repository contains a project analyzing **Boston 311 “Needle Pickup” service calls** as a proxy for opioid/drug-use risk and street-level public health harm. The goal is to build an ML-driven workflow that helps policy stakeholders **predict where and when needle pickup activity is likely to be highest**, supporting more proactive cleanup, outreach, and resource allocation.

The project was extended beyond the original Random Forest workflow by incorporating **neural network models**, including a **Temporal Convolutional Neural Network (Temporal CNN)** for weekly spike detection and **TabNet** for tabular high-risk classification. The final analysis treats prediction as a decision-support problem: the model is not meant to automate response, but to help city teams prioritize neighborhoods and time periods for intervention. :contentReference[oaicite:0]{index=0}

## Decision context + research question

Needle pickups are both a public health and city-operations concern. If high-need neighborhoods or time periods can be anticipated in advance, city agencies can better target cleanup crews, outreach workers, and prevention resources.

**Research question:** *Can publicly available Boston 311 data be used to predict high-risk locations or periods in advance to support proactive resource allocation?*

## Approach (high level)

The project includes two related modeling tracks:

### 1. High-risk neighborhood-month classification

The original modeling workflow uses a **Random Forest classifier** to predict whether a **neighborhood-month** is high-risk for needle pickup activity.

- **Unit of analysis**: neighborhood × month
- **Target**: `high_risk`, defined using the 75th percentile of needle pickup counts
- **Baseline model**: Random Forest classifier
- **Operating goal**: prioritize high recall, since missing a high-risk area is more costly than over-flagging

The project was later expanded to compare additional models, including:

- XGBoost
- CatBoost
- TabNet neural network
- Soft-voting and hard-voting ensemble models

The soft-voting ensemble performed well as a policy-facing model because it combined strengths across multiple classifiers and supported neighborhood-level risk mapping.

### 2. Weekly spike detection

The extended project also adds a **Temporal CNN** to predict short-term weekly spikes in needle pickup activity.

- **Unit of analysis**: neighborhood × week
- **Target**: weekly spike indicator based on an 8-week rolling average and rolling standard deviation
- **Model**: Temporal CNN using each neighborhood’s recent 8-week history
- **Evaluation**: time-based train/test split, with 2024–2025 used as the holdout period
- **Policy focus**: balance spike detection against alert volume

The Temporal CNN improved spike detection compared with earlier models, but also showed that alert thresholds must be treated as an operational policy choice. Higher recall catches more true spikes, but generates more alerts; lower alert volume reduces workload but misses more spike events.

## Key findings

- Boston 311 needle pickup data contains useful predictive signal for identifying high-need neighborhoods.
- Random Forest provided a strong initial baseline for high-risk classification.
- Neural network extensions added two new capabilities:
  - **Temporal CNN**: better suited for detecting short-term weekly spike patterns.
  - **TabNet**: useful as a neural network model for tabular high-risk classification.
- The soft-voting ensemble provided a strong overall approach for high-risk classification.
- Threshold tuning is central to deployment because it controls the trade-off between missed high-risk events and operational alert burden.
- The model is best framed as a **decision-support tool** for proactive cleanup and outreach, not as a fully automated trigger.

## Repository contents

- **Notebooks**
  - `data_processing.ipynb`: data loading, cleaning, and building master datasets
  - `eda.ipynb`: exploratory data analysis to understand distributions, trends, and spatial patterns
  - `modeling.ipynb`: Random Forest classification, threshold tuning, and risk mapping
  - Neural network / final modeling notebooks: Temporal CNN, TabNet, boosted models, and ensemble analysis

- **Data (CSV)**
  - `2018.csv`, `2019.csv`, `2020.csv`, `2021.csv`, `2022.csv`, `2024.csv`, `2025.csv`: yearly 311 extracts
  - `master_311.csv`: combined dataset
  - `master_311_clean.csv`: cleaned dataset used for modeling

- **Docs**
  - `midterm_final_report.docx`: original midterm report
  - `Final Project Report.pdf`: final project report with neural network extension
  - `Midterm Group Project Guidelines and Rubric.pdf`: assignment spec
  - `Proposal of Ideas/`: proposal drafts
  - `Technology/`: documentation/links

## Setup

### Requirements

- Python 3.10+ recommended
- Jupyter Notebook or JupyterLab

### Create an environment + install packages

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter contextily
