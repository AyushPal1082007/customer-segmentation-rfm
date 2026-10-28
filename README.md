# 🛍️ E-Commerce Customer Segmentation using RFM, K-Means & DBSCAN

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red)
![License](https://img.shields.io/badge/License-MIT-green)

> End-to-end customer segmentation pipeline on real e-commerce transactions.
> Combines **RFM analysis**, **K-Means**, **DBSCAN**, **PCA**, **t-SNE/UMAP**, and **Isolation Forest** to uncover actionable marketing personas from raw transaction data.

---

## 📖 Project Overview

Online retailers generate millions of transactions but rarely know *who* their customers really are. This project applies unsupervised learning to segment customers based on purchase behavior, enabling:

- 🎯 **Targeted marketing** — send the right offer to the right customer
- 💰 **Retention** — spot churn risks before they leave
- ⭐ **VIP identification** — reward your best customers
- 📊 **Product strategy** — understand which segments buy what

**Business question:** *How can we segment customers to increase retention, revenue, and marketing ROI?*

---

## 📂 Dataset

**Online Retail II** — UCI Machine Learning Repository
📎 [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) · [UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

| Property | Value |
|---|---|
| Rows (raw) | ~1,067,000 |
| Rows (cleaned) | ~780,000 |
| Unique customers | ~5,900 |
| Time period | Dec 2009 – Dec 2011 |
| Country | United Kingdom (mostly) |

**Columns:** `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`

---

## 🔬 Methodology — 10-Step Pipeline
Raw Transactions
│
▼

Data Cleaning → drop nulls, returns, cancellations, £0 prices

RFM Aggregation → Recency, Frequency, Monetary per customer

Log + Scale → reduce skew, standardize features

PCA / Inc. PCA → dimensionality reduction

K-Means → elbow + silhouette to choose k

DBSCAN / OPTICS → density-based clustering (finds outliers)

Hierarchical → Ward linkage + dendrogram

Isolation Forest → anomaly detection

t-SNE / UMAP → 2D visualization of clusters

Profiles + Personas → business-ready segments

text

### Key RFM definitions

| Metric | Formula |
|---|---|
| **Recency** | Days since customer's last purchase |
| **Frequency** | Number of unique invoices |
| **Monetary** | Sum of `Quantity × UnitPrice` |
| **AOV** | `Monetary / Frequency` |

---

## 📊 Results

### Customer Segments (K-Means)

| Cluster | Persona | Customers | Avg Recency | Avg Frequency | Avg Monetary |
|---|---|---|---|---|---|
| 0 | ⭐ VIP / High-value loyal | — | — days | — | £— |
| 1 | 🌱 Recent high-potential | — | — days | — | £— |
| 2 | ⚠️ At-risk / churn | — | — days | — | £— |
| 3 | 😐 Regular / mid-value | — | — days | — | £— |

### Cluster Visualizations

| PCA 2D | t-SNE | UMAP |
|---|---|---|
| ![PCA](images/pca.png) | ![t-SNE](images/tsne.png) | ![UMAP](images/umap.png) |

### Anomaly Detection (Isolation Forest)

- Flagged a small percentage of customers as anomalies (~2%)
- These are typically extremely high spenders or unusual buying patterns
- Recommended action: **manual review** — not marketing automation

### DBSCAN vs K-Means — Key finding

> DBSCAN collapsed most customers into a single large cluster with many outliers. **K-Means was the better choice** for this dataset because customers form dense, overlapping groups rather than distinct density regions.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data** | pandas, numpy |
| **ML / Clustering** | scikit-learn |
| **Dimensionality Reduction** | PCA, IncrementalPCA, t-SNE, UMAP |
| **Anomaly Detection** | Isolation Forest |
| **Visualization** | matplotlib, seaborn, plotly |
| **Notebook** | Jupyter / Kaggle |

---

## 📁 Project Structure
customer-segmentation/
│
├── data/
│ └── online_retail_II.csv # downloaded from Kaggle
│
├── notebooks/
│ └── rfm_customer_segmentation.ipynb # main notebook
│
├── outputs/
│ ├── rfm_clustered.csv # customer-level results
│ └── cluster_personas.csv # cluster summaries
│
├── images/
│ ├── pca.png
│ ├── tsne.png
│ ├── umap.png
│ ├── elbow.png
│ └── dendrogram.png
│
├── requirements.txt
├── README.md
└── .gitignore

text

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/YOUR-USERNAME/customer-segmentation.git
cd customer-segmentation
2. Create a virtual environment
bash
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows
3. Install dependencies
bash
pip install -r requirements.txt
requirements.txt:

text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
umap-learn>=0.5
jupyter
4. Download the dataset
Place online_retail_II.csv in the data/ folder.
📎 Download from Kaggle
