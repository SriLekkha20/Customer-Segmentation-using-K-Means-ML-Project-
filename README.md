# Customer Segmentation using K-Means 👥

This project uses **K-Means clustering** to group customers into segments based on
demographic and behavioral attributes.

## Features

- Loads customer data from CSV
- Scales numerical features
- Uses K-Means to discover customer clusters
- Saves cluster assignments back to a new CSV
- Generates visualizations (Income vs Spending Score per cluster)

## Example Use Cases

- Targeted marketing campaigns
- Loyalty program design
- Product recommendations by segment

## Dataset

Expected columns in `data/customers.csv` (you can adjust):

- `CustomerID`
- `Gender`
- `Age`
- `AnnualIncome`
- `SpendingScore`

## Installation

```bash
git clone https://github.com/<your-username>/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
pip install -r requirements.txt
