# IRMAI Coding Challenge

This repository contains three graph-based anomaly detection projects focused on analyzing stock trades and financial transactions to identify outliers, missing trades, and potential risks.

## Projects Overview

### Project 1: Stock Trade Outlier Analysis
This project analyzes stock trade data using Plotly Dash to detect potential outliers and visualize insights for better decision-making.

#### Features
- Interactive dashboard for visualizing stock trade data
- Outlier detection to highlight potential anomalies
- Data filtering based on different criteria
- Real-time updates for dynamic graph visualization

### Project 2: Stock Trade Gap Analysis 
This project identifies stock trade price gaps and deviations using Neo4j for relationship modeling and Streamlit for visualization.

#### Features
- Loads stock trade data from a CSV file
- Stores trades as nodes in Neo4j
- Establishes consecutive and similar trade relationships
- Detects significant price gaps and deviations
- Visualizes trade anomalies and patterns

### Project 3: FMEA Analysis for Financial Transactions
This project applies Failure Mode and Effect Analysis (FMEA) to financial transactions to identify high-risk cases, such as unusually large amounts, and flags them for review.

#### Features
- Loads and analyzes financial transaction data from a CSV file
- Stores transactions as nodes in Neo4j
- Establishes relationships between consecutive transactions
- Identifies and classifies high-amount transactions as failure modes
- Visualizes flagged transactions using Matplotlib
- Generates an FMEA report summarizing identified risks

## Technologies Used
- Python
- Pandas
- NumPy
- Plotly
- Dash (Plotly Dash)
- Streamlit
- Neo4j
- Py2Neo
- Matplotlib
