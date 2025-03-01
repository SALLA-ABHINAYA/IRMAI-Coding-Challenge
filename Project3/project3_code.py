#Project 3: Failure Mode and Effect Analysis (FMEA) for Financial Transactions

import pandas as pd
import numpy as np
from py2neo import Graph, Node, Relationship
import matplotlib.pyplot as plt
import streamlit as st

def load_transaction_data(file):
    """Load transaction data from a CSV file."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def connect_to_neo4j():
    """Establish connection to Neo4j database."""
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
    st.success("Connected to Neo4j Successfully")
    return graph

def store_transactions(graph, transactions_df):
    """Store transaction data as nodes in Neo4j."""
    graph.delete_all()
    for _, row in transactions_df.iterrows():
        transaction_node = Node("Transaction", id=row['transaction_id'], amount=row['amount'],
                                type=row['transaction_type'], timestamp=row['timestamp'])
        graph.create(transaction_node)
    st.success("Created nodes in Neo4j Database")

def create_relationships(graph, transactions_df):
    """Create relationships between consecutive transactions."""
    for i in range(len(transactions_df) - 1):
        tx1 = graph.nodes.match("Transaction", id=transactions_df.iloc[i]['transaction_id']).first()
        tx2 = graph.nodes.match("Transaction", id=transactions_df.iloc[i + 1]['transaction_id']).first()
        relationship = Relationship(tx1, "NEXT", tx2)
        graph.create(relationship)
    st.success("NEXT relationship created")

def assign_failure_modes(graph):
    """Identify high-amount transactions as failure modes in Neo4j."""
    failure_query = """
    MATCH (t:Transaction)
    WHERE t.amount > 5000000
    MERGE (f:FailureMode {type: 'High Amount', effect: 'Potential Fraud'})
    MERGE (t)-[:HAS_FAILURE]->(f)
    RETURN t, f
    """
    graph.run(failure_query)
    st.success("Failure Modes assigned in Neo4j.")

def get_failure_modes(graph):
    """Retrieve transactions with failure modes."""
    verify_query = """
    MATCH (t:Transaction)-[:HAS_FAILURE]->(f:FailureMode)
    RETURN t.transaction_id, t.amount, f.type, f.effect
    """
    return graph.run(verify_query).data()

def perform_fmea_analysis(transactions_df):
    """Perform FMEA analysis on transactions."""
    failure_df = transactions_df[transactions_df["amount"] > 5000000].copy()
    failure_df["failure_mode"] = "High Transaction Amount"
    failure_df["effect"] = "Risk of Fraud or Regulatory Concern"
    return failure_df

def visualize_transactions(transactions_df, fmea_results):
    """Plot transactions and highlight failure modes."""
    plt.figure(figsize=(16, 8))
    plt.scatter(transactions_df.index, transactions_df['amount'], 
                color='lightblue', label='Normal Transactions', s=80, alpha=0.6)
    
    if not fmea_results.empty:
        failure_tx_ids = transactions_df[transactions_df['transaction_id'].isin(fmea_results['transaction_id'])].index
        failure_tx_amounts = transactions_df.loc[failure_tx_ids, 'amount']
        
        plt.scatter(failure_tx_ids, failure_tx_amounts, 
                    color='red', label='Failure Modes', s=150, edgecolors='black', alpha=0.8)

        for i, txt in enumerate(fmea_results['transaction_id']):
            if i % 3 == 0:
                plt.annotate(txt, (failure_tx_ids[i], failure_tx_amounts.iloc[i]),
                             textcoords="offset points", xytext=(0,10), ha='center',
                             fontsize=9, color='black', weight='bold')
    
    plt.title("Financial Transactions with Failure Modes Highlighted", fontsize=14)
    plt.xlabel("Transaction ID", fontsize=12)
    plt.ylabel("Transaction Amount", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks([], [])
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(plt)

def generate_report(transactions_df, fmea_results):
    """Generate an FMEA report summary."""
    report = "FMEA Report Summary\n" + "=" * 30 + "\n"
    report += f"Total Transactions Analyzed: {len(transactions_df)}\n"
    report += f"Total Failure Modes Identified: {len(fmea_results)}\n\n"
    
    if not fmea_results.empty:
        report += "Details of Failure Modes:\n" + "-" * 30 + "\n"
        for _, row in fmea_results.iterrows():
            report += f"Transaction ID: {row['transaction_id']}, Failure Mode: {row['failure_mode']}, Effect: {row['effect']}\n"
    else:
        report += "No failure modes identified.\n"
    
    return report

def execute_fmea_pipeline(file):
    """Main function to execute the process."""
    transactions_df = load_transaction_data(file)
    if transactions_df is not None:
        graph = connect_to_neo4j()
        
        store_transactions(graph, transactions_df)
        create_relationships(graph, transactions_df)
        assign_failure_modes(graph)
        failure_results = get_failure_modes(graph)
        
        fmea_results = perform_fmea_analysis(transactions_df)
        visualize_transactions(transactions_df, fmea_results)
        
        fmea_report = generate_report(transactions_df, fmea_results)
        st.text(fmea_report)

# Streamlit UI
st.title("FMEA Analysis for Financial Transactions")
st.write("Upload a CSV file containing transaction data.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    execute_fmea_pipeline(uploaded_file)
