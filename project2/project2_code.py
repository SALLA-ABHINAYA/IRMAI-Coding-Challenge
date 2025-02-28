import streamlit as st
import pandas as pd
import numpy as np
from py2neo import Graph, Node
import matplotlib.pyplot as plt
import seaborn as sns

def connect_to_neo4j():
    """Connect to the Neo4j database."""
    return Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def store_trades(graph, df):
    """Store trades in the Neo4j database."""
    for _, row in df.iterrows():
        timestamp_iso = row['timestamp'].isoformat()
        trade_node = Node("Trade", timestamp=timestamp_iso, trade_volume=row['trade_volume'], trade_price=row['trade_price'])
        graph.merge(trade_node, "Trade", "timestamp")
    st.success("Trades stored in Neo4j database.")

def create_consecutive_relationships(graph, df):
    """Create relationships between consecutive trades."""
    for i in range(len(df) - 1):
        timestamp1, timestamp2 = df.iloc[i]['timestamp'].isoformat(), df.iloc[i + 1]['timestamp'].isoformat()
        graph.run("""
        MATCH (t1:Trade {timestamp: $timestamp1})
        MATCH (t2:Trade {timestamp: $timestamp2})
        MERGE (t1)-[:CONSECUTIVE]->(t2);
        """, parameters={"timestamp1": timestamp1, "timestamp2": timestamp2})
    st.success("Consecutive relationships created in Neo4j.")

def create_similar_relationships(graph, price_std_dev, volume_std_dev, time_window, max_similar_relations=25):
    """Create relationships between similar trades."""
    query = """
    MATCH (t1:Trade)
MATCH (t2:Trade)
WHERE t1 <> t2
  AND abs(t1.trade_price - t2.trade_price) < $similarity_threshold_price
  AND abs(t1.trade_volume - t2.trade_volume) < $similarity_threshold_volume
WITH t1, t2, abs(t1.trade_price - t2.trade_price) + abs(t1.trade_volume - t2.trade_volume) AS score
ORDER BY score ASC
WITH t1, collect(t2)[0..$max_similar_relations] AS similar_trades
UNWIND similar_trades AS t2
MERGE (t1)-[:SIMILAR]->(t2);

    """
    graph.run(query, parameters={
        "similarity_threshold_price": float(price_std_dev),
        "similarity_threshold_volume": float(volume_std_dev),
        "time_window": int(time_window),
        "max_similar_relations": int(max_similar_relations)
    })
    st.success("Similar relationships created in Neo4j.")

def calculate_thresholds(df):
    """Calculate thresholds for gap detection and deviations."""
    price_diff = df['trade_price'].diff().abs().dropna()
    gap_threshold = price_diff.mean() + 1.5 * price_diff.std()
    time_window = np.percentile(df['timestamp'].diff().dt.total_seconds().dropna(), 90) / 60
    expected_volume_range = (df['trade_volume'].quantile(0.01), df['trade_volume'].quantile(0.99))
    expected_price_range = (df['trade_price'].quantile(0.01), df['trade_price'].quantile(0.99))
    price_std_dev = df['trade_price'].std()
    volume_std_dev = df['trade_volume'].std()
    
    return gap_threshold, price_std_dev, volume_std_dev, time_window, expected_volume_range, expected_price_range

def detect_gaps(df, gap_threshold):
    """Detect gaps in trade prices."""
    return [(df.iloc[i]['timestamp'], abs(df.iloc[i]['trade_price'] - df.iloc[i - 1]['trade_price']))
            for i in range(1, len(df)) if abs(df.iloc[i]['trade_price'] - df.iloc[i - 1]['trade_price']) > gap_threshold]

def label_gaps(graph, gaps):
    """Label gaps in the Neo4j database."""
    for timestamp, _ in gaps:
        query = """
        MATCH (t:Trade {timestamp: $timestamp})
        SET t:Gap
        """
        graph.run(query, parameters={"timestamp": timestamp.isoformat()})
    st.success("Gaps labeled in Neo4j database.")

def find_deviations(df, expected_volume_range, expected_price_range):
    """Find deviations in trade volume and price."""
    deviations = df[
        (df['trade_volume'] < expected_volume_range[0]) | (df['trade_volume'] > expected_volume_range[1]) |
        (df['trade_price'] < expected_price_range[0]) | (df['trade_price'] > expected_price_range[1])
    ]
    return deviations

def visualize_trades(df, gap_data):
    """Visualize trade prices and highlight gaps."""
    st.subheader("Trade Price Visualization")
    plt.figure(figsize=(16, 8))
    plt.plot(df['timestamp'], df['trade_price'], label='Trade Price', color='blue', linewidth=1)
    if gap_data:
        gap_timestamps = [gap[0] for gap in gap_data]
        gap_prices = [df.loc[df['timestamp'] == gap[0], 'trade_price'].values[0] for gap in gap_data]
        plt.scatter(gap_timestamps, gap_prices, color='red', label='Gap Detected', marker='o', s=50, edgecolor='black')
    plt.title('Stock Trade Prices with Gaps Highlighted')
    plt.xlabel('Timestamp')
    plt.ylabel('Trade Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

def visualize_deviations(df, deviations):
    """Visualize deviations in trade prices."""
    st.subheader("Trade Price Deviations")
    plt.figure(figsize=(16, 8))
    sns.boxplot(x=df["trade_price"], color="blue", width=0.5)
    sns.stripplot(x=deviations["trade_price"], color="red", size=6, jitter=True, label="Deviations")
    plt.xlabel("Trade Price")
    plt.title("Box Plot of Trade Prices with Deviations")
    plt.legend()
    st.pyplot(plt)

def execute_gap_analysis(file):
    """Main function to execute the gap analysis."""
    df = load_data(file)
    st.success("Data Loaded Successfully")
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    gap_threshold, price_std_dev, volume_std_dev, time_window, expected_volume_range, expected_price_range = calculate_thresholds(df)
    
    graph = connect_to_neo4j()
    store_trades(graph, df)
    create_consecutive_relationships(graph, df)
    create_similar_relationships(graph, price_std_dev, volume_std_dev, time_window)
    gaps = detect_gaps(df, gap_threshold)
    label_gaps(graph, gaps)
    
    deviations = find_deviations(df, expected_volume_range, expected_price_range)
    
    st.subheader("Analysis Metrics")
    st.write(f"### Number of Gaps Identified: {len(gaps)}")
    st.write(f"### Number of Deviations Identified: {len(deviations)}")
    
    visualize_trades(df, gaps)
    visualize_deviations(df, deviations)

# Streamlit UI
st.title("Stock Trade Gap Analysis ")
st.write("Upload a CSV file containing stock trade data.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
        execute_gap_analysis(uploaded_file)