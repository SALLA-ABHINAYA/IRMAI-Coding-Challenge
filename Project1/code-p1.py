#Stock Trade Outlier Analysis

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from py2neo import Graph, Node, Relationship
from scipy import stats
import dash
from dash import dcc, html
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Generate synthetic data
def generate_synthetic_data(n=1000):
    trade_dates = []
    while len(trade_dates) < n:
        random_days = random.randint(0, 273)
        potential_date = datetime(2024, 1, 1) + timedelta(days=random_days)
        if potential_date.weekday() < 5:  # 0 = Monday, ..., 4 = Friday (exclude 5,6)
            trade_dates.append(potential_date)
    data = {
        'trade_id': [f'TRADE_{i}' for i in range(n)],
        'currency_pair': ['EUR/USD'] * n,
        'trade_volume': np.random.normal(loc=100000, scale=20000, size=n).clip(0),
        'price': np.random.normal(loc=1.1, scale=0.05, size=n).clip(0),
        'timestamp': trade_dates
    }
    return pd.DataFrame(data)
#Save to csv file
trades_df = generate_synthetic_data()
trades_df.to_csv('synthetic_fx_trades.csv', index=False)

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
print("Connected to Neo4j Successfully")

# Insert trades into Neo4j
def insert_trades_to_neo4j(df):
    for _, row in df.iterrows():
        trade_node = graph.nodes.match("Trade", id=row['trade_id']).first()
        if not trade_node:
            trade_node = Node("Trade", 
                              id=row['trade_id'], 
                              volume=row['trade_volume'], 
                              price=row['price'], 
                              timestamp=row['timestamp'],
                              is_outlier=False)
            graph.create(trade_node)
insert_trades_to_neo4j(trades_df)
print("Trade nodes created in Neo4j")

# Create relationships for consecutive trades
def create_consecutive_relationships(df):
    for i in range(len(df) - 1):
        trade1 = graph.nodes.match("Trade", id=df.iloc[i]['trade_id']).first()
        trade2 = graph.nodes.match("Trade", id=df.iloc[i + 1]['trade_id']).first()
        if trade1 and trade2:
            rel = Relationship(trade1, "NEXT", trade2)
            graph.create(rel)
create_consecutive_relationships(trades_df)
print("Consecutive trade relationships created")


# Calculate Z-scores for outlier detection
trades_df['z_score'] = np.abs(stats.zscore(trades_df['trade_volume']))
# Identify outliers (Z-score > 2)
outliers = trades_df[trades_df['z_score'] > 2]
# Mark outliers in Neo4j
def mark_outliers_in_neo4j(df):
    for _, row in df.iterrows():
        trade_node = graph.nodes.match("Trade", id=row['trade_id']).first()
        if trade_node:
            trade_node.add_label("Outlier")
            trade_node['is_outlier'] = True
            graph.push(trade_node)
mark_outliers_in_neo4j(outliers)
print("Outlier label added to existing Trade nodes in Neo4j")

# Store z_score in DataFrame
trades_df['z_score'] = np.abs(stats.zscore(trades_df['trade_volume']))
# Update trades in Neo4j to include z_score
def update_z_scores_in_neo4j(df):
    for _, row in df.iterrows():
        trade_node = graph.nodes.match("Trade", id=row['trade_id']).first()
        if trade_node:
            trade_node['z_score'] = row['z_score']
            graph.push(trade_node)
update_z_scores_in_neo4j(trades_df)
print("Z-score updated in Neo4j")

#create similar relationships
def create_similar_trade_relationships_cypher(graph, price_threshold=0.02, volume_threshold=0.1, z_score_threshold=1.0, time_window=7):
    """
    Uses Cypher to identify and create SIMILAR relationships in Neo4j.
    """
    query = """
    MATCH (t1:Trade), (t2:Trade)
    WHERE t1.id <> t2.id  // Avoid self-matching
      AND abs(t1.price - t2.price) / t1.price < $price_threshold
      AND abs(t1.volume - t2.volume) / t1.volume < $volume_threshold
      AND abs(t1.z_score - t2.z_score) < $z_score_threshold
      AND duration.inDays(datetime(t1.timestamp), datetime(t2.timestamp)).days <= $time_window
    MERGE (t1)-[:SIMILAR]->(t2);
    """
    graph.run(query, parameters={
        "price_threshold": price_threshold,
        "volume_threshold": volume_threshold,
        "z_score_threshold": z_score_threshold,
        "time_window": time_window
    })
    print("Similar trade relationships created using Cypher successfully.")
create_similar_trade_relationships_cypher(graph)

# Function to analyze outlier trends
def analyze_outlier_trends(df):
    """
    Identifies the day with the most outliers, checks if it's a weekday/weekend, 
    and creates a bar chart showing outlier frequency per day.
    """
    df['date'] = df['timestamp'].dt.date
    df['z_score'] = np.abs((df['trade_volume'] - df['trade_volume'].mean()) / df['trade_volume'].std())
    # Identify outliers (Z-score > 2)
    outliers = df[df['z_score'] > 2]
    # Count outliers per day
    outlier_counts = outliers.groupby('date').size().reset_index(name='outlier_count')
    # Find the day with the most outliers
    if not outlier_counts.empty:
        most_outlier_day = outlier_counts.loc[outlier_counts['outlier_count'].idxmax()]
        most_outlier_date = most_outlier_day['date']
        most_outlier_count = most_outlier_day['outlier_count']
        weekday_name = pd.to_datetime(str(most_outlier_date)).strftime('%A')
    else:
        most_outlier_date = "No Outliers Found"
        most_outlier_count = 0
        weekday_name = "N/A"
    # Create a bar chart for outlier frequency per day
    fig_outliers = px.line(outlier_counts, x='date', y='outlier_count', 
                        title="Outlier Frequency Per Day",
                        labels={'date': 'Date', 'outlier_count': 'Number of Outliers'},
                        markers=True,  # Adds dots for clarity
                        color_discrete_sequence=['red'])  # Highlights outliers in red

    fig_outliers.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Outlier Count")
    return most_outlier_date, most_outlier_count, weekday_name, fig_outliers, outliers
#Strong values
most_outlier_date, most_outlier_count, weekday_name, fig_outliers, outliers = analyze_outlier_trends(trades_df)

# Bar Chart for volume over time
def create_bar_chart(df):
    fig = px.bar(df, x='timestamp', y='trade_volume',
                 title="Trade Volume Over Time",
                 labels={'trade_volume': 'Trade Volume'},
                 color_discrete_sequence=['#FF8C00'])  
    fig.update_layout(template="plotly_white", xaxis_title="Timestamp", yaxis_title="Trade Volume")
    return fig

# line Chart for Price Movements
def create_price_linechart(df):
    df['date'] = df['timestamp'].dt.date  
    daily_avg_price = df.groupby('date')['price'].mean().reset_index()
    fig = px.line(daily_avg_price, x='date', y='price',
                  title="Daily Price Movements",
                  labels={'price': 'Average Price'},
                  line_shape="spline",  # Smooth curves
                  markers=True)  # Adds dots at data points
    fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Avg Price")
    return fig

# Outlier line chart
def create_outlier_linechart(df):
    # Convert timestamp to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Aggregate trade volume by day (to avoid clutter)
    df['date'] = df['timestamp'].dt.date
    daily_volume = df.groupby('date')['trade_volume'].mean().reset_index()
    # Identify outliers on daily aggregated data
    daily_volume['z_score'] = (daily_volume['trade_volume'] - daily_volume['trade_volume'].mean()) / daily_volume['trade_volume'].std()
    daily_volume['is_outlier'] = np.where(daily_volume['z_score'] > 2, "Outlier", "Normal")
    # Create line chart (smoother trend)
    fig = px.line(daily_volume, x='date', y='trade_volume', 
                  title="Trade Volume Trends with Outliers",
                  labels={'trade_volume': 'Average Daily Trade Volume'},
                  color_discrete_sequence=['#1E90FF'])  # Blue line
    # Filter only outliers
    outlier_df = daily_volume[daily_volume['is_outlier'] == "Outlier"]
    # Add **red dots** for outliers
    fig.add_trace(go.Scatter(
        x=outlier_df['date'], 
        y=outlier_df['trade_volume'],
        mode='markers',
        marker=dict(color='red', size=8, symbol='circle'),
        name="Outliers"
    ))
    # Improve layout
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Trade Volume",
        legend_title="Legend",
        showlegend=True
    )
    return fig

# Dash Layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.layout = html.Div(style={'backgroundColor': 'white', 'color': 'black', 'padding': '20px'}, children=[
    html.H1("FX Trade Monitoring Dashboard", style={'textAlign': 'center', 'color': '#FFDD44'}),
    # Outlier analysis section
    html.Div([
        html.H3("Most Outliers Occurred On:", style={'color': '#000000'}),
        html.P(f"Date: {most_outlier_date}", style={'fontSize': '20px'}),
        html.P(f"Day: {weekday_name}", style={'fontSize': '20px'}),
        html.P(f"Total Outliers: {most_outlier_count}", style={'fontSize': '20px', 'color': '#FF4B4B'})
    ], style={'border': '2px solid #FFDD44', 'padding': '15px', 'margin': '10px', 'borderRadius': '10px'}),
    

    # Graphs: Two per row
    dbc.Row([
        dbc.Col(dcc.Graph(figure=create_bar_chart(trades_df)), width=6),

        dbc.Col(dcc.Graph(figure=create_price_linechart(trades_df)), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=create_outlier_linechart(trades_df)), width=6),
        dbc.Col(dcc.Graph(figure=fig_outliers), width=6)
     ])
    # dbc.Row([
    #     dbc.Col(dcc.Graph(figure=comparison_fig), width=12)
    # ])
])

if __name__ == '__main__':
    app.run_server(debug=True)

print("Neo4j Analysis Completed.")
