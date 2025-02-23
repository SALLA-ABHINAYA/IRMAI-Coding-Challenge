# IRMAI Internship Challenge Projects
### NOTE:  Above files are zipped,click on raw to view the files.

## Project 1: Stock Trade Outlier Analysis using Graph Database
**Objective:** Analyze FX trades, identify outliers, and compare actual trade graphs with expected guidelines using a graph database.

### Key Tasks:
1. **Data Collection:**
   - Collected historical FX trade data (e.g., EUR/USD, GBP/USD) from public APIs or CSV files.
   - Stored data in a Neo4j graph database.

2. **Graph Construction:**
   - Built a graph with nodes as trades and edges as relationships (e.g., consecutive or similar trades).
   - Enriched the graph with attributes like trade volume, price, and timestamp.

3. **Outlier Detection:**
   - Implemented outlier detection algorithms (e.g., Z-score, IQR) to identify unusual trades.
   - Highlighted outliers in the graph.

4. **Comparison with Guidelines:**
   - Defined expected trade patterns (e.g., normal volume ranges, price movements).
   - Compared actual trade graphs with guidelines to identify deviations.

5. **Visualization:**
   - Used Matplotlib/Plotly to visualize the trade graph and highlight outliers.
   - Generated a summary report of findings.

**Tools:** Python, Neo4j (Py2neo/Neo4j Python driver), Pandas, NumPy, Matplotlib/Plotly.

---

## Project 2: Stock Trade Gap Analysis using Graph Database
**Objective:** Perform gap analysis on stock trades and compare actual trade graphs with expected patterns using a graph database.

### Key Tasks:
1. **Data Collection:**
   - Collected historical stock trade data (e.g., AAPL, MSFT) from public APIs or CSV files.
   - Stored data in a Neo4j graph database.

2. **Graph Construction:**
   - Built a graph with nodes as trades and edges as relationships (e.g., consecutive or similar trades).
   - Enriched the graph with attributes like trade volume, price, and timestamp.

3. **Gap Analysis:**
   - Implemented gap analysis algorithms to identify gaps (e.g., missing trades, unusual price jumps).
   - Highlighted gaps in the graph.

4. **Comparison with Patterns:**
   - Defined expected trade patterns (e.g., normal volume ranges, price movements).
   - Compared actual trade graphs with patterns to identify deviations.

5. **Visualization:**
   - Used Matplotlib/Plotly to visualize the trade graph and highlight gaps.
   - Generated a summary report of findings.

**Tools:** Python, Neo4j (Py2neo/Neo4j Python driver), Pandas, NumPy, Matplotlib/Plotly.

---

## Project 3: Failure Mode and Effect Analysis (FMEA) for Financial Transactions
**Objective:** Perform FMEA on financial transactions and visualize results using a graph database.

### Key Tasks:
1. **Data Collection:**
   - Collected historical financial transaction data (e.g., bank transfers, credit card transactions) from public APIs or CSV files.
   - Stored data in a Neo4j graph database.

2. **Graph Construction:**
   - Built a graph with nodes as transactions and edges as relationships (e.g., consecutive or similar transactions).
   - Enriched the graph with attributes like transaction amount, type, and timestamp.

3. **FMEA Implementation:**
   - Implemented FMEA algorithms to identify potential failure modes and their effects.
   - Highlighted failure modes in the graph.

4. **Visualization:**
   - Used Matplotlib/Plotly to visualize the transaction graph and highlight failure modes.
   - Generated a summary report of findings.

**Tools:** Python, Neo4j (Py2neo/Neo4j Python driver), Pandas, NumPy, Matplotlib/Plotly.
