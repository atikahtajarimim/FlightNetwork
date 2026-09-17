import pandas as pd
import networkx as nx

df = pd.read_csv("data/bts_route_network_1990_2026.csv")

# Check available years
print("Available years:", sorted(df['YEAR'].unique()))

# STEP 1: Recent year data
year = 2026
df_year = df[df['YEAR'] == year]
print("Rows for this year:", df_year.shape)

# STEP 2: Top 50 airports by total passenger traffic 
origin_traffic = df_year.groupby('ORIGIN')['PASSENGERS'].sum()
dest_traffic = df_year.groupby('DEST')['PASSENGERS'].sum()
total_traffic = origin_traffic.add(dest_traffic, fill_value=0)
top_50_airports = total_traffic.sort_values(ascending=False).head(50).index.tolist()

print("Top 50 airports:", top_50_airports)

# STEP 3: Keep only routes where BOTH airports are in our top 50 
df_filtered = df_year[
    df_year['ORIGIN'].isin(top_50_airports) & 
    df_year['DEST'].isin(top_50_airports)
]

print("Filtered routes:", df_filtered.shape)

# STEP 4: Build the network graph
G = nx.from_pandas_edgelist(
    df_filtered,
    source='ORIGIN',
    target='DEST',
    edge_attr='PASSENGERS'
)

print("Number of nodes (airports):", G.number_of_nodes())
print("Number of edges (routes):", G.number_of_edges())

# STEP 5: Betweenness Centrality
betweenness = nx.betweenness_centrality(G)
top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 airports by betweenness centrality:")
for airport, score in top_betweenness:
    print(f"{airport}: {score:.4f}")

# STEP 6: Clustering Coefficient (small-world check)
avg_clustering = nx.average_clustering(G)
print(f"\nAverage clustering coefficient: {avg_clustering:.4f}")

# STEP 7: Average Path Length (small-world check)
avg_path_length = nx.average_shortest_path_length(G)
print(f"Average path length: {avg_path_length:.4f}")

# STEP 8: Global Efficiency
efficiency = nx.global_efficiency(G)
print(f"\nGlobal efficiency: {efficiency:.4f}")

# STEP 9: Airport Removal Test
baseline_efficiency = nx.global_efficiency(G)
print(f"\nBaseline efficiency: {baseline_efficiency:.4f}")

results = []
for airport in G.nodes():
    G_temp = G.copy()
    G_temp.remove_node(airport)
    eff_without = nx.global_efficiency(G_temp)
    pct_loss = (baseline_efficiency - eff_without) / baseline_efficiency * 100
    
    results.append({
        'airport': airport,
        'betweenness': betweenness[airport],
        'efficiency_loss_pct': pct_loss
    })

results_df = pd.DataFrame(results).sort_values('efficiency_loss_pct', ascending=False)
print("\nTop 10 most critical airports (by efficiency loss when removed):")
print(results_df.head(10))

# STEP 10: Regression - does betweenness predict efficiency loss?
import statsmodels.api as sm

X = results_df[['betweenness']]
X = sm.add_constant(X)
y = results_df['efficiency_loss_pct']
model = sm.OLS(y, X).fit()
print("\n", model.summary())