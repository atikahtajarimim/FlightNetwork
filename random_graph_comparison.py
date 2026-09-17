import pandas as pd
import networkx as nx

# STEP 1: Load and filter data (same as before)
df = pd.read_csv("data/bts_route_network_1990_2026.csv")

year = 2026
df_year = df[df['YEAR'] == year]

origin_traffic = df_year.groupby('ORIGIN')['PASSENGERS'].sum()
dest_traffic = df_year.groupby('DEST')['PASSENGERS'].sum()
total_traffic = origin_traffic.add(dest_traffic, fill_value=0)
top_50_airports = total_traffic.sort_values(ascending=False).head(50).index.tolist()

df_filtered = df_year[
    df_year['ORIGIN'].isin(top_50_airports) &
    df_year['DEST'].isin(top_50_airports)
]

G = nx.from_pandas_edgelist(df_filtered, source='ORIGIN', target='DEST', edge_attr='PASSENGERS')

# STEP 2: Real network efficiency
real_efficiency = nx.global_efficiency(G)
print(f"Real network efficiency: {real_efficiency:.4f}")

# STEP 3: Generate random graphs of the same size, average their efficiency
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

random_efficiencies = []
for i in range(10):
    random_G = nx.gnm_random_graph(num_nodes, num_edges, seed=i)
    random_efficiencies.append(nx.global_efficiency(random_G))

avg_random_efficiency = sum(random_efficiencies) / len(random_efficiencies)

print(f"Average random graph efficiency (10 runs): {avg_random_efficiency:.4f}")
print(f"Difference (real - random): {real_efficiency - avg_random_efficiency:.4f}")