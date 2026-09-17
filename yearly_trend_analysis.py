import pandas as pd
import networkx as nx

df = pd.read_csv("data/bts_route_network_1990_2026.csv")

years = sorted(df['YEAR'].unique())
yearly_results = []

for year in years:
    df_year = df[df['YEAR'] == year]
    
    # Top 50 airports by traffic, for this year
    origin_traffic = df_year.groupby('ORIGIN')['PASSENGERS'].sum()
    dest_traffic = df_year.groupby('DEST')['PASSENGERS'].sum()
    total_traffic = origin_traffic.add(dest_traffic, fill_value=0)
    top_50 = total_traffic.sort_values(ascending=False).head(50).index.tolist()
    
    df_filtered = df_year[
        df_year['ORIGIN'].isin(top_50) &
        df_year['DEST'].isin(top_50)
    ]
    
    G = nx.from_pandas_edgelist(df_filtered, source='ORIGIN', target='DEST', edge_attr='PASSENGERS')
    
    yearly_results.append({
        'year': year,
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'clustering': nx.average_clustering(G),
        'avg_path_length': nx.average_shortest_path_length(G),
        'efficiency': nx.global_efficiency(G)
    })

results_df = pd.DataFrame(yearly_results)
print(results_df)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(results_df['year'], results_df['efficiency'], marker='o', linewidth=2, color='#ff6b35')
plt.title('US Top-50 Airport Network Efficiency (1990-2026)', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Global Efficiency')
plt.ylim(0.85, 1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('efficiency_trend.png', dpi=300)
plt.show()