import pandas as pd

df = pd.read_csv("data/bts_route_network_1990_2026.csv")

years = sorted(df['YEAR'].unique())
trend_results = []

for year in years:
    df_year = df[df['YEAR'] == year]
    
    origin_traffic = df_year.groupby('ORIGIN')['PASSENGERS'].sum()
    dest_traffic = df_year.groupby('DEST')['PASSENGERS'].sum()
    total_traffic = origin_traffic.add(dest_traffic, fill_value=0)
    top_50 = total_traffic.sort_values(ascending=False).head(50).index.tolist()
    
    df_filtered = df_year[
        df_year['ORIGIN'].isin(top_50) &
        df_year['DEST'].isin(top_50)
    ]
    
    avg_load_factor = df_filtered['LOAD_FACTOR'].mean()
    avg_airtime_ratio = (df_filtered['AVG_AIR_TIME'] / df_filtered['AVG_RAMP_TO_RAMP']).mean()
    
    trend_results.append({
        'year': year,
        'avg_load_factor': avg_load_factor,
        'avg_airtime_ratio': avg_airtime_ratio
    })

trend_df = pd.DataFrame(trend_results)
print(trend_df)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(trend_df['year'], trend_df['avg_load_factor'], marker='o', linewidth=2, color='#ff6b35')
ax1.set_title('Load Factor Over Time')
ax1.set_xlabel('Year')
ax1.set_ylabel('Load Factor')
ax1.grid(True, alpha=0.3)

ax2.plot(trend_df['year'], trend_df['avg_airtime_ratio'], marker='o', linewidth=2, color='#1e88e5')
ax2.set_title('Airtime Ratio Over Time')
ax2.set_xlabel('Year')
ax2.set_ylabel('Airtime Ratio')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('profitability_trends.png', dpi=300)
plt.show()