import time

import pandas as pd
import plotly.express as px

csv_path = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\feature_df.csv'

df = pd.read_csv(csv_path, sep=',')
t1 = time.perf_counter()
# Create the sunburst plot
fig = px.sunburst(
    df,
    path=['target', 'sex', 'neutered'],
    values=None,  # No need to specify values if it's just counts
    color='target',  # Optional: color by 'target'
    labels={'target': 'ill'},
    branchvalues='total'
)
time_ctr = time.perf_counter() - t1
print('found after {}s'.format(time_ctr))

# Show the plot
fig.show()
