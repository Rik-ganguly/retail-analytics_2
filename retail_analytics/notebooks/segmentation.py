import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv('data/cleaned_data.csv')
print("✅ Data Loaded!")

# ================================
# 2. BUILD STORE FEATURES
# ================================
store_features = df.groupby('Store_ID').agg(
    Total_Revenue=('Revenue', 'sum'),
    Avg_Daily_Sales=('Revenue', 'mean'),
    Volatility=('Revenue', 'std'),
    Total_Units=('Units_Sold', 'sum'),
    Avg_Rating=('Store_Rating', 'mean'),
    Avg_Discount=('Discount_Percentage', 'mean')
).reset_index()

print("✅ Store features built!")
print(store_features.head())

# ================================
# 3. SCALE & CLUSTER
# ================================
scaler = StandardScaler()
numeric_cols = ['Total_Revenue','Avg_Daily_Sales',
                'Volatility','Total_Units',
                'Avg_Rating','Avg_Discount']
X = scaler.fit_transform(store_features[numeric_cols])

kmeans = KMeans(n_clusters=4, random_state=42)
store_features['Cluster'] = kmeans.fit_predict(X)

labels = {0: '🌟 High Performers',
          1: '📈 Growing Stores',
          2: '⚠️ Underperformers',
          3: '🔄 Seasonal Stores'}
store_features['Segment'] = store_features['Cluster'].map(labels)

print("\n✅ Stores Segmented!")
print(store_features[['Store_ID','Total_Revenue','Segment']])

# ================================
# 4. CLUSTER CHART
# ================================
plt.figure(figsize=(12,6))
sns.scatterplot(
    data=store_features,
    x='Total_Revenue',
    y='Avg_Rating',
    hue='Segment',
    size='Total_Units',
    sizes=(100,500),
    palette='Set1'
)
plt.title('Store Segmentation')
plt.xlabel('Total Revenue')
plt.ylabel('Average Rating')
plt.tight_layout()
plt.savefig('outputs/store_segments.png')
plt.show()
print("✅ Segmentation chart saved!")

# ================================
# 5. REVENUE BY SEGMENT
# ================================
plt.figure(figsize=(10,5))
seg_rev = store_features.groupby('Segment')['Total_Revenue'].mean().sort_values(ascending=False)
sns.barplot(x=seg_rev.index, y=seg_rev.values, palette='Set2')
plt.title('Average Revenue by Segment')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('outputs/segment_revenue.png')
plt.show()
print("✅ Segment revenue chart saved!")

# ================================
# 6. SAVE RESULTS
# ================================
store_features.to_csv('outputs/store_segments.csv', index=False)
print("\n🚀 Phase 4 Complete! Store Segmentation Done!")