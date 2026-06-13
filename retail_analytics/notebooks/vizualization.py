import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
df = pd.read_csv('data/cleaned_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("✅ Data loaded for visualization!")

# ================================
# 1. REVENUE BY REGION (Bar Chart)
# ================================
plt.figure(figsize=(10,5))
region_rev = df.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
sns.barplot(x=region_rev.index, y=region_rev.values, palette='Blues_d')
plt.title('Total Revenue by Region')
plt.xlabel('Region')
plt.ylabel('Revenue')
plt.tight_layout()
plt.savefig('outputs/revenue_by_region.png')
plt.show()
print("✅ Chart 1 saved!")

# ================================
# 2. REVENUE BY CATEGORY (Bar Chart)
# ================================
plt.figure(figsize=(10,5))
cat_rev = df.groupby('Product_Category')['Revenue'].sum().sort_values(ascending=False)
sns.barplot(x=cat_rev.index, y=cat_rev.values, palette='Greens_d')
plt.title('Revenue by Product Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/revenue_by_category.png')
plt.show()
print("✅ Chart 2 saved!")

# ================================
# 3. MONTHLY REVENUE TREND (Line Chart)
# ================================
plt.figure(figsize=(12,5))
monthly = df.groupby(['Year','Month'])['Revenue'].sum().reset_index()
monthly['Period'] = monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2)
sns.lineplot(data=monthly, x='Period', y='Revenue', marker='o', color='blue')
plt.title('Monthly Revenue Trend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/monthly_trend.png')
plt.show()
print("✅ Chart 3 saved!")

# ================================
# 4. TOP 10 STORES (Bar Chart)
# ================================
plt.figure(figsize=(12,5))
top_stores = df.groupby('Store_ID')['Revenue'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_stores.index, y=top_stores.values, palette='Oranges_d')
plt.title('Top 10 Stores by Revenue')
plt.tight_layout()
plt.savefig('outputs/top_stores.png')
plt.show()
print("✅ Chart 4 saved!")

# ================================
# 5. REVENUE BY PAYMENT MODE (Pie Chart)
# ================================
plt.figure(figsize=(8,8))
payment = df.groupby('Payment_Mode')['Revenue'].sum()
plt.pie(payment.values, labels=payment.index, autopct='%1.1f%%', startangle=90)
plt.title('Revenue by Payment Mode')
plt.tight_layout()
plt.savefig('outputs/payment_mode.png')
plt.show()
print("✅ Chart 5 saved!")

print("\n🚀 Phase 2 Complete! All charts saved in outputs folder!")