import pandas as pd

# Load your data
df = pd.read_excel('data/4877028-Retail_Sales_Data.xlsx')

# Check it loaded correctly
print("✅ Data loaded!")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head()) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_excel('data/4877028-Retail_Sales_Data.xlsx')
print("✅ Data Loaded!")
print(f"Shape: {df.shape}")

# ================================
# 2. CLEAN DATA
# ================================
# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Check missing values
print("\n📊 Missing Values:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# ================================
# 3. FEATURE ENGINEERING
# ================================
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Quarter'] = df['Date'].dt.quarter

print("\n✅ New columns added: Year, Month, Day, Quarter")

# ================================
# 4. BASIC ANALYSIS
# ================================
print("\n💰 Total Revenue:", df['Revenue'].sum().round(2))
print("📦 Total Sales:", df['Total_Sales'].sum().round(2))
print("🏪 Total Stores:", df['Store_ID'].nunique())
print("📦 Total Products:", df['Product_ID'].nunique())
print("🌍 Regions:", df['Region'].unique())

# ================================
# 5. REVENUE BY STORE
# ================================
store_revenue = df.groupby('Store_ID')['Revenue'].sum().sort_values(ascending=False)
print("\n🏪 Top 5 Stores by Revenue:")
print(store_revenue.head())

# ================================
# 6. REVENUE BY CATEGORY
# ================================
category_revenue = df.groupby('Product_Category')['Revenue'].sum().sort_values(ascending=False)
print("\n📦 Revenue by Category:")
print(category_revenue)

# ================================
# 7. REVENUE BY REGION
# ================================
region_revenue = df.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
print("\n🌍 Revenue by Region:")
print(region_revenue)

# ================================
# 8. SAVE CLEAN DATA
# ================================
df.to_csv('data/cleaned_data.csv', index=False)
print("\n✅ Cleaned data saved to data/cleaned_data.csv")
print("\n🚀 Phase 1 Complete!")