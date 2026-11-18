import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'houses.csv')
    charts_dir = os.path.join(base_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # 1. Load data
    print("=" * 60)
    print("1. LOADING DATASET")
    print("=" * 60)
    df = pd.read_csv(data_path)
    print(f"Data successfully loaded from: {data_path}")
    print()
    
    # 2. Dataset Shape
    print("=" * 60)
    print("2. DATASET SHAPE")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print()
    
    # 3. Dataset Info
    print("=" * 60)
    print("3. DATASET INFO")
    print("=" * 60)
    df.info()
    print()
    
    # 4. Summary Statistics (Describe)
    print("=" * 60)
    print("4. SUMMARY STATISTICS (DESCRIBE)")
    print("=" * 60)
    print(df.describe().to_string())
    print()
    
    # 5. Missing Values
    print("=" * 60)
    print("5. MISSING VALUES CHECK")
    print("=" * 60)
    missing = df.isnull().sum()
    print(missing.to_string())
    print(f"\nTotal missing values: {missing.sum()}")
    print()
    
    # Configure plotting style
    sns.set_theme(style="whitegrid")
    
    # 6. Chart 1: Price Histogram
    print("=" * 60)
    print("6. GENERATING CHARTS")
    print("=" * 60)
    
    plt.figure(figsize=(9, 5))
    sns.histplot(df['price_lakh'], kde=True, bins=35, color='#2b5c8f', edgecolor='black')
    plt.axvline(df['price_lakh'].median(), color='red', linestyle='--', label=f'Median: ₹{df["price_lakh"].median():.2f}L')
    plt.axvline(df['price_lakh'].mean(), color='orange', linestyle='-', label=f'Mean: ₹{df["price_lakh"].mean():.2f}L')
    plt.title('Distribution of House Prices in Jaipur (Lakhs INR)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Price (Lakhs ₹)', fontsize=12)
    plt.ylabel('Number of Houses', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    hist_path = os.path.join(charts_dir, 'price_histogram.png')
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Saved: {hist_path}")
    
    # 7. Chart 2: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', square=True, 
                linewidths=0.8, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Housing Features', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    heatmap_path = os.path.join(charts_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved: {heatmap_path}")
    
    # 8. Chart 3: Scatter Plot of Area vs Price
    plt.figure(figsize=(10, 6))
    palette = {'Standard': '#6baed6', 'Good': '#3182bd', 'Premium': '#08519c'}
    sns.scatterplot(data=df, x='area_sqft', y='price_lakh', hue='locality', 
                    palette=palette, alpha=0.75, s=60, edgecolor='w')
    plt.title('House Price vs. Area (sqft) by Locality', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Area (Square Feet)', fontsize=12)
    plt.ylabel('Price (Lakhs ₹)', fontsize=12)
    plt.legend(title='Locality', title_fontsize='11', fontsize=10)
    plt.tight_layout()
    scatter_path = os.path.join(charts_dir, 'area_vs_price_scatter.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_path}")
    
    print("\nEDA Completed Successfully! All charts saved in charts/")

if __name__ == '__main__':
    main()
