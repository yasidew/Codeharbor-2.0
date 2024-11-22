# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr, spearmanr
#
# # Load Developer Ratings and WCC Scores
# # Assuming a CSV with columns: 'Method', 'Maintainability', 'Readability', 'Complexity', 'WCC'
# data = pd.read_csv('developer_ratings_wcc.csv')
#
# # View first few rows of the dataset
# print(data.head())
#
# # Calculate correlations
# pearson_corr, pearson_p_value = pearsonr(data['WCC'], data['Complexity'])
# spearman_corr, spearman_p_value = spearmanr(data['WCC'], data['Complexity'])
#
# print(f"Pearson Correlation between WCC and Developer Complexity Ratings: {pearson_corr:.2f}, P-value: {pearson_p_value:.4f}")
# print(f"Spearman Correlation between WCC and Developer Complexity Ratings: {spearman_corr:.2f}, P-value: {spearman_p_value:.4f}")
#
# # You can repeat this for Maintainability and Readability if needed
# maintainability_corr, _ = pearsonr(data['WCC'], data['Maintainability'])
# readability_corr, _ = pearsonr(data['WCC'], data['Readability'])
#
# print(f"Correlation between WCC and Maintainability: {maintainability_corr:.2f}")
# print(f"Correlation between WCC and Readability: {readability_corr:.2f}")
#
# # Generate correlation matrix and heatmap
# correlation_matrix = data[['WCC', 'Maintainability', 'Readability', 'Complexity']].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix between WCC and Developer Ratings')
# plt.show()
#
# # Establish thresholds for complexity based on WCC scores
# # You can use quantiles to define low, medium, and high complexity levels
# thresholds = data['WCC'].quantile([0.33, 0.66])
# print(f"WCC Thresholds: Low < {thresholds[0.33]:.2f}, Medium < {thresholds[0.66]:.2f}, High")
#
# # Visualize the distribution of WCC Scores with thresholds
# plt.figure(figsize=(10, 6))
# sns.histplot(data['WCC'], bins=20, kde=True)
# plt.axvline(thresholds[0.33], color='blue', linestyle='--', label='Low Complexity Threshold')
# plt.axvline(thresholds[0.66], color='green', linestyle='--', label='Medium Complexity Threshold')
# plt.title('WCC Score Distribution with Complexity Thresholds')
# plt.legend()
# plt.show()
#
# # Scatter plot of WCC vs Developer Complexity Ratings
# plt.figure(figsize=(10, 6))
# plt.scatter(data['WCC'], data['Complexity'], alpha=0.5, c='blue')
# plt.title('WCC vs Developer Complexity Ratings')
# plt.xlabel('WCC Scores')
# plt.ylabel('Developer Complexity Ratings')
# plt.grid(True)
# plt.show()
#
# # Perform linear regression (if needed) to further analyze the relationship
# import statsmodels.api as sm
# X = data['WCC']
# y = data['Complexity']
# X = sm.add_constant(X)  # Add a constant (intercept) term
# model = sm.OLS(y, X).fit()
# print(model.summary())
#
# # Validate thresholds by grouping data into categories
# data['Complexity_Category'] = pd.cut(data['WCC'], bins=[-float('inf'), thresholds[0.33], thresholds[0.66], float('inf')],
#                                      labels=['Low', 'Medium', 'High'])
#
# # Analyze the distribution of developer ratings across these categories
# complexity_groups = data.groupby('Complexity_Category').agg({
#     'Maintainability': 'mean',
#     'Readability': 'mean',
#     'Complexity': 'mean'
# })
#
# print(complexity_groups)
#
# # Plot ratings across complexity categories
# complexity_groups.plot(kind='bar', figsize=(10, 6))
# plt.title('Average Developer Ratings by WCC Complexity Category')
# plt.ylabel('Average Rating')
# plt.show()
