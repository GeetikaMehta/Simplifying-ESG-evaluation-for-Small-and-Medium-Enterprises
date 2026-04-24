import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
data = pd.read_csv(" ")

# 1. Distribution of ESG Scores
data['total_score'].hist()
plt.title("Distribution of ESG Scores")
plt.xlabel("ESG Score")
plt.ylabel("Frequency")
plt.show()

# 2. Environmental vs Social
plt.scatter(data['environment_score'], data['social_score'])
plt.title("Environmental vs Social Score")
plt.xlabel("Environmental Score")
plt.ylabel("Social Score")
plt.show()

# 3. Box Plot (Detect Outliers)
data[['environment_score', 'social_score', 'governance_score']].plot.box()
plt.title("Box Plot of ESG Features")
plt.show()

# 4. Correlation Matrix
corr = data[['environment_score', 'social_score', 'governance_score', 'total_score']].corr()

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()

# 5. ESG Category Count (if classification exists)
if 'Category' in data.columns:
    data['Category'].value_counts().plot(kind='bar')
    plt.title("Safe vs Risk Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.show()