import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = pd.read_csv('banking77_train.csv')
test = pd.read_csv('banking77_test.csv')

print("=== DATASET INFO ===")
print(f"Training samples: {len(train)}")
print(f"Testing samples: {len(test)}")
print(f"Total intents: {train['category'].nunique()}")

print("\n=== SAMPLE DATA ===")
print(train.head())

print("\n=== TOP 10 INTENTS ===")
print(train['category'].value_counts().head(10))

# Plot top 20 intents
plt.figure(figsize=(12, 6))
train['category'].value_counts().head(20).plot(kind='bar', color='steelblue')
plt.title('Top 20 Banking Intents')
plt.xlabel('Intent')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('intent_distribution.png')
plt.show()
print("\nChart saved as intent_distribution.png")

# Query length analysis
train['query_length'] = train['text'].apply(lambda x: len(x.split()))
print(f"\n=== QUERY LENGTH ===")
print(f"Average words per query: {train['query_length'].mean():.1f}")
print(f"Max words: {train['query_length'].max()}")
print(f"Min words: {train['query_length'].min()}")