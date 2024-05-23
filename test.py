import pandas as pd

# Sample DataFrame with an 'embedding' column initialized to hold lists
df = pd.DataFrame({
    'text': ['sample text', 'another sample'],
    'embedding': [None] * 2  # Initialize with None or np.nan
})

# Assuming get_embedding returns a list
def get_embedding(text):
    # Returns a list of embeddings
    return [1.0, 2.0, 3.0]

# Correct way to assign a list to a specific cell in 'embedding' column
df.at[0, 'embedding'] = get_embedding(df.iloc[0]['text'])

print(df)