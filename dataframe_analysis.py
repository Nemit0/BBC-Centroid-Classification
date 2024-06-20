import os
import warnings
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rich import print

from module.utils import get_project_root

def main():
    root_path = os.path.join(get_project_root(), 'data', 'bbc')

    warnings.filterwarnings("ignore")
    class_id_map = {
        'business': 0,
        'entertainment': 1,
        'politics': 2,
        'sport': 3,
        'tech': 4
    }
    df = pd.read_csv(os.path.join(root_path, 'test_df.csv'))
    print(df.columns)
    print(df.head())
    print(f"Document count: {len(df)}")

    df_trimmed = df[df['candidate_categories'] == df['candidate_categories_2']]
    print(f"Document count after trimming: {len(df_trimmed)}")
    print(df_trimmed.head())

    df_trimmed['candidate_category'] = [-1] * len(df_trimmed)
    df_relation = {i: {j: 0 for j in range(5) if j != i} for i in range(5)}
    for j in range(len(df_trimmed)):
        candidate_categories = json.loads(df_trimmed['candidate_categories'].iloc[j])
        true_category = df_trimmed['classid'].iloc[j]
        if not isinstance(candidate_categories, list):
            candidate_categories = [candidate_categories]
        if true_category in candidate_categories:
            candidate_categories.remove(true_category)
        candidate_category = candidate_categories[0]
        df_trimmed['candidate_category'].iloc[j] = candidate_category

    # Calculate category relation
    for i in range(5):
        for j in range(len(df_trimmed)):
            true_category = df_trimmed['classid'].iloc[j]
            if i == true_category:
                candidate_category= df_trimmed['candidate_category'].iloc[j]
                df_relation[i][candidate_category] += 1
    
    # Plot as heatmap
    df_relation = pd.DataFrame(df_relation)
    df_relation.fillna(0, inplace=True)
    print(df_relation)
    plt.figure(figsize=(10, 10))
    sns.heatmap(df_relation, annot=True, cmap='viridis')
    plt.show()

    df_trimmed.to_csv(os.path.join(root_path, 'test_df_trimmed.csv'), index=False)

if __name__ == "__main__":
    main()