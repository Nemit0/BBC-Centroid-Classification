import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from rich import print
from module.utils import get_project_root

def main():
    project_root = get_project_root()
    model_path = os.path.join(project_root, "models")
    with open(os.path.join(model_path, "bow_token_frequency_token.json"), 'r') as f:
        token_frequency = json.load(f)
    
    # Extract top 1000 most frequent token
    print("Extracting top 1000 most frequent token")
    top_1000_token = dict(list(token_frequency.items())[:1000])
    # Plotting
    print("Plotting token frequency")
    fig, ax = plt.subplots()
    ax.barh(range(len(top_1000_token)), list(top_1000_token.values()))
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Token Index")
    ax.set_title("Token Frequency")
    plt.yticks([])  # This hides the y-axis labels
    plt.savefig(os.path.join(model_path, "bow_token_frequency.png"))

    # Get top 100 most frequent token
    print("Getting top 100 most frequent token")
    top_100_token = dict(list(token_frequency.items())[:100])
    with open(os.path.join(model_path, "bow_token_frequency_top100.json"), 'w') as f:
        json.dump(top_100_token, f)
    


if __name__ == "__main__":
    main()