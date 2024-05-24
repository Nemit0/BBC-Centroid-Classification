import os
import polars as pl
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from module.utils import get_project_root

def main():
    if 'category_count_combined.json' not in os.listdir(os.path.join(get_project_root(), "data")):
        print("category_count_combined.json not found, creating one")
        json_data_list = [file for  file in os.listdir(os.path.join(get_project_root(), "data")) if file.endswith(".json") and 'category_counts' in file]
        print(json_data_list)
        with open(os.path.join(get_project_root(), "data", 'category_list.json'), "r") as f:
            category_counts = json.load(f)
        print(len(category_counts))
        _category_count_frame = {category: 0 for category in category_counts}
        for file in tqdm(json_data_list):
            with open(os.path.join(get_project_root(), "data", file), "r") as f:
                data = json.load(f)
            for category, count in data.items():
                _category_count_frame[category] += count
        
        # sort by descending for count
        _category_count_frame = dict(sorted(_category_count_frame.items(), key=lambda x: x[1], reverse=True))

        # save to json
        with open(os.path.join(get_project_root(), "data", "category_count_combined.json"), "w") as f:
            json.dump(_category_count_frame, f)
    
    with open(os.path.join(get_project_root(), "data", "category_count_combined.json"), "r") as f:
        category_count = json.load(f)

    # Get only category counts as a list of numbers
    count = list(category_count.values())

    bins = [0, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]

    # Plot with a logarithmic scale
    plt.figure(figsize=(10, 5))
    plt.hist(count, bins=30, log=True, color='blue', edgecolor='black')
    plt.title('Log-Scale Distribution of Category Counts')
    plt.xlabel('Count')
    plt.ylabel('Log Frequency')
    plt.grid(True)
    plt.show()


    



if __name__ == "__main__":
    main()