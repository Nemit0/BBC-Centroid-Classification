import os
import json

import numpy as np
import polars as pl

from collections import Counter

from module.utils import get_project_root

def main():
    project_root = get_project_root()
    data_path = os.path.join(project_root, 'data')
    category_count_list = [file for file in os.listdir(data_path) if file.endswith("_category_counts.json")]
    with open(os.path.join(data_path, 'a.parquet_category_counts.json'), 'r') as f:
        category_list = list(json.load(f).keys())

    category_dict = {category:0 for category in category_list}
    category = Counter(category_dict)
    for file in category_count_list:
        with open(data_path, file):
            _category_count = json.load(f)
            category += Counter(_category_count)
    
    category = dict(category)
    with open(os.path.join(data_path, 'category_counts_new.json'), 'w') as f:
        json.dump(category, f, indent=4)
        
if __name__ == "__main__":
    main()