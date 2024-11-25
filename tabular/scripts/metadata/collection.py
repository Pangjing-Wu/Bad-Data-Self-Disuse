import argparse
import os
import sys

import pandas as pd

sys.path.append('.')
from configs.path import METADATA_DIR


parser = argparse.ArgumentParser(description="Metadata Collection.")
parser.add_argument("--dataset", required=True, help="Dataset name.")
args  = parser.parse_args()

input_folder = os.path.join(METADATA_DIR, args.dataset)
output_file = os.path.join(input_folder, 'metadata.csv')
if os.path.exists(output_file):
    raise FileExistsError('meta collection exists.')

collection = pd.DataFrame()
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path)
        if 'score-norm' in df.columns:
            col_name = os.path.splitext(file_name)[0][:-12]
            collection[col_name] = df['score-norm']
        else:
            print(f"'score-norm' column not found in {file_name}")

collection.to_csv(output_file, index=False)
print(f"combined scores saved to {output_file}")
