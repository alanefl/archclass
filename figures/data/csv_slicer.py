"""
Scratch-space for slicing and dicing result CSVs.
"""

import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True,
                    help="Path to file.")

args = parser.parse_args()


def get_max_column_val(file_name, col):
    max_val = None
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if not max_val:
                    max_val = float(row[col])
                else:
                    max_val = max(max_val, float(row[col]))
    print(max_val)


COL = 3

get_max_column_val(args.file, COL)
