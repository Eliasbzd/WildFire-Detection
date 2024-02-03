# The objective of this script is to create a new csv containing a binary classification between sounds containing a fire and others.
# To do so, the existing csv will be renamed .._old.csv and the new one will replace all categories not containing 'fire' to 'not_fire' and others to 'fire'

import os
import pandas as pd


def convert_mult_to_bin(csv_path, target, target_column_name):
    '''
    Input:
        csv_path: path to fetch csv original file
        target: label targets (all label containing this key will be set to 1 (T))
    Output:
        None
        Created CSV, renamed CSV
    '''
    # Open previous file

    dir_path = os.path.dirname(csv_path)
    previous_csv = pd.read_csv(csv_path)
    # Change boolean complementary to elements including target
    inverse = (previous_csv[target_column_name] == target)
    previous_csv.loc[~ inverse, target_column_name] = 0
    previous_csv.loc[inverse, target_column_name] = 1

    # Replace and rename files
    if not os.path.exists(os.path.join(dir_path, 'old.csv')):
        os.rename(csv_path, os.path.join(dir_path, 'old.csv'))
        # previous_csv = previous_csv.drop(columns=['Unnamed: 0'])
        previous_csv.to_csv(csv_path, index=False)
        print(
            f'File renamed from {csv_path}, new file written with positive labels containing target {target}')
    else:
        print('transformation classe multiple->classe binaire déjà faite')

if __name__ == "__main__":
    convert_mult_to_bin("data/esc50/meta/esc50.csv", 12, "target")
    # Check the path of ecs50 if error is encoutered
