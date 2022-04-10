import splitfolders
import os

input_folder = os.path.join(os.getcwd(),"GTAVDataset")
splitfolders.ratio(input_folder, output="dataset", seed=1275, ratio=(.8, .1, .1), group_prefix=None)