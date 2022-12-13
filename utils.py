"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import csv

import torch  # for torch ops


def log_training_events(array, file_name: str, reset: bool = False) -> None:
    with open(file_name, "w" if reset else "a") as fp:
        csv.writer(fp, delimiter=",").writerow(array)


"""-------------LAMBDA FUNCTIONS----------"""
DEVICE = lambda: "cuda" if torch.cuda.is_available() else "cpu"
