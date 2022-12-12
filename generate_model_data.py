"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from dataloader.utils import generate_model_data  # for generating data files
from dataloader.utils import load_dataset

ebid_data = load_dataset("data/ebid_buyer_item_seller_id_R_2.txt")
bonanza_data = load_dataset("data/bonanza_buyer_item_seller_id_R_2.txt")

generate_model_data(ebid_data, "processed_data/ebid")
generate_model_data(bonanza_data, "processed_data/bonanza")
