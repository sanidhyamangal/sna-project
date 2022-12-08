"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import matplotlib.pyplot as plt  # for plotting
import networkx as nx
from networkx.algorithms import bipartite

from analytics.utils import load_dataset, plot_degree_dist, create_degree_list

df_ebid = load_dataset("data/ebid_buyer_item_seller_id_R_2.txt")
deg_bi_buyer, deg_bi_item = create_degree_list(df_ebid, "buyer", "item")
deg_is_item, deg_is_seller = create_degree_list(df_ebid, "item", "seller")
deg_bs_buyer, deg_bs_seller = create_degree_list(df_ebid, "buyer", "seller")


plot_degree_dist(deg_bi_buyer, "plots/ebid_buyer_item_deg.png", "Items per Buyer")
plot_degree_dist(deg_bi_item, "plots/ebid_item_buyer_deg.png", "Buyers per Item")
plot_degree_dist(deg_is_item, "plots/ebid_item_seller_deg.png", "Sellers per Item")
plot_degree_dist(deg_is_seller, "plots/ebid_seller_item_deg.png", "Items per Seller")
plot_degree_dist(deg_bs_buyer, "plots/ebid_buyer_seller_deg.png", "Sellers per Buyer")
plot_degree_dist(deg_bs_seller, "plots/ebid_seller_buyer_deg.png", "Buyers per Seller")

df_bonanza = load_dataset("data/bonanza_buyer_item_seller_id_R_2.txt")

deg_bi_buyer, deg_bi_item = create_degree_list(df_bonanza, "buyer", "item")
deg_is_item, deg_is_seller = create_degree_list(df_bonanza, "item", "seller")
deg_bs_buyer, deg_bs_seller = create_degree_list(df_bonanza, "buyer", "seller")


plot_degree_dist(deg_bi_buyer, "plots/bonanza_buyer_item_deg.png", "Items per Buyer")
plot_degree_dist(deg_bi_item, "plots/bonanza_item_buyer_deg.png", "Buyers per Item")
plot_degree_dist(deg_is_item, "plots/bonanza_item_seller_deg.png", "Sellers per Item")
plot_degree_dist(deg_is_seller, "plots/bonanza_seller_item_deg.png", "Items per Seller")
plot_degree_dist(deg_bs_buyer, "plots/bonanza_buyer_seller_deg.png", "Sellers per Buyer")
plot_degree_dist(deg_bs_seller, "plots/bonanza_seller_buyer_deg.png", "Buyers per Seller")