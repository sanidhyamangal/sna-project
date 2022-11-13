"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from analytics.utils import load_dataset, plot_degree_dist
import networkx as nx
import matplotlib.pyplot as plt # for plotting



df_ebid = load_dataset("data/ebid_buyer_item_seller_id_R_2.txt")


G_bi = nx.from_pandas_edgelist(df_ebid, "buyer", "item")
G_is = nx.from_pandas_edgelist(df_ebid, "item", "seller")
G_bs = nx.from_pandas_edgelist(df_ebid, "buyer", "seller")

plot_degree_dist(G_is,"ebid_item_seller_deg.png","Item Seller Degree Dist")
plot_degree_dist(G_bi,"ebid_buyer_item_deg.png","Buyer Item Degree Dist")
plot_degree_dist(G_bs,"ebid_buyer_seller_deg.png","Buyer Seller Degree Dist")

df_bonanza = load_dataset("data/bonanza_buyer_item_seller_id_R_2.txt")


G_bi = nx.from_pandas_edgelist(df_bonanza, "buyer", "item")
G_is = nx.from_pandas_edgelist(df_bonanza, "item", "seller")
G_bs = nx.from_pandas_edgelist(df_bonanza, "buyer", "seller")

plot_degree_dist(G_is,"bonanza_item_seller_deg.png","Item Seller Degree Dist")
plot_degree_dist(G_bi,"bonanza_buyer_item_deg.png","Buyer Item Degree Dist")
plot_degree_dist(G_bs,"bonanza_buyer_seller_deg.png","Buyer Seller Degree Dist")