import matplotlib.pyplot as plt  # for plotting
import networkx as nx
from networkx.algorithms import bipartite

from analytics.utils import load_dataset, plot_degree_dist, create_degree_list
from dataloader.utils import get_transformed_dataframe

df_ebid = load_dataset("data/ebid_buyer_item_seller_id_R_2.txt")
new_ebid = get_transformed_dataframe(df_ebid)

deg_bi_buyer, deg_bi_pair = create_degree_list(new_ebid, "buyer", "pair")

plot_degree_dist(deg_bi_buyer, "plots_transform/ebid_buyer_pair_deg.png", "Item-Sellers per Buyer")
plot_degree_dist(deg_bi_pair, "plots_transform/ebid_pair_buyer_deg.png", "Buyers per Item-Seller")

df_bonanza = load_dataset("data/bonanza_buyer_item_seller_id_R_2.txt")
new_ebid = get_transformed_dataframe(df_bonanza)

deg_bi_buyer, deg_bi_pair = create_degree_list(new_ebid, "buyer", "pair")

plot_degree_dist(deg_bi_buyer, "plots_transform/bonanza_buyer_pair_deg.png", "Item-Sellers per Buyer")
plot_degree_dist(deg_bi_pair, "plots_transform/bonanza_pair_buyer_deg.png", "Buyers per Item-Seller")