import argparse
from pydoc import describe
from tkinter import W
from torch_geometric.data import HeteroData
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import pickle as pkl



def process(edge_file_path, save_file_path):
    print('edge_file_path=',edge_file_path)
    print('save_file_path=',save_file_path)
    edges_dic = {}

    with open(edge_file_path,'r',encoding='utf-8') as f:
        for l in tqdm(f):
            data = l.strip().split(",")
            source_id = int(data[0])
            dest_id = int(data[1])
            source_type = data[2]
            dest_type = data[3]
            edge_type = data[4]

            if edge_type not in edges_dic.keys():
                edges_dic[edge_type] = {}
            
            if source_id not in edges_dic[edge_type].keys():
                edges_dic[edge_type][source_id] = set([])
            
            edges_dic[edge_type][source_id].add(dest_id)
    

    with open(save_file_path, 'wb') as f:
        pkl.dump(edges_dic, f)


def get_graph_dic():

    print('loading graph_dic....')
    edge_file_path = '../data/session1/icdm2022_session1_edges.csv'
    save_file_path = '../data/session1/session1_edges_dic.pkl'

    process(edge_file_path, save_file_path)



    edge_file_path = '../data/session2/icdm2022_session2_edges.csv'
    save_file_path = '../data/session2/session2_edges_dic.pkl'

    process(edge_file_path, save_file_path)
