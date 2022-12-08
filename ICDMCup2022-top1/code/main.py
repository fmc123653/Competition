import os
import os.path as osp
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import RGCNConv, TransformerConv
from sklearn.metrics import average_precision_score
from log_model import Logger
import random
import pandas as pd
import numpy as np
from collections import Counter
import pickle as pkl
from load_graph import load_graph
from get_graph_dic import get_graph_dic
from get_bf_nodes import get_bf_nodes
from run_pretrain1 import run_pretrain1
from run_pretrain2 import run_pretrain2
from run_pretrain3 import run_pretrain3
from run_finetune import run_finetune
from run_infer import run_infer

def random_seed(seed):
    """setting random seed

    Args:
        seed (int): seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

#固定随机数种子
random_seed(4096)



def main():
    #加载初赛和复赛的图数据生成一个整图文件保存在同一个文件中
    load_graph()
    #加载初赛和复赛的图的字典，方便后续得到item的邻居节点b和f
    get_graph_dic()
    #加载得到和item关联的b和f节点，做数据增强
    get_bf_nodes()

    #开始第一次预训练，初赛阶段的预训练模型，仅仅只训练了item类型节点500轮
    run_pretrain1()
    #开始第2次预训练，初赛结束后发现item的b和f邻居节点和item存在正相关关系，于是把f和b节点带入预训练了300轮
    run_pretrain2()
    #开始第3次预训练，加载复赛的item、b和f节点一起预训练了85轮
    run_pretrain3()
    #下游训练，全量训练
    run_finetune()
    #加载下游训练好的模型推理
    run_infer()
    

main()