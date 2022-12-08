import os
import os.path as osp
import argparse
import json
import datetime
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
#from RGAT import RGATConv
import pickle as pkl


def run_infer():
    
    #加载日志的类
    logger = Logger('logs/',level='debug')

    logger.logger.info('run infer.....')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/graph.pt')
    #进入复赛，这里的测试集文件更换为复赛的测试集文件
    parser.add_argument('--test_file', type=str, default='../data/session2/icdm2022_session2_test_ids.txt')
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size. If -1, use full graph training.")
    parser.add_argument("--fanout", type=int, default=300,
                        help="Fan-out of neighbor sampling.")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--h-dim", type=int, default=768,
                        help="number of hidden units")
    parser.add_argument("--in-dim", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--n-bases", type=int, default=8,
                        help="number of filter weight matrices, default: -1 [use all]")
    #全量训练7轮
    parser.add_argument("--n-epoch", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device-id", type=str, default="0")

    args = parser.parse_args()
    logger.logger.info(str(args))

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:' + str(args.device_id)
    logger.logger.info('device='+str(device))

    hgraph = torch.load(args.dataset)

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

    random_seed(4096)

    logger.logger.info('test_file='+args.test_file)
    #加载复赛测试集节点
    test_id = [int(x) for x in open(args.test_file).readlines()]
    converted_test_id = []
    for i in test_id:
        #注意修改成maps2
        converted_test_id.append(hgraph['item'].maps2[i])
    test_idx = torch.LongTensor(converted_test_id)



    num_relations = len(hgraph.edge_types)

    class rgcn(torch.nn.Module):
        def __init__(self, in_channels, out_channels, num_relations, n_bases):
            super().__init__()
            self.conv = RGCNConv(in_channels, out_channels, num_relations, num_bases=n_bases)
            self.ly = nn.LayerNorm(out_channels)

        def forward(self, x, edge_index, edge_type):
            x = self.conv(x, edge_index, edge_type)
            x = self.ly(x)
            return x


    class RGCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.relu = F.relu
            self.convs.append(rgcn(in_channels, hidden_channels, num_relations, args.n_bases))
            for i in range(n_layers - 2):
                self.convs.append(rgcn(hidden_channels, hidden_channels, num_relations, args.n_bases))
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, args.n_bases))

            #特征重构层
            self.line_feat_1 = nn.Linear(hidden_channels, in_channels)
            self.line_feat_2 = nn.Linear(in_channels, out_channels)

            

        def forward(self, x, edge_index, edge_type):
            origin_x = x
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index, edge_type)
                if i < len(self.convs) - 1:
                    x = x.relu_()
                    x = F.dropout(x, p=0.4, training=self.training)
            out_feat = self.line_feat_1(x)
            feat_loss_vec = (out_feat - origin_x) * (out_feat - origin_x)
            out_feat = self.line_feat_2(out_feat)
            return feat_loss_vec, out_feat

    model = RGCN(in_channels=args.in_dim, hidden_channels=args.h_dim, out_channels=2, n_layers=5)

    #加载之前下游训练好的模型
    model_name = osp.join('../models', "finetune_all.pth")
    logger.logger.info('loading model = '+model_name)
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    model.to(device)
    

    logger.logger.info('infering.....')
    test_loader = NeighborLoader(hgraph, input_nodes=('item', test_idx),
                                 num_neighbors=[args.fanout] * args.n_layers,
                                 shuffle=False, batch_size=args.batch_size)
    logger.logger.info('complete test_loader.....')

    labeled_class = 'item'
    
    model.eval()
    with torch.no_grad():
        y_pred = []
        for batch in tqdm(test_loader):
            batch_size = batch[labeled_class].batch_size
            start = 0
            for ntype in batch.node_types:
                if ntype == labeled_class:
                    break
                start += batch[ntype].num_nodes

            batch = batch.to_homogeneous()
            _, out_feat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))
            out_feat = out_feat[start:start + batch_size]

            y_pred += list(F.softmax(out_feat, dim=1)[:, 1].detach().cpu().numpy())

    #保存预测结果的文件地址
    save_file_path = "../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".json"
    logger.logger.info('save file path='+save_file_path)
    with open(save_file_path, 'w+') as f:
        for i in range(len(test_id)):
            y_dict = {}
            y_dict["item_id"] = int(test_id[i])
            y_dict["score"] = float(y_pred[i])
            json.dump(y_dict, f)
            f.write('\n')
    



