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
#from RGAT import RGATConv
import pickle as pkl


def run_pretrain2():
    #加载日志的类
    logger = Logger('logs/',level='debug')

    logger.logger.info('start run pretrain2.....')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/graph.pt')
    parser.add_argument('--test_file', type=str, default='../data/session1/icdm2022_session1_test_ids.txt')
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
    parser.add_argument("--n-epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device-id", type=str, default="0")



    
    args = parser.parse_args()
    logger.logger.info(str(args))

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


    node_labels = hgraph['item'].y.numpy()

    # 得到训练集和验证集划分
    ids_0 = list(np.argwhere(node_labels == 0).reshape(1,-1)[0])
    ids_1 = list(np.argwhere(node_labels == 1).reshape(1,-1)[0])


    random.shuffle(ids_0)
    random.shuffle(ids_1)
    train_idx = ids_0[16161:] + ids_1[951:]
    val_idx = ids_0[:16161] + ids_1[:951]
    random.shuffle(train_idx)
    random.shuffle(val_idx)


    #得到item节点的训练id   
    train_idx = train_idx + val_idx
    logger.logger.info('session1 item number='+str(len(train_idx)))

    #加载初赛训练集的b和f的邻居节点
    session1_train_fnode_labels_path = '../data/session1/session1_train_fnode_labels.pkl'
    session1_train_bnode_labels_path = '../data/session1/session1_train_bnode_labels.pkl'
    
    with open(session1_train_fnode_labels_path,'rb') as f:
        session1_train_fnode_labels = pkl.load(f)
    with open(session1_train_bnode_labels_path,'rb') as f:
        session1_train_bnode_labels = pkl.load(f)
    
    session1_train_fnode = list(session1_train_fnode_labels.keys())
    session1_train_bnode = list(session1_train_bnode_labels.keys())

    #转换到pyg文件中对应的节点id，注意这里是初赛数据，映射字典用maps
    session1_train_fnode = [hgraph['f'].maps[nodeid] for nodeid in session1_train_fnode]
    session1_train_bnode = [hgraph['b'].maps[nodeid] for nodeid in session1_train_bnode]

    logger.logger.info('session1 train f number='+str(len(session1_train_fnode)))
    logger.logger.info('session1 train b number='+str(len(session1_train_bnode)))



    #加载初赛测试集节点
    logger.logger.info('loading session1 test file='+args.test_file)
    test_id = [int(x) for x in open(args.test_file).readlines()]
    converted_test_id = []
    for itemid in test_id:
        #注意初赛的item节点用maps
        converted_test_id.append(hgraph['item'].maps[itemid])

    test_idx = converted_test_id

    logger.logger.info('session1 item number='+str(len(test_idx)))

    #加载初赛测试集的b和f邻居节点
    session1_test_fnode_labels_path = '../data/session1/session1_test_fnode_labels.pkl'
    session1_test_bnode_labels_path = '../data/session1/session1_test_bnode_labels.pkl'


    with open(session1_test_fnode_labels_path,'rb') as f:
        session1_test_fnode_labels = pkl.load(f)
    with open(session1_test_bnode_labels_path,'rb') as f:
        session1_test_bnode_labels = pkl.load(f)
    
    session1_test_fnode = list(session1_test_fnode_labels.keys())
    session1_test_bnode = list(session1_test_bnode_labels.keys())

    #转换到pyg文件中对应的节点id，注意这里是初赛数据，映射字典用maps
    session1_test_fnode = [hgraph['f'].maps[nodeid] for nodeid in session1_test_fnode]
    session1_test_bnode = [hgraph['b'].maps[nodeid] for nodeid in session1_test_bnode]

    logger.logger.info('session1 test f number='+str(len(session1_test_fnode)))
    logger.logger.info('session1 test b number='+str(len(session1_test_bnode)))


    #合并初赛训练集和测试集节点
    train_idx = train_idx + test_idx
    train_f_idx = list(set(session1_train_fnode + session1_test_fnode))
    train_b_idx = list(set(session1_train_bnode + session1_test_bnode))

    #再打乱一下
    random.shuffle(train_idx)
    random.shuffle(train_f_idx)
    random.shuffle(train_b_idx)


    train_idx = torch.tensor(np.array(train_idx))
    train_f_idx = torch.tensor(np.array(train_f_idx))
    train_b_idx = torch.tensor(np.array(train_b_idx))


    logger.logger.info('train item number='+str(len(train_idx)))
    logger.logger.info('train f number='+str(len(train_f_idx)))
    logger.logger.info('train b number='+str(len(train_b_idx)))


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
    #这里需要加载之前已经预训练item类型节点500轮的结果继续训练
    model_name = osp.join('../models/', "pretrain_item.pth")
    logger.logger.info('loading model = '+model_name)
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    model.to(device)


    #----------------下面是模型的参数训练的一个调整----------
    param_optimizer = list(model.named_parameters())#得到模型的参数
    #----------------下面是模型的参数训练的一个调整----------
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    #这里是训练的优化器选择，AdamW，学习率是2e-5
    opt = torch.optim.AdamW(optimizer_grouped_parameters, 
                            lr=args.lr)
    #scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=3, T_mult=2, eta_min=2e-5, last_epoch=-1)


    def train(labeled_class, train_loader, epoch_id, epoch, batch_size):
        show_step = 10
        train_loader_size = train_loader.__len__()
        num_train_step = int(len(train_loader.dataset)/batch_size)

        model.train()
        y_pred_feat = []
        y_true = []

        total_loss = []
        for batch_id, batch in enumerate(train_loader):
            
            opt.zero_grad()
            batch_size = batch[labeled_class].batch_size
            #y = batch[labeled_class].y[:batch_size].to(device)

            start = 0
            for ntype in batch.node_types:
                if ntype == labeled_class:
                    break
                start += batch[ntype].num_nodes

            batch = batch.to_homogeneous()
            feat_loss_vec, out_feat = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device))
            feat_loss_vec = feat_loss_vec[start:start + batch_size]
            
            loss = torch.mean(torch.sum(feat_loss_vec,dim=1))
            loss.backward()
            opt.step()
            #及时清空残留的缓存
            torch.cuda.empty_cache()
            scheduler.step(epoch_id + batch_id / train_loader_size) 

            total_loss.append(loss.item())
            if batch_id % show_step == 0:
                log = 'labeled_class='+ labeled_class + ' epoch_id='+str(epoch_id)+'/'+str(epoch)+' batch_id='+str(batch_id)+'/'+str(num_train_step)+' lr='+str(opt.param_groups[-1]['lr'])+' loss_feat='+str(np.mean(total_loss))
                logger.logger.info(log)
                total_loss = []



    
    train_loader = NeighborLoader(hgraph, input_nodes=('item', train_idx),
                                num_neighbors=[args.fanout] * args.n_layers,
                                shuffle=True, batch_size=args.batch_size)
    logger.logger.info('complete train_loader.....')


    train_f_loader = NeighborLoader(hgraph, input_nodes=('f', train_f_idx),
                                num_neighbors=[300, 3],
                                shuffle=True, batch_size=args.batch_size)
    logger.logger.info('complete train_f_loader.....')


    train_b_loader = NeighborLoader(hgraph, input_nodes=('b', train_b_idx),
                                num_neighbors=[300, 300],
                                shuffle=True, batch_size=args.batch_size)
    logger.logger.info('complete train_b_loader.....')


    epoch = args.n_epoch
    batch_size = args.batch_size
    
    for epoch_id in range(epoch):
        train('f',train_f_loader, epoch_id, epoch, batch_size)
        train('b',train_b_loader, epoch_id, epoch, batch_size)
        train('item',train_loader, epoch_id, epoch, batch_size)
        #这里epoch_id%2的原因是更迭保存两个，避免保存一个的时候出现问题，模型文件损毁，预训练时间成本较高
        model_name = osp.join('../models', "pretrain_itembf_" + str(epoch_id%2) + ".pth")
        logger.logger.info('save model = '+model_name)
        torch.save(model.state_dict(), model_name)
    
    #如果中途顺利训练完毕，保存最后一个即可
    model_name = osp.join('../models', "pretrain_itembf.pth")
    logger.logger.info('save model = '+model_name)
    torch.save(model.state_dict(), model_name)





