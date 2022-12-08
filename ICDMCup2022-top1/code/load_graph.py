from torch_geometric.data import HeteroData
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import pickle



def load_graph():
    print('loading graph data...')
    #初赛的相关数据文件路径
    session1_node_feat_file_path = '../data/session1/icdm2022_session1_nodes.csv'
    session1_edge_file_path = '../data/session1/icdm2022_session1_edges.csv'
    session1_item_label_file_path = '../data/session1/icdm2022_session1_train_labels.csv'
    session1_test_ids_file_path = '../data/session1/icdm2022_session1_test_ids.csv'

    #复赛的相关数据文件路径
    session2_node_feat_file_path = '../data/session2/icdm2022_session2_nodes.csv'
    session2_edge_file_path = '../data/session2/icdm2022_session2_edges.csv'
    session2_test_ids_file_path = '../data/session2/icdm2022_session2_test_ids.csv'


    #保存的文件地址
    save_graph_path = '../data/graph.pt'



    session1_node_maps = {}
    session2_node_maps = {}


    node_feat_maps = {}
    edge_maps = {}


    lack_num = 0

    #加载初赛节点特征数据
    print('loading session1_node_feat_file_path=',session1_node_feat_file_path)
    with open(session1_node_feat_file_path,'r',encoding='utf-8') as f:
        for l in tqdm(f):
            data = l.strip().split(",")
            node_id = int(data[0])
            node_type = data[1]

            if len(data[2]) < 50:
                node_feat = np.zeros(256, dtype=np.float32)
                lack_num += 1
            else:
                node_feat = np.array([x for x in data[2].split(":")], dtype=np.float32)


            #node_feat = np.array(data[2].split(':'),dtype=np.float32)
            if node_type not in session1_node_maps.keys():
                session1_node_maps[node_type] = {}
            if node_type not in node_feat_maps.keys():
                node_feat_maps[node_type] = []
            session1_node_maps[node_type][node_id] = len(session1_node_maps[node_type])
            #顺序读取，使得特征和id对应
            node_feat_maps[node_type].append(node_feat)


    #加载复赛节点特征数据
    print('loading session2_node_feat_file_path=',session2_node_feat_file_path)
    with open(session2_node_feat_file_path,'r',encoding='utf-8') as f:
        for l in tqdm(f):
            data = l.strip().split(",")
            node_id = int(data[0])
            node_type = data[1]
            
            if len(data[2]) < 50:
                node_feat = np.zeros(256, dtype=np.float32)
                lack_num += 1
            else:
                node_feat = np.array([x for x in data[2].split(":")], dtype=np.float32)

            
            #node_feat = np.array(data[2].split(':'),dtype=np.float32)
            if node_type not in session2_node_maps.keys():
                session2_node_maps[node_type] = {}
            if node_type not in node_feat_maps.keys():
                node_feat_maps[node_type] = []
            #复赛和初赛的图拼在一起，因此复赛节点id是继承在初赛节点id的基础上进行累加
            session2_node_maps[node_type][node_id] = len(session2_node_maps[node_type]) + len(session1_node_maps[node_type])
            #顺序读取，使得特征和id对应
            node_feat_maps[node_type].append(node_feat)


    print('lack_num=',lack_num)

    #加载初赛的边信息
    print('loading session1_edge_file_path = ',session1_edge_file_path)
    with open(session1_edge_file_path,'r',encoding='utf-8') as f:
        for l in tqdm(f):
            data = l.strip().split(",")
            sour_id = int(data[0])
            dest_id = int(data[1])
            sour_type = data[2]
            dest_type = data[3]
            edge_type = data[4]

            pyg_edge_type = (sour_type, edge_type, dest_type)

            if pyg_edge_type not in edge_maps.keys():
                edge_maps[pyg_edge_type] = {}
                edge_maps[pyg_edge_type]['sour'] = []
                edge_maps[pyg_edge_type]['dest'] = []

            #得到新的节点id，这里注意要用初赛的节点映射字典
            new_sour_id = session1_node_maps[sour_type][sour_id]
            new_dest_id = session1_node_maps[dest_type][dest_id]

            edge_maps[pyg_edge_type]['sour'].append(new_sour_id)
            edge_maps[pyg_edge_type]['dest'].append(new_dest_id)

    #加载复赛的边信息
    print('loading session2_edge_file_path =',session2_edge_file_path)
    with open(session2_edge_file_path,'r',encoding='utf-8') as f:
        for l in tqdm(f):
            data = l.strip().split(",")
            sour_id = int(data[0])
            dest_id = int(data[1])
            sour_type = data[2]
            dest_type = data[3]
            edge_type = data[4]

            pyg_edge_type = (sour_type, edge_type, dest_type)

            if pyg_edge_type not in edge_maps.keys():
                edge_maps[pyg_edge_type] = {}
                edge_maps[pyg_edge_type]['sour'] = []
                edge_maps[pyg_edge_type]['dest'] = []
            
            #这里注意要用复赛的节点映射字典
            new_sour_id = session2_node_maps[sour_type][sour_id]
            new_dest_id = session2_node_maps[dest_type][dest_id]
            edge_maps[pyg_edge_type]['sour'].append(new_sour_id)
            edge_maps[pyg_edge_type]['dest'].append(new_dest_id)       


    graph = HeteroData()
    #将节点特征写入pyg图文件中
    for node_type in tqdm(node_feat_maps.keys()):
        graph[node_type].x = torch.tensor(np.array(node_feat_maps[node_type]))
        #添加原始的图节点id和在pyg文件中的节点id对应关系
        graph[node_type].maps = session1_node_maps[node_type]
        graph[node_type].maps2 = session2_node_maps[node_type]


    #将边的信息写入pyg图文件中
    for pyg_edge_type in tqdm(edge_maps.keys()):
        sour = torch.tensor(edge_maps[pyg_edge_type]['sour'], dtype=torch.long)
        dest = torch.tensor(edge_maps[pyg_edge_type]['dest'], dtype=torch.long)
        graph[pyg_edge_type].edge_index = torch.vstack([sour, dest])


    #添加初赛的item节点标签，复赛没有提供item标签
    item_labels = np.array([-1] * graph['item'].x.shape[0])
    with open(session1_item_label_file_path,'r',encoding='utf-8') as f:
        for l in tqdm(f):
            data = l.strip().split(",")
            itemid = int(data[0])
            label = int(data[1])
            new_itemid = graph['item'].maps[itemid]
            item_labels[new_itemid] = label

    graph['item'].y = torch.tensor(item_labels,dtype=torch.long)


    torch.save(graph, save_graph_path)

