import numpy as np
from tqdm import tqdm
import pickle as pkl


#处理初赛训练集的数据b和f节点
def get_session1_train_bfnodes():
    print('loading session1_train_bfnodes...')
    session1_item_label_file_path = '../data/session1/icdm2022_session1_train_labels.csv'
    session1_edges_dic_file_path = '../data/session1/session1_edges_dic.pkl'

    #保存的f和b节点的文件路径
    fnode_labels_file_path = '../data/session1/session1_train_fnode_labels.pkl'
    bnode_labels_file_path = '../data/session1/session1_train_bnode_labels.pkl'

    with open(session1_edges_dic_file_path,'rb') as f:
        session1_edges_dic = pkl.load(f)

    #把标签为1和0的分别统计起来
    itemlabels_dic = {}
    itemlabels_dic[0] = set([])
    itemlabels_dic[1] = set([])
    with open(session1_item_label_file_path,'r',encoding='utf-8') as f:
        for l in tqdm(f):
            data = l.strip().split(",")
            itemid = int(data[0])
            label = int(data[1])
            itemlabels_dic[label].add(itemid)

    print('session1 train number=',len(itemlabels_dic[0]),len(itemlabels_dic[1]))


    fnode_labels = {}
    bnode_labels = {}


    #先将正常的节点进行传播
    for itemid in tqdm(itemlabels_dic[0]):
        fid_ls = []
        bid_ls = []
        if itemid in session1_edges_dic['B_1'].keys():
            fid_ls = session1_edges_dic['B_1'][itemid]
        if itemid in session1_edges_dic['A'].keys():
            bid_ls = session1_edges_dic['A'][itemid]
        
        for fid in fid_ls:
            fnode_labels[fid] = 0
        for bid in bid_ls:
            bnode_labels[bid] = 0
            
     
    #再将异常的节点进行传播感染
    for itemid in tqdm(itemlabels_dic[1]):
        fid_ls = []
        bid_ls = []
        if itemid in session1_edges_dic['B_1'].keys():
            fid_ls = session1_edges_dic['B_1'][itemid]
        if itemid in session1_edges_dic['A'].keys():
            bid_ls = session1_edges_dic['A'][itemid]
        
        for fid in fid_ls:
            fnode_labels[fid] = 1
        for bid in bid_ls:
            bnode_labels[bid] = 1


    with open(fnode_labels_file_path, 'wb') as f:
        pkl.dump(fnode_labels, f)

    with open(bnode_labels_file_path, 'wb') as f:
        pkl.dump(bnode_labels, f)



#处理初赛测试集item相连接的b和f节点信息
def get_session1_test_bfnodes():
    print('loading session1_test_bfnodes...')

    session1_test_item_file_path = '../data/session1/icdm2022_session1_test_ids.txt'
    session1_edges_dic_file_path = '../data/session1/session1_edges_dic.pkl'

    #保存的f和b节点的文件路径
    fnode_labels_file_path = '../data/session1/session1_test_fnode_labels.pkl'
    bnode_labels_file_path = '../data/session1/session1_test_bnode_labels.pkl'

    with open(session1_edges_dic_file_path,'rb') as f:
        session1_edges_dic = pkl.load(f)
    
    test_id = [int(x) for x in open(session1_test_item_file_path).readlines()]

    print('session1 test number=',len(test_id))
    fnode_labels = {}
    bnode_labels = {}


    for itemid in tqdm(test_id):
        fid_ls = []
        bid_ls = []
        if itemid in session1_edges_dic['B_1'].keys():
            fid_ls = session1_edges_dic['B_1'][itemid]
        if itemid in session1_edges_dic['A'].keys():
            bid_ls = session1_edges_dic['A'][itemid]
        
        for fid in fid_ls:
            fnode_labels[fid] = -1
        for bid in bid_ls:
            bnode_labels[bid] = -1
    
    with open(fnode_labels_file_path, 'wb') as f:
        pkl.dump(fnode_labels, f)

    with open(bnode_labels_file_path, 'wb') as f:
        pkl.dump(bnode_labels, f)
    


#处理复赛测试集item相连接的b和f节点信息
def get_session2_test_bfnodes():
    print('loading session2_test_bfnodes...')
    session2_test_item_file_path = '../data/session2/icdm2022_session2_test_ids.txt'
    session2_edges_dic_file_path = '../data/session2/session2_edges_dic.pkl'

    #保存的f和b节点的文件路径
    fnode_labels_file_path = '../data/session2/session2_test_fnode_labels.pkl'
    bnode_labels_file_path = '../data/session2/session2_test_bnode_labels.pkl'

    with open(session2_edges_dic_file_path,'rb') as f:
        session2_edges_dic = pkl.load(f)
    
    test_id = [int(x) for x in open(session2_test_item_file_path).readlines()]

    print('session2 test number=',len(test_id))
    fnode_labels = {}
    bnode_labels = {}


    for itemid in tqdm(test_id):
        fid_ls = []
        bid_ls = []
        if itemid in session2_edges_dic['B_1'].keys():
            fid_ls = session2_edges_dic['B_1'][itemid]
        if itemid in session2_edges_dic['A'].keys():
            bid_ls = session2_edges_dic['A'][itemid]
        
        for fid in fid_ls:
            fnode_labels[fid] = -1
        for bid in bid_ls:
            bnode_labels[bid] = -1
    
    with open(fnode_labels_file_path, 'wb') as f:
        pkl.dump(fnode_labels, f)

    with open(bnode_labels_file_path, 'wb') as f:
        pkl.dump(bnode_labels, f)
    



def get_bf_nodes():

    print('loading bf nodes...')
    get_session1_train_bfnodes()
    get_session1_test_bfnodes()
    get_session2_test_bfnodes()
