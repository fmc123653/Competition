# ICDM 2022:大规模电商图上的风险商品检测——冠军方案
# 赛题介绍
- 赛题地址：[ICDM 2022:大规模电商图上的风险商品检测](https://tianchi.aliyun.com/competition/entrance/531976/introduction)
- 赛题任务：利用来源于真实场景中的大规模的、异构的、存在噪声和缺失的图数据，以及极不均衡的样本，进行风险商品检测。


# 环境配置
- 操作系统版本：Ubuntu 7.2.0-8ubuntu3.1
- Python 版本：3.8.0
- PyTorch 版本：1.11.0+cu113
- CUDA 版本：11.3
- 其他Python依赖：
```text
numpy==1.21.2
pandas==1.4.1
scikit_learn==1.1.2
torch==1.11.0+cu113
torch_geometric==2.0.4
torch_scatter==2.0.9
torch_sparse==0.6.13
tqdm==4.63.0
```
# 代码结构
```text
├── code
│   ├── get_bf_nodes.py     # 加载得到和item关联的b和f节点
│   ├── get_graph_dic.py    # 加载初赛和复赛的图的字典
│   ├── load_graph.py       # 加载初赛和复赛的图数据
│   ├── log_model.py        # logger工具类
│   ├── logs                # 存放logs文件夹
│   ├── main.py             # 主文件，运行整个算法流程，包括预训练、微调和推理结果
│   ├── readme.md           
│   ├── run_finetune.py     # 全量微调
│   ├── run_infer.py        # 推理
│   ├── run_pretrain1.py    # 预训练初赛500轮
│   ├── run_pretrain2.py    # f和b节点带入预训练了300轮
│   └── run_pretrain3.py    # 加载复赛的item、b和f节点一起预训练了85轮
├── data    
│   ├── session1            # 初赛数据集
│   └── session2            # 复赛数据集
├── models                  # 保存预训练模型和微调模型
│   └── readme.md   
├── README.md
├── requirements.txt        # python环境依赖
├── submit                  # 存放推理结果
│   └── readme.md

```

# 运行流程
- 请进入code文件夹，输入指令，预训练时间约1周，建议挂后台运行。
```shell
PYTHONHASHSEED=4096 python main.py
```
- 程序自动加载图数据开始处理，进行3次预训练然后下游finetune然后推理。

- 如果下游finetune存在波动需要多次训练，请到main.py文件中注释其他函数，仅仅保留run_finetune()和run_infer()即可，反复加载预训练好的模型进行下游训练推理。


# 比赛策略

## 初赛策略
- 初赛搭建5层RGCN加载初赛训练集和测试集数据，以特征重构的方式进行无监督预训练，训练了500轮。
- 使用预训练了500轮的模型进行下游finetune，发现下游训练到8轮左右验证集达到最优往后开始过拟合，此时初赛线上分数达到94.0+。
- 因划分验证集下游训练8轮最优，进一步采用不划分验证集，用所有带标签数据进行全量训练7轮，初赛线上分数达到94.4（但存在一定波动）。

## 复赛策略
- 初赛结束做进一步数据分析发现和item直接连接的f和b节点存在一定规律，因此我们认为如果f和b连接的所有item中只要有一个异常，那么就认为该f和b节点异常，赋予标签为1，否则认为f和b节点正常，赋标签为0。
- 因为RGCN模型认为图中所有节点都是同一类型，只是边类型不同，因此赋予b和f节点标签后引入和item节点一起训练，发现带来了极高的正向收益。
- 在500轮基础上加入了初赛的b和f节点预训练300轮，然后带入赋予标签后的b和f和item一起下游全量训练7轮，复赛线上分数92.1+，对比仅仅训练了item节点的模型线上仅仅91.89+。
- 继续引入了复赛的测试集item、b和f节点同初赛节点一起预训练然后下游finetune。
- 随着加入复赛数据不断预训练50、85、100、150轮的时候，复赛线上分数分别是：92.2+,92.4+,92.2+,91.9+。
- 结果出现了很大震荡，分析原因可能是复赛测试集数目较多，模型训练了初赛800多轮，加入复赛训练仅仅100多轮，初赛和复赛数据特征存在一定偏差，模型没有预训练充分，使得没有对初赛和复赛的数据特征学习达到一个平衡点；同时我们初赛500轮是在A6000显卡上进行预训练，后来使用v100接着预训练，也可能设备不同带来了波动。


# 引用
- Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European semantic web conference. Springer, Cham, 2018.
- Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
- https://github.com/EagleLab-ZJU/DGLD
- https://zhoushengisnoob.github.io/courses/index.html?course=gnn

