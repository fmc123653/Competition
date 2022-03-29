【2022阿里安全真实场景篡改图像检测挑战赛】

images_data为训练数据的文件夹，train1是此次比赛的训练数据，train2是上届比赛的训练数据，train3和train4是后面数据增强生成的，请自己另外下载数据集放置相应位置。

运行模型的相关配置请在config.py文件里修改，运行顺序：

1、python data_process.py

2、python train_768.py

3、python train_512.py