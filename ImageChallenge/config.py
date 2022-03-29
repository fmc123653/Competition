class Config:
    #512_first切片模型配置参数
    device_512 = "4,5,6,7,8"#训练模型设备ID
    save_model_name_512_first = "checkpoint-best_512_first.pth"#保存的模型文件名
    data_dir_512_first = "/files/dataset/train_val_split_512_first/"#数据路径
    outmodel_dir_512 = "/files/outmodel/"#保存模型的文件地址
    epochs_512 = 2000#训练的epoch
    batch_size_512 = 64#训练的batch_size

    #512_second切片模型配置参数
    device_512 = "4,5,6,7,8"#训练模型设备ID
    save_model_name_512_second = "checkpoint-best_512_second.pth"#保存的模型文件名
    data_dir_512_second = "/files/dataset/train_val_split_512_second/"#数据路径
    outmodel_dir_512 = "/files/outmodel/"#保存模型的文件地址
    epochs_512 = 1000#训练的epoch
    batch_size_512 = 64#训练的batch_size


    #768_first直接resize模型配置参数
    device_768 = "2,3"
    save_model_name_768_first = "checkpoint-best_768_first.pth"
    data_dir_768_first = "/files/dataset/train_val_split_768_first/"
    outmodel_dir_768 = "/files/outmodel/"
    epochs_768 = 2000
    batch_size_768 = 64


    #768_second直接resize模型配置参数
    device_768 = "2,3"
    save_model_name_768_second = "checkpoint-best_768_second.pth"
    data_dir_768_second = "/files/dataset/train_val_split_768_second/"
    outmodel_dir_768 = "/files/outmodel/"
    epochs_768 = 1000
    batch_size_768 = 64