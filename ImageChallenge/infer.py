import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7,8"
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch.cuda.amp import autocast
def get_infer_transform_768():
    transform = A.Compose([
        A.Resize(768, 768),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    return transform

def get_infer_transform_512():
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    return transform

class seg_qyl(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()
        self.model = smp.UnetPlusPlus(
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
            )
    #
    @autocast()
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x


#预测直接resize图像的方式
def process_1(model, batch_size, test_jpg_all, shape_all, mask_all):
    print('process_1......')
    print('batch_size=', batch_size)
    model.eval()
    print('predicting....')
    with torch.no_grad():
        for index in tqdm(np.arange(0, len(test_jpg_all), batch_size)):#同时预测多张图像，为了优化预测时间
            test_data = torch.tensor(np.array(test_jpg_all[index : index+batch_size])).cuda()
            out = model(test_data)
            out = F.softmax(out, dim=1) #softmax函数映射成0-1的概率
            out = out.cpu().data.numpy()[:, 1] #只保留1的概率
            for jpg_id in range(out.shape[0]):
                mask_all[index+jpg_id] = out[jpg_id]

    print('save mask....')
    #还原回原始的图像尺寸大小
    mask_all_2 = []
    for jpg_id in tqdm(np.arange(len(test_jpg_all))):
        wd = shape_all[jpg_id][0]
        ht = shape_all[jpg_id][1]
        img = cv2.resize(mask_all[jpg_id], (wd, ht), interpolation=cv2.INTER_NEAREST)
        mask_all_2.append(img)
    return mask_all_2#返回

#预测滑动切片的图像方式
def process_2(model, batch_size, test_jpg_all, position_all, superposition_all, mask_all):
    print('process_2....')
    print('batch_size=', batch_size)
    model.eval()
    print('predicting....')
    with torch.no_grad():
        for index in tqdm(np.arange(0, len(test_jpg_all), batch_size)):#同时预测多张图像，为了优化预测时间
            test_data = torch.tensor(np.array(test_jpg_all[index : index+batch_size])).cuda()
            out = model(test_data)
            out = F.softmax(out, dim=1) #softmax函数映射成0-1的概率
            out = out.cpu().data.numpy()[:, 1] #只保留1的概率
            for jpg_id in range(out.shape[0]):
                pos = position_all[index+jpg_id][0] #是原来的第几张图片的
                start_1 = position_all[index+jpg_id][1]
                start_2 = position_all[index+jpg_id][2]
                mask = out[jpg_id]#不需要翻转
                mask_all[pos][start_1:start_1+512, start_2:start_2+512] += mask #累加mask概率
                superposition_all[pos][start_1:start_1+512, start_2:start_2+512] += np.ones((512, 512))##表示这个范围重合次数+1
        print('取平均....')
        for i in tqdm(np.arange(len(mask_all))):
            mask_all[i] /= superposition_all[i]#重合部分取平均
    
    return mask_all

def load_model(checkpoint_dir):
    print('loading model....')
    model_name = 'efficientnet-b6'#efficientnet-b4
    n_class=2
    model=seg_qyl(model_name,n_class).cuda()
    model= torch.nn.DataParallel(model)
    print(checkpoint_dir)
    checkpoints=torch.load(checkpoint_dir)
    if 'state_dict' in checkpoints.keys():
        model.load_state_dict(checkpoints['state_dict'])
    else:
        model.load_state_dict(checkpoints)
    return model




#获取要预测的图像文件ID
def get_file_ids(file_path):
    file_names = []
    file_ids = []
    id = 0
    for dir in os.listdir(file_path):
        #获取图像文件的ID
        file_names.append(dir.split('.')[0])
        id += 1
        file_ids.append(id)
    return file_names, file_ids

#把图像转换成768直接resize格式
def data_process_768_resize(file_path, file_names):
    print('data_process_768_resize...')
    transform = get_infer_transform_768()
    mask_all = np.zeros((len(file_ids), 768, 768))
    shape_all = []
    test_jpg_all = []
    print('loading data...')
    for jpg_name in tqdm(file_names):
        img_dir = file_path + '/' + jpg_name + '.jpg'
        image = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ht, wd, _ = image.shape
        shape_all.append([wd, ht])
        img = transform(image=image)['image'].cpu().data.numpy()
        test_jpg_all.append(img)
    return test_jpg_all, shape_all, mask_all

#把图像转换成512直接resize格式
def data_process_512_resize(file_path, file_names):
    print('data_process_512_resize...')
    transform = get_infer_transform_512()
    mask_all = np.zeros((len(file_ids), 512, 512))
    shape_all = []
    test_jpg_all = []
    print('loading data...')
    for jpg_name in tqdm(file_names):
        img_dir = file_path + '/' + jpg_name + '.jpg'
        image = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ht, wd, _ = image.shape
        shape_all.append([wd, ht])
        img = transform(image=image)['image'].cpu().data.numpy()
        test_jpg_all.append(img)
    return test_jpg_all, shape_all, mask_all


#把图像转换成512切片的格式
def data_process_512_slice(file_path, file_names, file_ids):
    print('data_process_512_slice...')
    d = 128#重合的像素值
    transform = get_infer_transform_512()
    mask_all = []
    test_jpg_all = []
    position_all = []
    superposition_all = []#表示重合的部分
    print('loading data...')
    index = 0
    for jpg_name in tqdm(file_names):
        img_dir = file_path + '/' + jpg_name + '.jpg'
        jpg_id = file_ids[index]
        index += 1
        IMAGE = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
        ht, wd, _ = IMAGE.shape
        mask_all.append(np.zeros((ht, wd)))#储存下对应尺寸的mask
        superposition_all.append(np.zeros((ht, wd)))
        for h in range(0, ht, 512-d):
            for w in range(0, wd, 512-d):
                start_1 = h
                start_2 = w
                if start_1 + 512 >= ht:
                    start_1 = ht - 512
                if start_2 + 512 >= wd:
                    start_2 = wd - 512

                image = IMAGE[start_1:start_1+512, start_2:start_2+512,:]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                img = transform(image=image)['image'].cpu().data.numpy()
                img_1 = img
                test_jpg_all.append(img_1)
                #储存下这是第几张图像的切片，以及在图像中的位置
                position_all.append([jpg_id-1, start_1, start_2])
    return test_jpg_all, position_all, superposition_all, mask_all

if __name__=="__main__":
    threshold = 0.40
    print('threshold=', threshold)

    #待预测的文件夹路径
    input_file_path = '/home/yao/data/fdata/files/dataset/test/img/'
    #输出预测结果的文件夹路径
    out_file_path = '/home/yao/data/fdata/files/dataset/test/images/'


    #加载的模型文件名
    checkpoint_dir_768_first = '/home/yao/data/fdata/files/outmodel/efficientnet-b6/ckpt/checkpoint-best_19.pth'
    checkpoint_dir_768_second = '/home/yao/data/fdata/files/outmodel/efficientnet-b6/ckpt/checkpoint-best_32.pth'
    checkpoint_dir_512_first = '/home/yao/data/fdata/files/outmodel/efficientnet-b6/ckpt/checkpoint-best_20.pth'
    checkpoint_dir_512_second = '/home/yao/data/fdata/files/outmodel/efficientnet-b6/ckpt/checkpoint-best_42.pth'

    #预测每种图像尺寸的batch_size大小
    batch_size_768 = 64
    batch_size_512 = 128

    #获取预测的文件ID
    file_names, file_ids = get_file_ids(input_file_path)

    #------------------------预测推理直接768resize图像----------------------------------
    #处理数据格式
    test_jpg_all, shape_all, mask_all = data_process_768_resize(input_file_path, file_names)

    model_768_first = load_model(checkpoint_dir_768_first)
    mask_768_resize_first = process_1(model_768_first, batch_size_768, test_jpg_all, shape_all, mask_all)
    del model_768_first


    model_768_second = load_model(checkpoint_dir_768_second)
    mask_768_resize_second = process_1(model_768_second, batch_size_768, test_jpg_all, shape_all, mask_all)
    del model_768_second

    del test_jpg_all
    del shape_all
    del mask_all

    #----------------------预测推理直接512resize图像-------------------------------
    #处理数据格式
    test_jpg_all, shape_all, mask_all = data_process_512_resize(input_file_path, file_names)

    model_512_first = load_model(checkpoint_dir_512_first)
    mask_512_resize_first = process_1(model_512_first, batch_size_512, test_jpg_all, shape_all, mask_all)
    del model_512_first

    model_512_second = load_model(checkpoint_dir_512_second)
    mask_512_resize_second = process_1(model_512_second, batch_size_512, test_jpg_all, shape_all, mask_all)
    del model_512_second

    del test_jpg_all
    del shape_all
    del mask_all

    #---------------------预测推理512滑动切片图像------------------------------------
    #处理数据格式
    test_jpg_all, position_all, superposition_all, mask_all = data_process_512_slice(input_file_path, file_names, file_ids)


    model_512_first = load_model(checkpoint_dir_512_first)
    mask_512_slice_first = process_2(model_512_first, batch_size_512, test_jpg_all, position_all, superposition_all, mask_all)
    del model_512_first

    model_512_second = load_model(checkpoint_dir_512_second)
    mask_512_slice_second = process_2(model_512_second, batch_size_512, test_jpg_all, position_all, superposition_all, mask_all)
    del model_512_second

    del test_jpg_all
    del position_all
    del superposition_all
    del mask_all


    #--------------------------合并多模型预测结果---------------------------

    mask_all = []
    print('合并....')
    for i in tqdm(np.arange(len(file_names))):
        #按一定权重合并
        mask_768 = mask_768_resize_first[i] * 0.5 + mask_768_resize_second[i] * 0.5
        mask_512_1 = mask_512_resize_first[i] * 0.2 + mask_512_slice_first[i] * 0.8
        mask_512_2 = mask_512_resize_second[i] * 0.2 + mask_512_slice_second[i] * 0.8
        mask_512 = mask_512_1 * 0.5 + mask_512_2 * 0.5
        mask = mask_768 * 0.5 + mask_512 * 0.5
        mask_all.append(mask)
        

    print('final save mask....')
    for jpg_id in tqdm(np.arange(len(file_names))):
        img = ((mask_all[jpg_id] >= threshold) * 255).astype(np.uint8)
        out_path = out_file_path + '/' + file_names[jpg_id] + '.png'
        cv2.imwrite(out_path, img)
    