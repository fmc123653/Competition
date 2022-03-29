import imp
import cv2
import os
import numpy as np
import random
import math
from tqdm import tqdm
def cal_new_mask(new_img, img, mask):
    """
    new img: 二次篡改的图片
    img：原来训练集中的图片
    mask：二次篡改前的标签，0-255
    """
    diff_img = cv2.absdiff(new_img, img)
    diff = np.linalg.norm(diff_img, ord=np.inf, axis=2)
    #print(diff.shape, mask.shape)
    _, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
    
    new_mask = diff + mask
    new_mask = np.clip(new_mask, 0, 255)
    
    return new_mask

def rand_bbox(size):
    # opencv格式的size
    W = size[1]
    H = size[0]
        
    cut_rat_w = random.random()*0.1 + 0.05
    cut_rat_h = random.random()*0.1 + 0.05

    cut_w = int(W * cut_rat_w)
    cut_h = int(H * cut_rat_h)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W) # 左上
    bby1 = np.clip(cy - cut_h // 2, 0, H) # 左上
    bbx2 = np.clip(cx + cut_w // 2, 0, W) # 右下
    bby2 = np.clip(cy + cut_h // 2, 0, H) # 右下

    return bbx1, bby1, bbx2, bby2

def copy_move(img1, img2, msk, is_plot=False):
    img = img1.copy()
    size = img.shape # h,w,c
    W = size[1]
    H = size[0]

    if img2 is None: # 从自身复制粘贴
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)

        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))

        img[bby1+y_move:bby2+y_move, bbx1+x_move:bbx2+x_move, :] = img[bby1:bby2, bbx1:bbx2, :]
        
    else: # 从其他图像复制粘贴
        bbx1, bby1, bbx2, bby2 = rand_bbox(img2.shape)

        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))

        img[bby1+y_move:bby2+y_move, bbx1+x_move:bbx2+x_move, :] = img2[bby1:bby2, bbx1:bbx2, :]

    """ 
    这里改了一下dave的代码中直接根据修改区域计算mask，因为我发现有时候裁剪了一样的区域粘贴过来，
    计算方法是二次篡改的图片减去原图，有差异的地方叠加到原来的mask上
    """
    msk = cal_new_mask(img, img1, msk)
    if is_plot: # 标出二次窜改的区域，主要是为了debug，生成图像的时候记得改成false
        img =  cv2.rectangle(img, pt1=[bbx1+x_move, bby1+y_move], pt2=[bbx2+x_move, bby2+y_move], color=(255,0,0), thickness=3)       

    return img, msk

def erase(img1, msk, is_plot=False):
    img = img1.copy()
    size = img.shape # h,w,c
    W = size[1]
    H = size[0]

    def midpoint(x1, y1, x2, y2):
        x_mid = int((x1 + x2)/2)
        y_mid = int((y1 + y2)/2)
        return (x_mid, y_mid)

    bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
    # print(bbx1, bby1, bbx2, bby2)
    
    x_mid0, y_mid0 = midpoint(bbx1, bby1, bbx1, bby2)
    x_mid1, y_mid1 = midpoint(bbx2, bby1, bbx2, bby2)
    thickness = int(math.sqrt((bby2-bby1)**2))
    
    mask_ = np.zeros(img.shape[:2], dtype="uint8")    
    cv2.line(mask_, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)
    
    # cv2.imwrite("mask_.jpg", mask_)
    img = cv2.inpaint(img, mask_, 7, cv2.INPAINT_NS)
    
    msk = cal_new_mask(img1, img, msk)
    
    if is_plot:             
        img = cv2.rectangle(img, pt1=[bbx1, bby1], pt2=[bbx2, bby2], color=(255,0,0), thickness=3)
        
    return img, msk

if __name__ == "__main__":
    np.random.seed(2022)
    create_file_num = 1000
    train = np.random.choice(4000, create_file_num, replace=False)
    train2 = np.random.choice(2005, create_file_num, replace=False)

    print('create_file_num = ',create_file_num)
    for dir in tqdm(np.arange(0, create_file_num)):

        img_path = '/files/dataset/train/img/' + str(train[dir] + 1) + '.jpg'
        img_path2 = "/files/dataset/train2/img/" + str(train2[dir] + 1) + '.jpg'
        mask_path = '/files/dataset/train/mask/' + str(train[dir] + 1) + '.png'
        img = cv2.imread(img_path)
        img2 = cv2.imread(img_path2)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 从同一张图片中复制粘贴一块
        new_img, new_mask = copy_move(img, img2=None, msk=mask, is_plot=False)    

        new_img_path = '/files/dataset/new_train/img/new_one_' + str(dir+1) + '.jpg'
        new_mask_path = '/files/dataset/new_train/mask/new_one_' + str(dir+1) + '.png'
        cv2.imwrite(new_img_path, new_img)
        cv2.imwrite(new_mask_path, new_mask)
        
        #从另外一张图片中复制一块
        new_img, new_mask = copy_move(img, img2=img2, msk=mask, is_plot=False)    
        new_img_path = '/files/dataset/new_train/img/new_two_' + str(dir+1) + '.jpg'
        new_mask_path = '/files/dataset/new_train/mask/new_two_' + str(dir+1) + '.png'
        cv2.imwrite(new_img_path, new_img)
        cv2.imwrite(new_mask_path, new_mask)
        
        # 随机擦除当前图片中的一块
        new_img, new_mask = erase(img, msk=mask, is_plot=False)
        new_img_path = '/files/dataset/new_train/img/new_three_' + str(dir+1) + '.jpg'
        new_mask_path = '/files/dataset/new_train/mask/new_three_' + str(dir+1) + '.png'
        cv2.imwrite(new_img_path, new_img)
        cv2.imwrite(new_mask_path, new_mask)

