from config import Config
import os
from tqdm import tqdm
import numpy as np
import cv2

dir_ls = ['train_images/', 'val_images/', 'train_labels/', 'val_labels/']
for dir in dir_ls:
    if os.path.exists(Config.data_dir_512) == False:
        os.mkdir(Config.data_dir_512)
    if os.path.exists(Config.data_dir_768) == False:
        os.mkdir(Config.data_dir_768)

    if os.path.exists(Config.data_dir_512  + dir) == False:
        os.mkdir(Config.data_dir_512 + dir)

    if os.path.exists(Config.data_dir_768 + dir) == False:
        os.mkdir(Config.data_dir_768 + dir)



def get_file_ids(path):
    file_ids = []
    for dir in os.listdir(path):
        file_ids.append(dir.split('.')[0])
    return file_ids

def jpg_slice(IMAGE, MASK, d, file_id, key):
    ans = 0
    ht, wd, _ = IMAGE.shape

    #滑动切片，重合部分为d
    for h in range(0, ht, 512-d):
        for w in range(0, wd, 512-d):
            start_1 = h
            start_2 = w
            if start_1 + 512 >= ht:
                start_1 = ht - 512
            if start_2 + 512 >= wd:
                start_2 = wd - 512
            image = IMAGE[start_1: start_1+512, start_2: start_2+512]
            mask = MASK[start_1: start_1+512, start_2: start_2+512]

            ans += 1

            img_path = Config.data_dir_512 + '/train_images/' + str(file_id) + '_' + str(ans) + '.jpg'
            mask_path = Config.data_dir_512 + '/train_labels/' + str(file_id) + '_' + str(ans) + '.png'
            img_path_2 = Config.data_dir_512 + '/val_images/' + str(file_id) + '_' + str(ans) + '.jpg'
            mask_path_2 = Config.data_dir_512 + '/val_labels/' + str(file_id) + '_' + str(ans) + '.png'

            if np.max(mask) == 2:
                cv2.imwrite(img_path, image)
                cv2.imwrite(mask_path, mask)
                if key == 1:
                    cv2.imwrite(img_path_2, image)
                    cv2.imwrite(mask_path_2, mask)         

            if w + 512 == wd:
                break
        if h + 512 == ht:
            break
        
if __name__ == '__main__':
    file_ids_1 = get_file_ids(Config.source_file_path + '/train/img')
    file_ids_2 = get_file_ids(Config.source_file_path + '/train2/img')
    file_ids_3 = get_file_ids(Config.source_file_path + '/train3/img')
    file_ids_4 = get_file_ids(Config.source_file_path + '/train4/img')
    #---------------------768resize--------------------------
    print('loading 768...........')
    file_id = 0
    for jpg_id in tqdm(file_ids_1):
        img = cv2.imread(Config.source_file_path + '/train/img/' + jpg_id + '.jpg')
        mask = cv2.imread(Config.source_file_path + '/train/mask/' + jpg_id + '.png')
        mask[mask <= 127] = 1
        mask[mask > 127] = 2
        file_id += 1
        cv2.imwrite(Config.data_dir_768 + '/train_images/' + str(file_id) + '.jpg', img)
        cv2.imwrite(Config.data_dir_768 + '/train_labels/' + str(file_id) + '.png', mask)
    
    for jpg_id in tqdm(file_ids_2):
        img = cv2.imread(Config.source_file_path + '/train2/img/' + jpg_id + '.jpg')
        mask = cv2.imread(Config.source_file_path + '/train2/mask/' + jpg_id + '.png')
        mask[mask <= 127] = 1
        mask[mask > 127] = 2
        file_id += 1
        cv2.imwrite(Config.data_dir_768 + '/train_images/' + str(file_id) + '.jpg', img)
        cv2.imwrite(Config.data_dir_768 + '/train_labels/' + str(file_id) + '.png', mask)
    
    for jpg_id in tqdm(file_ids_3):
        img = cv2.imread(Config.source_file_path + '/train3/img/' + jpg_id + '.jpg')
        mask = cv2.imread(Config.source_file_path + '/train3/mask/' + jpg_id + '.png')
        mask[mask <= 127] = 1
        mask[mask > 127] = 2
        file_id += 1
        cv2.imwrite(Config.data_dir_768 + '/train_images/' + str(file_id) + '.jpg', img)
        cv2.imwrite(Config.data_dir_768 + '/train_labels/' + str(file_id) + '.png', mask)
    

    for jpg_id in tqdm(file_ids_4):
        img = cv2.imread(Config.source_file_path + '/train4/img/' + jpg_id + '.jpg')
        mask = cv2.imread(Config.source_file_path + '/train4/mask/' + jpg_id + '.png')
        mask[mask <= 127] = 1
        mask[mask > 127] = 2
        file_id += 1
        cv2.imwrite(Config.data_dir_768 + '/train_images/' + str(file_id) + '.jpg', img)
        cv2.imwrite(Config.data_dir_768 + '/train_labels/' + str(file_id) + '.png', mask)
        cv2.imwrite(Config.data_dir_768 + '/val_images/' + str(file_id) + '.jpg', img)
        cv2.imwrite(Config.data_dir_768 + '/val_labels/' + str(file_id) + '.png', mask)
    

    #----------------------------512滑动切片------------------------------------------
    print('loading 512...........')
    d = 128
    file_id = 0
    for jpg_id in tqdm(file_ids_1):
        IMAGE = cv2.imread(Config.source_file_path + '/train/img/' + jpg_id + '.jpg')
        MASK = cv2.imread(Config.source_file_path + '/train/mask/' + jpg_id + '.png')
        MASK[MASK <= 127] = 1
        MASK[MASK > 127] = 2
        file_id += 1
        jpg_slice(IMAGE, MASK, d, file_id, 0)
    for jpg_id in tqdm(file_ids_2):
        IMAGE = cv2.imread(Config.source_file_path + '/train2/img/' + jpg_id + '.jpg')
        MASK = cv2.imread(Config.source_file_path + '/train2/mask/' + jpg_id + '.png')
        MASK[MASK <= 127] = 1
        MASK[MASK > 127] = 2
        file_id += 1
        jpg_slice(IMAGE, MASK, d, file_id, 0)
    for jpg_id in tqdm(file_ids_3):
        IMAGE = cv2.imread(Config.source_file_path + '/train3/img/' + jpg_id + '.jpg')
        MASK = cv2.imread(Config.source_file_path + '/train3/mask/' + jpg_id + '.png')
        MASK[MASK <= 127] = 1
        MASK[MASK > 127] = 2
        file_id += 1
        jpg_slice(IMAGE, MASK, d, file_id, 0)

    for jpg_id in tqdm(file_ids_4):
        IMAGE = cv2.imread(Config.source_file_path + '/train4/img/' + jpg_id + '.jpg')
        MASK = cv2.imread(Config.source_file_path + '/train4/mask/' + jpg_id + '.png')
        MASK[MASK <= 127] = 1
        MASK[MASK > 127] = 2
        file_id += 1
        jpg_slice(IMAGE, MASK, d, file_id, 1)


