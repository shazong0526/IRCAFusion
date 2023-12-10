#####################################################################################
#  EN、SD、 SF     Refer to the following code
# VIF、AG、SCD     Refer to https://github.com/jiayi-ma/SeAFusion/tree/main/Evaluation
######################################################################################
import cv2
from os.path import join
from os import listdir
from Args import en_std_tno,en_std_flir,en_std_nir
import numpy as np

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images
def EN(image):
    grayscale_num = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grayscale_num[int(image[i][j])] += 1
    temp = 0
    for i in range(len(grayscale_num)):
        p = grayscale_num[i] / np.sum(grayscale_num)
        if p != 0:
            temp -= p * np.log2(p)
    return temp
def SD(image):
    avg = np.mean(image)
    temp = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp += np.square(image[i][j] - avg)

    return np.sqrt(temp / (image.shape[0] * image.shape[1]))
def SF(image):
    CF = 0
    RF = 0
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            RF += np.square(image[i][j] - image[i][j-1])
            CF += np.square(image[i][j] - image[i-1][j])
    RF = RF / (image.shape[0] * image.shape[1])
    CF = CF / (image.shape[0] * image.shape[1])
    return np.sqrt(RF + CF)

def AG(image):
    temp = 0
    for i in range(0, image.shape[0] - 1):
        for j in range(0, image.shape[1] - 1):
            temp += np.sqrt((np.square(image[i][j] - image[i][j+1]) + np.square(image[i][j] - image[i+1][j]))/2)
    AG = temp / (image.shape[0] * image.shape[1])
    return AG

def computer_result_new(path_fusion,path_ir="./Datasets/IR/",path_vi="./Datasets/VIS/",method='Sum',dataset=1,name=[]):

    # images_ir = list_images(path_ir)
    # images_vi = list_images(path_vi)
    images_fu = list_images(path_fusion)


    en_list=[]
    sd_list=[]
    sf_list=[]
    ag_list=[]
    len_str_f = len(path_fusion)
    #en_std_tno,en_std_flir,en_std_nir
    if dataset ==1:
        en_std = en_std_tno
    elif dataset == 2:
        en_std = en_std_flir
    elif dataset == 3:
        en_std = en_std_nir
    else:
        en_std = 5

    if 20>=en_std:
        for i in range(  0,len(images_fu) ):
        # if i%6==0:
            if i >= 0:
                # print(images_fu[i])
                # image_ir =  cv2.imread(images_ir[i], 0).astype(np.float32)
                # image_vi =  cv2.imread(images_vi[i], 0).astype(np.float32)
                image_F =   cv2.imread(images_fu[i], 0).astype(np.float32)
                str = images_fu[i]

                EN_temp = EN(image_F)
                en_list.append(EN_temp)
                # SD_temp  = 0
                SD_temp = SD(image_F)
                sd_list.append(SD_temp)
                # SF_temp = 0
                SF_temp = SF(image_F)
                sf_list.append(SF_temp)

                # print('%30s     %8.6f     %8.6f     %8.6f   '% (
                #            str[39:-1],EN_temp,SD_temp, SF_temp  ))

                # AG_temp = AG(image_F)
                # ag_list.append(AG_temp)

        sd_avg =     np.mean(sd_list)
        sf_avg =     np.mean(sf_list)
        en_avg =    np.mean(en_list)
        # vif_avg =   np.mean(vif_list)
        # ag_avg =     np.mean(ag_list)

    else:
        sd_avg =    0
        sf_avg =    0
        vif_avg = 0
        en_avg =    0

    print('%s        %f  %f  %f   '% (name,en_avg,sd_avg,sf_avg  ))
    return en_avg
    # print('MAX  EN:%7.4f  SD:%8.4f  SF:%8.4f  AG:%7.4f  '% (
    #            np.max(en_list),np.max(sd_list), np.max(sf_list), np.max(ag_list)  ))
    # print('MIN  EN:%7.4f  SD:%8.4f  SF:%8.4f  AG:%7.4f  '% (
    #            np.min(en_list),np.min(sd_list), np.min(sf_list), np.min(ag_list)  ))

def computer_result(path_fusion,path_ir="./Datasets/IR/",path_vi="./Datasets/VIS/",method='Sum',dataset=1):
    # path_fusion = "C:/0py/000baseline/OR_AUIF/Test_result/Fusion/"
    # images_ir = list_images(path_ir)
    # images_vi = list_images(path_vi)
    images_fu = list_images(path_fusion)
    #print(path_fusion)

    en_list=[]
    sd_list=[]
    sf_list=[]
    ag_list=[]
    qabf_list=[]
    len_str_f = len(path_fusion)
    #en_std_tno,en_std_flir,en_std_nir
    if dataset ==1:
        en_std = en_std_tno

    elif dataset == 2:
        en_std = en_std_flir

    elif dataset == 3:
        en_std = en_std_nir

    else:
        en_std = 6.5
    en_std = 0




    for i in range(  0,len(images_fu) ):

        image_F =   cv2.imread(images_fu[i], 0).astype(np.float32)
        EN_temp = EN(image_F)
        en_list.append(EN_temp)
    en_avg = np.mean(en_list)
    if en_avg>=en_std:
        for i in range(  0,len(images_fu) ):
        # if i%6==0:
            if i >= 0:
                # print(images_fu[i])
                # image_ir =  cv2.imread(images_ir[i], 0).astype(np.float32)
                # image_vi =  cv2.imread(images_vi[i], 0).astype(np.float32)
                image_F =   cv2.imread(images_fu[i], 0).astype(np.float32)
                str = images_fu[i]


                SD_temp = SD(image_F)
                sd_list.append(SD_temp)

                SF_temp = SF(image_F)
                sf_list.append(SF_temp)

                # AG_temp = AG(image_F)
                # ag_list.append(AG_temp)

            # QABF_temp = Qabf(image_ir,image_vi,image_F)
            # qabf_list.append(QABF_temp)
        sd_avg =     np.mean(sd_list)
        sf_avg =     np.mean(sf_list)
        # ag_avg =     np.mean(ag_list)

    else:
        sd_avg =    0
        sf_avg =    0
        # ag_avg =    0

    print('%12s        %f  %f  %f   '% (method,en_avg,sd_avg,sf_avg  ))
    return en_avg
    # print('MAX  EN:%7.4f  SD:%8.4f  SF:%8.4f  AG:%7.4f  '% (
    #            np.max(en_list),np.max(sd_list), np.max(sf_list), np.max(ag_list)  ))
    # print('MIN  EN:%7.4f  SD:%8.4f  SF:%8.4f  AG:%7.4f  '% (
    #            np.min(en_list),np.min(sd_list), np.min(sf_list), np.min(ag_list)  ))
