import numpy as np
import torch
import torch.nn.functional as F
import Fusion_strategy as fus
from Network import Encoder_Base,Encoder_Detail,Decoder,Encoder_Middle,Encoder_Base_Detail

device='cuda'

def output_img(x):
    return x.cpu().detach().numpy()[0,0,:,:]
def l1_addition(y1,y2,window_width=1):
      ActivityMap1 = y1.abs()
      ActivityMap2 = y2.abs()

      kernel = torch.ones(2*window_width+1,2*window_width+1)/(2*window_width+1)**2
      kernel = kernel.to(device).type(torch.float32)[None,None,:,:]
      kernel = kernel.expand(y1.shape[1],y1.shape[1],2*window_width+1,2*window_width+1)
      ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=window_width)
      ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=window_width)
      WeightMap1 = ActivityMap1/(ActivityMap1+ActivityMap2)
      WeightMap2 = ActivityMap2/(ActivityMap1+ActivityMap2)
      return WeightMap1*y1+WeightMap2*y2

def  Test_fusion_all(img_test1, img_test2, addition_mode='Sum',mix_mode=0,
                 Encoder_b_path=[],Encoder_m_path=[],Encoder_d_path=[],Decoder_path=[]  ):

    Encoder_Base_Test = Encoder_Base().to(device)
    Encoder_Base_Test.load_state_dict(torch.load(Encoder_b_path))
    Encoder_Middle_Test = Encoder_Middle().to(device)
    Encoder_Middle_Test.load_state_dict(torch.load(Encoder_m_path))
    Encoder_Detail_Test = Encoder_Detail().to(device)
    Encoder_Detail_Test.load_state_dict(torch.load(Encoder_d_path))
    Decoder_Test = Decoder().to(device)
    Decoder_Test.load_state_dict(torch.load(Decoder_path))

    Encoder_Base_Test.eval()
    Encoder_Middle_Test.eval()
    Encoder_Detail_Test.eval()
    Decoder_Test.eval()

    img_test1 = np.array(img_test1, dtype='float32') / 255  # 将其转换为一个矩阵
    img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))

    img_test2 = np.array(img_test2, dtype='float32') / 255  # 将其转换为一个矩阵
    img_test2 = torch.from_numpy(img_test2.reshape((1, 1, img_test2.shape[0], img_test2.shape[1])))

    img_test1 = img_test1.cuda()
    img_test2 = img_test2.cuda()

    with torch.no_grad():
        B_K_IR, _, _ = Encoder_Base_Test(img_test1)
        B_K_VIS, _, _ = Encoder_Base_Test(img_test2)
        D_K_IR, _, _ = Encoder_Detail_Test(img_test1)
        D_K_VIS, _, _ = Encoder_Detail_Test(img_test2)
        M_K_IR = Encoder_Middle_Test(img_test1)
        M_K_VIS= Encoder_Middle_Test(img_test2)

    if mix_mode == 0:
        if addition_mode   == 'Average':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'CHA':#不动
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'l1_norm':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'max':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'nuclear':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'Sum':#不动
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.add_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)

    # ---------------------------------------------------------
    elif mix_mode == 1:
        if addition_mode   == 'Average':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.avg_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'CHA':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'l1_norm':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.l1_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'max':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.max_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'nuclear':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.nuclear_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'Sum':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
    elif mix_mode == 2:
        if addition_mode   == 'Average':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.avg_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'CHA':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'l1_norm':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.l1_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'max':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'nuclear':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.nuclear_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'Sum':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.add_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
    elif mix_mode == 3:
        if addition_mode   == 'Average':
            F_b = fus.avg_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'CHA':
            F_b = fus.spatial_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'l1_norm':
            F_b = fus.l1_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'max':
            F_b = fus.max_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'nuclear':
            F_b = fus.nuclear_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'Sum':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
    elif mix_mode == 4:
        if addition_mode   == 'Average':
            F_b = fus.avg_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'CHA':
            F_b = fus.spatial_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'l1_norm':
            F_b = fus.l1_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'max':
            F_b = fus.max_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'nuclear':
            F_b = fus.nuclear_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'Sum':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
    elif mix_mode == 5:
        if addition_mode   == 'Average':
            F_b = fus.avg_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'CHA':
            F_b = fus.spatial_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'l1_norm':
            F_b = fus.l1_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'max':
            F_b = fus.max_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'nuclear':
            F_b = fus.nuclear_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'Sum':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
    elif mix_mode == 6:
        if addition_mode   == 'Average':
            F_b = fus.avg_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'CHA':
            F_b = fus.spatial_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'l1_norm':
            F_b = fus.l1_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'max':
            F_b = fus.max_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'nuclear':
            F_b = fus.nuclear_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)
        elif addition_mode == 'Sum':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_d = fus.max_fusion(D_K_IR, D_K_VIS)
            F_m = fus.spatial_fusion(M_K_IR, M_K_VIS)

    with torch.no_grad():
        Out = Decoder_Test(F_b, F_d,F_m)

    img_test1.cpu()
    img_test2.cpu()


    Encoder_Base_Test.eval()
    Encoder_Middle_Test.eval()
    Encoder_Detail_Test.eval()
    Decoder_Test.eval()

    Encoder_Base_Test.cpu()
    Encoder_Middle_Test.cpu()
    Encoder_Detail_Test.cpu()
    Decoder_Test.cpu()


    return output_img(Out)
def  Test_fusion(img_test1, img_test2, addition_mode='Sum',Encoder_b_d_path=[],Encoder_m_path=[],Decoder_path=[]):

    Encoder_Base_Detail_Test = Encoder_Base_Detail().to(device)
    Encoder_Base_Detail_Test.load_state_dict(torch.load(Encoder_b_d_path))
    Encoder_Middle_Test = Encoder_Middle().to(device)
    Encoder_Middle_Test.load_state_dict(torch.load(Encoder_m_path))
    Decoder_Test = Decoder().to(device)
    Decoder_Test.load_state_dict(torch.load(Decoder_path))

    Encoder_Base_Detail_Test.eval()
    Encoder_Middle_Test.eval()
    Decoder_Test.eval()

    img_test1 = np.array(img_test1, dtype='float32') / 255  # 将其转换为一个矩阵
    img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))

    img_test2 = np.array(img_test2, dtype='float32') / 255  # 将其转换为一个矩阵
    img_test2 = torch.from_numpy(img_test2.reshape((1, 1, img_test2.shape[0], img_test2.shape[1])))

    img_test1 = img_test1.cuda()
    img_test2 = img_test2.cuda()

    with torch.no_grad():
        # B_K_IR, _, _ = Encoder_Base_Test(img_test1)
        # B_K_VIS, _, _ = Encoder_Base_Test(img_test2)
        # D_K_IR, _, _ = Encoder_Detail_Test(img_test1)
        # D_K_VIS, _, _ = Encoder_Detail_Test(img_test2)
        B_K_IR, D_K_IR, _ = Encoder_Base_Detail_Test(img_test1)
        B_K_VIS, D_K_VIS, _ = Encoder_Base_Detail_Test(img_test2)

        M_K_IR  = Encoder_Middle_Test(img_test1)
        M_K_VIS= Encoder_Middle_Test(img_test2)

    if addition_mode == 'Sum':
        F_b = fus.add_fusion(B_K_IR, B_K_VIS)
        F_d = fus.add_fusion(D_K_IR, D_K_VIS)
        F_m = fus.add_fusion(M_K_IR, M_K_VIS)
    elif addition_mode == 'Average':
        F_b = fus.add_fusion(B_K_IR, B_K_VIS)
        F_d = fus.spatial_fusion(D_K_IR, D_K_VIS)
        F_m = fus.add_fusion(M_K_IR, M_K_VIS)
    elif addition_mode == 'l1_norm':
        F_b = fus.add_fusion(B_K_IR, B_K_VIS)
        F_d = fus.l1_fusion(D_K_IR, D_K_VIS)
        F_m = fus.add_fusion(M_K_IR, M_K_VIS)
    elif addition_mode == 'max':
        F_b = fus.add_fusion(B_K_IR, B_K_VIS)
        F_d = fus.max_fusion(D_K_IR, D_K_VIS)
        F_m = fus.add_fusion(M_K_IR, M_K_VIS)
    elif addition_mode == 'nuclear':
        F_b = fus.max_fusion(B_K_IR, B_K_VIS)
        F_d = fus.add_fusion(D_K_IR, D_K_VIS)
        F_m = fus.add_fusion(M_K_IR, M_K_VIS)
    elif addition_mode == 'CHA':
        F_b = fus.add_fusion(B_K_IR, B_K_VIS)
        F_d = fus.add_fusion(D_K_IR, D_K_VIS)
        F_m = fus.max_fusion(M_K_IR, M_K_VIS)


    with torch.no_grad():
        Out = Decoder_Test(F_b, F_d,F_m)

    img_test1.cpu()
    img_test2.cpu()


    # Encoder_Base_Test.eval()
    # Encoder_Detail_Test.eval()
    Encoder_Base_Detail_Test.eval()
    Encoder_Middle_Test.eval()
    Decoder_Test.eval()

    Encoder_Base_Detail_Test.cpu()
    Encoder_Middle_Test.cpu()
    # Encoder_Base_Test.cpu()
    # Encoder_Detail_Test.cpu()
    Decoder_Test.cpu()

    return output_img(Out)
