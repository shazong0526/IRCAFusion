import torch
from torch import nn
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import kornia
from Network import Encoder_Base,Encoder_Detail,Decoder,Encoder_Middle
from Network import Coord_Inter_res_3_Fusion
from Args import train_data_path,train_path,device,batch_size,lr,is_cuda,log_interval,img_size,epochs
from Args import loss_min_temp,exp_name,ssim_w,sobel_w,en_std_tno,en_std_flir,en_std_nir,mse_ir_w
from skimage.io import imsave
from Fusion_metrics import computer_result
import os
import shutil
import Fusion_strategy as fus

models_path = ""
fusion_model = ""
Encoder_b_path_TNO ="Encoder_Bt_Best_"+  "0.09_75_30_17"  +     ".model"
Encoder_m_path_TNO ="Encoder_Mt_Best_"+  "0.09_75_30_17"  +     ".model"
Encoder_d_path_TNO ="Encoder_Dt_Best_"+  "0.09_75_30_17"  +     ".model"
Decoder_path_TNO   ="Decoder_t_Best_"+   "0.09_75_30_17"  +     ".model"

Encoder_b_path_FLIR ="Encoder_Bt_Best_"+  "0.09_38_58_29"  +     ".model"
Encoder_m_path_FLIR ="Encoder_Mt_Best_"+  "0.09_38_58_29"  +     ".model"
Encoder_d_path_FLIR ="Encoder_Dt_Best_"+  "0.09_38_58_29"  +     ".model"
Decoder_path_FLIR   ="Decoder_t_Best_"+   "0.09_38_58_29"  +     ".model"

Encoder_b_path_NIR ="Encoder_Bt_Best_"+  "0.09_38_58_29"  +     ".model"
Encoder_m_path_NIR ="Encoder_Mt_Best_"+  "0.09_38_58_29"  +     ".model"
Encoder_d_path_NIR="Encoder_Dt_Best_"+  "0.09_38_58_29"  +     ".model"
Decoder_path_NIR  ="Decoder_t_Best_"+   "0.09_38_58_29"  +     ".model"

#Encoder_b_path = Encoder_b_path_FLIR
if fusion_model == "TNO":
    Encoder_b_path = Encoder_b_path_TNO
    Encoder_m_path = Encoder_m_path_TNO
    Encoder_d_path = Encoder_d_path_TNO
    Decoder_path = Decoder_path_TNO
elif fusion_model == "FLIR":
    Encoder_b_path = Encoder_b_path_FLIR
    Encoder_m_path = Encoder_m_path_FLIR
    Encoder_d_path = Encoder_d_path_FLIR
    Decoder_path = Decoder_path_FLIR
elif fusion_model == "NIR":
    Encoder_b_path = Encoder_b_path_NIR
    Encoder_m_path = Encoder_m_path_NIR
    Encoder_d_path = Encoder_d_path_NIR
    Decoder_path = Decoder_path_NIR
else:
    Encoder_b_path = Encoder_b_path_TNO
    Encoder_m_path = Encoder_m_path_TNO
    Encoder_d_path = Encoder_d_path_TNO
    Decoder_path = Decoder_path_TNO
# =============================================================================
# Test
def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
def deldir(path):
    mydir =  path
    try:
        shutil.rmtree(mydir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
def output_img(x):
    return x.cpu().detach().numpy()[0,0,:,:]
def Test_net_fusion(img_test1, img_test2,Net_Fusion_Path2=[],Test_data_choose=[],addition_mode=[]):
#def Test_net_fusion(img_test1, img_test2,Net_Fusion_Path0=[],Net_Fusion_Path1=[],Net_Fusion_Path2=[],Test_data_choose=[],addition_mode=[]):
    if Test_data_choose == 'Test_data_TNO':
        Encoder_b_path = Encoder_b_path_TNO
        Encoder_m_path = Encoder_m_path_TNO
        Encoder_d_path = Encoder_d_path_TNO
        Decoder_path   = Decoder_path_TNO
    elif Test_data_choose == 'Test_data_FLIR':
        Encoder_b_path = Encoder_b_path_FLIR
        Encoder_m_path = Encoder_m_path_FLIR
        Encoder_d_path = Encoder_d_path_FLIR
        Decoder_path = Decoder_path_FLIR
    elif Test_data_choose == 'Test_data_NIR_Country':
        Encoder_b_path = Encoder_b_path_NIR
        Encoder_m_path = Encoder_m_path_NIR
        Encoder_d_path = Encoder_d_path_NIR
        Decoder_path = Decoder_path_NIR
    else:
        Encoder_b_path = Encoder_b_path_FLIR
        Encoder_m_path = Encoder_m_path_FLIR
        Encoder_d_path = Encoder_d_path_FLIR
        Decoder_path = Decoder_path_FLIR


    Encoder_Base_Test = Encoder_Base().to(device)
    Encoder_Base_Test.load_state_dict(torch.load(   models_path +Encoder_b_path))
    Encoder_Middle_Test = Encoder_Middle().to(device)
    Encoder_Middle_Test.load_state_dict(torch.load(   models_path +Encoder_m_path))
    Encoder_Detail_Test = Encoder_Detail().to(device)
    Encoder_Detail_Test.load_state_dict(torch.load(   models_path +Encoder_d_path))
    Decoder_Test = Decoder().to(device)
    Decoder_Test.load_state_dict(torch.load(   models_path +Decoder_path))


    Fusion_Test_d = Coord_Inter_res_3_Fusion().to(device)
    Fusion_Test_d.load_state_dict(torch.load(Net_Fusion_Path2))

    Encoder_Base_Test.eval()
    Encoder_Middle_Test.eval()
    Encoder_Detail_Test.eval()
    Decoder_Test.eval()
    # Fusion_Test_b.eval()
    # Fusion_Test_m.eval()
    Fusion_Test_d.eval()

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

        # img_fusion = Fusion_Test(B_K_IR, B_K_VIS,D_K_IR,D_K_VIS,M_K_IR,M_K_VIS)
        # Out = Decoder_Test(img_fusion, 0, 0)
        #['Test_data_TNO', 'Test_data_FLIR', 'Test_data_NIR_Country']
        if addition_mode == 'IRC_IRC_IRC':
            # F_b = Fusion_Test_b(B_K_IR, B_K_VIS)
            # F_m = Fusion_Test_m(M_K_IR, M_K_VIS)
            F_d = Fusion_Test_d(D_K_IR, D_K_VIS)
        elif addition_mode == 'ADD_ADD_ADD':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
            F_d = fus.add_fusion(D_K_IR, D_K_VIS)
        elif addition_mode == 'IRC_ADD_ADD':
            # F_b = Fusion_Test_b(B_K_IR, B_K_VIS)
            # F_m = fus.add_fusion(M_K_IR, M_K_VIS)
            F_d = fus.add_fusion(D_K_IR, D_K_VIS)
        elif addition_mode == 'ADD_IRC_ADD':
            # F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            # F_m = Fusion_Test_m(M_K_IR, M_K_VIS)
            F_d = fus.add_fusion(D_K_IR, D_K_VIS)
        elif addition_mode == 'ADD_ADD_IRC':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_m = fus.add_fusion(M_K_IR, M_K_VIS)
            F_d = Fusion_Test_d(D_K_IR, D_K_VIS)
        elif addition_mode == 'ADD_MAX_IRC':
            F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            F_m = fus.max_fusion(M_K_IR, M_K_VIS)
            F_d = Fusion_Test_d(D_K_IR, D_K_VIS)
        elif addition_mode == 'ADD_IRC_IRC':
            # F_b = fus.add_fusion(B_K_IR, B_K_VIS)
            # F_m = Fusion_Test_m(M_K_IR, M_K_VIS)
            F_d = Fusion_Test_d(D_K_IR, D_K_VIS)

        Out = Decoder_Test(F_b, F_d, F_m)


    img_test1.cpu()
    img_test2.cpu()


    Encoder_Base_Test.eval()
    Encoder_Middle_Test.eval()
    Encoder_Detail_Test.eval()
    Decoder_Test.eval()
    # Fusion_Test_b.eval()
    # Fusion_Test_m.eval()
    Fusion_Test_d.eval()

    Encoder_Base_Test.cpu()
    Encoder_Middle_Test.cpu()
    Encoder_Detail_Test.cpu()
    Decoder_Test.cpu()
    # Fusion_Test_b.cpu()
    # Fusion_Test_m.cpu()
    Fusion_Test_d.cpu()



    return output_img(Out)
# def test_fusion_model(test_num,exp_num,Net_Fusion_Path0,Net_Fusion_Path1,Net_Fusion_Path2,path_sum_tmp):
def test_fusion_model(test_num, exp_num, Net_Fusion_Path2, path_sum_tmp):
    # Encoder_b_path = Encoder_b_path_FLIR
    if fusion_model == "TNO":
        Test_data_choose_arr = ['Test_data_TNO']
    elif fusion_model == "FLIR":
        #Test_data_choose_arr = ['Test_data_FLIR']
        Test_data_choose_arr = [ 'Test_data_FLIR']
    elif fusion_model == "NIR":
        Test_data_choose_arr = ['Test_data_NIR_Country']
    else:
        Test_data_choose_arr = ['Test_data_TNO', 'Test_data_FLIR', 'Test_data_NIR_Country']

    # addition_mode_arr = ['IRC_IRC_IRC', 'IRC_ADD_ADD', 'ADD_IRC_ADD', 'ADD_ADD_IRC','ADD_ADD_IRC','ADD_IRC_IRC']
    addition_mode_arr = ['ADD_ADD_ADD', 'ADD_ADD_IRC']
    path_sum = '.\\Test_result\\' + exp_name + '_exp_' + exp_num + '_'

    #print("\nTNO FLIR NIR      EN       SD        SF         ")

    for Test_data_choose in Test_data_choose_arr:
        if Test_data_choose == 'Test_data_TNO':
            test_data_path = '.\\Datasets\\Test_data_TNO\\'
            print("\n")
        elif Test_data_choose == 'Test_data_FLIR':
            test_data_path = '.\\Datasets\\Test_data_FLIR\\'
            print("\n")
        elif Test_data_choose == 'Test_data_NIR_Country':
            test_data_path = '.\\Datasets\\Test_data_NIR_Country\\'
            print("\n")
        else:
            test_data_path = ''
        print(path_sum_tmp + Test_data_choose)
        f_path_list = []
        Test_Image_Number = len(os.listdir(test_data_path))

        for addition_mode in addition_mode_arr:
            if addition_mode == "ADD_ADD_ADD" and test_num>=2:
                pass
            else:
                for i in range(int(Test_Image_Number/2)):
                    #循环读取测试图片
                    if Test_data_choose=='Test_data_TNO':
                        Test_IR = Image.open(test_data_path+'\IR'+str(i+1)+'.bmp') # infrared image
                        Test_Vis = Image.open(test_data_path+'\VIS'+str(i+1)+'.bmp') # visible image
                    elif Test_data_choose=='Test_data_NIR_Country':
                        Test_IR = Image.open(test_data_path+'\IR'+str(i+1)+'.png') # infrared image
                        Test_Vis = Image.open(test_data_path+'\VIS'+str(i+1)+'.png') # visible image
                    elif Test_data_choose=='Test_data_FLIR':
                        Test_IR = Image.open(test_data_path+'\IR'+str(i+1)+'.jpg') # infrared image
                        Test_Vis = Image.open(test_data_path+'\VIS'+str(i+1)+'.jpg') # visible image


                    f_path = path_sum + Test_data_choose + '//'+addition_mode+'//'

                    mkdir(f_path)

                    fusion_path = f_path+ Test_data_choose + '_' + addition_mode + '_F' + str(i + 1) + '.png'
                    Fusion_image=Test_net_fusion(Test_IR,Test_Vis,Net_Fusion_Path2,Test_data_choose,addition_mode)
                    imsave(fusion_path, (Fusion_image * 255).astype(np.uint8))

                    f_path_list.append(f_path)
                    #print(f_path_list)
                if Test_data_choose == 'Test_data_TNO':     #6.96
                    flag = 0
                    if computer_result(f_path, method=addition_mode, dataset=1) >= en_std_tno:
                        flag = 1
                    #if flag == 0:
                        # deldir(path_sum + 'Test_data_TNO')
                        # break
                if Test_data_choose == 'Test_data_FLIR':# 7.45
                    flag = 0
                    if computer_result(f_path, method=addition_mode, dataset=2) >= en_std_flir:
                        flag = 1

                    #if flag == 0:
                        # deldir(path_sum + 'Test_data_TNO')
                        # deldir(path_sum+'Test_data_FLIR')
                        # break
                if Test_data_choose == 'Test_data_NIR_Country':
                    flag = 0
                    if computer_result(f_path, method=addition_mode, dataset=3) >= en_std_nir:
                        flag = 1
                    #if flag == 0:  # 7.44
                        # deldir(path_sum + 'Test_data_TNO')
                        # deldir(path_sum+'Test_data_FLIR')
                        # deldir(path_sum+'Test_data_NIR_Country')
                        # break


# 导入数据库


Train_Image_Number_IR=len(os.listdir('.\\Datasets\\Train_data_FLIR_IR\\'+'FLIR\\'))
Iter_per_epoch_IR=(Train_Image_Number_IR % batch_size!=0)+Train_Image_Number_IR//batch_size
Train_Image_Number_VIS=len(os.listdir('.\\Datasets\\Train_data_FLIR_VIS\\'+'FLIR\\'))
Iter_per_epoch_VIS=(Train_Image_Number_VIS % batch_size!=0)+Train_Image_Number_VIS//batch_size

transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        ])

Data_IR = torchvision.datasets.ImageFolder('.\\Datasets\\Train_data_FLIR_IR\\',transform=transforms)
dataloader_IR = torch.utils.data.DataLoader(Data_IR, batch_size,shuffle=True)

Data_VIS = torchvision.datasets.ImageFolder('.\\Datasets\\Train_data_FLIR_VIS\\',transform=transforms)
dataloader_VIS = torch.utils.data.DataLoader(Data_IR, batch_size,shuffle=True)






Encoder_Base_Test = Encoder_Base().to(device)
Encoder_Base_Test.load_state_dict(torch.load(   models_path + Encoder_b_path  ))
Encoder_Middle_Test = Encoder_Middle().to(device)
Encoder_Middle_Test.load_state_dict(torch.load( models_path + Encoder_m_path  ))
Encoder_Detail_Test = Encoder_Detail().to(device)
Encoder_Detail_Test.load_state_dict(torch.load( models_path + Encoder_d_path  ))
Decoder_Test = Decoder().to(device)
Decoder_Test.load_state_dict(torch.load(        models_path + Decoder_path    ))
##############################################################################################
# Net_Fusion_Train  = Coord_Inter_Fusion().to(device)
# Net_Fusion_Train_b  =Coord_Inter_res_3_Fusion().to(device)
# Net_Fusion_Train_m  =Coord_Inter_res_3_Fusion().to(device)
Net_Fusion_Train_d  =Coord_Inter_res_3_Fusion().to(device)
# Net_Fusion_Train = Res_Fusion().to(device)


Encoder_Base_Test.eval()
Encoder_Middle_Test.eval()
Encoder_Detail_Test.eval()
Decoder_Test.eval()
# Net_Fusion_Train_b.eval()
# Net_Fusion_Train_m.eval()
Net_Fusion_Train_d.eval()

# optimizer1 = optim.Adam(Net_Fusion_Train_b.parameters(), lr = lr)
# scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [int(epochs*0.8), epochs], gamma=0.1)
# optimizer2 = optim.Adam(Net_Fusion_Train_m.parameters(), lr = lr)
# scheduler2= torch.optim.lr_scheduler.MultiStepLR(optimizer2, [int(epochs*0.8), epochs], gamma=0.1)
optimizer3 = optim.Adam(Net_Fusion_Train_d.parameters(), lr = lr)
scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer3, [int(epochs*0.8), epochs], gamma=0.1)

##-------------------------------------------------------损失函数定义
MSELoss = nn.MSELoss()
SmoothL1Loss=nn.SmoothL1Loss()
L1Loss=nn.L1Loss()
SSIMLoss = kornia.losses.SSIM(3, reduction='mean')
class Sobelxy(nn.Module):
    def __init__(self, ):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)
def Sobel_loss(image_source,generate_img):
    sobel = Sobelxy()
    source_grad = sobel(image_source)
    generate_grad = sobel(generate_img)
    loss_grad=F.l1_loss(source_grad,generate_grad)
    return loss_grad
class TV_Loss(torch.nn.Module):

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, IA, IF):
        r = IA - IF
        h = r.shape[2]
        w = r.shape[3]
        tv1 = torch.pow((r[:, :, 1:, :] - r[:, :, :h - 1, :]), 2).mean()
        tv2 = torch.pow((r[:, :, :, 1:] - r[:, :, :, :w - 1]), 2).mean()
        return tv1 + tv2





print('======================== Training Begins ===========================')
print('The total number of images is %d,     Need to cycle %d   times.'%(Train_Image_Number_IR,Iter_per_epoch_IR))
print('The log_interval           is %d,     MSE_ir_w is   %f '%(log_interval,mse_ir_w))

test_num = 0
for iteration in range(epochs):

    # Net_Fusion_Train_b.train()
    # Net_Fusion_Train_m.train()
    Net_Fusion_Train_d.train()
    #Net_Fusion_Train.train()

#   7.098552  48.508115  12.384375
    data_iter_input_IR = iter(dataloader_IR)
    data_iter_input_VIS = iter(dataloader_VIS)

    for step in range(Iter_per_epoch_IR):
        img_input_IR, _ = next(data_iter_input_IR)
        img_input_VIS, _ = next(data_iter_input_VIS)

        if is_cuda:
            img_input_IR = img_input_IR.cuda()
            img_input_VIS = img_input_VIS.cuda()

        # optimizer1.zero_grad()
        # optimizer2.zero_grad()
        optimizer3.zero_grad()

        # =====================================================================
        # Calculate loss
        # =====================================================================

        B_K_IR, _, _ = Encoder_Base_Test(img_input_IR)
        B_K_VIS, _, _ = Encoder_Base_Test(img_input_VIS)
        D_K_IR, _, _ = Encoder_Detail_Test(img_input_IR)
        D_K_VIS, _, _ = Encoder_Detail_Test(img_input_VIS)
        M_K_IR = Encoder_Middle_Test(img_input_IR)
        M_K_VIS = Encoder_Middle_Test(img_input_VIS)

        # img_concat = torch.cat([B_K_IR, B_K_VIS,D_K_IR,D_K_VIS,M_K_IR,M_K_VIS], 1)

        # F_b = Net_Fusion_Train_b(B_K_IR, B_K_VIS)
        # F_m = Net_Fusion_Train_m(M_K_IR, M_K_VIS)
        # F_d = Net_Fusion_Train_d(D_K_IR, D_K_VIS)



        F_b = fus.add_fusion(B_K_IR, B_K_VIS)
        F_m = fus.add_fusion(M_K_IR, M_K_VIS)
        F_d = Net_Fusion_Train_d(D_K_IR, D_K_VIS)


        img_fusion =F_d

        #img_fusion = Net_Fusion_Train(D_K_IR, D_K_VIS)
        # with torch.no_grad():
        img_recon = Decoder_Test(F_b,F_d,F_m)

        #         img_input_IR          img_input_VIS


        # Total loss
        mse_ir = MSELoss(img_input_IR, img_recon)
        mse_vis = MSELoss(img_input_VIS, img_recon)

        ssim_ir = SSIMLoss(img_input_IR, img_recon)
        ssim_vis = SSIMLoss(img_input_VIS, img_recon)
        l1_loss_vis = Sobel_loss(img_input_VIS, img_recon)        #4.5 x
        # l1_loss_vis = F.l1_loss(img_input_VIS,img_recon)
        # loss = mse_ir_w*mse_ir +mse_vis +l1_loss_vis
        loss = mse_ir_w*mse_ir +mse_vis +sobel_w*l1_loss_vis

        #
        # loss = mse + ssim_w * ssim

        loss.backward()
        # optimizer1.step()
        # optimizer2.step()
        optimizer3.step()


        los = loss.item()
        mse_i = mse_ir.item()
        mse_v = mse_vis.item()
        ssim_i = ssim_ir.item()
        ssim_v = ssim_vis.item()
        l1_v   = l1_loss_vis.item()
        # sobel_l = sobel.item()

        # if (step + 1) % log_interval == 0 :
        #     print('Epoch/step: %d/%d, MSE:%.5f  SSIM:%.5f  sobel:%.5f  total_loss:%.5f, lr: %f' % (
        #     iteration + 1, step + 1, mse_l, ssim_l, sobel_l,los, optimizer1.state_dict()['param_groups'][0]['lr']))

        if (step + 1) % int(log_interval/6) == 0:
            print('Epo/s: %3d/%3d, mse_i:%.4f mse_v:%.4f   ssim_i:%.4f   ssim_v:%.4f   l1_v:%.4f  loss:%.4f  , lr: %.4f' % (
                iteration + 1, step + 1, mse_i, mse_v, ssim_i,ssim_v,l1_v,los, optimizer3.state_dict()['param_groups'][0]['lr']))


            if iteration>=5:
                # 保存临时最佳模型
                # if loss_min_temp >= mse_i + mse_v:
                epo_temp = iteration
                name = [ 'Net_Fusion_d_Best_']
                # name = ['Net_Fusion_b_Best_','Net_Fusion_m_Best_','Net_Fusion_d_Best_']
                # save_model_filename0 = name[0] + str(mse_ir_w)+"_"+ str(sobel_w )+ "_"+str(test_num) + '_' + str(iteration) + '_' + str(step) + ".model"
                # save_model_path0 = os.path.join(train_path, save_model_filename0)
                # torch.save(Net_Fusion_Train_b.state_dict(), save_model_path0)
                #
                # save_model_filename1 = name[1] + str(mse_ir_w)+"_"+ str(sobel_w )+ "_" +str(test_num) + '_' + str(iteration) + '_' + str(step) + ".model"
                # save_model_path1 = os.path.join(train_path, save_model_filename1)
                # torch.save(Net_Fusion_Train_m.state_dict(), save_model_path1)

                save_model_filename2 = name[0] +  str(mse_ir_w)+"_"+ str(sobel_w )+ "_"+str(test_num) + '_' + str(iteration) + '_' + str(step) + ".model"
                save_model_path2 = os.path.join(train_path, save_model_filename2)
                torch.save(Net_Fusion_Train_d.state_dict(), save_model_path2)


                exp_num_tmp = str(test_num) + '_epo' + str(epo_temp)
                #path_sum_tmp = '.\\Test_result\\' + exp_name + '_exp_' + exp_num_tmp + '_'
                path_sum_tmp =exp_name + '_exp_' + exp_num_tmp + '_'

                #print("Test    saved at:", path_sum_tmp)
                print("Trained saved at:", save_model_path2)
                print('Loss_min_temp   :%.8f' % (mse_i + mse_v))
                test_fusion_model(test_num, exp_num_tmp, save_model_path2,path_sum_tmp)
                # test_fusion_model(test_num,exp_num_tmp, save_model_path0,save_model_path1,save_model_path2,path_sum_tmp)
                test_num = test_num + 1

    # scheduler1.step()
    # scheduler2.step()
    scheduler3.step()
