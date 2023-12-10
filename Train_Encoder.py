import os
import shutil
import torch
import kornia
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import numpy as np
from PIL import Image
from skimage.io import imsave
from torchvision import transforms
from Args import train_data_path,train_path,device,batch_size,channel,lr,is_cuda,log_interval,img_size,epochs
from Args import loss_min_temp,exp_name,ssim_w,sobel_w,en_std_tno,en_std_flir,en_std_nir,cl_w,con_sim_w,con_sim_w_str
from Network import Encoder_Base,Encoder_Detail,Decoder,Encoder_Middle,Encoder_Base_Detail
from Utils_Network import Test_fusion
from Fusion_metrics import computer_result_new


device='cuda'

def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
def deldir(path):
    mydir =  path
    try:
        shutil.rmtree(mydir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
def test_model(exp_num,Encoder_b_d_path,Encoder_m_path,Decoder_path,path_sum_tmp):

    # strategy_type_list = [ 'Sum', 'Average','l1_norm','max','nuclear','CHA']
    Test_data_choose_arr=  ['Test_data_TNO']


    path_sum = '.\\Test_result\\' + exp_name + '_exp_' + exp_num + '_'


    for Test_data_choose in Test_data_choose_arr:
        if Test_data_choose == 'Test_data_TNO':
            test_data_path = '.\\Datasets\\Test_data_TNO'
            strategy_type_list = [ 'Sum']

        elif Test_data_choose == 'Test_data_FLIR':
            test_data_path = '.\\Datasets\\Test_data_FLIR\\'
            strategy_type_list = [ 'Sum']

        elif Test_data_choose == 'Test_data_NIR_Country':
            test_data_path = '.\\Datasets\\Test_data_NIR_Country'
            strategy_type_list = [ 'Sum']

        else:
            test_data_path = ''
            strategy_type_list = []

        f_path_list = []
        Test_Image_Number = len(os.listdir(test_data_path))


        for addition_mode in strategy_type_list:
            for i in range(int(Test_Image_Number/2)):
                if Test_data_choose=='Test_data_TNO':
                    Test_IR = Image.open(test_data_path+'\IR'+str(i+1)+'.bmp') # infrared image
                    Test_Vis = Image.open(test_data_path+'\VIS'+str(i+1)+'.bmp') # visible image
                elif Test_data_choose=='Test_data_NIR_Country':
                    Test_IR = Image.open(test_data_path+'\IR'+str(i+1)+'.png') # infrared image
                    Test_Vis = Image.open(test_data_path+'\VIS'+str(i+1)+'.png') # visible image
                elif Test_data_choose=='Test_data_FLIR':
                    Test_IR = Image.open(test_data_path+'\IR'+str(i+1)+'.jpg') # infrared image
                    Test_Vis = Image.open(test_data_path+'\VIS'+str(i+1)+'.jpg') # visible image

                if addition_mode == "Sum":
                    f_path1 = f_path = path_sum+Test_data_choose+'\\SUM\\'
                elif addition_mode == "Average":
                    f_path2 = f_path = path_sum+Test_data_choose+'\\AVG\\'
                elif addition_mode == "l1_norm":
                    f_path3 = f_path = path_sum+Test_data_choose+'\\L1\\'
                elif addition_mode == "max":
                    f_path4 = f_path = path_sum+Test_data_choose+'\\MAX\\'
                elif addition_mode == "nuclear":
                    f_path5 = f_path = path_sum + Test_data_choose + '\\NUC\\'
                elif addition_mode == "CHA":
                    f_path6 = f_path = path_sum + Test_data_choose + '\\CHA\\'

                mkdir(f_path)

                fusion_path = f_path+ Test_data_choose + '_' + addition_mode + '_F' + str(i + 1) + '.png'
                Fusion_image=Test_fusion(Test_IR,Test_Vis,addition_mode,Encoder_b_d_path,Encoder_m_path,Decoder_path)

                imsave(fusion_path, (Fusion_image * 255).astype(np.uint8))

            f_path_list.append(f_path)


        if Test_data_choose == 'Test_data_FLIR':

            flag = 0
            for i in range(len(strategy_type_list)):
                if computer_result_new(f_path_list[i], method=strategy_type_list[i], dataset=2,name=path_sum_tmp+'Test_data_FLIR') >= 7.3:
                    flag = 1

            if flag == 0:
                #deldir(path_sum + 'Test_data_TNO')
                deldir(path_sum+'Test_data_FLIR')

                #break
        if Test_data_choose == 'Test_data_NIR_Country':

            flag = 0
            for i in range(len(strategy_type_list)):
                if computer_result_new(f_path_list[i], method=strategy_type_list[i], dataset=3,name=path_sum_tmp+'Test_data_NIR_Country') >= 7.3:
                    flag = 1

            if flag == 0:
                #deldir(path_sum + 'Test_data_TNO')
                #deldir(path_sum+'Test_data_FLIR')
                deldir(path_sum+'Test_data_NIR_Country')

                #break
        if Test_data_choose == 'Test_data_TNO':
            flag = 0
            for i in range(len(strategy_type_list)):
                if computer_result_new(f_path_list[i], method=strategy_type_list[i], dataset=1,name=path_sum_tmp+'Test_data_TNO') >= 5:
                    flag = 1


            if flag == 0:
                deldir(path_sum + 'Test_data_TNO')
                #break
        # print("\n")
    # print("\n")



Train_Image_Number=len(os.listdir(train_data_path+'FLIR\\'))
Iter_per_epoch=(Train_Image_Number % batch_size!=0)+Train_Image_Number//batch_size



transforms = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        ])
Data = torchvision.datasets.ImageFolder(train_data_path,transform=transforms)
dataloader = torch.utils.data.DataLoader(Data, batch_size,shuffle=True)




Encoder_Base_Detail_Train = Encoder_Base_Detail()
Encoder_Middle_Train   = Encoder_Middle()
Decoder_Train=Decoder()


if is_cuda:

    Encoder_Base_Detail_Train = Encoder_Base_Detail_Train.cuda()
    Encoder_Middle_Train    = Encoder_Middle_Train.cuda()
    Decoder_Train           = Decoder_Train.cuda()



optimizer1 = optim.Adam(Encoder_Base_Detail_Train.parameters(), lr = 1*lr)
optimizer3 = optim.Adam(Decoder_Train.parameters(), lr = lr)
optimizer4 = optim.Adam(Encoder_Middle_Train.parameters(), lr = lr)

scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [int(epochs*0.8), epochs], gamma=0.1)
scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer3, [int(epochs*0.8), epochs], gamma=0.1)
scheduler4 = torch.optim.lr_scheduler.MultiStepLR(optimizer4, [int(epochs*0.8), epochs], gamma=0.1)

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

# Training=============================================================================

print('======================== Training Begins ===========================')
print('The total number of images is %d,     Need to cycle %d times.'%(Train_Image_Number,Iter_per_epoch))
print('loss_min_temp:%f     ssim_w:%d   cl_w:%f'%(loss_min_temp,ssim_w,cl_w))#con_sim_w
print('con_sim_w:',con_sim_w)
test_num = 0
for iteration in range(epochs):

    # Encoder_Base_Train.train()
    # Encoder_Detail_Train.train()
    Encoder_Base_Detail_Train.train()
    Decoder_Train.train()
    Encoder_Middle_Train.train()

    data_iter_input = iter(dataloader)
    
    for step in range(Iter_per_epoch):
        img_input,_ =next(data_iter_input)
        
          
        if is_cuda:
            img_input=img_input.cuda()
        #Encoder_Base_Detail_Train
        # optimizer1.zero_grad()
        # optimizer2.zero_grad()
        optimizer1.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        # Calculate loss
        # B_K,eta_B,theta_B = Encoder_Base_Train  (img_input)
        # D_K,eta_D,theta_D = Encoder_Detail_Train(img_input)
        B_K,D_K,Con_sim_list = Encoder_Base_Detail_Train(img_input)
        M_K = Encoder_Middle_Train(img_input)

        img_recon=Decoder_Train(B_K,D_K,M_K)

        # Total loss
        mse=MSELoss(img_input,img_recon)
        ssim=SSIMLoss(img_input,img_recon)
        similarity = torch.abs(torch.cosine_similarity(B_K, D_K, dim=0)).mean()
        # similarity = 0.0
        # for i in range(len(Con_sim_list)):
        #     similarity = similarity + Con_sim_list[i]*con_sim_w[i]

        # sobel = Sobel_loss(img_input, img_recon)
        # loss = mse + ssim_w*ssim +sobel_w*sobel
        loss = mse + ssim_w*ssim + 0*similarity

        loss.backward()
        optimizer1.step()
        optimizer3.step()
        optimizer4.step()

        los     = loss.item()
        mse_l   = mse.item()
        ssim_l  = ssim.item()
        cl_l = similarity.item()
        los_tem = ssim_l+ mse_l
        # sobel_l = sobel.item()

        if (step + 1) % int(log_interval/2) == 0 :
            if  iteration<19:
                print('Epoch/step: %3s/%3s, MSE:%.5f  SSIM:%.5f   CL:%.5f       los_tem:%.5f  total_loss:%.5f  , lr: %.4f' % (
            iteration + 1, step + 1, mse_l, ssim_l,cl_l,los_tem,los, optimizer1.state_dict()['param_groups'][0]['lr']))
            flag = 0
            if 19<iteration<epochs:
                flag = 1

            if  flag == 1 or   epochs < iteration:
                # print('Epoch/step: %3s/%3s, MSE:%.5f  SSIM:%.5f   CL:%.5f       los_tem:%.5f  total_loss:%.5f  , lr: %.4f' % (
                # iteration + 1, step + 1, mse_l, ssim_l,cl_l,los_tem,los, optimizer1.state_dict()['param_groups'][0]['lr']))
                epo_temp = iteration


                name = ['Encoder_Bt_Dt_Best_','Encoder_Mt_Best_',  'Decoder_t_Best_']

                save_model_filename0 = name[0] +str(ssim_w)+'_'+con_sim_w_str+'_'+str(test_num)+'_'+  str(iteration)+  '_' + str(step) + ".model"
                save_model_filename1 = name[1] +str(ssim_w)+'_'+con_sim_w_str+'_'+str(test_num)+'_' + str(iteration) + '_' + str(step) + ".model"
                save_model_filename2 = name[2] +str(ssim_w)+'_'+con_sim_w_str+'_'+str(test_num)+'_' + str(iteration) + '_' + str(step) + ".model"
                save_model_path0 = os.path.join(train_path, save_model_filename0)
                save_model_path1 = os.path.join(train_path, save_model_filename1)
                save_model_path2= os.path.join(train_path, save_model_filename2)

                torch.save(Encoder_Base_Detail_Train.state_dict(), save_model_path0)
                torch.save(Encoder_Middle_Train.state_dict(), save_model_path1)
                torch.save(Decoder_Train.state_dict(), save_model_path2)


                exp_num_tmp = str(test_num)+'_epo'+str(epo_temp)
                path_sum_tmp =  exp_name + '_exp_' + exp_num_tmp + '_'

                print("Trained saved at:", save_model_path0)
                test_model( exp_num_tmp , save_model_path0,save_model_path1,save_model_path2 ,path_sum_tmp)
                test_num = test_num+1


    scheduler1.step()
    scheduler3.step()
    scheduler4.step()

# Save models
Encoder_Base_Detail_Train.eval()
Encoder_Base_Detail_Train.cpu()
Encoder_Middle_Train.eval()
Encoder_Middle_Train.cpu()
Decoder_Train.eval()
Decoder_Train.cpu()