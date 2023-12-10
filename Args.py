Train_data_choose='FLIR'
train_data_path = '.\\Datasets\\Train_data_FLIR\\'
train_path = '.\\Train_result\\'
device = "cuda"
channel=64

lr = 1*1e-2
is_cuda = True
img_size=128

epochs =        120   #
batch_size=     20
log_interval = int(360/batch_size)

layer_numb_b = 7
layer_numb_d = 12
layer_numb_m = 0


kb_list =   [    2,     3,      4,      5,      6,      7 ]                            # 计算余弦相似度的 渐进层
kd_list =   [    2,     4,      6,      8,     10,     12 ]
con_sim_w = [ 0.021,  0.026,   0.031,   0.036 ,  0.041,   0.046  ]
con_sim_w_str =  str(int( con_sim_w[0]*1000 ))+"_"+str(int( con_sim_w[1]*1000 ))+"_"+str(int( con_sim_w[2]*1000 ))+"_"+str(int( con_sim_w[3]*1000 ))+"_"+str(int( con_sim_w[4]*1000 ))+"_"+str(int( con_sim_w[5]*1000 ))

loss_min_temp = 0.0081   #0.0081
loss_min =int(loss_min_temp*100000)

ssim_w = 5.501
cl_w = 0
exp_name = "LAP-7-12-DRGN_SSIM_"+str(ssim_w)



mse_ir_w = 5#3.4
sobel_w =0.81#0.8

en_std_tno = 5
en_std_flir = 7
en_std_nir = 7