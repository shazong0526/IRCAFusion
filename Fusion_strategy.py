import torch
import torch.nn.functional as F
import kornia
from Args import channel,img_size

#定义初始化滤波
Laplace = kornia.filters.Laplacian(19)#高通滤波
Blur = kornia.filters.BoxBlur((11, 11))#低通滤波
planes = 64
device='cuda'
EPSILON = 1e-10
#######################################

# ---------------------------------------------①    add_fusion strategy
def add_fusion(tensor1, tensor2):
    return (tensor1 + tensor2)
# ---------------------------------------------②    avg_fusion
def avg_fusion(tensor1, tensor2):
    return (tensor1 + tensor2)/2
# ---------------------------------------------③    max_fusion
def max_fusion(tensor1, tensor2):
    return torch.max(tensor1 ,tensor2)
# ---------------------------------------------④    l1_fusion
def l1_fusion(y1, y2, window_width=1):
    ActivityMap1 = y1.abs()
    ActivityMap2 = y2.abs()

    kernel = torch.ones(2 * window_width + 1, 2 * window_width + 1) / (2 * window_width + 1) ** 2
    kernel = kernel.to(device).type(torch.float32)[None, None, :, :]
    kernel = kernel.expand(y1.shape[1], y1.shape[1], 2 * window_width + 1, 2 * window_width + 1)
    ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=window_width)
    ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=window_width)
    WeightMap1 = ActivityMap1 / (ActivityMap1 + ActivityMap2)
    WeightMap2 = ActivityMap2 / (ActivityMap1 + ActivityMap2)
    return WeightMap1 * y1 + WeightMap2 * y2



# pooling function
def nuclear_pooling(tensor, kernel_size=None):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors
# ---------------------------------------------⑤    nuclear_fusion
def nuclear_fusion(tensor1, tensor2):
    shape = tensor1.size()
    # calculate channel attention
    global_p1 = nuclear_pooling(tensor1)
    global_p2 = nuclear_pooling(tensor2)
    # get weight map
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

    tensor_fusion = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_fusion
#----------------------------------------------⑥    CHA_fusion
def CHA_fusion(f1,f2,is_test=True,save_mat=False):
    if is_test:
        fp1 = (((f1.mean(2)).mean(2)).unsqueeze(2)).unsqueeze(3)
        fp2 = (((f2.mean(2)).mean(2)).unsqueeze(2)).unsqueeze(3)
    else:
        fp1 = F.avg_pool2d(f1, f1.size(2))
        fp2 = F.avg_pool2d(f2, f2.size(2))
    mask1 = fp1 / (fp1 + fp2)
    mask2 = 1 - mask1
    if save_mat:
        import scipy.io as io
        mask = mask1.cpu().detach().numpy()
        io.savemat("./outputs/fea/mask.mat", {'mask': mask})
    return f1 * mask1 + f2 * mask2



def channel_fusion(tensor1, tensor2, p_type='attention_avg'):
    # global max pooling
    shape = tensor1.size()
    # calculate channel attention
    global_p1 = channel_attention(tensor1, p_type)
    global_p2 = channel_attention(tensor2, p_type)

    # get weight map
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f
def spatial_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f
def attention_fusion(tensor1, tensor2, p_type='attention_avg'):
    # avg, max, nuclear
    f_channel = channel_fusion(tensor1, tensor2,  p_type)
    f_spatial = spatial_fusion(tensor1, tensor2)

    tensor_f = (f_channel + f_spatial) / 2
    return tensor_f

# select channel


# channel attention
def channel_attention(tensor, pooling_type='attention_avg'):
    # global pooling
    shape = tensor.size()
    pooling_function = F.avg_pool2d

    if pooling_type is 'attention_avg':
        pooling_function = F.avg_pool2d
    elif pooling_type is 'attention_max':
        pooling_function = F.max_pool2d
    elif pooling_type is 'attention_nuclear':
        pooling_function = nuclear_pooling
    global_p = pooling_function(tensor, kernel_size=shape[2:])
    return global_p
# spatial attention
def spatial_attention(tensor, spatial_type='mean'):
    t_min = tensor.min()
    t_max = tensor.max()
    if t_min < 0 :
        tensor += torch.abs(t_min)
        t_min = tensor.min()
    t_st = t_max - t_min
    tensor = (tensor - t_min).true_divide(t_st)

    spatial = []
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial

