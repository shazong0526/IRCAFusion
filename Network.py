import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import numpy as np
from Args import channel, img_size, layer_numb_b,layer_numb_d,layer_numb_m
# from skimage.filters import difference_of_gaussians, window

# 定义初始化滤波
Laplace3 = kornia.filters.Laplacian(101)  # 高通滤波
Laplace9 = kornia.filters.Laplacian(9)  # 高通滤波
Laplace13 = kornia.filters.Laplacian(13)  # 高通滤波
Laplace = kornia.filters.Laplacian(19)  # 高通滤波

Blur = kornia.filters.BoxBlur((11, 11))  # 低通滤波
Blur1 = kornia.filters.BoxBlur((1, 1))  # 低通滤波
Blur2 = kornia.filters.BoxBlur((2, 2))  # 低通滤波
Blur3 = kornia.filters.BoxBlur((3, 3))  # 低通滤波
Blur16 = kornia.filters.BoxBlur((16, 16))  # 低通滤波



class BCL(nn.Module):
    def __init__(self):
        super(BCL, self).__init__()
        self.kernel = nn.Parameter(torch.randn(size=[64, 1, 3, 3]))
        self.zero_pad = nn.ReflectionPad2d(1)
        self.rest = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.PReLU()
        )
        self.eta = nn.Parameter(
            nn.init.normal_(
                torch.empty(1).cuda(), mean=0.1, std=0.03
            ))
        self.theta = nn.Parameter(
            nn.init.normal_(
                torch.empty(1).cuda(), mean=1e-3, std=1e-4
            ))

    def forward(self, x_in, img):
        x_in_2 = self.zero_pad(x_in)
        x_1 = F.conv2d(x_in_2, self.kernel, padding=0)
        x_2 = self.zero_pad(x_1)
        kernel2 = torch.rot90(self.kernel, 2, [-1, -2]).transpose(0, 1)
        x_3 = F.conv2d(x_2, kernel2, padding=0)
        x_out = self.rest(x_3)
        x_out_2 = x_in - self.eta * (x_out - self.theta * (img - x_in))
        return x_out_2, img, self.eta.item(), self.theta.item()
class DCL(nn.Module):
    def __init__(self):
        super(DCL, self).__init__()
        self.kernel = nn.Parameter(torch.randn(size=[64, 1, 3, 3]))
        self.zero_pad = nn.ReflectionPad2d(1)
        self.rest = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.PReLU()
        )
        self.eta = nn.Parameter(
            nn.init.normal_(
                torch.empty(1).cuda(), mean=0.1, std=0.03
            ))
        self.theta = nn.Parameter(
            nn.init.normal_(
                torch.empty(1).cuda(), mean=1e-3, std=1e-4
            ))

    def forward(self, x_in, img):
        x_in_2 = self.zero_pad(x_in)
        x_1 = F.conv2d(x_in_2, self.kernel, padding=0)
        x_2 = self.zero_pad(x_1)
        kernel2 = torch.rot90(self.kernel, 2, [-1, -2]).transpose(0, 1)
        x_3 = F.conv2d(x_2, kernel2, padding=0)
        x_out = self.rest(x_3)
        x_out_2 = x_in - self.eta * (x_out - self.theta * (img - x_in))
        return x_out_2, img, self.eta.item(), self.theta.item()
class MCL(nn.Module):
    def __init__(self):
        super(MCL, self).__init__()
        self.kernel = nn.Parameter(torch.randn(size=[64, 1, 3, 3]))
        self.zero_pad = nn.ReflectionPad2d(1)
        self.rest = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.PReLU()
        )
        self.eta = nn.Parameter(
            nn.init.normal_(
                torch.empty(1).cuda(), mean=0.1, std=0.03
            ))
        self.theta = nn.Parameter(
            nn.init.normal_(
                torch.empty(1).cuda(), mean=1e-3, std=1e-4
            ))

    def forward(self, x_in, img):
        x_in_2 = self.zero_pad(x_in)
        x_1 = F.conv2d(x_in_2, self.kernel, padding=0)
        x_2 = self.zero_pad(x_1)
        kernel2 = torch.rot90(self.kernel, 2, [-1, -2]).transpose(0, 1)
        x_3 = F.conv2d(x_2, kernel2, padding=0)
        x_out = self.rest(x_3)
        x_out_2 = x_in - self.eta * (x_out - self.theta * (img - x_in))
        return x_out_2, img, self.eta.item(), self.theta.item()


class Encoder_Base(nn.Module):
    def __init__(self, size=img_size):
        super(Encoder_Base, self).__init__()
        self.numb = layer_numb_b
        self.conv1 = nn.ModuleList([BCL() for i in range(self.numb)])

    def forward(self, img):
        img_blur = Blur(img)
        # img_blur1 = Blur1(img)
        # img_blur2 = Blur2(img)
        # img_blur3 = Blur3(img)
        eta_list_base = []
        theta_list_base = []
        for layer in self.conv1:
            img_blur, img, eta, theta = layer(img_blur, img)
            eta_list_base.append(eta)
            theta_list_base.append(theta)
        return img_blur, eta_list_base, theta_list_base
class Encoder_Detail(nn.Module):
    def __init__(self, size=img_size):
        super(Encoder_Detail, self).__init__()
        self.numb = layer_numb_d
        self.conv2 = nn.ModuleList([DCL() for i in range(self.numb)])

    def forward(self, img):
        img_laplace = Laplace(img)
        # img_laplace1 = Laplace1(img)
        # img_laplace2 = Laplace2(img)
        # img_laplace3 = Laplace3(img)

        eta_list_detail = []
        theta_list_detail = []
        for layer in self.conv2:
            img_laplace, img, eta, theta = layer(img_laplace, img)
            eta_list_detail.append(eta)
            theta_list_detail.append(theta)
        return img_laplace, eta_list_detail, theta_list_detail
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 1, 3, padding=0, bias=False),  # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.cbam = CBAM1(3)
    def forward(self, fm_b, fm_d, fm_m):
        fm = fm_b + fm_d + fm_m
        # f_sum =  torch.cat([fm_b , fm_d , fm_m], 1) # 16+32+64
        # fm = self.cbam(f_sum)
        return self.decoder(fm)

class Encoder_Base_Detail(nn.Module):
    def __init__(self, size=img_size):
        super(Encoder_Base_Detail, self).__init__()
        self.numb1 = layer_numb_b
        self.conv1 = nn.ModuleList([BCL() for i in range(self.numb1)])
        self.numb2 = layer_numb_d
        self.conv2 = nn.ModuleList([DCL() for i in range(self.numb2)])

    def forward(self, img):
        img_blur = Blur(img)

        eta_list_base = []
        theta_list_base = []
        img_list_base = []
        for layer in self.conv1:
            img_blur, img, eta, theta = layer(img_blur, img)
            eta_list_base.append(eta)
            theta_list_base.append(theta)
            img_list_base.append(img_blur)


        img_laplace = Laplace(img)

        eta_list_detail = []
        theta_list_detail = []
        img_list_detail = []
        for layer in self.conv2:
            img_laplace, img, eta, theta = layer(img_laplace, img)
            eta_list_detail.append(eta)
            theta_list_detail.append(theta)
            img_list_detail.append(img_laplace)

        #计算每层相似度
        contrast_list = []
        kb_list = [ 2, 3, 4, 5,  6,  7]  #计算余弦相似度的 渐进层
        kd_list = [ 2, 4, 6, 8, 10, 12]
        for k in range(0,len(kb_list)):
            contrast_sim = torch.abs(torch.cosine_similarity(img_list_base[kb_list[k]-1], img_list_detail[kd_list[k]-1], dim=0)).mean()
            contrast_list.append(contrast_sim)

        return img_blur,img_laplace ,contrast_list
class Encoder_Middle(nn.Module):
    def __init__(self):
        super(Encoder_Middle, self).__init__()

        # 第一层卷积 拓展维度到 16
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)

        self.cbam1 = CBAM(32)
        self.cbam2 = CBAM(64)


        self.psa = PSA(16+32, 32)
        self.cha = ChannelAttentionModule(16+32)
        self.spa = SpatialAttentionModule()
        self.cbam = CBAM(16+32)
        self.bottle = nn.Conv2d(16+32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, img):
        img_bp = Blur2(img)-Blur16(img)  # 滤波
        out = self.conv_input(img_bp)  # 拓宽维度到 1*1 卷积 channel 32
        LR = out

        out = self.rgbd1(out)
        concat1 = out

        out = torch.cat([LR, concat1], 1)
        out = self.psa(out)

        out = self.cbam(out)

        out = self.bottle(out)

        # out =self.con1x1(out)
        # out = self.convt(self.conv(out))
        # out = self.conv_output(out)

        return out



## CBAM(channel)     # channel  16的倍数
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):  # channel // 16
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        # out = self.channel_attention(x) * x
        # out = self.spatial_attention(x) * x


        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out

class ChannelAttentionModule_MSRB(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule_MSRB, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
class SpatialAttentionModule_MSRB(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule_MSRB, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out



class CBAM1(nn.Module):
    def __init__(self, channel):  # channel // 16
        super(CBAM1, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel,ratio=1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(x) * x
        return out



## PSA(inplans, planes)     # planes  4的倍数
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=8):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

class PSA(nn.Module):
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 2, 4, 8]):
        super(PSA, self).__init__()
        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


## RGBD(in_channels, out_channels)     # planes  4的倍数
class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)
class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)
class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x
class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x


class DenseBlock2(nn.Module):
    def __init__(self, channels):
        super(DenseBlock2, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)
        self.conv3 = ConvLeakyRelu2d(3 * channels, channels)
        # self.conv4 = ConvLeakyRelu2d(4 * channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        x = torch.cat((x, self.conv3(x)), dim=1)
        # x = torch.cat((x, self.conv4(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=1):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
class Coord_Inter_Att(nn.Module):
    def __init__(self, inp, oup, reduction=1):
        super(Coord_Inter_Att, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)

        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # out = identity * a_w * a_h

        return  a_w , a_h

EPSILON = 1e-10
class Coord_Inter_Fusion(nn.Module):
    def __init__(self):
        super(Coord_Inter_Fusion, self).__init__()
        self.convup = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True)
        self.coord_ir = Coord_Inter_Att(16,16)
        self.coord_vi = Coord_Inter_Att(16,16)
        self.bottle = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)



    def forward(self, D_ir,D_vi ):
        x_ir = self.convup(D_ir)
        x_vi = self.convup(D_vi)

        ir_temp_w,ir_temp_h =  self.coord_ir(x_ir)
        vi_temp_w,vi_temp_h =  self.coord_vi(x_vi)

        wight_ir_w =  ir_temp_w/(ir_temp_w+vi_temp_w+EPSILON)
        wight_vi_w =  vi_temp_w/(ir_temp_w+vi_temp_w+EPSILON)
        wight_ir_h =  ir_temp_h/(ir_temp_h+vi_temp_h+EPSILON)
        wight_vi_h =  vi_temp_h/(ir_temp_h+vi_temp_h+EPSILON)


        x_ir = x_ir*wight_ir_w*wight_ir_h
        x_vi = x_vi*wight_vi_w*wight_vi_h
        output = torch.cat([x_ir, x_vi], 1)
        output = self.bottle(output)
        return output

class Coord_Inter_res_3_Fusion(nn.Module):
    def __init__(self):
        super(Coord_Inter_res_3_Fusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)

        self.coord_ir = Coord_Inter_Att(16,16)
        self.coord_vi = Coord_Inter_Att(16,16)
        self.bottle = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)



    def forward(self, D_ir,D_vi ):
        x_res = torch.cat([D_ir, D_vi], 1)
        x_res = self.conv2(x_res)
        x_ir = self.conv1(D_ir)
        x_vi = self.conv1(D_vi)

        ir_temp_w,ir_temp_h =  self.coord_ir(x_ir)
        vi_temp_w,vi_temp_h =  self.coord_vi(x_vi)

        wight_ir_w =  ir_temp_w/(ir_temp_w+vi_temp_w+EPSILON)
        wight_vi_w =  vi_temp_w/(ir_temp_w+vi_temp_w+EPSILON)
        wight_ir_h =  ir_temp_h/(ir_temp_h+vi_temp_h+EPSILON)
        wight_vi_h =  vi_temp_h/(ir_temp_h+vi_temp_h+EPSILON)


        x_ir = x_ir*wight_ir_w*wight_ir_h
        x_vi = x_vi*wight_vi_w*wight_vi_h
        x_out = torch.cat([x_ir, x_vi], 1)

        output = x_res+x_out
        output = self.bottle(output)
        return output

class Coord2_Fusion(nn.Module):
    def __init__(self):
        super(Coord2_Fusion, self).__init__()
        self.coord = CoordAtt(2, 2)
        self.bottle = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, D_ir, D_vi):
        output = torch.cat([D_ir, D_vi], 1)


        output = self.coord(output)+output

        output = self.bottle(output)
        return output



