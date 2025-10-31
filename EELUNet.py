import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from models.testblocks.testblock1 import Test_block1
# from models.testblocks.testblock2 import Test_Block2
from models.testblocks.simple_testblock2 import Test_Block2
from models.testblocks.test_block3 import Test_Block3
from models.testblocks.sobel_extract_edge import EdgeExtractionModule

from models.testblocks.ScConv import ScConv

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in//4
        k_size=3
        pad=(k_size-1) // 2

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')

        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        #----------xy----------#
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        #----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(F.interpolate(params_zx, size=x2.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        #----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(F.interpolate(params_zy, size=x3.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        #----------dw----------#
        x4 = self.dw(x4)
        #----------concat----------#
        x = torch.cat([x1,x2,x3,x4],dim=1)
        #----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x


class EELUNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
            # Test_block1(c_list[0], c_list[1]),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            # Test_block1(c_list[1], c_list[2]),
            # TODO: 将2d卷积替换为Test_block1，尝试解决过拟合的问题.
        )
        self.encoder4 = nn.Sequential(
            #Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3]),# todo： 加一个egeunet的Grouped_multi_axis_Hadamard_Product_Attention模块
            Test_block1(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            #Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4]),
            Test_block1(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            #Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5]),
            Test_block1(c_list[4], c_list[5]),
        )


        if bridge:
            # self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            # self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            # self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            # self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            # self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
            #todo: 修改为test_block3
            self.connnect1 = Test_Block2(c_list[0])
            self.connnect2 = Test_Block2(c_list[1])
            self.connnect3 = Test_Block2(c_list[2])
            self.connnect4 = Test_Block2(c_list[3])
            self.connnect5 = Test_Block2(c_list[4])

            # self.scconv = ScConv(c_list[2]) #todo: 20250531加一个注意力模块

            print('testblock2 was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')

        self.decoder1 = nn.Sequential(
            #Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4]),
            Test_block1(c_list[5], c_list[4]),
        )
        self.decoder2 = nn.Sequential(
            #Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3]),
            Test_block1(c_list[4], c_list[3]),
        )
        self.decoder3 = nn.Sequential(
            #Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2]),
            Test_block1(c_list[3], c_list[2]),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
            # Test_block1(c_list[2], c_list[1]),
            # ScConv(c_list[1]), #todo: 使用scconv模块协助提升精度
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )

        self.extractedge = EdgeExtractionModule()

        # self.decoder6 = nn.Sequential(
        #     Test_block1(c_list[3], c_list[2]),
        # )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32

        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        t6 = out

        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        if self.gt_ds:
            gt_pre5 = self.gt_conv1(out5)

            edge_pre5 = self.extractedge(gt_pre5)
            t5 = t5 + t5 * torch.sigmoid(edge_pre5)

            t5 = self.connnect5(t5 , gt_pre5 , edge_pre5, 0.1, 0)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode ='bilinear', align_corners=True)
            edge_pre5 = F.interpolate(edge_pre5, scale_factor=32, mode ='bilinear', align_corners=True)
        else: t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(out4)

            edge_pre4 = self.extractedge(gt_pre4)
            t4 = t4 + t4 * torch.sigmoid(edge_pre4)


            t4 = self.connnect4(t4 , gt_pre4 , edge_pre4, 0.2, 0)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
            edge_pre4 = F.interpolate(edge_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        else:t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(out3)

            edge_pre3 = self.extractedge(gt_pre3)
            t3 = t3 + t3 * torch.sigmoid(edge_pre3) + t3 * torch.sigmoid(gt_pre3)

            t3 = self.connnect3(t3 , gt_pre3 , edge_pre3, 0.3, 0.1)

            # t3 = self.scconv(t3) #todo: 20250531,应用这个注意力模块，在connect5后初始化过了

            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
            edge_pre3 = F.interpolate(edge_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        else: t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)

            edge_pre2 = self.extractedge(gt_pre2)
            t2 = t2 + t2 * torch.sigmoid(edge_pre2) + t2 * torch.sigmoid(gt_pre2)

            t2 = self.connnect2(t2 , gt_pre2 , edge_pre2, 0.4, 0.2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
            edge_pre2 = F.interpolate(edge_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        else: t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(out1)

            edge_pre1 = self.extractedge(gt_pre1)
            t1 = t1 + t1 * torch.sigmoid(edge_pre1) + t1 * torch.sigmoid(gt_pre1)

            t1 = self.connnect1(t1 , gt_pre1 , edge_pre1, 0.5, 0.3)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
            edge_pre1 = F.interpolate(edge_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        else: t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W

        if self.gt_ds:

            # # 可视化gt_pre和edge_pre
            # import matplotlib.pyplot as plt
            # import numpy as np
            # import os
            #
            # # 指定保存路径
            # save_dir = "/home/RAID0/wtx/project/A6000_UltraLightUNet/data/isic1718/isic2017/val/pics"
            # os.makedirs(save_dir, exist_ok=True)  # 确保文件夹存在
            #
            # # 保存gt_pre图片
            # gt_preds = [gt_pre1, gt_pre2, gt_pre3, gt_pre4, gt_pre5]
            # for i, gt_pred in enumerate(gt_preds):
            #     # 转换为numpy并取第一个batch和通道
            #     img = gt_pred[0, 0].detach().cpu().numpy()
            #
            #     plt.figure(figsize=(8, 8))
            #     plt.imshow(img, cmap='gray')
            #     plt.title(f'GT_Pre{i+1}')
            #     plt.axis('off')
            #
            #     # 保存单独的gt_pre图片
            #     save_path = os.path.join(save_dir, f'gt_pre{i+1}.png')
            #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
            #     plt.close()
            #
            # # 保存edge_pre图片
            # edge_preds = [edge_pre1, edge_pre2, edge_pre3, edge_pre4, edge_pre5]
            # for i, edge_pred in enumerate(edge_preds):
            #     # 转换为numpy并取第一个batch和通道
            #     img = edge_pred[0, 0].detach().cpu().numpy()
            #
            #     plt.figure(figsize=(8, 8))
            #     plt.imshow(img, cmap='gray')
            #     plt.title(f'Edge_Pre{i+1}')
            #     plt.axis('off')
            #
            #     # 保存单独的edge_pre图片
            #     save_path = os.path.join(save_dir, f'edge_pre{i+1}.png')
            #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
            #     plt.close()
            #
            # print(f'所有图片已保存到: {save_dir}')

            return (
                (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2), torch.sigmoid(gt_pre1)),
                (torch.sigmoid(edge_pre5), torch.sigmoid(edge_pre4), torch.sigmoid(edge_pre3), torch.sigmoid(edge_pre2), torch.sigmoid(edge_pre1)),
                torch.sigmoid(out0)
            )
        else:
            return torch.sigmoid(out0)


if __name__ == '__main__':
    tensor = torch.randn(1, 3, 256, 256)
    model = EELUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64] , bridge=True , gt_ds=True)
    gt_pre, edge, out = model(tensor)

    # 打印gt_pre中每个张量的形状
    print("GT预测形状:")
    for i, t in enumerate(gt_pre):
        print(f"gt_pre[{i}].shape = {t.shape}")

    # 打印edge中每个张量的形状
    print("\nEdge预测形状:")
    for i, t in enumerate(edge):
        print(f"edge[{i}].shape = {t.shape}")

    # 打印最终输出形状
    print(f"\n最终输出形状: {out.shape}")
