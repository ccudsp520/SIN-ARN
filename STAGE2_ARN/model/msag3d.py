import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('/media/user/DATA/GLTA/code_ori/MS_G3D/')
from utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP
from model.activation import activation_factory
# sys.path.append('/media/user/DATA/GLTA/')
from model.MSA_ATnet import ATNet

def spatial_cross(x, fidx1, fidx2):
    x_spc1 = np.transpose(x[:, :, np.array(fidx1) - 1] - x, (1, 2, 0))  # (300, 25, 3)
    x_spc2 = np.transpose(x[:, :, np.array(fidx2) - 1] - x, (1, 2, 0))
    x_spcc = 100 * np.cross(x_spc1, x_spc2)
    # print(x_spcc.shape)  # (300, 25, 3)
    return x_spcc


def edge(x):

    x = np.transpose(x, [0, 4, 1, 2, 3])
    N = x.shape[0]
    M = x.shape[1]
    C = x.shape[2]
    T = x.shape[3]
    V = x.shape[4]

    ori_list = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8),
                (24, 25), (25, 12)]

    x_spc_n = []
    for n in range(N):
        x_spc_m = []
        for m in range(M):
            x_spc_r = []

            for r in range(len(ori_list)):
                x_m = x[n, m, :, :, :]
                # print('x_m :', x_m.shape)
                x_spc1 = np.transpose(
                    x_m[:, :, np.array(ori_list[r][0]) - 1] - x_m[:, :, np.array(ori_list[r][1] - 1)], (1, 0))
                x_spcc = np.copy(x_spc1)
                x_spc_r.append(x_spcc)
            x_spc_m.append(x_spc_r)
        x_spc_n.append(x_spc_m)

    x_spc_n_to_array = np.array(x_spc_n)
    # print(x_spc_n_to_array.shape)  # (16, 2, 300, 300, 3)
    x_spc_n_to_array = x_spc_n_to_array.transpose(0, 4, 3, 2, 1)
    # print('real', x_spc_n_to_array.shape)  # (16, 3, 300, 300, 2)
    # x_spc_n_to_array = torch.tensor(x_spc_n_to_array).cuda()

    # print(x_spc_n_to_array.shape)  #(16, 3, 300, 25, 2)

    return x_spc_n_to_array


def surface(x):

    x = np.transpose(x, [0, 4, 1, 2, 3])
    # print(data.shape) #(40091, 2, 3, 300, 25)

    # Spatial cross
    # fidx0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    fidx1 = [17, 21, 4, 21, 6, 5, 6, 7, 21, 11, 12, 11, 1, 13, 16, 14, 18, 17, 18, 19, 22, 8, 8, 12, 12]
    fidx2 = [13, 1, 21, 3, 21, 7, 8, 23, 10, 9, 10, 25, 14, 15, 14, 15, 1, 19, 20, 18, 9, 23, 22, 25, 24]

    N = x.shape[0]  # (40091, 2, 3, 300, 25)
    M = x.shape[1]
    C = x.shape[2]
    T = x.shape[3]
    V = x.shape[4]

    x_spc_n = []
    for n in range(N):
        x_spc_list = []
        for m in range(M):
            x_m = x[n, m, :, :, :]
            # print('xm',x_m.shape) #(3, 300, 25)
            x_spc_list.append(spatial_cross(x_m, fidx1, fidx2))
            x_spc_list_to_array = np.array(x_spc_list)
            # print(x_spc_list_to_array.shape)  # (40091, 2, 300, 25, 3)
        x_spc_n.append(x_spc_list)

    x_spc_n_to_array = np.array(x_spc_n)
    # print(x_spc_n_to_array.shape)  # (40091, 2, 300, 25, 3)
    x_spc_n_to_array = x_spc_n_to_array.transpose(0, 4, 2, 3, 1)
    # print('x_spc_n_to_array', x_spc_n_to_array)
    # print('real', x_spc_n_to_array.shape)  # (40091, 3, 300, 25, 2)

    return x_spc_n_to_array


def motion(x):
    x = np.transpose(x, [0, 4, 2, 1, 3])
    N = x.shape[0]  # (16, 2, 3, 300, 25)
    M = x.shape[1]
    T = x.shape[2]
    C = x.shape[3]
    V = x.shape[4]

    x_spc_n = []
    for n in range(N):
        x_spc_m = []
        for m in range(M):
            x_spc_r = []
            for r in range(T-1):
                x_m = x[n, m, :, :, :]  # (40091, 2, 300, 3, 25)
                x_spc1 = x_m[r + 1, :, :] - x_m[r, :, :]
                x_spcc = np.copy(x_spc1)
                x_spc_r.append(x_spcc)
            x_last = x_m[0, :, :] - x_m[0, :, :]
            x_spc_r.append(x_last)
            x_spc_m.append(x_spc_r)
        x_spc_n.append(x_spc_m)

    x_spc_n_to_array = np.array(x_spc_n)
    x_spc_n_to_array = x_spc_n_to_array.transpose(0, 3, 2, 4, 1)

    return x_spc_n_to_array


def velocity(x):

    x = np.transpose(x, [0, 4, 1, 2, 3])
    # print(x.shape) #(40091, 2, 3, 300, 25)
    N = x.shape[0]
    M = x.shape[1]
    C = x.shape[2]
    T = x.shape[3]
    V = x.shape[4]

    x_spc_n = []
    for n in range(N):
        x_spc_list = []
        for m in range(M):
            x_m = x[n, m, :, :, :]
            x_spt_list = []
            x_spt_list.append(np.transpose(x_m[:, 0, :] - x_m[:, 0, :], (1, 0)))
            for t in range(T - 2):
                # print('xm',x_m.shape) #(3, 300, 25)
                x_spc1 = np.transpose(x_m[:, t, :] - x_m[:, t + 1, :], (1, 0))  # (25, 3)
                x_spc2 = np.transpose(x_m[:, t + 2, :] - x_m[:, t + 1, :], (1, 0))  # (25, 3)
                x_spcc = x_spc1 + x_spc2
                x_spt_list.append(x_spcc)
                # print(np.array(x_spt_list).shape)
            x_spt_list.append(np.transpose(x_m[:, 0, :] - x_m[:, 0, :], (1, 0)))
            x_spc_list.append(x_spt_list)
        x_spc_n.append(x_spc_list)
    x_spc_n_to_array = np.array(x_spc_n)
    # print(x_spc_n_to_array.shape)  # (40091, 2, 300, 25, 3)
    x_spc_n_to_array = x_spc_n_to_array.transpose(0, 4, 2, 3, 1)
    # print('x_spc_n_to_array', x_spc_n_to_array)
    # print('real', x_spc_n_to_array.shape)  # (40091, 3, 300, 25, 2)

    return x_spc_n_to_array


class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3,5],
                 window_stride=1,
                 window_dilations=[1,1]):

        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        # Input shape: (N, C, T, V)
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        # no activation
        return out_sum


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels

        c1 = 138
        c2 = c1 * 2  # 276
        # c3 = c2 * 2  # 384

        # c1 = 126
        # c2 = c1 * 2  # 252
        # c3 = c2 * 2  # 384

        # c1 = 96
        # c2 = c1 * 2  # 192
        # c3 = c2 * 2  # 384

        # c1 = 18
        # c2 = 42  # 32  
        # c3 = 66  # 66

        # c1 = 84
        # c2 = 162  # 32  
        # c3 = 240  # 66

        # c1 = 96
        # c2 = 96  # 32  
        # c3 = 96  # 66

        # c1 = 84
        # c2 = 84  # 32  
        # c3 = 84  # 66




        # r=3 STGC blocks
        self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)
        

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        # self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        # self.sgcn3 = nn.Sequential(
        #     MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
        #     MS_TCN(c2, c3, stride=2),
        #     MS_TCN(c3, c3))
        # self.sgcn3[-1].act = nn.Identity()
        # self.tcn3 = MS_TCN(c3, c3)

        # self.fc = nn.Linear(c3, num_class)

        # self.fc2 = nn.Linear(1875, 300*25)
        # self.tfc2 = nn.Linear(75, 300)
        self.tfc2 = nn.Linear(150, 300)
        self.AttentionTemporalNet = ATNet(2, 300)   #  MSA

        from model.va import ViewAdaptive
        self.va = ViewAdaptive()

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        ''''''
        # ---------------------------------------------va--------------------------------------------------#
        ''''''
        # x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # x = x.permute(0, 2, 1)  # (16, 300, 150)
        # x = self.va(x)  # (16, 300, 150)
        # x = x.permute(0, 2, 1)  # (16, 150, 300)
        #
        # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # (32, 3, 300, 25)
        ''''''
        # ------------------------------------------(J_E_S_D_A)--------------------------------------------#
        ''''''
        # x = x.cpu().detach().numpy()  # torch to numpy
        #
        # S2 = edge(x)  # 16, 3, 300, 25, 2) E
        # S3 = surface(x)  # (16, 3, 300, 25, 2) S
        # T2 = motion(x)  # (16, 3, 300, 25, 2) D
        # T3 = velocity(x)
        #
        # x = np.concatenate((x, S2, S3, T2, T3), axis=1)
        # N = x.shape[0]
        # C = x.shape[1]
        # T = x.shape[2]
        # V = x.shape[3]
        # M = x.shape[4]
        #
        # x = torch.Tensor(x)
        # x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # (16, 150, 300)
        # x = self.data_bn(x.cuda())
        # x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # (32, 3, 300, 25)

        ''''''
        # # ------------------------------------J+va+preprocess(E_S_D_A)--------------------------------------#
        ''''''
        # x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # x = x.permute(0, 2, 1)  # (16, 300, 150) (N, M * V * C, T)
        # x = self.va(x)  # (16, 300, 150) (N, M * V * C, T)
        # x = x.permute(0, 2, 1)  # (16, 150, 300)
        # x = x.view(N, M, V, C, T).permute(0, 1, 2, 3, 4).contiguous().view(N, M, V, C, T)  # (16, 2, 25, 3, 300)
        # x = x.permute(0, 3, 4, 2, 1)  # (16, 3, 300, 25, 2)
        # x = x.cpu().detach().numpy()  # torch to numpy
        #
        # S2 = edge(x)  # (16, 3, 300, 25, 2) E
        # S3 = surface(x)  # (16, 3, 300, 25, 2) S
        # T2 = motion(x)  # (16, 3, 300, 25, 2) D
        # T3 = velocity(x)
        #
        # x = np.concatenate((x, S2, S3, T2, T3), axis=1)
        #
        # N = x.shape[0]
        # C = x.shape[1]
        # T = x.shape[2]
        # V = x.shape[3]
        # M = x.shape[4]
        # x = torch.Tensor(x)
        # x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # (16, 150, 300)
        # x = self.data_bn(x.cuda())
        # x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # (32, 3, 300, 25)

        ''''''

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)        # (N, T, C*V) ([16, output channel, 300, 25])
        # print("stgc1 output",x.shape)  #   ()
        ################MSA######################
        x = self.AttentionTemporalNet(x)
        ########################################

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)   # [16, output channel, 150, 25])
        # print("stgc2 output",x.shape) # ()

        # print(x.shape)
        # x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        # x = self.tcn3(x)

        # print('check',x.shape)
        # # LayerNorm
        # ln = nn.LayerNorm(x.size()[1:], elementwise_affine=False)
        # x = ln(x)

        out = x
        out_channels = out.size(1)
        # out = out.view(N, M, out_channels, -1)
        # out = out.view(N, M, out_channels, -1) # [N, M,  out_channels=96, 1875->(frame75 * joint25)] #the frame size were reduced by stride
        out = out.view(N, M, out_channels, 150,25).permute(0 , 1 , 2 , 4 , 3 )
        # print(out.shape)  
        out = self.tfc2(out)
        # print(out.shape)
        out = out.view(N, M, out_channels, 25,300).permute(0 , 1 , 2 , 4 , 3 )
        # print("out shape", out.shape)
        return out
        # print('check2', out.shape)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence
        # print('check2', out.shape)


        out = self.fc(out)
        # print('check3', out.shape)
        # return out


if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 6, 3, 50, 25, 2
    x = torch.randn(N,C,T,V,M)
    model.forward(x)

    print('Model total # params:', count_params(model))
