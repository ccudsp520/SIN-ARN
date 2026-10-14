import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import pandas as pd
import sys
# sys.path.append('/home/antony/lin/GLTA/')
sys.path.append('/media/user/DATA/GLTA/')
sys.path.append("/path/to/DLNest/Common")
from Dataset.Datasets import NTUDataSet
from tqdm import tqdm
from torchvision import transforms
from src.dataprocessor import *
from src.graph import Graph
import GCN_Unsupervised.GCNEncoder.AGCNEncoder_3layer_action_atu as AGCNEncoder
# from GCN_Unsupervised.unsupervised_partGCN import Model # global+local gcn (ok)
# from GCN_Unsupervised.unsupervised_globalgcn import Model #change globalgcn (have problem)
# from GCN_Unsupervised.unsupervised_globalgcn_3layer import Model #change 3layer
# from GCN_Unsupervised.unsupervised_globalgcn_10layer_action import Model  #change 10layer+action
from unsupervised_globalgcn_3layer_action_KD_atu import Model  # change 3layer+action
from unsupervised_globalgcn_3layer_action_KD_atu import Decoder
from unsupervised_globalgcn_3layer_action_KD_atu import AGCNClassifier
from model.msag3d import Model as MSA_G3D
from model.msg3d import  Model as MSG3D

from GCN_Unsupervised.unsupervised_globalgcn_3layer_action_KD_atu import Model_teacher  # change 3layer+action
# from unsupervised_globalgcn_3layer_action_KD_atu import Model_student
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import RandomSampler
import torch.nn.functional as F
import random

######################random seed set###############
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#####################################################
np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True, linewidth=800)
# 定義訓練參數
num_epochs = 60
vali_epoch = 1
train_batch_size = 16
test_batch_size = 1
learning_rate = 0.1
frame = 300
downsample_rate = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:0')
config = {
    "model_file_path": "//home/dsp520/lin/GLTA/GCN_Unsupervised/Model.py",
    "save_root": "/home/dsp520/lin/GLTA/partGCNsaves",
    "other_file_paths": ["/home/dsp520/lin/GLTA/GCN_Unsupervised/unsupervised_partGCN.py"],
    "dataset_config": {
        "directory": "/home/dsp520/lin/2s-AGCN/data/ntu/",
        "partition": "xsub",
        "num_workers": 8,
        "batchsize": 16,
        "use_mmap": True,
        "debug": False
    },
    "model_config": {
        "model_path": "GCN_Unsupervised.unsupervised_partGCN",
        "device": [0],
        "step": [30, 50, 60],
        "model_args": {
            "prediction_mask": 0,
            "predict_seg": 1,
            "if_rotation": False,
            "seg_num": 1,
            "if_vibrate": False,
            "T": 150,
            "num_class": 60,
            "num_point": 25,
            "num_person": 2,
            "graph": None,
            "graph_args": None,
            "in_channels": 3,
            "GCNEncoder": "shiftGCN",
            "ATU_layer": 2,
            "self_supervised_mask": 1
        },
        "base_lr": 0.1,
        "knn_k": 1,
        "R-rotation": False,
        "R-rotation-loss": 0,
        "R-drop": False,
        "R-drop-loss": 0,
        "test_epoch_interval": 1,
        "autoencoder": False,
        "loss": "CustomizeL2loss",
        "innerautoencoder_loss": False,
        "joint_loss": 1,
        "bone_length_loss": 0,
        "motion_loss": 0,
        "joint_direction_loss": 0,
        "colorization": 0
    },
    "ckpt_load": "",
    "epoch": 70,
    "is_test": False,
    "save_dir": "/home/dsp520/lin/GLTA/npz_save/151_wholeBranch"
}

args = {
    "dataset_config": {
        "directory": "../hong_MSG3D/data/",
        "partition": "xsub",
        "use_mmap": True,
        "batchsize": 16,
        "num_workers": 4,
        "debug": False,
        "occlusion_part": [4],
        # choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], help='1:left arm, 2:right arm, 3:two hands, 4:two legs, 5:trunk'
        "downsample_rate": 1
    }
}
args = args["dataset_config"]
train_dataset_config = {
    # "data_path":self.args["directory"]+self.args["partition"]+"/train_data_joint.npy",#change
    "data_path": args["directory"] + args["partition"] + "/train_data_joint.npy",
    # "data_path":args["directory"]+args["partition"]+"/train_multi_joint_notnormalize2.npy",
    "label_path": args["directory"] + args["partition"] + "/train_label.pkl",
    "use_mmap": args["use_mmap"],
    "debug": args["debug"],
    "downsample_rate": args["downsample_rate"]
}
val_dataset_config = {
    # "data_path":self.args["directory"]+self.args["partition"]+"/val_data_joint.npy",#change
    "data_path": args["directory"] + args["partition"] + "/val_data_joint.npy",
    # "data_path":args["directory"]+args["partition"]+"/val_multi_joint_notnormalize2.npy",
    "label_path": args["directory"] + args["partition"] + "/val_label.pkl",
    "use_mmap": args["use_mmap"],
    "debug": args["debug"],
}

def mpjpe(predicted, target, reverse_center=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    if predicted.shape != target.shape:
        print('predicted.shape', predicted.shape)  # [16, 3, 300, 25, 2]
        print('target.shape', target.shape)
    assert predicted.shape == target.shape
    error = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))  # dim=4
    # print('error',error)
    return error


transform = transforms.Compose([
    Occlusion_part(args["occlusion_part"]),
])


class Model_student(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,if_rotation=False,seg_num=1,if_vibrate=False,prediction_mask=0,GCNEncoder="AGCN",ATU_layer=2,T=300,predict_seg=1, occ=1):
        super(Model_student, self).__init__()
        # print('in_channels: ',in_channels) #3
        # self.full_body = AGCNEncoder.AGCNEncoder(num_class, 25, num_person, graph, graph_args,64) #change in_channels 3 -> 64
        # self.full_body = AGCNEncoder.AGCNEncoder(num_class, 25, num_person, graph, graph_args,in_channels, adaptive=True, attention=True,ATU_layer=ATU_layer) #change
        self.full_body = MSA_G3D(num_class=60,num_point=25,num_person=2,num_gcn_scales=13,num_g3d_scales=6,graph='graph.ntu_rgb_d.AdjMatrixGraph') #MSA_G3D


        self.FULL_BODY = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
        if GCNEncoder == "AGCN":
            self.Encoder = AGCNEncoder.AGCNEncoder(num_class, num_point, num_person, graph, graph_args,64,ATU_layer=ATU_layer) #change 3layer
            # self.Encoder = AGCNEncoder.AGCNEncoder(num_class, num_point, num_person, graph, graph_args,256) #change 10layer
        self.Decoder = Decoder()
        
        self.if_rotation = if_rotation
        self.if_vibrate = if_vibrate
        # self.AttentionTemporalNet = ATNet(ATU_layer,T)

        self.predict_seg = predict_seg

        self.Classifier = AGCNClassifier()

        # self.AAGCN = AGCNEncoder.AAGCN()
        # self.AAGCN = AGCNEncoder.MSAAGCN()
        # self.AAGCN.requires_grad = False

        self.MSG3D = MSG3D(num_class=60, num_point=25, num_person=2, num_gcn_scales=13, num_g3d_scales=6, in_channels=15, graph='graph.ntu_rgb_d.AdjMatrixGraph')

        self.parts = dict()
        self.parts[1] = np.array([5, 6, 7, 8, 22, 23]) - 1                      # left arm
        self.parts[2] = np.array([9, 10, 11, 12, 24, 25]) - 1                   # right arm
        self.parts[3] = np.array([22, 23, 24, 25]) - 1                          # two hands
        self.parts[4] = np.array([13, 14, 15, 16, 17, 18, 19, 20]) - 1          # two legs
        self.parts[5] = np.array([1, 2, 3, 4, 21]) - 1                          # trunk
        self.parts[6] = np.array([5, 6, 7, 8, 13, 14, 15, 16, 22, 23]) - 1      # left body
        self.parts[7] = np.array([9, 10, 11, 12, 17, 18, 19, 20, 24, 25]) - 1   # right body
        self.parts[8] = np.array([], dtype=np.int32)  # no occlusion
        # self.parts[9] = np.array([], dtype=np.int32)
        # self.parts[10] = np.array([], dtype=np.int32)
        # self.parts[11] = np.array([], dtype=np.int32)
        # self.parts[12] = np.array([], dtype=np.int32)
        # self.parts[13] = np.array([], dtype=np.int32)
        miss_rate = 0.6
        self.joint_list=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])-1
        self.joint_occlusion_num=int(len(self.joint_list) *miss_rate)

        #permutation

        self.parts_history = []
        self.occ = occ
        self.init_parts_9_13()

    def init_parts_9_13(self):
        self.parts[9] = np.array([], dtype=np.int32)
        self.parts[10] = np.array([], dtype=np.int32)
        self.parts[11] = np.array([], dtype=np.int32)
        self.parts[12] = np.array([], dtype=np.int32)
        self.parts[13] = np.array([], dtype=np.int32)
        # 分別給定特定的遮擋比例
        self.parts[9] = np.random.choice(self.joint_list, int(len(self.joint_list) * 0.2), replace=False) - 1
        self.parts[10] = np.random.choice(self.joint_list, int(len(self.joint_list) * 0.3), replace=False) - 1
        self.parts[11] = np.random.choice(self.joint_list, int(len(self.joint_list) * 0.4), replace=False) - 1
        self.parts[12] = np.random.choice(self.joint_list, int(len(self.joint_list) * 0.5), replace=False) - 1
        self.parts[13] = np.random.choice(self.joint_list, int(len(self.joint_list) * 0.6), replace=False) - 1


    def select_part_for_epoch(self, epoch):    
        num_parts = 13
        if epoch == 0 or epoch % 13 == 0:
            # 第一個 epoch 或每 13 個 epoch 重新選擇
            selected_part = np.random.choice(np.arange(1, num_parts + 1))
        else:
            # 接下來的 12 個 epoch 選擇與前一次不同的部位
            last_selected_part = self.parts_history[-1]
            available_parts = list(set(range(1, num_parts + 1)) - {last_selected_part})
            selected_part = np.random.choice(available_parts)

        # 將選擇的部位記錄下來
        self.parts_history.append(selected_part)

        # 返回實際的部位索引
        return selected_part



    def forward(self,x, epoch, is_test=False):
        N, C, T, V, M = x.size()
        ori_data = x.clone()
        joint_occlusion = np.random.choice(self.joint_list, self.joint_occlusion_num, replace=False)
        x_mask = np.ones((x.shape[0],3, 300, 25,2))
        num_parts = 13 #len(parts)
        part_indices = np.random.choice(num_parts, size=N, replace=True)

        # x[:,:,:, self.parts[1], :] = 0
        # x_mask[:, :,:, self.parts[1], :] = 0 #change occlusion 

        # x[:,:,:, joint_occlusion, :] = 0
        # x_mask[:, :, :, joint_occlusion, :] = 0 #change random occlusion

        

    # def forward(self, x, is_test=False):
    #     N, C, T, V, M = x.size()
    #     ori_data = x.clone()
    #     joint_occlusion = np.random.choice(self.joint_list, self.joint_occlusion_num, replace=False)
    #     x_mask = np.ones((x.shape[0], 3, 300, 25, 2))
    #     num_parts = 13  # len(parts)
    #     part_indices = np.random.choice(num_parts, size=N, replace=True)

        # if self.occ == 1 or self.occ == 2 or self.occ == 3 or self.occ == 4 or self.occ == 5 or self.occ == 6 or self.occ == 7 or self.occ == 8 \
        # or self.occ == 9 or self.occ == 10 or self.occ == 11 or self.occ == 12 or self.occ == 13 :
        #     x[:,:,:, self.parts[self.occ], :] = 0
        #     x_mask[:, :,:, self.parts[self.occ], :] = 0 #change occlusion

        if self.occ == 1 or self.occ == 2 or self.occ == 3 or self.occ == 4 or self.occ == 5 or self.occ == 6 or self.occ == 7 or self.occ == 8 \
:
            x[:,:,:, self.parts[self.occ], :] = 0
            x_mask[:, :,:, self.parts[self.occ], :] = 0 #change occlusion

        # x[:,:,:, joint_occlusion, :] = 0
        # x_mask[:, :, :, joint_occlusion, :] = 0 #change random occlusion

        # # 遮擋數據
        # for i_b in range(x.shape[0]):
        #     # skeletons = x[i_b]
        #     # 隨機選擇要遮擋的部位
        #     part_index = part_indices[i_b]
        #     part = self.parts[part_index+1] # 加1是因為部位索引從1開始
        #     if part_index<=8:
        #         # 遮擋所選部位
        #         x[i_b, :, :, part, :] = 2
        #         x_mask[i_b, :, :, part, :] = 0
        #     else:
        #         miss_rate = np.random.choice([0.2,0.3,0.4,0.5,0.6])#0,0.2,0.3,0.4,0.5,0.6
        #         for t in range(x.shape[2]):
        #             joint_occlusion_num = int(len(self.joint_list) * miss_rate)
        #             joint_occlusion = np.random.choice(self.joint_list, joint_occlusion_num, replace=False)
        #             x[i_b, :, t, joint_occlusion, :] = 2
        #             x_mask[i_b,:,t, joint_occlusion, :] = 0
            
        elif self.occ == 0.2 or self.occ == 0.3 or self.occ == 0.4 or self.occ == 0.5 or self.occ == 0.6:
            for i_b in range(x.shape[0]):
                miss_rate = np.random.choice([self.occ])  # 0,0.2,0.3,0.4,0.5,0.6
                for t in range(x.shape[2]):
                    joint_occlusion_num = int(len(self.joint_list) * miss_rate)
                    joint_occlusion = np.random.choice(self.joint_list, joint_occlusion_num, replace=False)
                    x[i_b, :, t, joint_occlusion, :] = 0
                    x_mask[i_b, :, t, joint_occlusion, :] = 0

        # 定义GCN_feature的维度为(N,M,64,T,V)
        # GCN_feature = torch.empty(N, M, 64, T, V).cuda().to(torch.float32)  # N,M,C,T,V  [16, 2, 64, 300, 25] #3layer
        # GCN_feature = torch.empty(N,M,256,T,V).cuda().to(torch.float32) # N,M,C,T,V  [16, 2, 64, 300, 25] #10layer

        # 將各部分特徵分別賦值給GCN_feature
        # GCN_feature[:, :, :, :, self.FULL_BODY - 1] = self.full_body(
            # x[:, :, :, self.FULL_BODY - 1, :])  # torch.Size([16, 2, 64, 300, 25])
        
        # GCN_feature = torch.empty(N,M,66,T,V).cuda().to(torch.float32) # N,M,C,T,V  [16, 2, 384, 300, 25] #MSA_G3D
        # GCN_feature = torch.empty(N,M,384,T,V).cuda().to(torch.float32) # N,M,C,T,V  [16, 2, 384, 300, 25] #MSA_G3D
        GCN_feature = torch.empty(N,M,276,T,V).cuda().to(torch.float32) # N,M,C,T,V  [16, 2, 384, 300, 25] #MSA_G3D
        # GCN_feature = torch.empty(N,M,84,T,V).cuda().to(torch.float32) # N,M,C,T,V  [16, 2, 384, 300, 25] #MSA_G3D

        # 將各部分特徵分別賦值給GCN_feature
        GCN_feature[:,:,:,:,self.FULL_BODY-1] = self.full_body(x[:,:,:,self.FULL_BODY-1,:]) #torch.Size([16, 2, 64, 300, 25])

        # GCN_feature = GCN_feature.permute(0,2,3,4,1) #N,C,T,V,M #torch.Size([16, 64, 300, 25, 2])
        # GCN_feature = self.Encoder(GCN_feature)

        # # Attention Temporal Net
        # GCN_feature = self.AttentionTemporalNet(GCN_feature)

        # Decoder，输出維度為(N, C, T, V, M)
        output = self.Decoder(GCN_feature)

        x_mask = torch.tensor(x_mask, dtype=torch.float32, requires_grad=False).to(device)
        output_mask = output * (1 - x_mask) + ori_data * (x_mask)

        # output_action = self.AAGCN(output_mask)#output_mask
        output_action = self.MSG3D(output_mask)
        return output, GCN_feature, output_action, x_mask, None



'''
class Model_student(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 if_rotation=False, seg_num=1, if_vibrate=False, prediction_mask=0, GCNEncoder="AGCN", ATU_layer=2,
                 T=300, predict_seg=1, occ=1):
        super(Model_student, self).__init__()
        # print('in_channels: ',in_channels) #3
        # self.full_body = AGCNEncoder.AGCNEncoder(num_class, 25, num_person, graph, graph_args,64) #change in_channels 3 -> 64
        self.full_body = AGCNEncoder.AGCNEncoder(num_class, 25, num_person, graph, graph_args, in_channels,
                                                 adaptive=True, attention=True, ATU_layer=ATU_layer)  # change
        self.FULL_BODY = np.asarray(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        if GCNEncoder == "AGCN":
            self.Encoder = AGCNEncoder.AGCNEncoder(num_class, num_point, num_person, graph, graph_args, 64,
                                                   ATU_layer=ATU_layer)  # change 3layer
            # self.Encoder = AGCNEncoder.AGCNEncoder(num_class, num_point, num_person, graph, graph_args,256) #change 10layer

        self.Decoder = Decoder()
        self.occ = occ
        self.if_rotation = if_rotation
        self.if_vibrate = if_vibrate
        # self.AttentionTemporalNet = ATNet(ATU_layer,T)
        self.predict_seg = predict_seg
        self.Classifier = AGCNClassifier()

        # self.AAGCN = AGCNEncoder.AAGCN()
        self.AAGCN = AGCNEncoder.MSAAGCN()

        self.parts = dict()
        self.parts[1] = np.array([5, 6, 7, 8, 22, 23]) - 1  # left arm
        self.parts[2] = np.array([9, 10, 11, 12, 24, 25]) - 1  # right arm
        self.parts[3] = np.array([22, 23, 24, 25]) - 1  # two hands
        self.parts[4] = np.array([13, 14, 15, 16, 17, 18, 19, 20]) - 1  # two legs
        self.parts[5] = np.array([1, 2, 3, 4, 21]) - 1  # trunk
        self.parts[6] = np.array([5, 6, 7, 8, 13, 14, 15, 16, 22, 23]) - 1  # left body
        self.parts[7] = np.array([9, 10, 11, 12, 17, 18, 19, 20, 24, 25]) - 1  # right body
        self.parts[8] = np.array([], dtype=np.int32)  # no occlusion
        self.parts[9] = np.array([], dtype=np.int32)
        self.parts[10] = np.array([], dtype=np.int32)
        self.parts[11] = np.array([], dtype=np.int32)
        self.parts[12] = np.array([], dtype=np.int32)
        self.parts[13] = np.array([], dtype=np.int32)

        miss_rate = 0.2
        self.joint_list = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]) - 1
        self.joint_occlusion_num = int(len(self.joint_list) * miss_rate)

    def forward(self, x, is_test=False):
        N, C, T, V, M = x.size()
        ori_data = x.clone()
        joint_occlusion = np.random.choice(self.joint_list, self.joint_occlusion_num, replace=False)
        x_mask = np.ones((x.shape[0], 3, 300, 25, 2))
        num_parts = 13  # len(parts)
        part_indices = np.random.choice(num_parts, size=N, replace=True)

        if self.occ == 1 or self.occ == 2 or self.occ == 3 or self.occ == 4 or self.occ == 5 or self.occ == 6 or self.occ == 7 or self.occ == 8:
            x[:,:,:, self.parts[self.occ], :] = 0
            x_mask[:, :,:, self.parts[self.occ], :] = 0 #change occlusion

        # x[:,:,:, joint_occlusion, :] = 0
        # x_mask[:, :, :, joint_occlusion, :] = 0 #change random occlusion

        # # 遮擋數據
        # for i_b in range(x.shape[0]):
        #     # skeletons = x[i_b]
        #     # 隨機選擇要遮擋的部位
        #     part_index = part_indices[i_b]
        #     part = self.parts[part_index+1] # 加1是因為部位索引從1開始
        #     if part_index<=8:
        #         # 遮擋所選部位
        #         x[i_b, :, :, part, :] = 2
        #         x_mask[i_b, :, :, part, :] = 0
        #     else:
        #         miss_rate = np.random.choice([0.2,0.3,0.4,0.5,0.6])#0,0.2,0.3,0.4,0.5,0.6
        #         for t in range(x.shape[2]):
        #             joint_occlusion_num = int(len(self.joint_list) * miss_rate)
        #             joint_occlusion = np.random.choice(self.joint_list, joint_occlusion_num, replace=False)
        #             x[i_b, :, t, joint_occlusion, :] = 2
        #             x_mask[i_b,:,t, joint_occlusion, :] = 0
        elif self.occ == 0.2 or self.occ == 0.3 or self.occ == 0.4 or self.occ == 0.5 or self.occ == 0.6:
            for i_b in range(x.shape[0]):
                miss_rate = np.random.choice([self.occ])  # 0,0.2,0.3,0.4,0.5,0.6
                for t in range(x.shape[2]):
                    joint_occlusion_num = int(len(self.joint_list) * miss_rate)
                    joint_occlusion = np.random.choice(self.joint_list, joint_occlusion_num, replace=False)
                    x[i_b, :, t, joint_occlusion, :] = 0
                    x_mask[i_b, :, t, joint_occlusion, :] = 0

        # 定义GCN_feature的维度为(N,M,64,T,V)
        GCN_feature = torch.empty(N, M, 64, T, V).cuda().to(torch.float32)  # N,M,C,T,V  [16, 2, 64, 300, 25] #3layer
        # GCN_feature = torch.empty(N,M,256,T,V).cuda().to(torch.float32) # N,M,C,T,V  [16, 2, 64, 300, 25] #10layer

        # 將各部分特徵分別賦值給GCN_feature
        GCN_feature[:, :, :, :, self.FULL_BODY - 1] = self.full_body(
            x[:, :, :, self.FULL_BODY - 1, :])  # torch.Size([16, 2, 64, 300, 25])

        # GCN_feature = GCN_feature.permute(0,2,3,4,1) #N,C,T,V,M #torch.Size([16, 64, 300, 25, 2])
        # GCN_feature = self.Encoder(GCN_feature)

        # # Attention Temporal Net
        # GCN_feature = self.AttentionTemporalNet(GCN_feature)

        # Decoder，输出維度為(N, C, T, V, M)
        output = self.Decoder(GCN_feature)

        x_mask = torch.tensor(x_mask, dtype=torch.float32, requires_grad=False).to(device)
        output_mask = output * (1 - x_mask) + ori_data * (x_mask)

        output_action = self.AAGCN(output_mask)#output_mask
        return output, GCN_feature, output_action, x_mask, None
'''

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


inform = {
    "occ_index": [], "total_acc": [], "total_error": [], "total_mpjpe_all": [], "total_mpjpe": [], "data.shape": [],
    "video_number": [], "label": [], "predict_label": []
}

inform = pd.DataFrame(inform)

# 載入資料

# train_dataset = NTUDataSet(train_dataset_config,transform=transform) #change
train_dataset = NTUDataSet(train_dataset_config)
randomSampler_train = RandomSampler(train_dataset, replacement=True,
                                    num_samples=int(len(train_dataset) / downsample_rate))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size,
                              shuffle=True)  # , shuffle=False,sampler=randomSampler_train
# train_dataloader = Dataset(config)

gcn_kernel_size = [5, 2]  # change
graph = Graph(max_hop=gcn_kernel_size[1])  # change
A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False).to(device)  # change

occ_list = [1,2,3,4,5,6,7,8,0.2,0.3,0.4,0.5,0.6]

for occ in occ_list:
    print("occ_index:",occ)
    # 定義模型
    # model = Model(num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,self_supervised_mask=1,if_rotation=False,seg_num=1,if_vibrate=False,prediction_mask=0,GCNEncoder="AGCN",ATU_layer=2,T=300,predict_seg=1)
    model_teacher = Model(num_class=60, num_point=25, num_person=2, graph=A,
                          graph_args=dict())  # graph=ntu_rgb_d.Graph,graph_args=dict()
    model_student = Model_student(num_class=60, num_point=25, num_person=2, graph=A, graph_args=dict(), ATU_layer=2,
                                  in_channels=3, if_rotation=False, seg_num=1, if_vibrate=False, prediction_mask=0,
                                  GCNEncoder="AGCN", T=300, predict_seg=1, occ=occ)  # change
    # 定義損失函數和優化器
    criterion_mse = nn.MSELoss()
    criterion_crossentropy = nn.CrossEntropyLoss()
    criterion_kld = nn.KLDivLoss(reduction='batchmean')

    # optimizer = torch.optim.SGD(model_student.AAGCN.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001,
    #                             nesterov=True)
    # optimizer = torch.optim.SGD(model.AAGCN.parameters(), lr=learning_rate,momentum=0.9, weight_decay=0.0001, nesterov=True) #only指定AAGCN的参数

    optimizer = torch.optim.SGD(model_student.MSG3D.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001,
                                nesterov=True)

    # step = [30, 40]
    step = [30, 50]
    warm_up_epoch = 5
    lr_scheduler_pre = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=step, gamma=0.1)
    lr_scheduler = GradualWarmupScheduler(optimizer, total_epoch=warm_up_epoch,
                                          after_scheduler=lr_scheduler_pre)
    # 初始化最低loss為正無限大
    lowest_loss = float('inf')

    # 測試模型
    test_dataset = NTUDataSet(val_dataset_config)
    # test_dataset = NTUDataSet(val_dataset_config,transform=transform) #change
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)  # , shuffle=True

    # model_student.load_state_dict(torch.load('./model_weight_atu/model_best_epoch46_parts12345678_0.2to0.6_occinnet_global_3layer_downsample1_action_noAE_atul2head8ingcnl1_1encoder_post1.pth'))
    # model_student.load_state_dict(torch.load('./model_weight_atu/model_best_parts12345678_0.2to0.6_occinnet_global_3layer_downsample1_action_noAE_atul2head8ingcnl1_1encoder_post1_epoch54.pth'))

    model_student.load_state_dict(torch.load(
       './weight_hong/msag3d_epoch73.pth'))

    # model_student.load_state_dict(torch.load('./model_weight_rich/model_best_parts12345678_occinnet_global_3layer_downsample1_action_noAE_atul2head8ingcnl1_1encoder_rich_post1_epoch48.pth'))

    model_student.to(device)  # 加入這行程式
    model_student.eval()

    for epoch in range(vali_epoch):
        with torch.no_grad():
            total_error = 0.0
            total_error_all = 0.0
            total_mpjpe = 0.0
            total_mpjpe_all = 0.0
            total_acc = 0.0
            tqdm_time = 0

            # 計時
            num_batches = 0
            num_samples = 0
            t_fwd_total = 0.0     # model forward (純推論)
            t_post_total = 0.0    # 後處理(重建/遮罩/MPJPE/acc)
            # 計時

            for data in tqdm(test_dataloader):
                tqdm_time += 1
                # data = data.to(device)
                label = data['label'].to(device)
                data = data['data'].to(device)  # [16, 3, 300, 25, 2]#change
                data = data[:, :, :frame, :, :]  # change
                # data_orig = data.clone()

                mask_data = data.clone()  # [16, 3, 300, 25, 2] #.clone()複製data到不同儲存區,與原本data互不相關
                # joint_occlusion = np.random.choice(joint_list, joint_occlusion_num, replace=False)

                # 計時
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                # 計時

                reconstructed = model_student(mask_data, epoch)  # change

                # 計時
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                # 計時

                predict_action_class = reconstructed[2]
                x_mask = reconstructed[3]

                mask_data = mask_data * x_mask
                reconstructed_all = reconstructed[0]
                reconstructed = reconstructed[0] * (1 - x_mask) + data * (x_mask)  # change [0]:decoder output ,[1]:hidden_feature,[2]:GCN_feature,[4]:autoencoder_output_feature
                # reconstructed = reconstructed[0]

                # error_orig = mpjpe(data, data,reverse_center=False)
                # error_recon = mpjpe(reconstructed_orig, data_orig,mask)
                error_recon_all = mpjpe(reconstructed_all, data)
                error_recon = mpjpe(reconstructed, data)

  
                predict_label = torch.max(predict_action_class, 1)[1]
                acc = torch.mean((predict_label == label).float())
                
                # 計時
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                # 計時

                total_error_all += error_recon_all.item()
                total_error += error_recon.item()
                total_mpjpe_all = total_error_all / tqdm_time
                total_mpjpe = total_error / tqdm_time
                total_acc += acc.item()
                
                # 時間累積
                batch_size = data.size(0)
                num_batches += 1
                num_samples += batch_size
                t_fwd_total += (t1 - t0)
                t_post_total += (t2 - t1)
                # 時間累積

                f = open('xxxxxxxxxx_test', 'a')
                f.write(str(acc))
                f.write(str(predict_label))
                f.write(str(label))
                f.write('\n')
                f.close()


        # do something with reconstructed and hidden_feature
    total_acc = total_acc / len(test_dataloader.dataset) * test_batch_size
    print('total_acc', total_acc)
    print('total_error', total_error)
    print('total_mpjpe_all', total_mpjpe_all)
    print('total_mpjpe', total_mpjpe)

    print('data.shape: ', data.shape)
    video_number = data.shape[0]
    print('video_number: ', video_number)
    print('label', label, label.shape)
    print('predict_label', predict_label, predict_label.shape)

    print('t_fwd_total:',t_fwd_total)
    print('t_post_total:',t_post_total)
#     inform = inform.append({
#         "occ_index": occ, "total_acc": total_acc, "total_error": total_error, "total_mpjpe_all": total_mpjpe_all,
#         "total_mpjpe": total_mpjpe, "data.shape": data.shape, "video_number": video_number, "label": label,
#         "predict_label": predict_label}, ignore_index=1)
# '''label = label
# predict_label = predict_label
# np.save('label', label.cpu().detach().numpy())
# np.save('predict_label', predict_label.cpu().detach().numpy())
# # data = data[video_number,:,:,:,0] 
# data = data[:, :, :, :, :]

# # data = data.permute(1, 2, 0).contiguous().view(frame,25,3 )  # 改變張量形狀
# data = data.permute(0, 2, 3, 1, 4).contiguous().view(video_number, frame, 25, 3, data.shape[4])  # 改變張量形狀
# np.save('3d_pose_ori', data.cpu().detach().numpy())
# f = open('data', 'w')
# f.write(str(data.cpu().detach().numpy()))
# f.close()
# # mask_data = mask_data[video_number,:,:,:,0]
# mask_data = mask_data[:, :, :, :, :]
# # mask_data = mask_data.permute(1, 2, 0).contiguous().view(frame,25,3 )  # 改變張量形狀
# mask_data = mask_data.permute(0, 2, 3, 1, 4).contiguous().view(video_number, frame, 25, 3, data.shape[4])  # 改變張量形狀
# np.save('3d_pose0', mask_data.cpu().detach().numpy())

# # reconstructed= reconstructed[video_number,:,:,:,0] #change  #one dim is different video
# # reconstructed = reconstructed[0,:,:,:,:]
# reconstructed = reconstructed[:, :, :, :, :]  # change  #one dim is different video
# # reconstructed = reconstructed.permute(1, 2, 0).contiguous().view(frame,25,3 )  # 改變張量形狀
# reconstructed = reconstructed.permute(0, 2, 3, 1, 4).contiguous().view(video_number, frame, 25, 3,
#                                                                        data.shape[4])  # 改變張量形狀
# np.save('3d_pose', reconstructed.cpu().detach().numpy())
# # print('reconstructed.shape:',reconstructed.shape) #[7, 3, 100, 25, 2]
# f = open('reconstructed', 'w')
# f.write(str(reconstructed.cpu().detach().numpy()))
# f.close()'''
# inform.to_csv('Ting_occ_test_0621.csv')
