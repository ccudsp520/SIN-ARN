import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
# sys.path.append('/home/antony/lin/GLTA/')
sys.path.append("/path/to/DLNest/Common")
from Dataset.Datasets import NTUDataSet
from tqdm import tqdm
from torchvision import transforms
from src.dataprocessor import *
from src.graph import Graph
# from GCN_Unsupervised.unsupervised_partGCN import Model # global+local gcn (ok)
# from GCN_Unsupervised.unsupervised_globalgcn import Model #change globalgcn (have problem)
# from GCN_Unsupervised.unsupervised_globalgcn_3layer import Model #change 3layer
# from GCN_Unsupervised.unsupervised_globalgcn_10layer_action import Model  #change 10layer+action
# from Model.msag3d import Model  #change 3layer+action
from unsupervised_globalgcn_3layer_action_KD_atu import Model_teacher  #change 3layer+action
# from Model.msag3d import Model_student
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import RandomSampler
import torch.nn.functional as F
import random
import pandas as pd
######################random seed set###############
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#####################################################
np.set_printoptions(threshold=sys.maxsize,precision=3,suppress=True,linewidth=800)
# 定義訓練參數
num_epochs = 60
# vali_epoch = 1
train_batch_size = 16
test_batch_size = 1
learning_rate = 0.1
frame=300
downsample_rate=20
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = {   
    "model_file_path":"//home/dsp520/lin/GLTA/GCN_Unsupervised/Model.py",
    "save_root":"/home/dsp520/lin/GLTA/partGCNsaves",
    "other_file_paths":["/home/dsp520/lin/GLTA/GCN_Unsupervised/unsupervised_partGCN.py"],
    "dataset_config" : {
        "directory":"/home/dsp520/lin/2s-AGCN/data/ntu/",
        "partition":"xsub",
        "num_workers":8,
        "batchsize":16,
        "use_mmap": True,
        "debug":False
    },
    "model_config":{
        "model_path":"GCN_Unsupervised.unsupervised_partGCN",
        "device":[0],
        "step":[30,50,60],
        "model_args":{
            "prediction_mask":0,
            "predict_seg":1,
            "if_rotation":False,
            "seg_num":1,
            "if_vibrate":False,
            "T":150,
            "num_class": 60,
            "num_point": 25,
            "num_person":2,
            "graph":None,
            "graph_args":None,
            "in_channels":3,
            "GCNEncoder":"shiftGCN",
            "ATU_layer":2,
            "self_supervised_mask":1
        },
        "base_lr": 0.1,
        "knn_k":1,
        "R-rotation":False,
        "R-rotation-loss":0,
        "R-drop":False,
        "R-drop-loss":0,
        "test_epoch_interval":1,
        "autoencoder":False,
        "loss":"CustomizeL2loss",
        "innerautoencoder_loss":False,
        "joint_loss":1,
        "bone_length_loss":0,
        "motion_loss":0,
        "joint_direction_loss":0,
        "colorization":0
    },
    "ckpt_load" :"",
    "epoch":70,
    "is_test": False,
    "save_dir":"/home/dsp520/lin/GLTA/npz_save/151_wholeBranch"
}

args= {
    "dataset_config" : {
        "directory":"../hong_MSG3D/data/",
        "partition":"xsub",
        "use_mmap": True,
        "batchsize":16,
        "num_workers":4,
        "debug":False,
        "occlusion_part":[4], # choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], help='1:left arm, 2:right arm, 3:two hands, 4:two legs, 5:trunk'
        "downsample_rate":1
    }
}
args = args["dataset_config"]
train_dataset_config = {
    # "data_path":self.args["directory"]+self.args["partition"]+"/train_data_joint.npy",#change
    "data_path":args["directory"]+args["partition"]+"/train_data_joint.npy",
    # "data_path":args["directory"]+args["partition"]+"/train_multi_joint_notnormalize2.npy",
    "label_path":args["directory"]+args["partition"]+"/train_label.pkl",
    "use_mmap":args["use_mmap"],
    "debug":args["debug"],
    "downsample_rate":args["downsample_rate"]
}
val_dataset_config = {
    # "data_path":self.args["directory"]+self.args["partition"]+"/val_data_joint.npy",#change
    "data_path":args["directory"]+args["partition"]+"/val_data_joint.npy",
    # "data_path":args["directory"]+args["partition"]+"/val_multi_joint_notnormalize2.npy",
    "label_path":args["directory"]+args["partition"]+"/val_label.pkl",
    "use_mmap":args["use_mmap"],
    "debug":args["debug"],   
}
 

inform = {
    "epoch":[],"total_acc":[],"total_error":[],"total_mpjpe_all":[],"total_mpjpe":[],"data.shape":[],"video_number":[],"label":[],"predict_label":[]
}
 
inform = pd.DataFrame(inform)

def mpjpe(predicted, target,reverse_center=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    if predicted.shape != target.shape:
        print('predicted.shape', predicted.shape) #[16, 3, 300, 25, 2]
        print('target.shape', target.shape)
    assert predicted.shape == target.shape
    error = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1)) #dim=4
    # print('error',error)
    return error

transform = transforms.Compose([
            Occlusion_part(args["occlusion_part"]),
        ])
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
# # 載入資料

# # train_dataset = NTUDataSet(train_dataset_config,transform=transform) #change
# train_dataset = NTUDataSet(train_dataset_config)
# randomSampler_train = RandomSampler(train_dataset, replacement=True, num_samples=int(len(train_dataset)/downsample_rate))
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True) #, shuffle=False,sampler=randomSampler_train
# # train_dataloader = Dataset(config)

gcn_kernel_size = [5,2] #change
graph = Graph(max_hop=gcn_kernel_size[1]) #change
A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False).to(device) #change

# 定義模型
# model = Model(num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,self_supervised_mask=1,if_rotation=False,seg_num=1,if_vibrate=False,prediction_mask=0,GCNEncoder="AGCN",ATU_layer=2,T=300,predict_seg=1)
# model_teacher = Model(num_class=60, num_point=25, num_person=2, graph=A, graph_args=dict())#graph=ntu_rgb_d.Graph,graph_args=dict()
model_student = Model_teacher(num_class=60, num_point=25, num_person=2, graph=A, graph_args=dict(), in_channels=3,
                      if_rotation=False, seg_num=1, ATU_layer=2, if_vibrate=False, prediction_mask=0, GCNEncoder="AGCN",
                      T=300, predict_seg=1) #change
# 定義損失函數和優化器
criterion_mse = nn.MSELoss()
criterion_crossentropy = nn.CrossEntropyLoss()
criterion_kld = nn.KLDivLoss(reduction='batchmean')

# optimizer = torch.optim.SGD(model_student.AAGCN.parameters(), lr=learning_rate,momentum=0.9, weight_decay=0.0001, nesterov=True)
# optimizer = torch.optim.SGD(model.AAGCN.parameters(), lr=learning_rate,momentum=0.9, weight_decay=0.0001, nesterov=True) #only指定AAGCN的参数

optimizer = torch.optim.SGD(model_student.MSG3D.parameters(), lr=learning_rate,momentum=0.9, weight_decay=0.0001, nesterov=True)

# step=[30, 40]
step=[30, 50]
warm_up_epoch=5
lr_scheduler_pre = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=step, gamma=0.1)
lr_scheduler = GradualWarmupScheduler(optimizer, total_epoch=warm_up_epoch,
                                            after_scheduler=lr_scheduler_pre)
# 初始化最低loss為正無限大
lowest_loss = float('inf')


# 測試模型
test_dataset = NTUDataSet(val_dataset_config)
# test_dataset = NTUDataSet(val_dataset_config,transform=transform) #change
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,shuffle=False)#, shuffle=True

# model_student.load_state_dict(torch.load('./model_weight_atu/model_best_epoch46_parts12345678_0.2to0.6_occinnet_global_3layer_downsample1_action_noAE_atul2head8ingcnl1_1encoder_post1.pth'))
#model_student.load_state_dict(torch.load('model_TEST_post.pth'))
for epoch in range(60,80):
    print(epoch+1,"\n")
    # model_student.load_state_dict(torch.load("./model_weight_mask=1.2/model_best_parts12345678_0.2to0.6_occinnet_global_3layer_downsample10_action_noAE_atul2ingcnl1head8_1encoder_pre1_mask=1.2_epoch{}.pth".format(epoch+1)))
    model_student.load_state_dict(torch.load("./weight_hong/msag3d_epoch{}.pth".format(epoch+1)))
    # model_student.load_state_dict(torch.load('./model_weight_rich/model_best_parts12345678_occinnet_global_3layer_downsample1_action_noAE_atul2head8ingcnl1_1encoder_rich_post1_epoch48.pth'))

    model_student.to(device)  # 加入這行程式
    model_student.eval()
    
    with torch.no_grad():
        total_error =0.0
        total_error_all=0.0
        total_mpjpe =0.0
        total_mpjpe_all =0.0
        total_acc =0.0
        tqdm_time = 0
        for data in tqdm(test_dataloader):
            tqdm_time += 1
            # data = data.to(device)
            label = data['label'].to(device) 
            data = data['data'].to(device) #[16, 3, 300, 25, 2]#change
            data = data[:, :, :frame, :, :] #change 
            # data_orig = data.clone()

            mask_data = data.clone() #[16, 3, 300, 25, 2] #.clone()複製data到不同儲存區,與原本data互不相關
            # joint_occlusion = np.random.choice(joint_list, joint_occlusion_num, replace=False)

            reconstructed = model_student(mask_data, epoch) #change
            predict_action_class = reconstructed[2]
            # reconstructed = reconstructed[0]
            # reconstructed = mask_data
            x_mask=reconstructed[3]

            mask_data = mask_data *x_mask
            reconstructed_all = reconstructed[0]
            reconstructed = reconstructed[0]*(1-x_mask)+data*(x_mask) #change [0]:decoder output ,[1]:hidden_feature,[2]:GCN_feature,[4]:autoencoder_output_feature

            #     data_orig[i_b] = skeletons_clone
            #     reconstructed_orig[i_b] = reconstructed_skeletons
            #     # f = open('mask_data', 'w')
            #     # f.write(str(data_orig[i_b].cpu().numpy()))
            #     # f.write('\n')  
            #     # f.close()
            #     # time.sleep(10)
            # error_orig = mpjpe(data, data,reverse_center=False)
            # error_recon = mpjpe(reconstructed_orig, data_orig,mask)
            error_recon_all = mpjpe(reconstructed_all,data)
            error_recon = mpjpe(reconstructed, data)

            total_error_all += error_recon_all.item()
            total_error += error_recon.item()
            total_mpjpe_all = total_error_all/tqdm_time   
            total_mpjpe = total_error/tqdm_time  
            predict_label = torch.max(predict_action_class, 1)[1]
            acc = torch.mean((predict_label == label).float())
            f = open('xxxxxxxxxx_test', 'a')
            f.write(str(acc))
            f.write(str(predict_label))
            f.write(str(label))
            f.write('\n')  
            f.close()
            total_acc += acc.item()

            # do something with reconstructed and hidden_feature
    total_acc = total_acc/len(test_dataloader.dataset)*test_batch_size
    print('total_acc',total_acc)
    print('total_error',total_error)
    print('total_mpjpe_all',total_mpjpe_all) 
    print('total_mpjpe',total_mpjpe) 

    print('data.shape: ',data.shape)
    video_number = data.shape[0]
    print('video_number: ',video_number)
    print('label',label,label.shape)
    print('predict_label',predict_label,predict_label.shape)
#     inform = inform.append({
#     "epoch":(epoch+1),"total_acc":total_acc,"total_error":total_error,"total_mpjpe_all":total_mpjpe_all,"total_mpjpe":total_mpjpe,"data.shape":data.shape,"video_number":video_number,"label":label,"predict_label":predict_label}, ignore_index=1)
#     label=label
#     predict_label=predict_label


#     data = data[:,:,:,:,:] 


#     data = data.permute(0,2, 3, 1,4).contiguous().view(video_number,frame,25,3,data.shape[4] )  # 改變張量形狀
#     np.save('3d_pose_ori',data.cpu().detach().numpy())
#     f = open('data', 'w')
#     f.write(str(data.cpu().detach().numpy()))
#     f.close()

#     mask_data = mask_data[:,:,:,:,:]

#     mask_data = mask_data.permute(0,2, 3, 1,4).contiguous().view(video_number,frame,25,3,data.shape[4] )  # 改變張量形狀



#     reconstructed= reconstructed[:,:,:,:,:] #change  #one dim is different video

#     reconstructed = reconstructed.permute(0,2, 3, 1,4).contiguous().view(video_number,frame,25,3,data.shape[4] )  # 改變張量形狀

# inform.to_csv('Ting_msag3d_84_epoch_list0621.csv')