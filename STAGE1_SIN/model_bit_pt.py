import torch

# # 載入模型權重檔 (state_dict 格式)
# state_dict = torch.load("./weight_hong/msag3d_epoch73.pth", map_location="cpu")

# # 有些權重檔是 {'state_dict': {...}}，所以要判斷一下
# if "state_dict" in state_dict:
#     state_dict = state_dict["state_dict"]

# # 計算總參數量
# total_params = sum(p.numel() for p in state_dict.values())
# print(f"Total parameters: {total_params:,}")

# # 計算大小（以 float32 為例）
# size_mb = total_params * 4 / (1024 ** 2)
# print(f"Estimated size (float32): {size_mb:.2f} MB")

from model.msag3d import Model as MSA_G3D
# from unsupervised_globalgcn_3layer_action_KD_atu import Model_teacher
# from unsupervised_globalgcn_3layer_action_KD_atu import Model_student
from unsupervised_globalgcn_3layer_action_KD_atu import Model_Imputation
# from model.AGCNEncoder_3layer_action_atu import ATNet
# from GCN_Unsupervised.GCNEncoder.AGCNEncoder_3layer_action_atu import ATNet

def count_params_and_size(model, dtype=torch.float32):
    model_params = 0
    size_mb = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params/1000000, 'Million')
    
    return model_params, size_mb

if __name__ == "__main__":

    # model = MSG3D(
    #     num_class=60,         # 動作類別數，例如 NTU RGB+D 60
    #     num_point=25,         # 骨架點數 (NTU RGB+D 用 25)
    #     num_person=2,         # 每個序列的人數
    #     num_gcn_scales=13,    # GCN 尺度數 (依 repo 預設)
    #     num_g3d_scales=6,     # G3D 尺度數 (依 repo 預設)
    #     in_channels=15,
    #     graph='graph.ntu_rgb_d.AdjMatrixGraph'     # 圖結構 (依照資料集設定)
    # )

    # model = MSA_G3D(
    #     num_class=60,         # 動作類別數，例如 NTU RGB+D 60
    #     num_point=25,         # 骨架點數 (NTU RGB+D 用 25)
    #     num_person=2,         # 每個序列的人數
    #     num_gcn_scales=13,    # GCN 尺度數 (依 repo 預設)
    #     num_g3d_scales=6,     # G3D 尺度數 (依 repo 預設)
    #     in_channels=15,
    #     graph='graph.ntu_rgb_d.AdjMatrixGraph'     # 圖結構 (依照資料集設定)
    # )

    # model = MSA_G3D()
    # model = Model_teacher()
    # model = Model_student()
    model = Model_Imputation()
    # model = ATNet(2,300)

    total_params, size_mb = count_params_and_size(model, dtype=torch.float32)

    print(f"Total parameters: {total_params:,}")
    print(f"Estimated size (float32): {size_mb:.2f} MB")

    _, size_fp16 = count_params_and_size(model, dtype=torch.float16)
    _, size_int8 = count_params_and_size(model, dtype=torch.int8)
    print(f"Estimated size (float16): {size_fp16:.2f} MB")
    print(f"Estimated size (int8): {size_int8:.2f} MB")
