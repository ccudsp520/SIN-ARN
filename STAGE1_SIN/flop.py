# pip install fvcore
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from unsupervised_globalgcn_3layer_action_KD_atu import Model_Imputation
from model.msag3d import Model as MSA_G3D
from model.msg3d import  Model as MSG3D

# ========= 基本設定 =========
# 你的模型 forward 內硬編了 T=300, V=25, M=2 的遮擋用 x_mask
N, C, T, V, M = 1, 3, 300, 25, 2
epoch = 60

# 必須使用 GPU，因為你的 forward 裡面有 .cuda()
# assert torch.cuda.is_available(), "此模型 forward 內有 .cuda()，請在有 GPU 的環境執行，或改成 .to(x.device)"

# device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MSA_G3D(60,25,2,6,13,graph='graph.ntu_rgb_d.AdjMatrixGraph').eval().to(device)
# model = MSG3D(60,25,2,6,13,graph='graph.ntu_rgb_d.AdjMatrixGraph').eval().to(device)

# 補上缺的屬性，讓 forward 不會噴 AttributeError
# joint_list：通常是 0..V-1 的關節索引
# joint_occlusion_num：你程式一開始會用到，就給 0（隨後迴圈會用 miss_rate 重新抽樣）
if not hasattr(model, "joint_list"):
    model.joint_list = list(range(V))
if not hasattr(model, "joint_occlusion_num"):
    model.joint_occlusion_num = 0

# 假輸入：骨架張量 (N, C, T, V, M)
x = torch.randn(N, C, T, V, M, device=device)

with torch.no_grad():
    # fvcore：回傳的是 MACs（乘加合併當 1 次）
    flops = FlopCountAnalysis(model, x)
    macs = flops.total()

print("Total MACs:", macs)
print("Approx FLOPs (mul+add):", macs * 2)  # 若你要 FLOPs（乘、加分開），通常 ≈ 2 * MACs

# 可選：列出參數統計與各模組拆解
# print(parameter_count_table(model))
# print(flops.by_module())          # 各子模組的 MACs 細項
# print(flops.by_operator())        # 依算子聚合
