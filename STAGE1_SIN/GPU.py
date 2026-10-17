import torch
print(torch.__version__)
print(torch.cuda.is_available())   # True 才代表有抓到 GPU
# print(torch.cuda.device_count())   # 有幾張 GPU
# print(torch.cuda.get_device_name(0))  # 顯示 GPU 名稱
