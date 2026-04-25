import torch

# 检查 CUDA 是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查 CUDA 版本
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
