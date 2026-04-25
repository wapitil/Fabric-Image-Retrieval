# DINOv2 Fabric-Image-Retrieval

一个基于 DINOv2 的本地图像反查小工具，支持：

- 使用单张参考图在图库中检索相似图片
- 使用 GUI 对结果做 `Accepted` / `Rejected` 人工反馈
- 将 `Accepted` 图片复制到指定目录
- 在确认后删除已经复制对应的图库源图
- 输出检索结果、日志和历史记录到结果目录

运行效果如下：
![alt text](./assets/image.png)
## 目录结构

- `retrieval_gui.py`
  PySide6 图形界面主程序
- `dinov2_patch_retrieval_fast.py`
  检索核心逻辑，包含特征提取、缓存和重排序
- `cuda_check.py`
  简单 CUDA 环境检查脚本
- `dinov2_retrieval_3class_result/`
  运行输出目录，包含日志、历史记录、CSV 结果等
- `dinov2_vitb14_pretrain.pth`
  DINOv2 模型权重文件

## 运行环境

建议环境：

- Python 3.10+
- PyTorch
- torchvision
- pandas
- Pillow
- PySide6
- tqdm

如果启用 GPU，还需要可用的 CUDA 环境。

## 运行方式

先根据你的实际路径修改 [dinov2_patch_retrieval_fast.py](/g:/images/ZSY/dinov2_patch_retrieval_fast.py) 里的默认配置：

- `reference_image_path`
- `gallery_dir`
- `output_dir`
- 模型权重路径及其他检索参数

然后启动 GUI：

```bash
python retrieval_gui.py
```

如果只是检查 CUDA：

```bash
python cuda_check.py
```

## 使用流程

1. 选择参考图、图库目录和保存目录。
2. 点击“开始检索 / 应用反馈”。
3. 在结果区中将正确结果标记为 `Accepted`，不需要的结果可批量标为 `Rejected`。
4. 点击“复制 Accepted 图片”，将结果复制到保存目录。
5. 复制后的源图会在后续检索中被排除，方便继续查漏。
6. 确认这一批已处理完成后，点击“删除已复制源图”，删除图库中的对应源文件。


## 版本控制建议

已配套提供 `.gitignore`，会忽略：

- Python 缓存和虚拟环境
- IDE 配置和日志
- `dinov2_retrieval_3class_result/`
- `*.pth`

