import os
import random
from pathlib import Path


def link_images(src_dir, dst_dir, num_samples=500):
    """
    随机选择 num_samples 个文件创建软连接
    """

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有文件
    files = [f for f in src_dir.iterdir() if f.is_file()]

    # 随机采样
    # files = random.sample(files, min(num_samples, len(files)))

    for img_path in files:
        link_path = dst_dir / img_path.name

        # 已存在则跳过
        if link_path.exists():
            continue

        os.symlink(img_path.resolve(), link_path)

        print(f"link: {link_path}")


if __name__ == "__main__":
    # src = Path(r"G:\images\fabric\images")
    src = Path("/media/wapiti/MyFile/Datasets/Dataset_Split/test/DHW_Good")
    dst = Path("Data/images")

    # dst = Path(r"Data\labels")

    link_images(src, dst, num_samples=500)
