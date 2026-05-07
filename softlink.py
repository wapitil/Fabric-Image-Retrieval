from pathlib import Path
import os


def link_images(src_dir, dst_dir):
    """
    将 src_dir 下所有图片软连接到 dst_dir
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue

        link_path = dst_dir / img_path.name
        # 已存在则跳过
        if link_path.exists():
            continue

        os.symlink(img_path.resolve(), link_path)

        print(f"link: {link_path}")

if __name__ == "__main__":

    # src = Path(r"G:\images\fabric\images")
    # dst = Path(f"Data\images")

    src = Path(r"G:\images\fabric\labels")
    dst = Path(f"Data\labels")

    link_images(src, dst)
