import tifffile
import zarr
import numpy as np
import cv2
import os

def generate_ground_truth(image_path, output_path, max_side=2048):
    """
    高效读取 WSI 大图并生成一张低分辨率的 Ground Truth 图片
    
    Args:
        image_path: TIF 文件路径
        output_path: 结果保存路径 (.png/.jpg)
        max_side: 输出图片的最大边长 (默认 2048)
    """
    print(f"正在处理文件: {image_path}")
    
    # --- 策略 1: 尝试直接读取 WSI 的金字塔低分辨率层 ---
    try:
        with tifffile.TiffFile(image_path) as tif:
            print(f"  - 检测到 {len(tif.pages)} 个层级 (Levels)")
            
            best_page = None
            best_diff = float('inf')
            
            for i, page in enumerate(tif.pages):
                # 获取该层分辨率
                h, w = page.shape[0], page.shape[1] 
                # 注意：有些 metadata 可能会让 shape 变动，这里做个简单防御
                if page.ndim == 3 and page.shape[0] < 10: # (C, H, W)
                     h, w = page.shape[1], page.shape[2]
                
                print(f"    Level {i}: {h}x{w}")
                
                # 寻找最接近目标尺寸 max_side，但又不小于 max_side 的层
                # 这样可以保证画质清晰，最后再 resize 一下即可
                long_side = max(h, w)
                
                # 我们倾向于找一个比 max_side 稍微大一点的层，这样 resize 效果好
                if long_side >= max_side:
                    if long_side < best_diff:
                        best_diff = long_side
                        best_page = page
            
            # 如果找到了合适的中间层
            if best_page is not None and len(tif.pages) > 1:
                print(f"  -> 选中最佳层级: {best_page.shape}，直接读取...")
                img = best_page.asarray()
            else:
                # --- 策略 2: 如果没有金字塔，或者只有一层，使用稀疏采样 ---
                print("  -> 未找到合适的金字塔层，将对 Level 0 进行稀疏采样读取...")
                img = read_level0_strided(image_path, max_side)
                
    except Exception as e:
        print(f"  [Warning] TiffFile 读取失败，尝试 Zarr 稀疏采样回退: {e}")
        img = read_level0_strided(image_path, max_side)

    if img is None:
        print("错误：无法读取图像数据")
        return

    # --- 后处理: 格式标准化与 Resize ---
    print("正在处理图像格式...")
    
    # 1. 维度调整 (C, H, W) -> (H, W, C)
    if img.ndim == 3:
        if img.shape[0] < 10: # Channel first
            img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2: # 灰度
        # 保持灰度或转 RGB 都可以，OpenCV 处理灰度没问题
        pass

    # 2. 数值转换 (uint16 -> uint8)
    if img.dtype == np.uint16:
        print("  - 检测到 16-bit 图像，归一化到 8-bit")
        img = (img / 65535.0 * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        print(f"  - 检测到 {img.dtype}，min-max 归一化到 8-bit")
        img_float = img.astype(np.float32)
        img = ((img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8) * 255).astype(np.uint8)

    # 3. 最终 Resize 到指定的 max_side
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"  - 缩放: {w}x{h} -> {new_w}x{new_h}")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 4. 保存 (OpenCV 使用 BGR 顺序，如果读取的是 RGB 需要转换)
    # Tifffile 通常读取 RGB，cv2.imwrite 需要 BGR
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, img)
    print(f"✅ Ground Truth 已保存至: {output_path}")

def read_level0_strided(image_path, target_size):
    """
    使用 Zarr 接口对 Level 0 进行间隔采样（Strided Reading）
    原理：image[::step, ::step]
    """
    store = tifffile.imread(image_path, aszarr=True)
    z_root = zarr.open(store, mode='r')
    
    # 解包 Group
    if isinstance(z_root, zarr.Group):
        if '0' in z_root: z_arr = z_root['0']
        else: z_arr = z_root[list(z_root.keys())[0]]
    else:
        z_arr = z_root
        
    # 判断形状
    shape = z_arr.shape
    if shape[0] < 10: # (C, H, W)
        c, h, w = shape
        channel_first = True
    elif len(shape) == 3: # (H, W, C)
        h, w, c = shape
        channel_first = False
    else: # (H, W)
        h, w = shape
        channel_first = False
        
    # 计算步长 step
    # 比如原图 10000，目标 2000 -> step = 5
    step = int(np.ceil(max(h, w) / target_size))
    print(f"  - 原始尺寸 {h}x{w}，采样步长: {step}")
    
    # 稀疏读取
    if channel_first:
        img = z_arr[:, ::step, ::step]
        img = np.transpose(img, (1, 2, 0))
    elif len(shape) == 3:
        img = z_arr[::step, ::step, :]
    else:
        img = z_arr[::step, ::step]
        
    return img

if __name__ == "__main__":
    # 在这里修改你的路径
    tif_path = "./data/test_001.tif" 
    save_path = "ground_truth_preview.png"
    
    if os.path.exists(tif_path):
        generate_ground_truth(tif_path, save_path, max_side=2048)
    else:
        print(f"文件不存在: {tif_path}")