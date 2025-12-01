import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import tifffile
import cv2
import zarr
from ipdb import set_trace as st

def get_mgrid(sidelen, dim=2):
    '''
    生成网格坐标
    sidelen: int (生成正方形) 或 tuple (height, width)
    '''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

class ImageFitting(Dataset):
    def __init__(self, image_path, sidelength=None):
        super().__init__()
        img = Image.open(image_path)
        
        if sidelength is not None:
            img = img.resize((sidelength, sidelength), Image.LANCZOS)

        self.sidelength = (img.height, img.width)
        self.n_channels = 3 # 假设是RGB，如果包含Alpha通道这里可能需要改
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.pixels = transform(img) # (C, H, W)
        self.pixels = self.pixels.permute(1, 2, 0).view(-1, self.pixels.shape[0]) # (H*W, C)
        self.coords = get_mgrid(self.sidelength, dim=2) # (H*W, 2)

    def __len__(self):
        return self.pixels.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.pixels[idx]

class WSIFitting(Dataset):
    def __init__(self, image_path, sidelength=None, level=0):
        """
        Args:
            image_path (str): .tif 文件的路径
            sidelength (int, optional): 如果指定，会将图像强制缩放到 (sidelength, sidelength)。
                                      如果不指定，则保持原始尺寸（慎用，WSI可能非常大）。
            level (int): TIF 如果是多页/多分辨率金字塔，指定读取第几层。默认为0（最高分辨率）。
        """
        super().__init__()
        
        # 1. 使用 tifffile 读取 (支持 16-bit, 多页 TIF)
        try:
            with tifffile.TiffFile(image_path) as tif:
                if len(tif.pages) > 1 and level < len(tif.pages):
                    img = tif.pages[level].asarray()
                else:
                    img = tifffile.imread(image_path)
        except Exception as e:
            print(f"读取指定 level 失败，尝试直接读取: {e}")
            img = tifffile.imread(image_path)

        # 2. 格式标准化处理 (H, W, C)
        if img.ndim == 2: # 灰度图 (H, W)
            img = img[:, :, None] # -> (H, W, 1)
        elif img.ndim == 3:
            if img.shape[0] < 5: # 假设通道数小于5，认为是 (C, H, W) -> 转为 (H, W, C)
                img = np.transpose(img, (1, 2, 0))
        
        # 3. 数值归一化 (处理 16-bit / 8-bit)
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # 4. 调整大小 (Resize)
        original_h, original_w = img.shape[:2]
        if sidelength is not None:
            img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)
            if img.ndim == 2: img = img[:, :, None] 
        
        self.sidelength = (img.shape[0], img.shape[1])
        self.n_channels = img.shape[2]
        
        print(f"[WSIFitting] 加载完成: 原始 {original_h}x{original_w} -> 当前 {self.sidelength}")
        
        # 5. 数据转换: [0, 1] -> [-1, 1]
        self.pixels = torch.from_numpy(img)
        self.pixels = (self.pixels - 0.5) * 2.0 
        
        self.pixels = self.pixels.view(-1, self.n_channels)
        self.coords = get_mgrid(self.sidelength, dim=2)

    def __len__(self):
        return self.pixels.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.pixels[idx]

class ChunkedWSIFitting(Dataset):
    def __init__(self, image_path, patch_size=256, steps_per_epoch=1000, target_level=None, downsample=1):
        """
        Args:
            image_path (str): TIF 文件路径
            patch_size (int): 训练时的 patch 大小
            steps_per_epoch (int): 每个 epoch 迭代次数
            target_level (int, optional): 如果 TIF 有金字塔，指定读取第几层 (0是原图, 1是缩小版...)
            downsample (int, optional): 如果没有金字塔，手动指定下采样倍数 (例如 16 表示长宽各缩小 16 倍)
        """
        super().__init__()
        self.image_path = image_path
        self.patch_size = patch_size
        self.steps_per_epoch = steps_per_epoch
        
        # 1. 初始化 Zarr 接口
        try:
            self.store = tifffile.imread(self.image_path, aszarr=True)
            z_root = zarr.open(self.store, mode='r')
        except Exception as e:
            raise RuntimeError(f"无法打开 TIF: {e}")

        # 2. 选择合适的层级 (Level)
        self.zarr_img = None
        
        # 如果是 Group，说明有多层
        if isinstance(z_root, zarr.Group):
            # 打印所有可用层级
            keys = list(z_root.keys())
            print(f"[WSI] 检测到金字塔层级: {keys}")
            
            # 如果用户指定了层级
            if target_level is not None:
                key = str(target_level)
                if key in z_root:
                    self.zarr_img = z_root[key]
                    print(f"[WSI] 选中 Level {key}")
                else:
                    print(f"[WSI] 警告: Level {key} 不存在，回退到 Level 0")
                    self.zarr_img = z_root['0'] if '0' in z_root else z_root[keys[0]]
            else:
                # 默认选 0
                self.zarr_img = z_root['0'] if '0' in z_root else z_root[keys[0]]
        else:
            self.zarr_img = z_root
            print("[WSI] 图像只有单层")

        # 3. 处理形状与通道
        self.shape = self.zarr_img.shape
        self.dtype = self.zarr_img.dtype
        self.ndim = self.zarr_img.ndim
        
        if self.ndim == 2:
            self.H, self.W = self.shape; self.C = 1; self.channel_dim = None
        else:
            if self.shape[0] < 10: self.C, self.H, self.W = self.shape; self.channel_dim = 0
            else: self.H, self.W, self.C = self.shape; self.channel_dim = -1

        # 4. 处理手动下采样 (Strided Sampling)
        # 如果用户没选 level 但选了 downsample，或者只有单层大图想跑低分辨率
        self.downsample = downsample
        if self.downsample > 1:
            # 逻辑分辨率变小了
            self.H = self.H // self.downsample
            self.W = self.W // self.downsample
            print(f"[WSI] 启用下采样: 原始数据每隔 {downsample} 个点读一次")
            print(f"      虚拟分辨率: {self.H} x {self.W}")
        
        self.sidelength = (self.H, self.W)
        self.n_channels = self.C

    def __len__(self):
        return self.steps_per_epoch

    def get_coords(self, start_h, start_w, height, width):
        # 坐标生成逻辑不变，依然生成相对于"当前虚拟分辨率"的 [-1, 1] 坐标
        i_coords, j_coords = np.meshgrid(np.arange(start_h, start_h + height), np.arange(start_w, start_w + width), indexing='ij')
        coord_h = (i_coords.astype(np.float32) / (self.H - 1)) * 2 - 1
        coord_w = (j_coords.astype(np.float32) / (self.W - 1)) * 2 - 1
        return torch.from_numpy(np.stack([coord_h, coord_w], axis=-1)).view(-1, 2)

    def __getitem__(self, idx):
        # 1. 在"缩小后"的尺寸上随机选位置
        max_h = self.H - self.patch_size
        max_w = self.W - self.patch_size
        
        if max_h < 0 or max_w < 0:
             # 如果缩小后比 patch 还小，就随机裁一部分，或者报错
             # 这里简单处理：重置 patch size 适应图片
             actual_h = min(self.H, self.patch_size)
             actual_w = min(self.W, self.patch_size)
             start_h, start_w = 0, 0
        else:
             actual_h, actual_w = self.patch_size, self.patch_size
             start_h = np.random.randint(0, max_h + 1)
             start_w = np.random.randint(0, max_w + 1)

        # 2. 映射回"原始大图"的读取坐标
        # 如果 downsample=16, 我们要读取原始数据的 [start*16 : (start+len)*16 : 16]
        src_start_h = start_h * self.downsample
        src_start_w = start_w * self.downsample
        src_end_h = (start_h + actual_h) * self.downsample
        src_end_w = (start_w + actual_w) * self.downsample
        
        # 3. 读取数据 (使用 step 切片)
        # 关键：zarr 支持 slice(start, stop, step)
        ds = self.downsample
        
        if self.channel_dim == 0: 
            patch = self.zarr_img[:, src_start_h:src_end_h:ds, src_start_w:src_end_w:ds]
            patch = np.transpose(patch, (1, 2, 0))
        elif self.channel_dim == -1: 
            patch = self.zarr_img[src_start_h:src_end_h:ds, src_start_w:src_end_w:ds, :]
        else:
            patch = self.zarr_img[src_start_h:src_end_h:ds, src_start_w:src_end_w:ds]
            patch = patch[:, :, None]

        # 4. 归一化
        if self.dtype == np.uint16: patch = patch.astype(np.float32) / 65535.0
        elif self.dtype == np.uint8: patch = patch.astype(np.float32) / 255.0
        else: patch = patch.astype(np.float32)

        patch_tensor = torch.from_numpy(patch)
        pixels = (patch_tensor - 0.5) * 2.0
        pixels = pixels.reshape(-1, self.n_channels)
        
        coords = self.get_coords(start_h, start_w, actual_h, actual_w)
        
        return coords, pixels

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # 测试代码
    try:
        # 注意：请确保路径存在，否则可以改为测试 ImageFitting
        # dataset = ImageFitting('./data/graenys.png', sidelength=512)
        print("正在测试数据集类...")
        # 假设当前目录下有测试图，否则请自行调整路径测试
    except Exception as e:
        print(e)