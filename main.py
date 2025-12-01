import os
import sys
# 确保路径包含项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import configargparse
import numpy as np
import torchvision
from torch.utils.data import DataLoader

# 假设 data 和 network 是你本地的模块
import data
import network

def get_config():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--logging_root', type=str, default='./logs', help='日志和结果保存的根目录')
    parser.add_argument('--experiment_name', type=str, required=True, help='实验名称，用于创建子文件夹')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10000)

    parser.add_argument('--epochs_til_ckpt', type=int, default=10, help='每多少个epoch保存一次')
    
    parser.add_argument('--model_type', type=str, default='sine')
    
    # --- 数据相关参数 ---
    parser.add_argument('--image_path', type=str, default='./data/graenys.png')
    # 修改: 默认值为 -1，表示"使用原始分辨率"，如果设置正整数则强制 resize
    parser.add_argument('--sidelength', type=int, default=-1, help='如果为-1则使用原始分辨率，否则强制resize到该数值')
    
    # 注意: out_features 不再作为必须参数，因为我们会从数据集中自动检测
    
    args = parser.parse_args()
    return args

import math
import os
import torchvision

def get_mgrid(sidelen, dim=2):
    '''
    生成标准网格坐标，范围 [-1, 1]。
    '''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        # 生成 (H, W, 2) 的网格
        tensors = tuple([torch.linspace(-1, 1, steps=l) for l in sidelen])
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
        mgrid = mgrid.reshape(-1, dim) # (H*W, 2)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
        
    return mgrid

def batched_inference(model, coords, batch_size=40960):
    '''
    分批次推理，防止显存溢出
    '''
    res = []
    # 将巨大的坐标列表切分成小块
    for i in range(0, coords.shape[0], batch_size):
        # 获取当前批次的坐标
        batch_coords = coords[i : i + batch_size]
        
        # 推理
        with torch.no_grad():
            output = model(batch_coords)
        
        # 将结果移回 CPU 以节省显存，并收集
        res.append(output.cpu())
        
    # 拼接所有结果
    return torch.cat(res, dim=0)

def render_and_save(model, resolution, out_channels, output_dir, epoch, step, device, max_render_side=2048):
    '''
    渲染图片并保存。
    优化点：
    1. 自动计算缩略图尺寸，避免生成几万像素的大图撑爆显存。
    2. 使用分块推理 (Batching)，进一步降低峰值显存。
    
    args:
        resolution: 原始 WSI 的 (H, W)
        max_render_side: 渲染出的图片最大边长 (默认 2048)。
                         这足以看清整体结构，又不会 OOM。
    '''
    model.eval()
    
    h, w = resolution
    
    # --- 1. 计算缩略图尺寸 ---
    # 保持长宽比，将最大边限制在 max_render_side 以内
    scale = max_render_side / max(h, w)
    
    # 如果原图比 2048 小，就保持原样；否则缩小
    if scale < 1.0:
        render_h = int(h * scale)
        render_w = int(w * scale)
        print(f"[Render] 原始分辨率 {h}x{w} 过大，生成缩略图: {render_h}x{render_w}")
    else:
        render_h, render_w = h, w
        print(f"[Render] 生成全分辨率图: {render_h}x{render_w}")

    # --- 2. 生成缩略图的网格坐标 ---
    # SIREN 的特性：只要坐标范围是 [-1, 1]，无论分辨率多少，它都能预测
    render_resolution = (render_h, render_w)
    coords = get_mgrid(render_resolution, dim=2).to(device)
    
    # --- 3. 分块推理 ---
    # 即使是 2048x2048 也有 400万像素，为了稳妥，使用 batch 推理
    with torch.no_grad():
        # model_output 会在 CPU 上
        model_output = batched_inference(model, coords, batch_size=4096*4)
    
    # --- 4. 形状变换与保存 ---
    # model_output: (H*W, C) -> (C, H, W)
    img_tensor = model_output.view(render_h, render_w, out_channels).permute(2, 0, 1)
    
    # 反归一化: [-1, 1] -> [0, 1]
    img_tensor = (img_tensor + 1) / 2
    img_tensor = torch.clamp(img_tensor, 0, 1)

    # 保存
    filename = os.path.join(output_dir, f"render_e{epoch}_s{step}.png")
    torchvision.utils.save_image(img_tensor, filename)
    
    print(f"[Render] 图片已保存至: {filename}")
    model.train()


def train(model, train_dataloader, args, img_resolution, img_channels):
    epochs = args.num_epochs
    lr = args.lr
    
    # --- 0. 目录准备 ---
    root_path = os.path.join(args.logging_root, args.experiment_name)
    ckpt_dir = os.path.join(root_path, 'checkpoints')
    results_dir = os.path.join(root_path, 'results')
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"实验目录已创建: {root_path}")

    # 1. 初始化优化器
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    
    # 2. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"开始训练... 设备: {device}, Epochs: {epochs}, LR: {lr}")
    
    total_steps = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1): 
        for step, (coords, pixels) in enumerate(train_dataloader):
            
            coords = coords.to(device).reshape(-1, 2)
            pixels = pixels.to(device).reshape(-1, img_channels)

            pred = model(coords)

            loss = ((pred - pixels) ** 2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            if total_steps % 10 == 0:
                print(f"Epoch {epoch} | Step {total_steps} | Loss: {loss.item():.6f}")

            total_steps += 1

        # --- VALIDATION / CHECKPOINT ---
        if epoch % args.epochs_til_ckpt == 0:
            ckpt_path = os.path.join(ckpt_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] 模型已保存: {ckpt_path}")

            # 传入这一步检测到的真实分辨率 img_resolution 和通道数 img_channels
            render_and_save(
                model=model, 
                resolution=img_resolution, 
                out_channels=img_channels,
                output_dir=results_dir, 
                epoch=epoch, 
                step=total_steps, 
                device=device
            )

    print(f"训练结束. 总耗时: {time.time() - start_time:.2f}s")



# dataset = data.ImageFitting(args.image_path, sidelength=target_sidelength)

if __name__ == "__main__":
    args = get_config()
    
    # 使用分块 Dataset
    # patch_size=512 意味着每个 step 训练 512x512 = 26万个像素
    # 这比之前全图训练要快得多，且显存占用极低
    # 假设 Level 2 是大概 4K 分辨率
    dataset = data.ChunkedWSIFitting(
        args.image_path, 
        patch_size=512, 
        steps_per_epoch=500,
        target_level=6  # <--- 修改这里：尝试读取 Level 2
    )
    
    # 这里的 batch_size 设为 1 即可，因为 Dataset 每次已经返回了一大块数据 (patch_size*patch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    # 获取真实信息
    actual_resolution = dataset.sidelength # (H, W) 
    actual_channels = dataset.n_channels

    print(f"WSI 尺寸: {actual_resolution}, 通道: {actual_channels}")

    model = network.Siren(
        in_features=2,
        hidden_features=256, 
        hidden_layers=3,
        out_features=actual_channels 
    ).cuda()

    # --- 4. 开始训练 ---
    # 将检测到的分辨率和通道数显式传给 train 函数，用于渲染
    train(
        model=model,
        train_dataloader=dataloader,
        args=args,
        img_resolution=actual_resolution, # <--- 传入真实分辨率
        img_channels=actual_channels      # <--- 传入真实通道数
    )