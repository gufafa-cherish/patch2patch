import os
import time
import argparse
import shutil
import numpy as np
from PIL import Image
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.init as init

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import einops


class MultiscalePixelBank:
    def __init__(self, scales=[(7, 40), (5, 30), (3, 20)]):
        self.scales = scales  # [(patch_size, window_size), ...]
    
    def construct_multiscale_banks(self, args):
        """构建多尺度像素银行"""
        multiscale_banks = {}
        
        for ps, ws in self.scales:
            print(f"Constructing pixel banks for scale ({ps}, {ws}) from noisy images ...")
            # 临时修改参数
            original_ps, original_ws = args.ps, args.ws
            args.ps, args.ws = ps, ws
            
            # 构建该尺度的像素银行
            bank_dir = os.path.join(args.save, args.dataset, f'scale_{ps}_{ws}', '_'.join(str(i) for i in [ws, ps, args.nn, args.loss]))
            os.makedirs(bank_dir, exist_ok=True)
            
            noisy_folder = os.path.join(args.data_path, args.dataset, args.Noisy)
            image_files = sorted(os.listdir(noisy_folder))
            
            for image_file in image_files:
                bank_data = self._construct_single_scale_bank(image_file, args, ps, ws)
                file_name_without_ext = os.path.splitext(image_file)[0]
                np.save(os.path.join(bank_dir, file_name_without_ext), bank_data.cpu())
            
            print(f"Pixel bank construction completed for scale_{ps}_{ws}.")
            
            multiscale_banks[f'scale_{ps}_{ws}'] = bank_dir
            
            # 恢复原始参数
            args.ps, args.ws = original_ps, original_ws
        print("Pixel bank construction completed for all images and scales.")
        return multiscale_banks
    
    def _construct_single_scale_bank(self, image_file, args, patch_size, window_size):
        """构建单个尺度的像素银行 - 只存储噪声像素块"""
        image_path = os.path.join(args.data_path, args.dataset, args.Noisy, image_file)
        
        # 加载噪声图像
        img = Image.open(image_path)
        img = transform(img).unsqueeze(0).cuda()
        start_time = time.time()
        
        pad_sz = window_size // 2 + patch_size // 2
        center_offset = window_size // 2
        blk_sz = 64
        
        # 填充图像
        img_pad = F.pad(img, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
        img_unfold = F.unfold(img_pad, kernel_size=patch_size, padding=0, stride=1)
        H_new = img.shape[-2] + window_size
        W_new = img.shape[-1] + window_size
        img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)
        print(f"Scale ({patch_size}, {window_size}) - Image {image_file} - shape after unfolding: {img_unfold.shape}")
        
        num_blk_w = img.shape[-1] // blk_sz
        num_blk_h = img.shape[-2] // blk_sz
        is_window_size_even = (window_size % 2 == 0)
        topk_list = []
        
        # 处理每个块
        for blk_i in range(num_blk_w):
            for blk_j in range(num_blk_h):
                start_h = blk_j * blk_sz
                end_h = (blk_j + 1) * blk_sz + window_size
                start_w = blk_i * blk_sz
                end_w = (blk_i + 1) * blk_sz + window_size
                
                sub_img_uf = img_unfold[..., start_h:end_h, start_w:end_w]
                sub_img_shape = sub_img_uf.shape
                
                if is_window_size_even:
                    sub_img_uf_inp = sub_img_uf[..., :-1, :-1]
                else:
                    sub_img_uf_inp = sub_img_uf
                
                patch_windows = F.unfold(sub_img_uf_inp, kernel_size=window_size, padding=0, stride=1)
                patch_windows = einops.rearrange(
                    patch_windows,
                    'b (c k1 k2 k3 k4) (h w) -> b (c k1 k2) (k3 k4) h w',
                    k1=patch_size, k2=patch_size, k3=window_size, k4=window_size,
                    h=blk_sz, w=blk_sz
                )
                
                img_center = einops.rearrange(
                    sub_img_uf,
                    'b (c k1 k2) h w -> b (c k1 k2) 1 h w',
                    k1=patch_size, k2=patch_size,
                    h=sub_img_shape[-2], w=sub_img_shape[-1]
                )
                img_center = img_center[..., center_offset:center_offset + blk_sz, center_offset:center_offset + blk_sz]
                
                # 计算L2距离并选择最相似的patches
                l2_dis = torch.sum((img_center - patch_windows) ** 2, dim=1)
                _, sort_indices = torch.topk(l2_dis, k=args.nn, largest=False, sorted=True, dim=-3)
                
                patch_windows_reshape = einops.rearrange(
                    patch_windows,
                    'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                    k1=patch_size, k2=patch_size, k3=window_size, k4=window_size
                )
                patch_center = patch_windows_reshape[:, :, patch_windows_reshape.shape[2] // 2, ...]
                topk = torch.gather(patch_center, dim=-3,
                                    index=sort_indices.unsqueeze(1).repeat(1, 3, 1, 1, 1))
                topk_list.append(topk)
        
        # 合并所有块的结果
        topk = torch.cat(topk_list, dim=0)
        topk = einops.rearrange(topk, '(w1 w2) c k h w -> k c (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
        topk = topk.permute(2, 3, 0, 1)
        elapsed = time.time() - start_time
        print(f"Scale ({patch_size}, {window_size}) - Processed {image_file} in {elapsed:.2f} seconds. Pixel bank shape: {topk.shape}")
        
        return topk


class MultiscaleFusion(nn.Module):
    def __init__(self, scales, base_dim=64, num_layers=2):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.num_layers = num_layers
        
        # 每个尺度的编码器
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, base_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_dim, base_dim, 3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in scales
        ])
        
        # 尺度间注意力
        self.scale_attention = nn.Sequential(
            nn.Conv2d(base_dim * self.num_scales, base_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, self.num_scales, 1),
            nn.Softmax(dim=1)
        )
        
        # 融合网络 - 根据层数动态构建
        fusion_layers = []
        in_channels = base_dim * self.num_scales
        
        # 第一层：输入层
        if num_layers == 1:
            # 只有1层：直接输出
            fusion_layers.append(nn.Conv2d(in_channels, 3, 1))
        else:
            # 第一层：扩展通道
            fusion_layers.append(nn.Conv2d(in_channels, base_dim * 2, 3, padding=1))
            fusion_layers.append(nn.ReLU(inplace=True))
            
            # 中间层：根据层数添加
            for i in range(num_layers - 2):
                if i % 2 == 0:
                    # 偶数层：保持或减少通道
                    fusion_layers.append(nn.Conv2d(base_dim * 2, base_dim, 3, padding=1))
                else:
                    # 奇数层：扩展通道
                    fusion_layers.append(nn.Conv2d(base_dim, base_dim * 2, 3, padding=1))
                fusion_layers.append(nn.ReLU(inplace=True))
            
            # 最后一层：输出层
            if (num_layers - 2) % 2 == 0:
                # 如果中间层数是偶数，当前通道是 base_dim * 2
                fusion_layers.append(nn.Conv2d(base_dim * 2, 3, 1))
            else:
                # 如果中间层数是奇数，当前通道是 base_dim
                fusion_layers.append(nn.Conv2d(base_dim, 3, 1))
        
        self.fusion = nn.Sequential(*fusion_layers)
    
    def forward(self, multiscale_banks):
        """
        multiscale_banks: dict of tensors with shape [B, H, W, C]
        """
        features = []
        
        # 按照预定义的尺度顺序处理特征
        for i, (ps, ws) in enumerate(self.scales):
            scale_name = f'scale_{ps}_{ws}'
            if scale_name not in multiscale_banks:
                continue
                
            bank = multiscale_banks[scale_name]
            # 转换维度 [B, H, W, C] -> [B, C, H, W]
            bank_tensor = bank.permute(0, 3, 1, 2)
            feat = self.scale_encoders[i](bank_tensor)
            features.append(feat)
        
        # 拼接所有尺度的特征
        combined = torch.cat(features, dim=1)  # [B, base_dim*num_scales, H, W]
        
        # 计算尺度注意力权重
        scale_weights = self.scale_attention(combined)  # [B, num_scales, H, W]
        
        # 加权融合
        weighted_features = []
        for i, feat in enumerate(features):
            weight = scale_weights[:, i:i+1, :, :]
            weighted_feat = feat * weight
            weighted_features.append(weighted_feat)
        
        # 最终融合
        final_combined = torch.cat(weighted_features, dim=1)
        
        # 通过融合网络得到残差信号
        residual = self.fusion(final_combined)
        
        return residual


parser = argparse.ArgumentParser('Patch2Patch')
parser.add_argument('--data_path', default='./data', type=str, help='Path to the data')
parser.add_argument('--dataset', default='SIDD', type=str, help='Dataset name')
parser.add_argument('--GT', default='GT', type=str, help='Folder name for ground truth images')
parser.add_argument('--Noisy', default='Noisy', type=str, help='Folder name for noisy images')
parser.add_argument('--save', default='./results', type=str, help='Directory to save pixel bank results')
parser.add_argument('--out_image', default='./results_image', type=str, help='Directory to save denoised images')
parser.add_argument('--ws', default=40, type=int, help='Window size')
parser.add_argument('--ps', default=7, type=int, help='Patch size')
parser.add_argument('--nn', default=100, type=int, help='Number of nearest neighbors to search')
parser.add_argument('--mm', default=20, type=int, help='Number of pixel banks to use for training')
parser.add_argument('--loss', default='L2', type=str, help='Loss function type')
args = parser.parse_args()

# 设置随机种子
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = args.ws
PATCH_SIZE = args.ps
NUM_NEIGHBORS = args.nn
loss_type = args.loss

transform = transforms.Compose([transforms.ToTensor()])

# 损失函数
loss_f = nn.L1Loss() if args.loss == 'L1' else nn.MSELoss()


def construct_multiscale_pixel_bank():
    """构建多尺度像素银行"""
    scales = [(7, 40), (5, 30), (3, 20)]
    multiscale_banks = {}
    
    # 检查银行是否已经存在
    all_exist = True
    for ps, ws in scales:
        scale_name = f'scale_{ps}_{ws}'
        bank_dir = os.path.join(args.save, args.dataset, scale_name, '_'.join(str(i) for i in [ws, ps, args.nn, args.loss]))
        if os.path.exists(bank_dir):
            multiscale_banks[scale_name] = bank_dir
        else:
            all_exist = False
            break
    
    if all_exist:
        return multiscale_banks
    
    multiscale_bank = MultiscalePixelBank(scales=scales)
    multiscale_banks = multiscale_bank.construct_multiscale_banks(args)
    return multiscale_banks


def mse_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(gt, pred)


def train_multiscale(fusion_net, optimizer, multiscale_bank_data):
    """多尺度融合网络训练 - 学习像素银行中两个伪实例之间的映射关系"""
    # 找到最小的空间维度
    min_H = min(img_bank.shape[1] for img_bank in multiscale_bank_data.values())
    min_W = min(img_bank.shape[2] for img_bank in multiscale_bank_data.values())
    
    # 构建多尺度输入（第一个伪实例）
    multiscale_input = {}
    input_indices = {}
    for scale_name, img_bank in multiscale_bank_data.items():
        N, H, W, C = img_bank.shape
        
        # 第一次随机采样 - 作为输入
        index_input = torch.randint(0, N, size=(min_H, min_W), device=device)
        index_input_exp = index_input.unsqueeze(0).unsqueeze(-1).expand(1, min_H, min_W, C)
        img_input = torch.gather(img_bank, 0, index_input_exp)
        
        multiscale_input[scale_name] = img_input
        input_indices[scale_name] = index_input
    
    # 构建多尺度目标（第二个伪实例）- 从像素银行中独立采样
    multiscale_target = {}
    for scale_name, img_bank in multiscale_bank_data.items():
        N, H, W, C = img_bank.shape
        
        # 第二次随机采样 - 作为目标（与输入独立采样）
        index_target = torch.randint(0, N, size=(min_H, min_W), device=device)
        index_input = input_indices[scale_name]
        eq_mask = (index_target == index_input)
        if eq_mask.any():
            index_target[eq_mask] = (index_target[eq_mask] + 1) % N
        index_target_exp = index_target.unsqueeze(0).unsqueeze(-1).expand(1, min_H, min_W, C)
        img_target = torch.gather(img_bank, 0, index_target_exp)
        
        # 转换为 [B, C, H, W] 格式用于损失计算
        multiscale_target[scale_name] = img_target.permute(0, 3, 1, 2)
    
    # 前向传播
    pred_residual = fusion_net(multiscale_input)
    
    # 使用第一个尺度作为残差基准
    base_scale = list(multiscale_bank_data.keys())[0]
    target = multiscale_target[base_scale][:, :, :min_H, :min_W]
    input_base = multiscale_input[base_scale].permute(0, 3, 1, 2)
    input_base = input_base[:, :, :min_H, :min_W]
    residual_target = target - input_base
    
    # 计算损失：学习残差
    loss = loss_f(pred_residual, residual_target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def denoise_images_multiscale(num_layers=None):
    """使用多尺度像素银行进行去噪
    
    Args:
        num_layers: 融合网络的层数，固定为2
    """
    # 结果保存文件（将打印内容同步写入 txt）

    results_file = os.path.join(args.out_image, f'{args.dataset}_results.txt')
    os.makedirs(args.out_image, exist_ok=True)
    with open(results_file, 'w') as f:
        f.write(f"Single image independent denoising strategy (Zero-shot)\n")
    
    # 构建多尺度像素银行
    multiscale_banks = construct_multiscale_pixel_bank()
    # 将像素银行构建完成的信息写入文件
    with open(results_file, 'a') as f:
        f.write("Pixel bank construction completed for all images and scales.\n")
    
    gt_folder = os.path.join(args.data_path, args.dataset, args.GT)
    gt_files = sorted(os.listdir(gt_folder))

    scales = [(7, 40), (5, 30), (3, 20)]
    num_layers = 2 if num_layers is None else num_layers
    
    max_epoch = 3000
    lr = 0.001
    avg_PSNR = 0
    avg_SSIM = 0
    
    for image_file in gt_files:
        # 加载干净图像
        image_path = os.path.join(gt_folder, image_file)
        clean_img = Image.open(image_path)
        clean_img_tensor = transform(clean_img).unsqueeze(0).to(device)
        clean_img_np = io.imread(image_path)

        # 加载多尺度像素银行
        multiscale_bank_data = {}
        for scale_name, bank_dir in multiscale_banks.items():
            bank_path = os.path.join(bank_dir, os.path.splitext(image_file)[0])
            if not os.path.exists(bank_path + '.npy'):
                print(f"Pixel bank for {image_file} at {scale_name} not found, skipping.")
                continue
            
            try:
                img_bank_arr = np.load(bank_path + '.npy')
                if img_bank_arr.ndim == 3:
                    img_bank_arr = np.expand_dims(img_bank_arr, axis=1)
                # Transpose to (k, H, W, c)
                img_bank = img_bank_arr.astype(np.float32).transpose((2, 0, 1, 3))
                # Use only the first mm banks for training
                img_bank = img_bank[:args.mm]
                multiscale_bank_data[scale_name] = torch.from_numpy(img_bank).to(device)
            except Exception as e:
                print(f"Error loading {scale_name}: {e}")
                continue

        if not multiscale_bank_data:
            print(f"No pixel banks found for {image_file}, skipping.")
            continue

        # Create a new model instance for each image (single image independent training)
        fusion_net = MultiscaleFusion(scales, num_layers=num_layers).to(device)
        num_params = sum(p.numel() for p in fusion_net.parameters() if p.requires_grad)
        param_line = f"Image: {image_file} | Number of parameters: {num_params}"
        print(param_line)
        with open(results_file, 'a') as f:
            f.write(param_line + "\n")

        # 使用基准尺度的第一个银行作为噪声输入
        base_scale = list(multiscale_bank_data.keys())[0]
        noisy_img = multiscale_bank_data[base_scale][0].unsqueeze(0).permute(0, 3, 1, 2)

        # 训练多尺度融合网络
        n_chan = clean_img_tensor.shape[1]
        optimizer = optim.AdamW(fusion_net.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[1500, 2000, 2500], gamma=0.5)

        for epoch in range(max_epoch):
            # 训练步骤 - 只使用像素银行中的伪实例，不使用干净图像
            loss = train_multiscale(fusion_net, optimizer, multiscale_bank_data)
            scheduler.step()
            
        # 测试
        with torch.no_grad():
            # 准备多尺度输入
            multiscale_input = {}
            for scale_name, bank in multiscale_bank_data.items():
                multiscale_input[scale_name] = bank[0:1]  # 只取第一个银行
            
            residual = fusion_net(multiscale_input)
            base_input = multiscale_input[base_scale].permute(0, 3, 1, 2)
            base_input = base_input[:, :, :residual.shape[-2], :residual.shape[-1]]
            pred = torch.clamp(base_input + residual, 0, 1)
            
            mse_val = mse_loss(clean_img_tensor, pred).item()
            PSNR = 10 * np.log10(1 / mse_val)

        # 保存结果
        out_img_pil = to_pil_image(pred.squeeze(0))
        out_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + f'_multiscale_layers{num_layers}.png')
        out_img_pil.save(out_img_save_path)

        noisy_img_pil = to_pil_image(noisy_img.squeeze(0))
        noisy_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '_noisy.png')
        noisy_img_pil.save(noisy_img_save_path)

        out_img_loaded = io.imread(out_img_save_path)
        SSIM, _ = compare_ssim(clean_img_np, out_img_loaded, full=True, multichannel=True, win_size=3)
        line = f"Image: {image_file} | PSNR: {PSNR:.2f} dB | SSIM: {SSIM:.4f}"
        print(line)
        # 将单张图像的结果追加写入 txt 文件
        with open(results_file, 'a') as f:
            f.write(line + "\n")
        avg_PSNR += PSNR
        avg_SSIM += SSIM

    avg_PSNR /= len(gt_files)
    avg_SSIM /= len(gt_files)
    avg_line = f"Average PSNR: {avg_PSNR:.2f} dB, Average SSIM: {avg_SSIM:.4f}"
    print(avg_line)
    # 将平均结果写入 txt 文件
    with open(results_file, 'a') as f:
        f.write(avg_line + "\n")

    # 将结果图像文件夹打包为 zip 文件，文件名为「数据集名称_时间戳」，防止覆盖
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    zip_base = f"{args.dataset}_{timestamp}"
    shutil.make_archive(zip_base, 'zip', root_dir=args.out_image)
    print(f"Saved results folder as: {zip_base}.zip")

    return avg_PSNR, avg_SSIM


def test_multiple_layers():
    """测试固定为2层的神经网络去噪效果"""
    results = {}
    layer_range = [2]
    
    for num_layers in layer_range:
        try:
            avg_psnr, avg_ssim = denoise_images_multiscale(num_layers=num_layers)
            results[num_layers] = {
                'PSNR': avg_psnr,
                'SSIM': avg_ssim
            }
        except Exception as e:
            results[num_layers] = {
                'PSNR': 0.0,
                'SSIM': 0.0,
                'error': str(e)
            }
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    best_psnr_layer = None
    best_ssim_layer = None
    if valid_results:
        best_psnr_layer = max(valid_results.keys(), key=lambda k: valid_results[k]['PSNR'])
        best_ssim_layer = max(valid_results.keys(), key=lambda k: valid_results[k]['SSIM'])
    
    results_file = os.path.join(args.out_image, 'layer_test_results.txt')
    with open(results_file, 'w') as f:
        f.write("神经网络层数测试结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'层数':<8} {'PSNR (dB)':<15} {'SSIM':<15}\n")
        f.write("-" * 80 + "\n")
        for num_layers in sorted(results.keys()):
            result = results[num_layers]
            if 'error' in result:
                f.write(f"{num_layers:<8} {'ERROR':<15} {result['error']:<15}\n")
            else:
                f.write(f"{num_layers:<8} {result['PSNR']:<15.2f} {result['SSIM']:<15.4f}\n")
        if valid_results:
            f.write("\n最佳结果:\n")
            f.write(f"  最高PSNR: {best_psnr_layer}层网络 - {valid_results[best_psnr_layer]['PSNR']:.2f} dB\n")
            f.write(f"  最高SSIM: {best_ssim_layer}层网络 - {valid_results[best_ssim_layer]['SSIM']:.4f}\n")
    
    return {
        'results': results,
        'best_psnr_layer': best_psnr_layer,
        'best_ssim_layer': best_ssim_layer,
        'results_file': results_file
    }


if __name__ == "__main__":
    # 保证输出目录存在
    os.makedirs(args.out_image, exist_ok=True)
    denoise_images_multiscale()
