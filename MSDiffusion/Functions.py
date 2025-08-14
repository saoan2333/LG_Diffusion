import torch
from setuptools.sandbox import save_path
from skimage import morphology, filters
from inspect import isfunction
import numpy as np
from PIL import Image
from pathlib import Path

# 混合精度 Mixed Precision,这是旧版apex的amp,待更新!!!!!!!!!!!
# try:
#     from torch.cuda import amp
#     AMP_AVAILABLE = True
# except:
#     AMP_AVAILABLE = False


# 混合精度下（FP16）的loss反向传播, amp机制待更新！！！！！！！！！！！
# def loss_backwards(fp16, loss, optimizer, **kwargs):
#     if fp16:
#         with amp.scale_loss(loss, optimizer) as scaled_loss:
#             scaled_loss.backward(**kwargs)
#     else:
#         loss.backward(**kwargs)

from functools import partial
from contextlib import contextmanager

class AMP:
    def __init__(self, model , fp16):
        self.model = model
        self.fp16 = fp16 and torch.cuda.is_available()
        if self.fp16:
            self.autocast = partial(torch.amp.autocast, device_type="cuda")
            self.scaler = torch.amp.GradScaler(enabled=True)

        else:
            self.autocast = _dummy_cm
            self.scaler = torch.amp.GradScaler(enabled=False)
        # self.autocast = torch.amp.autocast if self.fp16 else _dummy_cm
        # self.scaler = torch.amp.GradScaler(enabled=self.fp16)

    def backward(self, loss, optimizer, *,  accumulate_grad = 1, **kwargs):
        loss = loss / accumulate_grad
        self.scaler.scale(loss).backward(**kwargs)

    def step(self, optimizer):
        if not self.fp16:
            optimizer.step()
        else:
            self.scaler.step(optimizer)
            self.scaler.update()

# 空过函数
@contextmanager
def _dummy_cm():
    yield


# 掩码预处理(扩张边缘+高斯平滑）
def mask_preprocess(mask, mode):
    # 根据Mode选择不同的画笔size
    if mode == "harmonization":
        element = morphology.disk(radius=7)
    if mode == "editing":
        element = morphology.disk(radius=20)
    # 提取通道数据，mask处理不需要后两项的数据（高和宽）
    mask = mask.permute((1, 2, 0))
    mask = mask[:, :, 0]
    # 扩张边缘 dilate mask
    mask = morphology.binary_dilation(mask, selem=element)
    # 高斯平滑
    mask = filters.gaussian(mask, sigma=5)
    # 整理mask结构
    mask = mask[:, :, None, None]
    mask = mask.transpose(3, 2, 0, 1)
    # 归一化
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask

# 从图像中取出某个矩阵size的数据，并计算该矩阵数据的mean和std
def stat_matrix(img, matrix):
    y, x, h, w = matrix
    mean = torch.mean(img[:, :, y:y+h, x:x+w], dim=(2, 3), keepdim=True)
    std = torch.std(img[:, :, y:y+h, x:x+w], dim=(2, 3), keepdim=True)
    return [mean, std]

# 从图像中提取特定size矩阵的数据
def extract_img(img, matrix):
    y, x, h, w = matrix
    img_new = img[:, :, y:y+h, x:x+w]
    return img_new

# 梯度稀疏化sparsification, 减轻梯度回传时的压力，通过裁剪掉变化值低于特定阈值的梯度变化的方式。
# 在这里主要用于CLIP时减少模型训练压力。
# CLIP（Contrastive Language–Image Pre-training） 是 OpenAI 提出的一个模型，可以“理解图像和文字之间的关系”。
def grad_filter(grad, quantile):
    # 计算梯度变化量
    grad_value = torch.norm(grad, dim=1)
    # 展平梯度，方便后续计算阈值
    grad_value_reshaped = torch.reshape(grad_value, (grad_value.shape[0], -1))
    # 阈值计算
    threshold = torch.quantile(grad_value_reshaped, quantile, dim = 1, interpolation='nearest')[:, None, None]
    # 筛选高于阈值的像素
    grad_threshold = grad_value - threshold
    grad_mask = (grad_threshold > 0)[:, None, :, :]
    # 将低于阈值的梯度变化调整至0
    grad_filtered = torch.clamp(grad_threshold, min= 0)[:, None, :, :]
    # 把每个梯度向量单位化，得到“方向”。
    grad_direction = grad / grad_value[:, None, None]
    # 避免nan值（除于0）
    grad_direction[torch.isnan(grad_direction)] = 0
    # 关注保留下来的梯度变化的变化方向
    grad_spares = grad_filtered * grad_direction
    return grad_spares, grad_mask

# 简单辅助函数集

# 检查x是不是None值
def exists(x):
    return x is not None

# 默认值设置
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# dataset加载器
def dataloader(datasets):
    while True:
        for data in datasets:
            yield data

# 将一组数组分成若干小组
def spilt_groups(num, div):
    groups = num // div
    rest = num % div
    new_groups = [div] * groups
    if rest > 0:
        new_groups.append(rest)
    return new_groups


# 提取特定timestep的数据，后续用于如timestep embedding的技术中
# Timestep embedding =》虽然timestep被划分为若干整数（如1000步），但是模型并不好理解整数，因此将timestep投射到高维的连续空间中方便模型理解。
def extract(a, timestep, x_shape):
    b, *_ = timestep.shape
    output = a.gather(-1, timestep)
    # reshape是为了后续的广播操作不会因为shape操作而报错？
    return output.reshape(b, *((1,) * (len(x_shape) - 1)))

# 噪音生成函数
def noise_generator(shape, device, repeat=False):
    # 生成相同的noise
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    # 随机生成不同的noise
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

# 余弦噪音调度器
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

# 多尺度下的图像金字塔机制+不同尺度下的信息丢失率评估
def muti_scales_img(foldername, filename, scale_factor = 1.411, image_size = None, create = False, auto_scale = None, single_scale = False):
    # 获取img_size
    orig_img = Image.open(foldername + filename)
    filename = filename.rsplit( ".", 1 )[ 0 ] + '.png'
    if image_size is None:
        # 直接用原图的size
        image_size = orig_img.size
    if auto_scale is not None:
        # 如果开启了自动缩放，则使用缩放后的size
        scaler = np.sqrt((image_size[0] * image_size[1])/auto_scale)
        if scaler > 1:
            image_size = (int(image_size[0]/scaler), int(image_size[1]/scaler))

    sizes = []
    downscaled_img = []
    upsample_images = []
    rescale_loss = []

    if single_scale:
        if create:
            save_path = foldername + 'scale_0/'
            Path(save_path).mkdir(parents = True, exist_ok = True)
            orig_img.save(save_path + filename.rsplit( ".", 1 )[ 0 ] + '.png')
        return [image_size], [0, 0], 1.0, 1

    # 模型的感受野为：Receptive Field， 简写为rf_net
    # rf_net = 35，意思是网络的某一层最终每个输出像素，受输入图像上一个35×35的区域影响。
    # area_scale_min参数用于确认感受野在最低尺度下覆盖图像的面积，计算方式为rf_net^2/area_scale_min
    # 根据计算rf_net = 35, area_scale_min = 3110, 以上参数配置下，感受野约占最底尺度下图像面积的40%
    area_scale_min = 3110
    short_boundary =min(image_size[0], image_size[1])
    long_boundary = max(image_size[0], image_size[1])
    # 根据上面三个参数计算出最小尺度下的短边的尺寸
    scale_min_short_boundary = int(round(np.sqrt(area_scale_min*short_boundary/long_boundary)))
    # 限制最小尺寸在42-55，防止过小导致的失真
    scale_min_short_boundary = 42 if scale_min_short_boundary < 42 else (55 if scale_min_short_boundary >55 else scale_min_short_boundary)
    min_b= scale_min_short_boundary
    min_b_img = min(image_size[0], image_size[1])
    # 根据最小尺寸计算出金字塔所需要的层数
    n_scales = int(round( (np.log(min_b_img/min_b)) / (np.log(scale_factor)) ) + 1)
    # 校准经过计算的scale_factor
    scale_factor = np.exp((np.log(min_b_img / min_b)) / (n_scales - 1))

    # 构建多层级金字塔
    # 下采样
    for i in range(n_scales):
        current_img_size = (int(round(image_size[0] / np.power(scale_factor, n_scales - i - 1))),
                            int(round(image_size[1] / np.power(scale_factor, n_scales - i - 1))))
        current_img = orig_img.resize(current_img_size, Image.Resampling.LANCZOS)
        save_path = foldername + 'scale_' + str(i) + '/'
        if create:
            Path(save_path).mkdir(parents = True, exist_ok = True)
            current_img.save(save_path+filename)
        downscaled_img.append(current_img)
        sizes.append(current_img_size)
    # 上采样
    for i in range(n_scales-1):
        upsample_img = downscaled_img[i].resize(sizes[i + 1], Image.Resampling.BILINEAR)
        upsample_images.append(upsample_img)
        # 计算各尺度下的loss
        rescale_loss.append(np.linalg.norm(np.subtract(downscaled_img[i + 1], upsample_img)) / np.asarray(upsample_img).size)
        if create:
            save_path = foldername + 'scale_' + str(i+1) + '_upsample/'
            Path(save_path).mkdir(parents = True, exist_ok = True)
            upsample_img.save(save_path+filename)

    return sizes, rescale_loss, scale_factor, n_scales








