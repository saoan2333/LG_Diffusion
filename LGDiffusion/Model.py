# from datetime import dim

from sympy.strategies.branch import condition

from LGDiffusion.Functions import *
import math
from torch import nn
from einops import rearrange
from functools import partial
import torch.nn.functional as F
from torchvision import utils
from matplotlib import pyplot as plt
from tqdm import tqdm

# EMA 是 "Exponential Moving Average"（指数滑动平均）的缩写,它常用于模型参数的平滑更新，比如在训练神经网络时，用 EMA 来得到更加稳定的模型版本，有助于提升泛化能力。
class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_params(self, old, new):
        return old*self.beta + (1-self.beta)*new if old is not None else new

    def update_model(self, mean_model, current_model):
        for c_param, m_param in zip(current_model.parameters(), mean_model.parameters()):
            old_weight, new_weight = m_param.data, c_param.data
            m_param.data = self.update_params(old_weight, new_weight)

# CNN相关
# 正弦位置编码（Sinusoidal Positional Embedding）模块，常用于 Transformer、Diffusion 等模型中为输入添加位置信息。
# 位置信息的编码计算方式来自Transformer给出的公式（原文Attention is All You Need）
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb

# 卷积块定义（ConvBlock）
# 将时间信息（time_emb）通过广播加法调制到了原有的通道特征中。
class ConvbBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=1, small_rf = False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )if exists(time_emb_dim) else None

        self.time_reshape = nn.Conv2d(time_emb_dim, dim, kernel_size=1)
        self.deep_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out*mult, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out*mult, dim_out, kernel_size=3, padding=1)
        )

        # k_deep = 3 if small_rf else 5
        # k_main = 1 if small_rf else 3
        # self.deep_conv = nn.Conv2d(dim, dim, kernel_size=k_deep, padding=k_deep // 2, groups=dim)
        # self.net = nn.Sequential(
        #     nn.Conv2d(dim, dim_out*mult, kernel_size=k_main, padding=k_main // 2),
        #     nn.GELU(),
        #     nn.Conv2d(dim_out*mult, dim_out, kernel_size=k_main, padding=k_main // 2)
        # )

        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        y = self.deep_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'Miss: Time Emb'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            condition = self.time_reshape(condition)
            y = y + condition

        y = self.net(y)
        return y + self.res_conv(x)

# CNN神经网络结构定义
class Net(nn.Module):
    def __init__(self, dim, out_dim = None, channels = 3, time_emb = True, multiscale = False, device = None):
        super().__init__()
        self.device = device
        self.channels = channels
        self.multiscale = multiscale

        if time_emb:
            time_dim = 32

            if multiscale:
                self.TimeEmb = SinusoidalPositionalEncoding(time_dim)
                self.ScaleEmb = SinusoidalPositionalEncoding(time_dim)
                self.time_mlp = nn.Sequential(
                    nn.Linear(time_dim * 2, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
            else:
                self.time_mlp = nn.Sequential(
                    SinusoidalPositionalEncoding(time_dim),
                    nn.Linear(time_dim, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )

        else:
            time_dim = None
            self.time_mlp = None

        half_dim = int(dim / 2)
        small_rf = not multiscale

        self.l1 = ConvbBlock(channels, half_dim, time_emb_dim=time_dim, small_rf = small_rf)
        self.l2 = ConvbBlock(half_dim, dim, time_emb_dim=time_dim, small_rf = small_rf)
        self.l3 = ConvbBlock(dim, dim, time_emb_dim=time_dim, small_rf = small_rf)
        self.l4 = ConvbBlock(dim, half_dim, time_emb_dim=time_dim, small_rf = small_rf)

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(half_dim, out_dim, kernel_size=1)
        )

    def forward(self, x, time, scale=None):
        if self.multiscale:
            scale_tensor = torch.ones(size=time.shape).to(device=self.device) * scale
            t = self.TimeEmb(time)
            s = self.ScaleEmb(scale_tensor)
            t_s_vec = torch.cat([t, s], dim=1)
            cond_vec = self.time_mlp(t_s_vec)
        else:
            t = self.time_mlp(time) if exists(self.time_mlp) else None
            cond_vec = t

        x= self.l1(x, cond_vec)
        x = self.l2(x, cond_vec)
        x = self.l3(x, cond_vec)
        x = self.l4(x, cond_vec)

        return self.final_conv(x)

# UNET相关

class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            if hasattr(layer, 'forward') and 'emb' in layer.forward.__code__.co_varnames:
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps, dim, max_period = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Upsample(nn.Module):
    def __init__(self, channels, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dims = dims
        self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        self.op = conv_nd(
            dims, self.channels, self.out_channels, 3, stride=stride, padding=1
        )

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, dims)
            self.x_upd = Upsample(channels, dims)
        elif down:
            self.h_upd = Downsample(channels,  dims)
            self.x_upd = Downsample(channels, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            )
        )

        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=self.out_channels)
        self.out_act = nn.SiLU()
        self.out_drop = nn.Dropout(p=dropout)
        self.out_conv = zero_module(
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            kernel_size = 3 if use_conv else 1
            padding = 1 if use_conv else 0
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

# Unet
class UNet(nn.Module):
    def __init__(
            self,
            # channels
            in_channels,
            # dim
            model_channels,
            # out_dim
            out_channels = 3,
            num_res_blocks = 2,
            dropout=0,
            # channel_mult=(1, 2, 4, 8),
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dtype = torch.float32

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                if level == len(channel_mult) - 1 and i == 0:
                    ich = 0
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, scale=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if scale is not None and self.num_classes is not None:
            scale_tensor = torch.tensor(scale, device=emb.device).long()
            if scale_tensor.ndim == 0:
                scale_tensor = scale_tensor.expand(emb.shape[0])
            emb = emb + self.label_emb(scale_tensor)

        h =  x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        # for level, module in enumerate(self.output_blocks):
        #     if level == 0:
        #         h = hs.pop()
        #     else:
        #         h = torch.cat([h, hs.pop()], dim=1)
        #     h = module(h, emb)
        for level, module in enumerate(self.output_blocks):
            if level == 0:
                h = hs.pop()
            else:
                h_skip = hs.pop()
                if h.shape[-2:] != h_skip.shape[-2:]:
                    # align spatial dims: H, W
                    h_skip = F.interpolate(h_skip, size=h.shape[-2:], mode="nearest")
                h = torch.cat([h, h_skip], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:], mode="nearest")
        return self.out(h)


# 模型结构定义
class Diffusion(nn.Module):
    def __init__(
            self,
            denoise_net,
            *,
            save_history=False,
            output_folder= '/output',
            n_scales,
            scale_factor,
            image_sizes,
            scale_mul=(1, 1),
            channels=3,
            timesteps=100,
            full_train=False,
            scale_losses=None,
            loss_factor=1,
            loss_type='l1',
            betas=None,
            device=None,
            reblurring=True,
            sample_limited_t=False,
            omega=0,
    ):
        super().__init__()
        # 基础base
        self.device = device
        self.save_history = save_history
        self.output_folder = Path(output_folder)
        self.channels = channels

        # 尺度总数
        self.n_scales = n_scales
        # 每个尺度之间的缩放因子
        self.scale_factor = scale_factor
        # 每个尺度下的图片size
        self.image_sizes = ()
        # 额外的尺度缩放因子
        self.scale_mul = scale_mul

        # 是否只使用部分时间步进行采样（加速采样）
        self.sample_limited_t = sample_limited_t

        # reblurring用于控制是否上下不同尺度之间的信息
        self.reblurring = reblurring

        self.img_prev_upsample = None

        # 采样方式一:CLIP
        # Clip采样用的属性
        self.clip_guided_sampling = False
        self.guidance_sub_iters = None
        self.stop_guidance = None
        self.quantile = 0.8
        self.clip_model = None
        self.clip_strength = None
        self.clip_text = ''
        self.text_embedds = None
        self.text_embedds_hr = None
        self.text_embedds_lr = None
        self.clip_text_features = None
        self.clip_score = []
        self.clip_mask = None
        self.llambda = 0
        self.x_recon_prev = None

        # omega参数用于在采样时添加随机性,即模型是否完全根据学到的内容来采样的程度
        self.omega = omega

        # 在Clip采样时指定ROI, ROI（Region of Interest） 指的是 感兴趣区域
        self.clip_roi_bb = []

        # 采样方式二:ROI
        # ROI引导采样
        self.roi_guided_sampling = False
        # ROI区域位置信息标记 [y,x,h,w]
        self.roi_bbs = []
        # ROI区域的信息统计 [mean_tensor[1,3,1,1], std_tensor[1,3,1,1]]
        self.roi_bbs_stat = []
        # 特定区域的patch指定
        self.roi_target_patch = []

        # 反转图片的x与y，为了对齐pytorch的图像张量格式[batch_size, channels, height, width]
        for i in range(n_scales):
            self.image_sizes = self.image_sizes + ((image_sizes[i][1], image_sizes[i][0]),)

        self.denoise_net = denoise_net

        # 如果给出了噪音betas那么就用输入的betas，否则使用余弦调度器来完成噪音调度
        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        # 根据采样公式与输入的beta计算出图像保留率alpha
        alphas = 1 - betas
        # cumprod是cumulative product（累积乘积）
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # 设定模型的timesteps以及loss类型
        timesteps,  = betas.shape
        self.num_timesteps = int(timesteps)
        self.num_timesteps_trained = []
        self.num_timesteps_ideal = []
        self.num_timesteps_trained.append(self.num_timesteps)
        self.num_timesteps_ideal.append(self.num_timesteps)
        self.loss_type = loss_type

        # 设立一个简易函数用于后续快捷转换np数据为tensor类型
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # 提前设立Buffer完成静态数据的计算,减少非训练目标参数的计算占比与资源消耗
        # 大部分的计算公式都源自DDPM
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',to_torch(alphas_cumprod_prev))

        # 由于训练前已经确定了噪音调度,因此Buffer扩散阶段的静态参数
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # 由于训练前已经确定了噪音调度,因此Buffer推理阶段的静态参数
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        sigma_t = np.sqrt(1. - alphas_cumprod) / np.sqrt(alphas_cumprod)

        # 根据各尺度的模糊程度参数scale_losses,为每个尺度分配时间步数
        if scale_losses is not None:
            for i in range(n_scales-1):
                self.num_timesteps_ideal.append(
                    int(np.argmax(sigma_t > loss_factor * scale_losses[i]))
                )
                if full_train:
                    self.num_timesteps_trained.append(
                    int(timesteps)
                )
                else:
                    self.num_timesteps_trained.append(self.num_timesteps_ideal[i+1])


        # 控制上一尺度的图像在融合进下一尺度前的模糊处理程度,由gammacabs参数控制.
        gammas = torch.zeros(size=(n_scales - 1, self.num_timesteps), device=self.device)
        for i in range(n_scales - 1):
            gammas[i, :] = (torch.tensor(sigma_t, device=self.device) / (loss_factor * scale_losses[i])).clamp(min=0, max=1)
        self.register_buffer('gammas', gammas)

    # ROI采样
    # patch融合函数, eta参数控制融合程度.
    # 计算方式出自SinDDM
    def roi_patch_modification(self, x_recon, scale=0, eta=0.8):
        x_modified = x_recon
        for bb in self.roi_bbs:
            bb = [int(bb_i / np.power(self.scale_factor, self.n_scales - scale - 1)) for bb_i in bb]
            bb_y, bb_x, bb_h, bb_w = bb
            target_patch_resize = F.interpolate(self.roi_target_patch[scale], size=(bb_h, bb_w))
            x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w] = eta * target_patch_resize + (
                        1 - eta) * x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w]
        return x_modified

    # 计算正向扩散时xt的mean与std
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # 反向传播的过程的第一步,根据xt直接估计初始图像x0,计算公式出自DDPM,加入了SinDDM的reblurring策略
    def predict_start_from_noise(self, x_t, t, s, noise):
        x_recon_ddpm = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        # 引入高分辨率模糊融合策略(reblurring),以增强细节与稳定性,计算方式出自SinDDM
        if not self.reblurring or s == 0:
            # 在关闭reblurring或最底层时不进行高分辨率模糊
            return x_recon_ddpm, x_recon_ddpm
        else:
            cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)
            x_tm1_mix = (x_recon_ddpm - extract(cur_gammas, t, x_recon_ddpm.shape) * self.img_prev_upsample) / (
                        1 - extract(cur_gammas, t, x_recon_ddpm.shape))

            # 未处理版本备份
            x_t_mix = x_recon_ddpm
            return x_tm1_mix, x_t_mix

    # 后验分布计算,计算方式来自DDPM,融入了SinDDM的reblurring机制
    def q_posterior(self, x_start, x_t_mix, x_t, t, s):
        if not self.reblurring or s == 0:
            # regular DDPM
            posterior_mean = (
                    extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                    extract(self.posterior_mean_coef2, t, x_t.shape) * x_t

            )
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        elif t[0] > 0:
            x_tm1_mix = x_start
            posterior_variance_low = torch.zeros(x_t.shape,
                                                 device=self.device)
            posterior_variance_high = 1 - extract(self.alphas_cumprod, t - 1, x_t.shape)
            omega = self.omega
            posterior_variance = (1-omega) * posterior_variance_low + omega * posterior_variance_high
            posterior_log_variance_clipped = torch.log(posterior_variance.clamp(1e-20,None))

            var_t = posterior_variance

            posterior_mean = extract(self.sqrt_alphas_cumprod, t-1, x_t.shape) * x_tm1_mix + \
                                    torch.sqrt(1-extract(self.alphas_cumprod, t-1, x_t.shape) - var_t) * \
                                    (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t_mix) / \
                                    extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        else:
            posterior_mean = x_start  # for t==0 no noise added
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 根据q_posterior计算出的后验分布计算下一步的数据分布
    @torch.enable_grad()
    def p_mean_variance(self, x, t, s, clip_denoised:bool):
        pred_noise = self.denoise_net(x, t, scale=s)
        x_recon, x_t_mix = self.predict_start_from_noise(x, t=t, s=s, noise=pred_noise)
        # cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)

        if self.save_history:
            final_results_folder = Path(str(self.output_folder / f'mid_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (x_recon.clamp(-1., 1.) + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'denoised_t-{t[0]:03}_s-{s}.png'),
                             nrow=4)
        # CLIP引导采样
        if self.clip_guided_sampling and (self.stop_guidance <= t[0] or s < self.n_scales - 1) and self.guidance_sub_iters[s] > 0:
            if clip_denoised:
                x_recon.clamp_(-1., 1.)

            # 确保 CLIP 引导的图像修改效果在不同子步骤间逐步融合、平滑过渡，同时开启自动微分以继续优化图像。
            if self.clip_mask is not None:
                x_recon = x_recon * (1 - self.clip_mask) + (
                        (1 - self.llambda) * self.x_recon_prev + self.llambda * x_recon) * self.clip_mask
            # 开启自动微分(auto-diff)
            x_recon.requires_grad_(True)

            x_recon_renorm = (x_recon + 1) * 0.5
            for i in range(self.guidance_sub_iters[s]):
                self.clip_model.zero_grad()
                # 根据当前图像的尺度s选择引导方式,s>0高分辨率文本引导, 否则低分辨率文本嵌入.
                if s > 0:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.text_embedds_hr)
                else:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.text_embedds_lr)

                clip_grad = torch.autograd.grad(score, x_recon, create_graph=False)[0]

                # 根据梯度的强度，创建一个“CLIP引导的掩码区域”,使模型更关注被标注的这个mask区域
                if self.clip_mask is None:
                    clip_grad, clip_mask = grad_filter(grad=clip_grad, quantile=self.quantile)
                    self.clip_mask = clip_mask.float()

                if self.save_history:
                    final_results_folder = Path(str(self.output_folder / f'mid-samples_scale_{s}'))
                    final_results_folder.mkdir(parents=True, exist_ok=True)
                    final_mask = self.clip_mask.type(torch.float64)

                    utils.save_image(final_mask,
                                     str(final_results_folder / f'clip_mask_s-{s}.png'),
                                     nrow=4)
                    utils.save_image((x_recon.clamp(-1., 1.) + 1) * 0.5,
                                     str(final_results_folder / f'clip_out_s-{s}_t-{t[0]}_subiter_{i}.png'),
                                     nrow=4)

                # 梯度归一化,normalize gradients
                division_norm = torch.linalg.vector_norm(x_recon * self.clip_mask, dim=(1,2,3), keepdim=True) / torch.linalg.vector_norm(
                    clip_grad * self.clip_mask, dim=(1,2,3), keepdim=True)

                # 基于CLIP更新x
                x_recon += self.clip_strength * division_norm * clip_grad * self.clip_mask
                x_recon.clamp_(-1., 1.)

                # 将图像转换到[0, 1]以用于下一次迭代的计算
                x_recon_renorm = (x_recon + 1) * 0.5
                # 保存本次迭代的CLip相似性得分,detach() 是为了断开 autograd 链条，不再反向传播；
                self.clip_score.append(score.detach().cpu())

            self.x_recon_prev = x_recon.detach()

            # Clip loss可视化
            plt.rcParams['figure.figsize'] = [16, 8]
            plt.plot(self.clip_score)
            plt.grid(True)
            plt.savefig(str(self.results_folder / 'clip_score'))
            plt.clf()

        # ROI引导采样
        elif self.roi_guided_sampling and (s < self.n_scales-1):
            x_recon = self.roi_patch_modification(x_recon, scale=s)

        # 无引导采样
        if int(s) > 0 and t[0] > 0 and self.reblurring:
            cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)
            x_tm1_mix = extract(cur_gammas, t - 1, x_recon.shape) * self.img_prev_upsample + \
                        (1 - extract(cur_gammas, t - 1, x_recon.shape)) * x_recon
        else:
            x_tm1_mix = x_recon

        if clip_denoised:
            x_tm1_mix.clamp_(-1., 1.)
            x_t_mix.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_tm1_mix, x_t_mix=x_t_mix,
                                                                                  x_t=x, t=t, s=s,
                                                                                  )
        return model_mean, posterior_variance, posterior_log_variance

    # 前向扩散阶段逐步加噪的函数
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # 基于p_mean_variance计算出来的数据分布,对x进行单步反向采样.
    @torch.no_grad()
    def p_sample(self, x, t, s, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, clip_denoised=clip_denoised)

        noise = noise_generator(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        nonzero_mask_s = torch.tensor([True], device=self.device).float()

        return model_mean + nonzero_mask_s * nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # 采样循环,从纯噪音图开始逐步去噪直到生成最终图像,s为尺度索引
    @torch.no_grad()
    def p_sample_loop(self, shape, s):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        if self.save_history:
            final_results_folder = Path(str(self.output_folder / f'mid_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'input_noise_s-{s}.png'),
                             nrow=4)
        if self.sample_limited_t and s < (self.n_scales-1):
            t_min = self.num_timesteps_ideal[s+1]
        else:
            t_min = 0
        for i in tqdm(reversed(range(t_min, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), s)
            if self.save_history:
                final_img = (img + 1) * 0.5
                utils.save_image(final_img,
                                 str(final_results_folder / f'output_t-{i:03}_s-{s}.png'),
                                 nrow=4)
        return img

    # 采样Function封装
    @torch.no_grad()
    def sample(self, batch_size=16, scale_0_size=None, s=0):
        if scale_0_size is not None:
            image_size = scale_0_size
        else:
            image_size = self.image_sizes[0]
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size[0], image_size[1]), s=s)

    # 先对上一尺度的图加噪再采样还原的采样策略
    @torch.no_grad()
    def p_sample_via_scale_loop(self, batch_size, img, s, custom_t=None):
        device = self.betas.device
        if custom_t is None:
            total_t = self.num_timesteps_ideal[min(s, self.n_scales-1)]-1
        else:
            total_t = custom_t
        b = batch_size
        self.img_prev_upsample = img

        if self.save_history:
            final_results_folder = Path(str(self.output_folder / f'mid_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'clean_input_s_{s}.png'),
                             nrow=4)
        # add noise
        img = self.q_sample(x_start=img, t=torch.Tensor.expand(torch.tensor(total_t, device=device), batch_size), noise=None)

        if self.save_history:
            final_results_folder = Path(str(self.output_folder / f'mid_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'noisy_input_s_{s}.png'),
                             nrow=4)

        if self.clip_mask is not None:
            if s > 0:
                mul_size = [int(self.image_sizes[s][0]* self.scale_mul[0]), int(self.image_sizes[s][1]* self.scale_mul[1])]
                self.clip_mask = F.interpolate(self.clip_mask, size=mul_size, mode='bilinear')
                self.x_recon_prev = F.interpolate(self.x_recon_prev, size=mul_size, mode='bilinear')
            else:
                self.clip_mask = None

        if self.sample_limited_t and s < (self.n_scales - 1):
            t_min = self.num_timesteps_ideal[s + 1]
        else:
            t_min = 0
        for i in tqdm(reversed(range(t_min, total_t)), desc='sampling loop time step', total=total_t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), s)
            if self.save_history:
                final_img = (img + 1) * 0.5
                utils.save_image(final_img,
                                 str(final_results_folder / f'output_t-{i:03}_s-{s}.png'),
                                 nrow=4)
        return img

    # 从特定尺度s开始采样,输入为上一尺度的输出
    @torch.no_grad()
    def sample_via_scale(self, batch_size, img, s, scale_mul=(1, 1), custom_sample=False, custom_img_size_idx=0, custom_t=None, custom_image_size=None):
        if custom_sample:
            if custom_img_size_idx >= self.n_scales:
                size = self.image_sizes[self.n_scales - 1]
                factor = self.scale_factor ** (custom_img_size_idx + 1 - self.n_scales)
                size = (int(size[0] * factor), int(size[1] * factor))
            else:
                size = self.image_sizes[custom_img_size_idx]
        else:
            size = self.image_sizes[s]
        image_size = (int(size[0] * scale_mul[0]), int(size[1] * scale_mul[1]))
        if custom_image_size is not None:
            image_size = custom_image_size

        img = F.interpolate(img, size=image_size, mode='bilinear')
        return self.p_sample_via_scale_loop(batch_size, img, s, custom_t=custom_t)

    # 前向扩散过程中的Loss计算
    def p_losses(self, x_start, t, s, noise=None, x_orig=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        if int(s) > 0:
            cur_gammas = self.gammas[s - 1].reshape(-1)
            x_mix = extract(cur_gammas, t, x_start.shape) * x_start + \
                    (1 - extract(cur_gammas, t, x_start.shape)) * x_orig
            x_noisy = self.q_sample(x_start=x_mix, t=t, noise=noise)
            x_recon = self.denoise_net(x_noisy, t, s)

        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_net(x_noisy, t, s)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        elif self.loss_type == 'l1_pred_img':
            if int(s) > 0:
                cur_gammas = self.gammas[s - 1].reshape(-1)
                if t[0]>0:
                    x_mix_prev = extract(cur_gammas, t-1, x_start.shape) * x_start + \
                            (1 - extract(cur_gammas, t-1, x_start.shape)) * x_orig
                else:
                    x_mix_prev = x_orig
            else:
                x_mix_prev = x_start
            loss = (x_mix_prev-x_recon).abs().mean()
        else:
            raise NotImplementedError()

        return loss

    # 前向扩散函数
    def forward(self, x, s, *args, **kwargs):
        if int(s) > 0:  # no deblurring in scale=0
            x_orig = x[0]
            x_recon = x[1]
            b, c, h, w = x_orig.shape
            device = x_orig.device
            img_size = self.image_sizes[s]
            assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()
            return self.p_losses(x_recon, t, s, x_orig=x_orig, *args, **kwargs)

        else:

            b, c, h, w = x[0].shape
            device = x[0].device
            img_size = self.image_sizes[s]
            assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()
            return self.p_losses(x[0], t, s, *args, **kwargs)


# if __name__ == '__main__':
#     model = UNet(
#         model_channels=160,
#         in_channels=3,
#     )
#     print('model loaded')
