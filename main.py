import torch
import numpy as np
import argparse
import os
import torchvision
from fsspec.registry import default

from MSDiffusion.Functions import muti_scales_img
from MSDiffusion.Model import Net, Diffusion
from MSDiffusion.Trainer import MutiScaleTrainer
# python main.py  --mode train --timesteps 10 --train_num_steps 10 --avg_window 1 --save_and_sample_every 5 --AMP --SinScale --step_start_ema 10
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--scope", help='choose training mark num.', default='pyramids')
    parser.add_argument("--mode", help='choose mode: train, sample')
    parser.add_argument("--input_image", help='content image for style transfer or harmonization.',
                        default='seascape_composite_dragon.png')
    parser.add_argument("--start_t_harm", help='default=5, starting T at last scale for harmonization', default=5, type=int)
    parser.add_argument("--start_t_style", help='default=15, starting T at last scale for style transfer', default=15, type=int)

    # 图像嵌入（harmonization）时需要的输入，融合图像的掩码mask
    parser.add_argument("--harm_mask", help='harmonization mask.', default='seascape_mask_dragon.png')

    # Dataset
    parser.add_argument("--dataset_folder", help='choose dataset folder.', default='./datasets/pyramids/')
    parser.add_argument("--image_name", help='choose image name.', default='pyramids.png')
    parser.add_argument("--results_folder", help='choose results folder.', default='./results/')

    # Net
    parser.add_argument("--dim", help='widest channel dimension for conv blocks.', default=160, type=int)

    # diffusion params
    parser.add_argument("--scale_factor", help='downscaling step for each scale.', default=1.411, type=float)

    # 混合精度AMP
    parser.add_argument("--AMP", help='Automatically Mixed Precision, default = False, True/False.', action="store_true")

    # 单尺度训练
    parser.add_argument("--SinScale", help='Enable SinSacle Mode default = False, True/False.', action="store_true")

    # ema_start_step
    parser.add_argument("--step_start_ema", help='start step ema.', default=2000, type=int)
    # training params
    # 总时间步
    parser.add_argument("--timesteps", help='total diffusion timesteps.', default=100, type=int)

    # 训练步数
    parser.add_argument("--train_num_steps", help='total training steps.', default=1, type=int)
    parser.add_argument("--train_batch_size", help='batch size during training.', default=32, type=int)
    parser.add_argument("--grad_accumulate", help='gradient accumulation (bigger batches).', default=1, type=int)
    parser.add_argument("--save_and_sample_every", help='n. steps for checkpointing model.', default=10, type=int)
    parser.add_argument("--avg_window", help='window size for averaging loss (visualization only).', default=100, type=int)
    parser.add_argument("--train_lr", help='starting lr.', default=1e-3, type=float)

    # 学习率在t*1000步后会下降的机制
    parser.add_argument("--sched_k_milestones", nargs="+", help='lr scheduler steps x 1000.',
                        default=[20, 40, 70, 80, 90, 110], type=int)
    parser.add_argument("--load_milestone", help='load specific milestone.', default=0, type=int)

    # sampling params
    parser.add_argument("--sample_batch_size", help='batch size during sampling.', default=4, type=int)
    parser.add_argument("--scale_mul", help='image size retargeting modifier.', nargs="+", default=[1, 1], type=float)
    parser.add_argument("--sample_t_list", nargs="+", help='Custom list of timesteps corresponding to each scale (except scale 0).', type=int)

    # device num
    parser.add_argument("--device_num", help='use specific cuda device.', default=0, type=int)

    # DEV. params - do not modify
    parser.add_argument("--sample_limited_t", help='limit t in each scale to stop at the start of the next scale', action='store_true')
    parser.add_argument("--omega", help='sigma=omega*max_sigma.', default=0, type=float)
    parser.add_argument("--loss_factor", help='ratio between MSE loss and starting diffusion step for each scale.', default=1, type=float)

    args = parser.parse_args()

    print('num devices: '+ str(torch.cuda.device_count()))
    device = f"cuda:{args.device_num}"
    scale_mul = (args.scale_mul[0], args.scale_mul[1])
    sched_milestones = [val * 1000 for val in args.sched_k_milestones]
    results_folder = args.results_folder + '/' + args.scope

    # 保存中间的数据
    save_interm = False

    # rescale_losses 是损失缩放因子，避免误差累计的手段之一（通过缩放大尺度时的图片损失来缓解）
    sizes, rescale_losses, scale_factor, n_scales = muti_scales_img(args.dataset_folder, args.image_name,
                                                                                  scale_factor=args.scale_factor,
                                                                                  create=True,
                                                                                  auto_scale=50000, # limit max number of pixels in image
                                                                                  single_scale=args.SinScale
                                                                                  )

    model = Net(
        dim=args.dim,
        multiscale=not args.SinScale,
        device=device,
    )
    model.to(device)


    diffusion = Diffusion(
        denoise_net=model,
        save_history=save_interm,
        output_folder=results_folder,
        n_scales=n_scales,
        scale_factor=scale_factor,
        image_sizes=sizes,
        scale_mul=scale_mul,
        channels=3,
        timesteps=args.timesteps,
        full_train=True,
        scale_losses=rescale_losses,
        loss_factor=args.loss_factor,
        loss_type='l1',
        betas=None,
        device=device,
        reblurring=True,
        sample_limited_t=args.sample_limited_t,
        omega=args.omega,
    ).to(device)

    if args.sample_t_list is None:
        sample_t_list = diffusion.num_timesteps_ideal[1:]
    else:
        sample_t_list = args.sample_t_list

    ScalerTrainer = MutiScaleTrainer(
        diffusion,
        folder=args.dataset_folder,
        n_scales=n_scales,
        scale_factor=scale_factor,
        image_sizes=sizes,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.grad_accumulate,
        step_start_ema=args.step_start_ema,

        # ema衰减率
        ema_decay=0.9999,

        # 混合精度
        fp16=args.AMP,
        save_and_sample_every=args.save_and_sample_every,
        avg_window=args.avg_window,
        sched_milestones=sched_milestones,
        results_folder=results_folder,
        device=device,
    )

    if args.load_milestone > 0:
        ScalerTrainer.load(milestone=args.load_milestone)
    if args.mode == "train":
        ScalerTrainer.train()
        ScalerTrainer.sample_scales(
            scale_mul=(1, 1),
            custom_sample=True,
            image_name=args.image_name,
            batch_size=args.sample_batch_size,
            custom_t_list=sample_t_list,
        )
    elif args.mode == "sample":
        ScalerTrainer.sample_scales(scale_mul=scale_mul,
                                   custom_sample=True,
                                   image_name=args.image_name,
                                   batch_size=args.sample_batch_size,
                                   custom_t_list=sample_t_list,
                                   save_unbatched=True
                                   )


if __name__ == '__main__':
    main()
    quit()