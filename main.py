import torch
import numpy as np
import argparse
import os
import torchvision
from fsspec.registry import default

from LGDiffusion.Functions import muti_scales_img
from LGDiffusion.Model import Net, Diffusion
from LGDiffusion.Trainer import MutiScaleTrainer
from text2live_util.clip_extractor import ClipExtractor
# python main.py  --mode train --timesteps 10 --train_num_steps 10 --avg_window 1 --save_and_sample_every 5 --AMP --SinScale --step_start_ema 10
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--scope", help='choose training mark num.', default='pyramids')
    parser.add_argument("--mode", help='choose mode: train, sample, clip_content, clip_style_gen, clip_style_trans, clip_roi, harmonization, style_transfer, roi')
    parser.add_argument("--input_image", help='content image for style transfer or harmonization.',
                        default='seascape_composite_dragon.png')
    parser.add_argument("--start_t_harm", help='default=5, starting T at last scale for harmonization', default=5, type=int)
    parser.add_argument("--start_t_style", help='default=15, starting T at last scale for style transfer', default=15, type=int)

    # 图像嵌入（harmonization）时需要的输入，融合图像的掩码mask
    parser.add_argument("--harm_mask", help='harmonization mask.', default='seascape_mask_dragon.png')

    # CLIP指导时需要输入的参数。可删除，只保留ROI
    parser.add_argument("--clip_text", help='enter CLIP text.', default='Fire in the Forest')
    parser.add_argument("--fill_factor",
                        help='Dictates relative amount of pixels to be changed. Should be between 0 and 1.', type=float)
    parser.add_argument("--strength",
                        help='Dictates the relative strength of CLIPs gradients. Should be between 0 and 1.',
                        type=float)
    parser.add_argument("--roi_n_tar", help='Defines the number of target ROIs in the new image.', default=1, type=int)

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

# CLIP系列
    elif args.mode == 'clip_content':
        text_input = args.clip_text
        clip_cfg = {"clip_model_name": "ViT-B/32",
                    "clip_affine_transform_fill": True,
                    "n_aug": 16}
        t2l_clip_extractor = ClipExtractor(clip_cfg)
        clip_custom_t_list = sample_t_list

        guidance_sub_iters = [0]
        for i in range(n_scales - 1):
            guidance_sub_iters.append(1)

        assert args.strength is not None and 0 <= args.strength <= 1, f"Strength value should be between 0 & 1. Got: {args.strength} "
        assert args.fill_factor is not None and 0 <= args.fill_factor <= 1, f"fill_factor value should be between 0 & 1. Got: {args.fill_factor} "
        strength = args.strength
        quantile = 1. - args.fill_factor

        llambda = 0.2
        stop_guidance = 3  # at the last scale, disable the guidance in the last x steps in order to avoid artifacts from CLIP
        ScalerTrainer.ema_model.reblurring = False
        ScalerTrainer.clip_sampling(clip_model=t2l_clip_extractor,
                                   text_input=text_input,
                                   strength=strength,
                                   sample_batch_size=args.sample_batch_size,
                                   custom_t_list=clip_custom_t_list,
                                   quantile=quantile,
                                   guidance_sub_iters=guidance_sub_iters,
                                   stop_guidance=stop_guidance,
                                   save_unbatched=True,
                                   scale_mul=scale_mul,
                                   llambda=llambda
                                   )

    elif args.mode == 'clip_style_trans' or args.mode == 'clip_style_gen':
        # CLIP
        text_input = args.clip_text + ' Style'
        clip_cfg = {"clip_model_name": "ViT-B/32",
                    "clip_affine_transform_fill": True,
                    "n_aug": 16}
        t2l_clip_extractor = ClipExtractor(clip_cfg)
        clip_custom_t_list = sample_t_list

        guidance_sub_iters = []
        for i in range(n_scales-1):
            guidance_sub_iters.append(0)
        guidance_sub_iters.append(1)

        strength = 0.3
        quantile = 0.0
        llambda = 0.05
        stop_guidance = 3
        if args.mode == 'clip_style_gen':
            start_noise = True
        else:  # mode == 'clip_style_trans':
            start_noise = False
        image_name = args.image_name.rsplit( ".", 1 )[ 0 ] + '.png'
        ScalerTrainer.ema_model.reblurring = False
        ScalerTrainer.clip_sampling(clip_model=t2l_clip_extractor,
                                   text_input=text_input,
                                   strength=strength,
                                   sample_batch_size=args.sample_batch_size,
                                   custom_t_list=clip_custom_t_list,
                                   quantile=quantile,
                                   guidance_sub_iters=guidance_sub_iters,
                                   stop_guidance=stop_guidance,
                                   save_unbatched=True,
                                   scale_mul=scale_mul,
                                   llambda=llambda,
                                   start_noise=start_noise,
                                   image_name=image_name,
                                   )

    elif args.mode == 'clip_roi':
        # CLIP_ROI
        text_input = args.clip_text
        clip_cfg = {"clip_model_name": "ViT-B/32",
                    "clip_affine_transform_fill": True,
                    "n_aug": 16}
        t2l_clip_extractor = ClipExtractor(clip_cfg)
        strength = 0.1
        num_clip_iters = 100
        num_denoising_steps = 3
        # select from the finest scale
        dataset_folder = os.path.join(args.dataset_folder, f'scale_{n_scales - 1}/')
        image_name = args.image_name.rsplit(".", 1)[0] + '.png'
        import cv2
        image_to_select = cv2.imread(dataset_folder+image_name)
        roi = cv2.selectROI(image_to_select)
        roi_perm = [1, 0, 3, 2]
        roi = [roi[i] for i in roi_perm]
        ScalerTrainer.ema_model.reblurring = False
        ScalerTrainer.clip_roi_sampling(clip_model=t2l_clip_extractor,
                                       text_input=text_input,
                                       strength=strength,
                                       sample_batch_size=args.sample_batch_size,
                                       num_clip_iters=num_clip_iters,
                                       num_denoising_steps=num_denoising_steps,
                                       clip_roi_bb=roi, #[90,75,50,50],
                                       save_unbatched=True,
                                       )


    elif args.mode == 'roi':
        import cv2
        image_path = os.path.join(args.dataset_folder, f'scale_{n_scales - 1}', args.image_name.rsplit(".", 1)[0] + '.png')
        image_to_select = cv2.imread(image_path)
        roi = cv2.selectROI(image_to_select)
        image_to_select = cv2.cvtColor(image_to_select, cv2.COLOR_BGR2RGB)
        roi_perm = [1, 0, 3, 2]
        target_roi = [roi[i] for i in roi_perm]
        tar_y, tar_x, tar_h, tar_w = target_roi
        roi_bb_list = []
        n_targets = args.roi_n_tar  # number of target patches
        target_h = int(image_to_select.shape[0] * scale_mul[0])
        target_w = int(image_to_select.shape[1] * scale_mul[1])
        empty_image = np.ones((target_h, target_w, 3))
        target_patch_tensor = torchvision.transforms.ToTensor()(
            image_to_select[tar_y:tar_y + tar_h, tar_x:tar_x + tar_w, :])

        for i in range(n_targets):
            roi = cv2.selectROI(empty_image)
            roi_reordered = [roi[i] for i in roi_perm]
            roi_bb_list.append(roi_reordered)
            y, x, h, w = roi_reordered
            target_patch_tensor_resize = torch.nn.functional.interpolate(target_patch_tensor[None, :, :, :],
                                                                         size=(h, w))
            empty_image[y:y + h, x:x + w, :] = target_patch_tensor_resize[0].permute(1, 2, 0).numpy()

        empty_image = torchvision.transforms.ToTensor()(empty_image)
        torchvision.utils.save_image(empty_image, os.path.join(args.results_folder, args.scope, f'roi_patches.png'))

        ScalerTrainer.roi_sampling(custom_t_list=sample_t_list,
                                         target_roi=target_roi,
                                         roi_bb_list=roi_bb_list,
                                         save_unbatched=True,
                                         batch_size=args.sample_batch_size,
                                         scale_mul=scale_mul)

    elif args.mode == 'style_transfer' or args.mode == 'harmonization':

        i2i_folder = os.path.join(args.dataset_folder, 'i2i')
        if args.mode == 'style_transfer':
            # start diffusion from last scale
            start_s = n_scales - 1
            # start diffusion from t - increase for stronger prior on the original image
            start_t = args.start_t_style
            use_hist = True
        else:
            # start diffusion from last scale
            start_s = n_scales - 1
            # start diffusion from t - increase for stronger prior on the original image
            start_t = args.start_t_harm
            use_hist = False
        custom_t = []
        for i in range(n_scales-1):
            custom_t.append(0)
        custom_t.append(start_t)
        # use the histogram of the original image for histogram matching
        hist_ref_path = f'{args.dataset_folder}scale_{start_s}/'

        ScalerTrainer.ema_model.reblurring = True
        ScalerTrainer.imgtoimg(input_folder=i2i_folder, input_file=args.input_image, mask=args.harm_mask, hist_ref_path=hist_ref_path,
                                 batch_size=args.sample_batch_size,
                                 image_name=args.image_name, start_s=start_s, custom_t=custom_t, scale_mul=(1, 1),
                                 device=device, use_hist=use_hist, save_unbatched=True, auto_scale=50000, mode=args.mode)
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    main()
    quit()