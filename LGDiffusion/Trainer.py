import copy
import os
import datetime
from functools import partial

from LGDiffusion.Functions import *
from LGDiffusion.Model import EMA

from torch.utils import data
from torchvision import transforms, utils
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from matplotlib import pyplot as plt
from skimage.exposure import match_histograms
from text2live_util.util import get_augmentations_template
from tqdm import tqdm

# from Functions import AMP

# # 待更新
# try:
#     from torch.cuda import amp
#
#     AMP_AVAILABLE = True
# except:
#     AMP_AVAILABLE = False


# 数据集class， datasets
class Dataset(data.Dataset):
    def __init__(self, folder, image_size, blurry_img=False, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.blurry_img = blurry_img
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        if blurry_img:
            self.folder_recon = folder + '_upsample/'
            self.paths_recon = [p for ext in exts for p in Path(f'{self.folder_recon}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # 归一化到[-1, 1]
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        # *128是手动扩大样本数，以应对单图生成任务的挑战之一，单图
        return len(self.paths) * 128

    def __getitem__(self, index):
        path = self.paths[0]
        img = Image.open(path).convert('RGB')
        if self.blurry_img:
            path_recon = self.paths_recon[0]
            img_recon = Image.open(path_recon).convert('RGB')
            return self.transform(img), self.transform(img_recon)
        return self.transform(img)

class MutiScaleTrainer(object):
    def __init__(
            self,
            ms_diffusion_model,
            folder,
            *,
            ema_decay=0.9999,
            n_scales=None,
            scale_factor=1,
            image_sizes=None,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=25000,
            avg_window=100,
            sched_milestones=None,
            results_folder='./results',
            device=None
    ):
        super().__init__()
        self.device = device
        if sched_milestones is None:
            self.sched_milestones = [10000, 30000, 60000, 80000, 90000]
        else:
            self.sched_milestones = sched_milestones
        if image_sizes is None:
            image_sizes = []
        self.model = ms_diffusion_model.to(device)
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.avg_window = avg_window

        self.batch_size = train_batch_size
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.input_paths = []
        self.ds_list = []
        self.dl_list = []
        self.data_list = []
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        # 为每个尺度初始化数据加载器
        for i in range(n_scales):
            self.input_paths.append(folder + 'scale_' + str(i))
            blurry_img = True if i > 0 else False
            self.ds_list.append(Dataset(self.input_paths[i], image_sizes[i], blurry_img))
            self.dl_list.append(
                dataloader(data.DataLoader(self.ds_list[i], batch_size=train_batch_size, shuffle=True, pin_memory=True)))

            if i > 0:
                Data = next(self.dl_list[i])
                self.data_list.append((Data[0].to(self.device), Data[1].to(self.device)))
            else:
                self.data_list.append(
                    (next(self.dl_list[i]).to(self.device), next(self.dl_list[i]).to(self.device)))

        self.fp16 = fp16
        self.amp = AMP(self.model, self.fp16)
        self.opt = Adam(ms_diffusion_model.parameters(), lr=train_lr)

        self.scheduler = MultiStepLR(self.opt, milestones=self.sched_milestones, gamma=0.5)

        self.step = 0
        self.running_loss = []
        self.running_scale = []
        self.avg_t = []

        # 待更新
        # assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'
        # if fp16:
        #     (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt,
        #                                                             opt_level='O1')

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'sched': self.scheduler.state_dict(),
            'running_loss': self.running_loss,
            'running_scale': self.running_scale
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        plt.rcParams['figure.figsize'] = [16, 8]

        plt.plot(self.running_loss)
        plt.grid(True)
        # plt.ylim((0, 0.2))
        plt.savefig(str(self.results_folder / 'running_loss'))
        plt.clf()

        np.save(str(self.results_folder / 'running_loss.npy'), data['running_loss'])

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.scheduler.load_state_dict(data['sched'])
        self.running_loss = data['running_loss']


    def train(self):
        # backwards = partial(loss_backwards, self.fp16)
        loss_avg = 0
        s_weights = torch.tensor(self.model.num_timesteps_trained, device=self.device, dtype=torch.float)

        # 主循环
        while self.step < self.train_num_steps:
            s = torch.multinomial(input=s_weights, num_samples=1)
            for i in range(self.gradient_accumulate_every):
                data = self.data_list[s]
                with self.amp.autocast():
                    loss = self.model(data, s)
                # loss = self.model(data, s)
                loss_avg = loss_avg + loss.item()
                # backwards(loss / self.gradient_accumulate_every, self.opt)
                self.amp.backward(loss, self.opt, accumulate_grad=self.gradient_accumulate_every)


            if self.step % self.avg_window == 0:
                print(f'step:{self.step} loss:{loss_avg/self.avg_window}')
                self.running_loss.append(loss_avg/self.avg_window)
                loss_avg = 0

            self.amp.step(self.opt)
            self.opt.zero_grad()
            # self.opt.step()
            # self.opt.zero_grad()

            # 更新EMA
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            self.scheduler.step()
            self.step = self.step + 1

            # check point
            if self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = spilt_groups(16, self.batch_size)
                all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=4)
                self.save(milestone)
        print('training completed')

    # 多尺度采样pipeline
    def sample_scales(
            self,
            scale_mul=None,
            batch_size=16,
            custom_sample=False,
            custom_image_size_idxs=None,
            custom_scales=None,
            image_name='',
            start_noise=True,
            custom_t_list=None,
            desc=None,
            save_unbatched=True
    ):
        if desc is None:
            desc = f'sample_{str(datetime.datetime.now()).replace(":", "_")}'
        if self.n_scales == 1 and (custom_t_list is None or len(custom_t_list) == 0):
            custom_t_list = []
        if self.ema_model.reblurring:
            desc = desc + '_rblr'
        if self.ema_model.sample_limited_t:
            desc = desc + '_t_lmtd'
        if custom_t_list is None:
            custom_t_list = self.ema_model.num_timesteps_ideal[1:]
        if custom_scales is None:
            custom_scales = [*range(self.n_scales)]
            n_scales = self.n_scales
        else:
            n_scales = len(custom_scales)
        if custom_image_size_idxs is None:
            custom_image_size_idxs = [*range(self.n_scales)]

        samples_from_scales = []
        final_results_folder = Path(str(self.results_folder / 'final_samples'))
        final_results_folder.mkdir(parents=True, exist_ok=True)
        if scale_mul is not None:
            scale_0_size = (
                int(self.model.image_sizes[custom_image_size_idxs[0]][0] * scale_mul[0]),
                int(self.model.image_sizes[custom_image_size_idxs[0]][1] * scale_mul[1]))
        else:
            scale_0_size = None
        t_list = [self.ema_model.num_timesteps_trained[0]] + custom_t_list
        res_sub_folder = '_'.join(str(e) for e in t_list)
        final_img = None
        for i in range(n_scales):
            if start_noise and i == 0:
                samples_from_scales.append(
                    self.ema_model.sample(batch_size=batch_size, scale_0_size=scale_0_size, s=custom_scales[i]))

            # 如果start_noise = False, 那么将图片作为scale_0的输入来进行采样
            elif i == 0:
                orig_sample_0 = Image.open((self.input_paths[custom_scales[i]] + '/' + image_name)).convert("RGB")

                samples_from_scales.append((transforms.ToTensor()(orig_sample_0) * 2 - 1).repeat(batch_size, 1, 1, 1).to(self.device))

            else:
                samples_from_scales.append(self.ema_model.sample_via_scale(batch_size,
                                                                           samples_from_scales[i - 1],
                                                                           s=custom_scales[i],
                                                                           scale_mul=scale_mul,
                                                                           custom_sample=custom_sample,
                                                                           custom_img_size_idx=custom_image_size_idxs[i],
                                                                           custom_t=custom_t_list[int(custom_scales[i])-1],
                                                                           ))
            final_img = (samples_from_scales[i] + 1) * 0.5

            utils.save_image(final_img, str(final_results_folder / res_sub_folder) + f'_out_s{i}_{desc}_sm_{scale_mul[0]}_{scale_mul[1]}.png', nrow=4)

        if save_unbatched:
            final_results_folder = Path(str(self.results_folder / f'final_samples_unbatched_{desc}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            for b in range(batch_size):
                utils.save_image(final_img[b], str(final_results_folder / res_sub_folder) + f'_out_b{b}.png')
