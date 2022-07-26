import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
# from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    """
    clip이 True이면 픽셀값을 -1~1 구간으로 두고,
    clip이 False이면 그 구간에 있던걸 0~1구간으로 변환
    """
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    # elif beta_schedule == "sigmoid":
    #     betas = np.linspace(-6, 6, num_diffusion_timesteps)
    #     betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        s = 0.008
        steps = num_diffusion_timesteps + 1
        x = torch.linspace(0, num_diffusion_timesteps, steps, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)

    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas




class Diffusion(nn.Module):
    # def __init__(self, args, config, device=None):
    #     self.args = args
    #     self.config = config
    def __init__(self,
        model,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        # ddim_sampling_eta = 1.
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0) # or alphas.cumprod(dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        # alphas_cumprod_prev = torch.cat(
        #     [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        # )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.loss_type = loss_type
        # 이걸 해줘야할지 모르겠네..
        # timesteps, = betas.shape 
        # self.num_timesteps = int(timesteps)
        # assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = 0
        # self.is_ddim_sampling = self.sampling_timesteps < timesteps
        # self.ddim_sampling_eta = ddim_sampling_eta

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        # def set_gpu(gpu_num):
        #     print ('현재 이용가능한 gpu 수: ', torch.cuda.device_count())
        #     device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
        #     torch.cuda.set_device(device)
        #     print (f'현재 할당된 gpu: {print(torch.cuda.get_device_name(gpu_num))} ({torch.cuda.current_device()}번)')

        #     return device 


        self.model_var_type = config.model.var_type # ???
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        #1. set configs
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        
        #2. set data
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        #3. set model
        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        #4. optimizer
        optimizer = get_optimizer(self.config, model.parameters())

        #**TODO**: ema 사용해야하는데, ddpm,ddim 서로 다른듯. 확인필요.
        # if self.config.model.ema:
        #     ema_helper = EMAHelper(mu=self.config.model.ema_rate)
        #     ema_helper.register(model)
        # else:
        #     ema_helper = None

        #0. training
        start_epoch, step = 0, 0
        # if self.args.resume_training:
        #     states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
        #     model.load_state_dict(states[0])

        #     states[1]["param_groups"][0]["eps"] = self.config.optim.eps
        #     optimizer.load_state_dict(states[1])
        #     start_epoch = states[2]
        #     step = states[3]
        #     if self.config.model.ema:
        #         ema_helper.load_state_dict(states[4])
        
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    # def sample_fid(self, model):
    #     config = self.config
    #     img_id = len(glob.glob(f"{self.args.image_folder}/*"))
    #     print(f"starting from image {img_id}")
    #     total_n_samples = 50000
    #     n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

    #     with torch.no_grad():
    #         for _ in tqdm.tqdm(
    #             range(n_rounds), desc="Generating image samples for FID evaluation."
    #         ):
    #             n = config.sampling.batch_size
    #             x = torch.randn(
    #                 n,
    #                 config.data.channels,
    #                 config.data.image_size,
    #                 config.data.image_size,
    #                 device=self.device,
    #             )

    #             x = self.sample_image(x, model)
    #             x = inverse_data_transform(config, x)

    #             for i in range(n):
    #                 tvu.save_image(
    #                     x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
    #                 )
    #                 img_id += 1

    # def sample_sequence(self, model):
    #     config = self.config

    #     x = torch.randn(
    #         8,
    #         config.data.channels,
    #         config.data.image_size,
    #         config.data.image_size,
    #         device=self.device,
    #     )

    #     # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
    #     with torch.no_grad():
    #         _, x = self.sample_image(x, model, last=False)

    #     x = [inverse_data_transform(config, y) for y in x]

    #     for i in range(len(x)):
    #         for j in range(x[i].size(0)):
    #             tvu.save_image(
    #                 x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
    #             )

    # def sample_interpolation(self, model):
    #     config = self.config

    #     def slerp(z1, z2, alpha):
    #         theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    #         return (
    #             torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
    #             + torch.sin(alpha * theta) / torch.sin(theta) * z2
    #         )

    #     z1 = torch.randn(
    #         1,
    #         config.data.channels,
    #         config.data.image_size,
    #         config.data.image_size,
    #         device=self.device,
    #     )
    #     z2 = torch.randn(
    #         1,
    #         config.data.channels,
    #         config.data.image_size,
    #         config.data.image_size,
    #         device=self.device,
    #     )
    #     alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
    #     z_ = []
    #     for i in range(alpha.size(0)):
    #         z_.append(slerp(z1, z2, alpha[i]))

    #     x = torch.cat(z_, dim=0)
    #     xs = []

    #     # Hard coded here, modify to your preferences
    #     with torch.no_grad():
    #         for i in range(0, x.size(0), 8):
    #             xs.append(self.sample_image(x[i : i + 8], model))
    #     x = inverse_data_transform(config, torch.cat(xs, dim=0))
    #     for i in range(x.size(0)):
    #         tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    # def sample_image(self, x, model, last=True):
    #     try:
    #         skip = self.args.skip
    #     except Exception:
    #         skip = 1

    #     if self.args.sample_type == "generalized":
    #         if self.args.skip_type == "uniform":
    #             skip = self.num_timesteps // self.args.timesteps
    #             seq = range(0, self.num_timesteps, skip)
    #         elif self.args.skip_type == "quad":
    #             seq = (
    #                 np.linspace(
    #                     0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
    #                 )
    #                 ** 2
    #             )
    #             seq = [int(s) for s in list(seq)]
    #         else:
    #             raise NotImplementedError
    #         from functions.denoising import generalized_steps

    #         xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
    #         x = xs
    #     elif self.args.sample_type == "ddpm_noisy":
    #         if self.args.skip_type == "uniform":
    #             skip = self.num_timesteps // self.args.timesteps
    #             seq = range(0, self.num_timesteps, skip)
    #         elif self.args.skip_type == "quad":
    #             seq = (
    #                 np.linspace(
    #                     0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
    #                 )
    #                 ** 2
    #             )
    #             seq = [int(s) for s in list(seq)]
    #         else:
    #             raise NotImplementedError
    #         from functions.denoising import ddpm_steps

    #         x = ddpm_steps(x, seq, model, self.betas)
    #     else:
    #         raise NotImplementedError
    #     if last:
    #         x = x[0][-1]
    #     return x

    # def test(self):
    #     pass