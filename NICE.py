import itertools
import time
from glob import glob
from pathlib import Path

import torch_fidelity
# import torch.utils.tensorboard as tensorboardX
from thop import clever_format, profile
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageFolder
from networks import *
from seed import get_seeded_generator, seed_everything, seeded_worker_init_fn
from utils import *


class NICE(object):
  def __init__(self, args):
    self.light = args.light

    if self.light:
      self.model_name = 'NICE_light'
    else:
      self.model_name = 'NICE'

    self.result_dir = args.result_dir
    self.dataset = args.dataset

    self.iteration = args.iteration
    self.decay_flag = args.decay_flag

    self.batch_size = args.batch_size
    self.print_freq = args.print_freq
    self.val_freq = args.val_freq
    self.save_freq = args.save_freq

    self.lr = args.lr
    self.weight_decay = args.weight_decay
    self.ch = args.ch

    """ Weight """
    self.adv_weight = args.adv_weight
    self.cycle_weight = args.cycle_weight
    self.recon_weight = args.recon_weight

    """ Generator """
    self.n_res = args.n_res

    """ Discriminator """
    self.n_dis = args.n_dis

    self.img_size = args.img_size
    self.img_ch = args.img_ch

    self.device = args.device
    self.benchmark_flag = args.benchmark_flag
    self.resume = args.resume
    self.seed = args.seed
    self.ckpt = args.ckpt

    self.start_iter = 1
    self.smallest_val_fid = float('inf')
    self.fid = 1000
    self.fid_A = 1000
    self.fid_B = 1000
    if torch.backends.cudnn.enabled and self.benchmark_flag:
      print('set benchmark !')
      torch.backends.cudnn.benchmark = True

    print()

    print("##### Information #####")
    print("# light : ", self.light)
    print("# dataset : ", self.dataset)
    print("# batch_size : ", self.batch_size)
    print("# iteration per epoch : ", self.iteration)
    print("# the size of image : ", self.img_size)
    print("# the size of image channel : ", self.img_ch)
    print("# base channel number per layer : ", self.ch)

    print()

    print("##### Generator #####")
    print("# residual blocks : ", self.n_res)

    print()

    print("##### Discriminator #####")
    print("# discriminator layers : ", self.n_dis)

    print()

    print("##### Weight #####")
    print("# adv_weight : ", self.adv_weight)
    print("# cycle_weight : ", self.cycle_weight)
    print("# recon_weight : ", self.recon_weight)

  ##################################################################################
  # Model
  ##################################################################################

  def build_model(self):
    """ Seed everything """
    seed_everything(self.seed)
    trainA_dataloader_generator = get_seeded_generator(self.seed)
    trainB_dataloader_generator = get_seeded_generator(self.seed)

    """ DataLoader """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((self.img_size + 30, self.img_size+30)),
        transforms.RandomCrop(self.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((self.img_size, self.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    self.trainA = ImageFolder(os.path.join(
        'dataset', self.dataset, 'trainA'), train_transform)
    self.trainB = ImageFolder(os.path.join(
        'dataset', self.dataset, 'trainB'), train_transform)
    self.valA = ImageFolder(os.path.join(
        'dataset', self.dataset, 'valA'), test_transform
    )
    self.valB = ImageFolder(os.path.join(
        'dataset', self.dataset, 'valB'), test_transform
    )
    self.testA = ImageFolder(os.path.join(
        'dataset', self.dataset, 'testA'), test_transform)
    self.testB = ImageFolder(os.path.join(
        'dataset', self.dataset, 'testB'), test_transform)
    self.trainA_loader = DataLoader(
        self.trainA, batch_size=self.batch_size, shuffle=True, pin_memory=True,
        generator=trainA_dataloader_generator,)
    self.trainB_loader = DataLoader(
        self.trainB, batch_size=self.batch_size, shuffle=True, pin_memory=True,
        generator=trainB_dataloader_generator,)
    self.valA_loader = DataLoader(
        self.valA, batch_size=1, shuffle=False, pin_memory=True)
    self.valB_loader = DataLoader(
        self.valB, batch_size=1, shuffle=False, pin_memory=True)
    self.testA_loader = DataLoader(
        self.testA, batch_size=1, shuffle=False, pin_memory=True)
    self.testB_loader = DataLoader(
        self.testB, batch_size=1, shuffle=False, pin_memory=True)

    """ Define Generator, Discriminator """
    self.gen2B = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch,
                                 n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
    self.gen2A = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch,
                                 n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
    self.disA = Discriminator(
        input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)
    self.disB = Discriminator(
        input_nc=self.img_ch, ndf=self.ch, n_layers=self.n_dis).to(self.device)

    print('-----------------------------------------------')
    input = torch.randn([1, self.img_ch, self.img_size,
                        self.img_size]).to(self.device)
    macs, params = profile(self.disA, inputs=(input, ))
    macs, params = clever_format([macs*2, params*2], "%.3f")
    print('[Network %s] Total number of parameters: ' % 'disA', params)
    print('[Network %s] Total number of FLOPs: ' % 'disA', macs)
    print('-----------------------------------------------')
    _, _, _,  _, real_A_ae = self.disA(input)
    macs, params = profile(self.gen2B, inputs=(real_A_ae, ))
    macs, params = clever_format([macs*2, params*2], "%.3f")
    print('[Network %s] Total number of parameters: ' % 'gen2B', params)
    print('[Network %s] Total number of FLOPs: ' % 'gen2B', macs)
    print('-----------------------------------------------')

    """ Define Loss """
    self.L1_loss = nn.L1Loss().to(self.device)
    self.MSE_loss = nn.MSELoss().to(self.device)

    """ Trainer """
    self.G_optim = torch.optim.Adam(itertools.chain(self.gen2B.parameters(
    ), self.gen2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
    self.D_optim = torch.optim.Adam(itertools.chain(self.disA.parameters(
    ), self.disB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

  def train(self):
    # writer = tensorboardX.SummaryWriter(os.path.join(self.result_dir, self.dataset, 'summaries/Allothers'))
    self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()

    self.start_iter = 1
    if self.resume:
      params = self.load('params_latest')
      self.start_iter = params['start_iter']+1
      if self.decay_flag and self.start_iter > (self.iteration // 2):
        self.G_optim.param_groups[0]['lr'] -= (self.lr / (
            self.iteration // 2)) * (self.start_iter - self.iteration // 2)
        self.D_optim.param_groups[0]['lr'] -= (self.lr / (
            self.iteration // 2)) * (self.start_iter - self.iteration // 2)
      print("ok")

    # training loop
    testnum = 4
    # for step in range(1, self.start_iter):
    #   if step % self.print_freq == 0:
    #     for _ in range(testnum):
    #       try:
    #         real_A, _ = testA_iter.next()
    #       except:
    #         testA_iter = iter(self.testA_loader)
    #         real_A, _ = testA_iter.next()

    #       try:
    #         real_B, _ = testB_iter.next()
    #       except:
    #         testB_iter = iter(self.testB_loader)
    #         real_B, _ = testB_iter.next()

    print("self.start_iter", self.start_iter)
    print('training start !')
    start_time = time.time()

    trainA_iter = iter(self.trainA_loader)
    trainB_iter = iter(self.trainB_loader)

    for step in range(self.start_iter, self.iteration + 1):
      if self.decay_flag and step > (self.iteration // 2):
        self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
        self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

      try:
        real_A, _ = next(trainA_iter)
      except:
        trainA_iter = iter(self.trainA_loader)
        real_A, _ = next(trainA_iter)

      try:
        real_B, _ = next(trainB_iter)
      except:
        trainB_iter = iter(self.trainB_loader)
        real_B, _ = next(trainB_iter)

      real_A, real_B = real_A.to(self.device), real_B.to(self.device)

      # Update D
      self.D_optim.zero_grad()

      real_LA_logit, real_GA_logit, real_A_cam_logit, _, real_A_z = self.disA(
          real_A)
      real_LB_logit, real_GB_logit, real_B_cam_logit, _, real_B_z = self.disB(
          real_B)

      fake_A2B = self.gen2B(real_A_z)
      fake_B2A = self.gen2A(real_B_z)

      fake_B2A = fake_B2A.detach()
      fake_A2B = fake_A2B.detach()

      fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, _ = self.disA(fake_B2A)
      fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.disB(fake_A2B)

      D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(
          self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
      D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(
          self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
      D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(
          self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
      D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(
          self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
      D_ad_cam_loss_A = self.MSE_loss(real_A_cam_logit, torch.ones_like(real_A_cam_logit).to(
          self.device)) + self.MSE_loss(fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit).to(self.device))
      D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit).to(
          self.device)) + self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).to(self.device))

      D_loss_A = self.adv_weight * \
          (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
      D_loss_B = self.adv_weight * \
          (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

      Discriminator_loss = D_loss_A + D_loss_B
      Discriminator_loss.backward()
      self.D_optim.step()
      # writer.add_scalar('D/%s' % 'loss_A', D_loss_A.data.cpu().numpy(), global_step=step)
      # writer.add_scalar('D/%s' % 'loss_B', D_loss_B.data.cpu().numpy(), global_step=step)

      # Update G
      self.G_optim.zero_grad()

      _,  _,  _, _, real_A_z = self.disA(real_A)
      _,  _,  _, _, real_B_z = self.disB(real_B)

      fake_A2B = self.gen2B(real_A_z)
      fake_B2A = self.gen2A(real_B_z)

      fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, fake_A_z = self.disA(
          fake_B2A)
      fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, fake_B_z = self.disB(
          fake_A2B)

      fake_B2A2B = self.gen2B(fake_A_z)
      fake_A2B2A = self.gen2A(fake_B_z)

      G_ad_loss_GA = self.MSE_loss(
          fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
      G_ad_loss_LA = self.MSE_loss(
          fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
      G_ad_loss_GB = self.MSE_loss(
          fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
      G_ad_loss_LB = self.MSE_loss(
          fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

      G_ad_cam_loss_A = self.MSE_loss(
          fake_A_cam_logit, torch.ones_like(fake_A_cam_logit).to(self.device))
      G_ad_cam_loss_B = self.MSE_loss(
          fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).to(self.device))

      G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A)
      G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B)

      fake_A2A = self.gen2A(real_A_z)
      fake_B2B = self.gen2B(real_B_z)

      G_recon_loss_A = self.L1_loss(fake_A2A, real_A)
      G_recon_loss_B = self.L1_loss(fake_B2B, real_B)

      G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA) + \
          self.cycle_weight * G_cycle_loss_A + self.recon_weight * G_recon_loss_A
      G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB) + \
          self.cycle_weight * G_cycle_loss_B + self.recon_weight * G_recon_loss_B

      Generator_loss = G_loss_A + G_loss_B
      Generator_loss.backward()
      self.G_optim.step()
      # writer.add_scalar('G/%s' % 'loss_A', G_loss_A.data.cpu().numpy(), global_step=step)
      # writer.add_scalar('G/%s' % 'loss_B', G_loss_B.data.cpu().numpy(), global_step=step)

      print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step,
            self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))

      # for name, param in self.gen2B.named_parameters():
      #     writer.add_histogram(name + "_gen2B", param.data.cpu().numpy(), global_step=step)

      # for name, param in self.gen2A.named_parameters():
      #     writer.add_histogram(name + "_gen2A", param.data.cpu().numpy(), global_step=step)

      # for name, param in self.disA.named_parameters():
      #     writer.add_histogram(name + "_disA", param.data.cpu().numpy(), global_step=step)

      # for name, param in self.disB.named_parameters():
      #     writer.add_histogram(name + "_disB", param.data.cpu().numpy(), global_step=step)

      if step % self.save_freq == 0:
        self.save(ckpt_file_name='params_%07d.pt' % step, step=step)

      if step % self.print_freq == 0:
        print('current D_learning rate:{}'.format(
            self.D_optim.param_groups[0]['lr']))
        print('current G_learning rate:{}'.format(
            self.G_optim.param_groups[0]['lr']))
        self.save(ckpt_file_name='params_latest.pt', step=step)

      if step % self.val_freq == 0:
        self.val(step)

      if step % self.print_freq == 0:
        train_sample_num = testnum
        test_sample_num = testnum
        A2B = np.zeros((self.img_size * 5, 0, 3))
        B2A = np.zeros((self.img_size * 5, 0, 3))

        self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()

        self.gen2B, self.gen2A = self.gen2B.to('cpu'), self.gen2A.to('cpu')
        self.disA, self.disB = self.disA.to('cpu'), self.disB.to('cpu')
        for _ in range(train_sample_num):
          try:
            real_A, _ = next(trainA_iter)
          except:
            trainA_iter = iter(self.trainA_loader)
            real_A, _ = next(trainA_iter)

          try:
            real_B, _ = next(trainB_iter)
          except:
            trainB_iter = iter(self.trainB_loader)
            real_B, _ = next(trainB_iter)
          real_A, real_B = real_A.to('cpu'), real_B.to('cpu')
          # real_A, real_B = real_A.to(self.device), real_B.to(self.device)

          _, _,  _, A_heatmap, real_A_z = self.disA(real_A)
          _, _,  _, B_heatmap, real_B_z = self.disB(real_B)

          fake_A2B = self.gen2B(real_A_z)
          fake_B2A = self.gen2A(real_B_z)

          _, _,  _,  _,  fake_A_z = self.disA(fake_B2A)
          _, _,  _,  _,  fake_B_z = self.disB(fake_A2B)

          fake_B2A2B = self.gen2B(fake_A_z)
          fake_A2B2A = self.gen2A(fake_B_z)

          fake_A2A = self.gen2A(real_A_z)
          fake_B2B = self.gen2B(real_B_z)

          A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                     cam(tensor2numpy(
                                                         A_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_A2A[0]))),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_A2B[0]))),
                                                     RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

          B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                     cam(tensor2numpy(
                                                         B_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_B2B[0]))),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_B2A[0]))),
                                                     RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

        for _ in range(test_sample_num):
          try:
            real_A, _ = next(testA_iter)
          except:
            testA_iter = iter(self.testA_loader)
            real_A, _ = next(testA_iter)

          try:
            real_B, _ = next(testB_iter)
          except:
            testB_iter = iter(self.testB_loader)
            real_B, _ = next(testB_iter)
          real_A, real_B = real_A.to('cpu'), real_B.to('cpu')
          # real_A, real_B = real_A.to(self.device), real_B.to(self.device)

          _, _,  _, A_heatmap, real_A_z = self.disA(real_A)
          _, _,  _, B_heatmap, real_B_z = self.disB(real_B)

          fake_A2B = self.gen2B(real_A_z)
          fake_B2A = self.gen2A(real_B_z)

          _, _,  _,  _,  fake_A_z = self.disA(fake_B2A)
          _, _,  _,  _,  fake_B_z = self.disB(fake_A2B)

          fake_B2A2B = self.gen2B(fake_A_z)
          fake_A2B2A = self.gen2A(fake_B_z)

          fake_A2A = self.gen2A(real_A_z)
          fake_B2B = self.gen2B(real_B_z)

          A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                     cam(tensor2numpy(
                                                         A_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_A2A[0]))),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_A2B[0]))),
                                                     RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

          B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                     cam(tensor2numpy(
                                                         B_heatmap[0]), self.img_size),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_B2B[0]))),
                                                     RGB2BGR(tensor2numpy(
                                                         denorm(fake_B2A[0]))),
                                                     RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

        cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                    'img', 'A2B_%07d.png' % step), A2B * 255.0)
        cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                    'img', 'B2A_%07d.png' % step), B2A * 255.0)

        self.gen2B, self.gen2A = self.gen2B.to(
            self.device), self.gen2A.to(self.device)
        self.disA, self.disB = self.disA.to(
            self.device), self.disB.to(self.device)

        self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()

  def save(self, ckpt_file_name: str, step: int):
    params = {}
    params['gen2B'] = self.gen2B.state_dict()
    params['gen2A'] = self.gen2A.state_dict()
    params['disA'] = self.disA.state_dict()
    params['disB'] = self.disB.state_dict()
    params['D_optimizer'] = self.D_optim.state_dict()
    params['G_optimizer'] = self.G_optim.state_dict()
    params['start_iter'] = step

    torch.save(
        params,
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            ckpt_file_name
        )
    )

  def load(self, cktpt_file_name: str):
    params = torch.load(
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            cktpt_file_name
        )
    )
    self.gen2B.load_state_dict(params['gen2B'])
    self.gen2A.load_state_dict(params['gen2A'])
    self.disA.load_state_dict(params['disA'])
    self.disB.load_state_dict(params['disB'])
    self.D_optim.load_state_dict(params['D_optimizer'])
    self.G_optim.load_state_dict(params['G_optimizer'])
    self.start_iter = params['start_iter']

    return params

  def val(self, step: int):

    model_val_translations_dir = Path(self.result_dir, self.dataset, 'val')
    if not os.path.exists(model_val_translations_dir):
      os.mkdir(model_val_translations_dir)

    model_with_step_translations_dir = Path(
        model_val_translations_dir,
        'params_%07d' % step
    )

    if not os.path.exists(model_with_step_translations_dir):
      os.mkdir(model_with_step_translations_dir)

    self.gen2B.eval()
    self.disA.eval()

    print('translating val...')
    for n, (real_A, _) in enumerate(self.valA_loader):
      real_A = real_A.to(self.device)
      img_path, _ = self.valA_loader.dataset.samples[n]
      img_name = Path(img_path).name.split('.')[0]
      _, _,  _, _, real_A_z = self.disA(real_A)
      fake_A2B = self.gen2B(real_A_z)
      cv2.imwrite(
          os.path.join(model_with_step_translations_dir,
                       f'{img_name}_fake_B.jpg'),
          RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
      )
    # compute metrics
    target_real_val_dir = os.path.join(
        'dataset',
        self.dataset,
        'valB'
    )
    metrics = torch_fidelity.calculate_metrics(
        input1=str(target_real_val_dir),
        input2=str(Path(model_with_step_translations_dir)),  # fake dir,
        isc=True,
        fid=True,
        verbose=False,
        rng_seed=self.seed,
        cuda=torch.cuda.is_available(),
    )
    with open(os.path.join(Path(self.result_dir, self.dataset), 'val_log.txt'), 'a') as tl:
      tl.write(f'step: {step}\n')
      tl.write(
          f'inception_score_mean: {metrics["inception_score_mean"]}\n'
      )
      tl.write(
          f'inception_score_std: {metrics["inception_score_std"]}\n',
      )
      tl.write(
          f'frechet_inception_distance: {metrics["frechet_inception_distance"]}\n'
      )

    if metrics['frechet_inception_distance'] < self.smallest_val_fid:
      self.smallest_val_fid = metrics['frechet_inception_distance']
      self.save(ckpt_file_name=f'params_smallest_val_fid.pt', step=step)
      smallest_val_fid_file = Path(
          self.result_dir,
          self.dataset, 'smallest_val_fid.txt'
      )
      if os.path.exists(smallest_val_fid_file):
        os.remove(smallest_val_fid_file)
      with open(smallest_val_fid_file, 'a') as tl:
        tl.write(
            f'step: {step}\n'
        )
        tl.write(
            f'frechet_inception_distance: {metrics["frechet_inception_distance"]}\n'
        )

  def test(self):
    self.load('params_latest.pt')
    print(self.start_iter)

    self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()
    for n, (real_A, real_A_path) in enumerate(self.testA_loader):
      real_A = real_A.to(self.device)
      _, _,  _, _, real_A_z = self.disA(real_A)
      fake_A2B = self.gen2B(real_A_z)

      A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
      print(real_A_path[0])
      cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                  'fakeB', real_A_path[0].split('/')[-1]), A2B * 255.0)

    for n, (real_B, real_B_path) in enumerate(self.testB_loader):
      real_B = real_B.to(self.device)
      _, _,  _, _, real_B_z = self.disB(real_B)
      fake_B2A = self.gen2A(real_B_z)

      B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))
      print(real_B_path[0])
      cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                  'fakeA', real_B_path[0].split('/')[-1]), B2A * 255.0)

  def translate(self):
    model_list = glob(
        os.path.join(
            self.result_dir,
            self.dataset,
            'model',
            '*.pt'
        )
    )
    if not len(model_list) == 0:
      self.load(cktpt_file_name=self.ckpt)
      print(" [*] Load SUCCESS")
    else:
      print(" [*] Load FAILURE")
      return

    if not os.path.exists('translations'):
      os.mkdir('translations')

    model_translations_dir = Path('translations', self.dataset)
    if not os.path.exists(model_translations_dir):
      os.mkdir(model_translations_dir)

    model_with_ckpt_translations_dir = Path(
        model_translations_dir,
        Path(self.ckpt).stem
    )
    if not os.path.exists(model_with_ckpt_translations_dir):
      os.mkdir(model_with_ckpt_translations_dir)

    train_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'train')
    if self.valA:
      val_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'val')
    test_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'test')
    full_translated_imgs_dir = Path(model_with_ckpt_translations_dir, 'full')

    if not os.path.exists(train_translated_imgs_dir):
      os.mkdir(train_translated_imgs_dir)
    if not os.path.exists(test_translated_imgs_dir):
      os.mkdir(test_translated_imgs_dir)
    if val_translated_imgs_dir and not os.path.exists(val_translated_imgs_dir):
      os.mkdir(val_translated_imgs_dir)
    if not os.path.exists(full_translated_imgs_dir):
      os.mkdir(full_translated_imgs_dir)

    self.gen2B.eval()
    self.disA.eval()
    self.disB.eval()

    print('translating train...')
    for n, (real_A, _) in enumerate(self.trainA_loader):
      real_A = real_A.to(self.device)
      img_path, _ = self.trainA_loader.dataset.samples[n]
      img_name = Path(img_path).name.split('.')[0]
      _, _,  _, _, real_A_z = self.disA(real_A)
      fake_A2B = self.gen2B(real_A_z)

      print(os.path.join(train_translated_imgs_dir, f'{img_name}_fake_B.jpg'))
      cv2.imwrite(
          os.path.join(train_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
          RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
      )
      cv2.imwrite(
          os.path.join(full_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
          RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
      )
    if val_translated_imgs_dir:
      print('translating val...')
      for n, (real_A, _) in enumerate(self.valA_loader):
        real_A = real_A.to(self.device)
        img_path, _ = self.valA_loader.dataset.samples[n]
        img_name = Path(img_path).name.split('.')[0]
        _, _,  _, _, real_A_z = self.disA(real_A)
        fake_A2B = self.gen2B(real_A_z)
        print(os.path.join(val_translated_imgs_dir, f'{img_name}_fake_B.jpg'))
        cv2.imwrite(
            os.path.join(val_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
        )
        cv2.imwrite(
            os.path.join(full_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
            RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
        )
    print('translating test...')
    for n, (real_A, _) in enumerate(self.testA_loader):
      real_A = real_A.to(self.device)
      img_path, _ = self.testA_loader.dataset.samples[n]
      img_name = Path(img_path).name.split('.')[0]
      _, _,  _, _, real_A_z = self.disA(real_A)
      fake_A2B = self.gen2B(real_A_z)
      print(os.path.join(test_translated_imgs_dir, f'{img_name}_fake_B.jpg'))
      cv2.imwrite(
          os.path.join(test_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
          RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
      )
      cv2.imwrite(
          os.path.join(full_translated_imgs_dir, f'{img_name}_fake_B.jpg'),
          RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
      )
