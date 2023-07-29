import argparse

from NICE import NICE
from NICE_CUT import NICE_CUT
from utils import *

"""parsing and configuration"""


def parse_args():
  desc = "Pytorch implementation of NICE-GAN"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('--phase', type=str, default='train',
                      help='[train / test]')
  parser.add_argument('--light', type=str2bool, default=False,
                      help='[NICE-GAN full version / NICE-GAN light version]')
  parser.add_argument('--dataset', type=str,
                      default='YOUR_DATASET_NAME', help='dataset_name')
  parser.add_argument('--cut', type=str2bool, default=False,
                      help='[CUT or NOT CUT]')
  parser.add_argument('--ckpt',
                      type=str,
                      default=None,
                      help='checkpoint to load.')
  parser.add_argument('--seed', type=int, default=269902365, help='Seed')

  parser.add_argument('--iteration', type=int, default=300000,
                      help='The number of training iterations')
  parser.add_argument('--batch_size', type=int, default=1,
                      help='The size of batch size')
  parser.add_argument('--print_freq', type=int, default=1000,
                      help='The number of image print freq')
  parser.add_argument('--val_freq', type=int, default=10000,
                      help='The number of image print freq')
  parser.add_argument('--save_freq', type=int, default=100000,
                      help='The number of model save freq')
  parser.add_argument('--decay_flag', type=str2bool,
                      default=True, help='The decay_flag')

  parser.add_argument('--lr', type=float, default=0.0001,
                      help='The learning rate')
  parser.add_argument('--weight_decay', type=float,
                      default=0.0001, help='The weight decay')
  parser.add_argument('--adv_weight', type=int,
                      default=1, help='Weight for GAN')
  parser.add_argument('--cycle_weight', type=int,
                      default=10, help='Weight for Cycle')
  parser.add_argument('--recon_weight', type=int, default=10,
                      help='Weight for Reconstruction')

  parser.add_argument('--ch', type=int, default=64,
                      help='base channel number per layer')
  parser.add_argument('--n_res', type=int, default=6,
                      help='The number of resblock')
  parser.add_argument('--n_dis', type=int, default=7,
                      help='The number of discriminator layer')

  parser.add_argument('--img_size', type=int, default=256,
                      help='The size of image')
  parser.add_argument('--img_ch', type=int, default=3,
                      help='The size of image channel')

  parser.add_argument('--result_dir', type=str, default='results',
                      help='Directory name to save the results')
  parser.add_argument('--device', type=str, default='cuda',
                      choices=['cpu', 'cuda', 'mps'], help='Set gpu mode; [cpu, cuda, mps]')
  parser.add_argument('--benchmark_flag', type=str2bool, default=False)
  parser.add_argument('--resume', type=str2bool, default=False)
  parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
  parser.add_argument('--n_downsampling', type=int, default=2,
                      help='The number of downsampling')

  parser.add_argument('--nce_weight', type=float, default=1.0,
                      help='weight for NCE loss: NCE(G(X), X)')
  parser.add_argument(
      '--nce_idt',
      type=str2bool,
      nargs='?',
      const=True,
      default=False,
      help='use NCE loss for identity mapping: NCE(G(Y), Y))'
  )
  parser.add_argument(
      '--nce_temperature',
      type=float,
      default=0.07,
      help='temperature for NCE loss'
  )
  parser.add_argument(
      '--nce_patch_embedding_dim',
      type=int,
      default=256
  )
  parser.add_argument(
      '--nce_n_patches',
      type=int,
      default=256,
      help='number of patches per layer'
  )
  parser.add_argument(
      '--nce_layers',
      type=str,
      default='0,2,3,4,8',
      help='layers contributing to NCE'
  )

  # full CUT
  # parser.set_defaults(nce_idt=True, nce_weight=1.0)

  # default for U-GAT-IT-CUT
  parser.set_defaults(
      lr=0.0001
  )
  parser.set_defaults(
      nce_idt=True,
      nce_layers='0,2,5',
      nce_weight=10.0
  )
  return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
  # --result_dir
  check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
  check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
  check_folder(os.path.join(args.result_dir, args.dataset, 'fakeA'))
  check_folder(os.path.join(args.result_dir, args.dataset, 'fakeB'))

  # --epoch
  try:
    assert args.epoch >= 1
  except:
    print('number of epochs must be larger than or equal to one')

  # --batch_size
  try:
    assert args.batch_size >= 1
  except:
    print('batch size must be larger than or equal to one')
  return args


"""main"""


def main():
  # parse arguments
  args = parse_args()
  if args is None:
    exit()

  # open session
  if args.cut:
    gan = NICE_CUT(args)
  else:
    gan = NICE(args)

  # build graph
  gan.build_model()

  if args.phase == 'train':
    gan.train()
    print(" [*] Training finished!")

  if args.phase == 'test':
    gan.test()
    print(" [*] Test finished!")

  if args.phase == 'translate':
    gan.translate()
    print(" [*] Translation finished!")


if __name__ == '__main__':
  main()
