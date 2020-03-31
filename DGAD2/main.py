from DGAD import DGAD
from Lis_GAD import GAD
from TEST_GAD import TEST
import argparse
from utils import *
import os
#from torch.backends import cudnn
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Data_dict = {'reddit_data': [100,50,3199,2411,300,64],
             'DBLP5':[6,4,6606,6606,100,32]}

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 3dgraphconv"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--dataset', type=str, default='DBLP5', help='dataset_name: reddit_data/DBLP5')
    parser.add_argument('--model', type=str, default='DGAD', help='DGAD/GAD/TEST')
    parser.add_argument('--dataset_setting', type=dict, default=Data_dict,
                        help='train_len, test_len, train_size, test_size, node_channel, conve_channel')

    parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')
    parser.add_argument('--denoising', type=float, default=0.1, help='denoising autoencoder')

    parser.add_argument('--decay_flag', type=bool, default=False, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=6, help='decay epoch')

    parser.add_argument('--epoch', type=int, default=3, help='The number of epochs to run')
    #parser.add_argument('--iteration', type=int, default=2000, help='The number of training iterations')##
    parser.add_argument('--new_start', type=bool, default=True, help='new_start')

    parser.add_argument('--lr', type=float, default=0.005, help='The learning rate')

    parser.add_argument('--ax_w', type=float, default=0.5, help='weight of edge reconstruction error')
    parser.add_argument('--loss_function', type=str, default='l2_loss', help='loss function type [l1_loss / l2_loss/ cross_entropy]')

    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=5, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=100, help='The number of ckpt_save_freq')
    parser.add_argument('--print_net', type=bool, default=False, help='print_net')
    parser.add_argument('--use_tensorboard', type=bool, default=False, help='use_tensorboard')

    parser.add_argument('--num_clips', type=int, default=3, help='The size of clip')
    parser.add_argument('--conv_ch', type=int, default=0, help='The base channel number per layer')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')


    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

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
def main(**setting):
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    #cudnn.benchmark = True

    if args.model == 'DGAD':
        gae = DGAD(args)
    elif args.model == 'GAD':
        gae = GAD(args)
    else:
        gae = TEST(args)
    print('Model: {}'.format(args.model))

    if args.phase == 'train':
        # launch the graph in a session
        gae.train()
        print(" [*] Training finished!")
        print("\n\n\n")
        gae.test()
        print(" [*] Test finished!")

    if args.phase == 'test':
        gae.test()
        print(" [*] Test finished!")

    if args.phase == 'test2':
        gae.test2()
        print(" [*] Test finished!")


if __name__ == '__main__':
    #main()
    main(phase='train', model='GAD', resume_iters=0, lr=0.005, epoch=10, num_clips=3, conv_ch=32, denoising=0.2)