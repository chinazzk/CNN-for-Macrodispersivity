import argparse
import torch
import json
import random
from pprint import pprint
from utils.misc import mkdirs

# always uses cuda if avaliable

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Dense Convolutional Encoder-Decoder Networks')
        self.add_argument('--exp-name', type=str, default='Dis_CNN4-0.2', help='experiment name')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')        
        self.add_argument('--post', action='store_true', default=False, help='post training analysis')
        
        self.add_argument('--num_class', type=int, default=1, help='number of output label')
        # data 
        self.add_argument('--data-dir', type=str, default="./dataset", help='directory to dataset')
        #self.add_argument('--kle', type=int, default=4225, help='num of KLE terms')
        self.add_argument('--var', type=float, default=0.5, help='variance of lnK fields')
        self.add_argument('--ntrain', type=int, default=4000, help="number of training data")
        self.add_argument('--ntest', type=int, default=1000, help="number of test data")
        self.add_argument('--cor', type=int, default=4, help="correlation length of field") 

        # training
        self.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
        self.add_argument('--lr', type=float, default=1e-3, help='learnign rate')
        # self.add_argument('--lr-scheduler', type=str, default='plateau', help="scheduler, plateau or step")
        self.add_argument('--weight-decay', type=float, default=5e-4, help="weight decay")
        self.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 16)')
        self.add_argument('--test-batch-size', type=int, default=64, help='input batch size for testing (default: 100)')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')
        
        # logging
        self.add_argument('--ckpt-epoch', type=int, default=200, help='which epoch of checkpoints to be loaded in post mode')
        self.add_argument('--ckpt-freq', type=int, default=50, help='how many epochs to wait before saving model')
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-freq', type=int, default=1, help='how many epochs to wait before plotting test output')
        #self.add_argument('--plot-fn', type=str, default='contourf', choices=['contourf', 'imshow'], help='plotting method')


    def parse(self):
        args = self.parse_args()
        
        args.run_dir = args.exp_dir + '/' + args.exp_name
        args.ckpt_dir = args.run_dir + '/checkpoints'
        mkdirs([args.run_dir, args.ckpt_dir])

        assert args.epochs % args.ckpt_freq == 0, 'epochs must'\
            'be dividable by ckpt_freq'

        # seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        print('Arguments:')
        pprint(vars(args))

        if not args.post:  # 如果为 False 则执行以下
            with open(args.run_dir + "/args.txt", 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        return args

# global
args = Parser().parse()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
