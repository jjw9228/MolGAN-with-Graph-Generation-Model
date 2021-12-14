import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import json
import argparse
from rdkit import RDLogger
from torch_geometric.data import DenseDataLoader
from DIG.dig.ggraph.dataset import QM9, ZINC250k, MOSES
from DIG.dig.ggraph.method import GraphDF
from DIG.dig.ggraph.evaluation import RandGenEvaluator
import copy


def str2bool(v):
    return v.lower() in ('true')

def main(config, config2):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Solver for training and testing StarGAN.
    solver = Solver(config, is_df=True, config2=config2)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='qm9', choices=['qm9', 'zinc250k', 'moses'], help='dataset name')
    parser.add_argument('--model_path', type=str, default='./saved_ckpts/rand_gen/rand_gen_qm9.pth',
                        help='The path to the saved model file')
    parser.add_argument('--num_mols', type=int, default=100, help='The number of molecules to be generated')
    parser.add_argument('--train', action='store_true', default=False,
                        help='specify it to be true if you are running training')


    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=8, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[128,256,512], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]], help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset')
    parser.add_argument('--log_dir', type=str, default='molgan/logs')
    parser.add_argument('--model_save_dir', type=str, default='molgan/models')
    parser.add_argument('--sample_dir', type=str, default='molgan/samples')
    parser.add_argument('--result_dir', type=str, default='molgan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    args = parser.parse_args()


    if args.data == 'qm9':
        with open('DIG/examples/ggraph/GraphDF/config/rand_gen_qm9_config_dict.json') as f:
            conf = json.load(f)
    elif args.data == 'zinc250k':
        with open('DIG/examples/ggraph/GraphDF/config/rand_gen_zinc250k_config_dict.json') as f:
            conf = json.load(f)
    elif args.data == 'moses':
        with open('DIG/examples/ggraph/GraphDF/config/rand_gen_moses_config_dict.json') as f:
            conf = json.load(f)
    else:
        print("Only qm9, zinc250k and moses datasets are supported!")
        exit()
    config2 = copy.deepcopy(conf)

    config = parser.parse_args()
    print(config)
    main(config, config2)
