import argparse
import os

from utils.utils import setup_seed
from utils.regression_trianer import RegTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=44, 
                        help='random seed')
    parser.add_argument('--save-dir', default='./model',
                        help='directory to save model')
    parser.add_argument('--log-dir', default='./logging',
                        help='directory to save logging')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='epochs for training without watermarking')
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1,
                        help="whether to output on the terminal")
    parser.add_argument('--device', default='0', 
                        help='assign device')
    parser.add_argument('--default', type=int, default=1, 
                        help='whether to use default hyperparameter, 0 or 1')
    # 断点重训
    parser.add_argument('--if-continue', type=bool, default=False,
                        help="whether to continue training")
    parser.add_argument('--saved-model-dir', default="/root/cgx/PaperCodeReproduction/EWE/model/0902-183934",
                        help="restarted model weight path")

    parser.add_argument('--dataset', type=str, default="mnist", 
                        help='mnist, fashion, speechcmd, cifar10, or cifar100')
    parser.add_argument('--model', type=str, default="2_conv", 
                        help='2_conv, lstm, or resnet')
    parser.add_argument('--layers', type=int, default=18, 
                        help='number of layers, only useful if model is resnet')
    
    parser.add_argument('--metric', type=str, default="cosine",
                        help='distance metric used in snnl, euclidean or cosine')
    parser.add_argument('--ratio', type=float, default=1.,
                        help='ratio of amount of legitimate data to watermarked data')
    parser.add_argument('--w_epochs', type=int, default=10, 
                        help='epochs for training with watermarking')
    parser.add_argument('--factors', nargs='+', type=float, default=[32, 32, 32],
                        help='weight factor for snnl')
    parser.add_argument('--temperatures', nargs='+', type=float, default=[1, 1, 1], 
                        help='temperature for snnl')
    parser.add_argument('--threshold', type=float, default=0.1, 
                        help='threshold for estimated false watermark rate, should be <= 1/num_class')
    parser.add_argument('--maxiter', type=int, default=10, 
                        help='iter of perturb watermarked data with respect to snnl')
    parser.add_argument('--w_lr', type=float, default=0.01,
                        help='learning rate for perturbing watermarked data')
    parser.add_argument('--t_lr', type=float, default=0.1,
                        help='learning rate for temperature')
    parser.add_argument('--source', type=int, default=1, 
                        help='source class of watermark')
    parser.add_argument('--target', type=int, default=7,
                        help='target class of watermark')
    parser.add_argument('--distrib', type=str, default='out',
                        help='use in or out of distribution watermark')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    # 设置仅可使用特定的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip() 

    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
    
