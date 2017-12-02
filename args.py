import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--x_dim', type=int, default=64, help='the h / w of the input image')
parser.add_argument('--z_dim', type=int, default=256, help='the size of the latent space')
parser.add_argument('--epochs', type=int, default=1000000, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train for')
parser.add_argument('--ckpt_step', type=int, default=5, help='epoch step for saving checkpoint')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate for generator')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimization default:SGD, option: Adam')
parser.add_argument('--alpha', type=float, default=10., help='learning rate for generator')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--l1_weight', type=float, default=0.05, help='weight for l1 loss')
parser.add_argument('--sample_size', type=int, default=64, help='sample size to export')
parser.add_argument('--sample_path', default='./samples', help='Where to store samples and models')
parser.add_argument('--model_path', default='./models', help='Where to store samples and models')
parser.add_argument('-gpu', '--gpu', type=bool, default=True, help='use gpu =1 , no gpu = 0')
parser.add_argument('-generator_type', '--gen_type', type=int, default=1,
                    help='use gpu =1 , no gpu = 0')
parser.add_argument('--resume', type=str, default=None,
                    help='input models name if you want to resmume the train from checkpoint'
                    )

opt = parser.parse_args()


def get_opt():
    return opt


def print_opt():
    tuples = vars(opt).items()
    for x in tuples:
        print(x)
