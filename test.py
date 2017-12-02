import torch
from torch.autograd import Variable
import torchvision
import os
from torch import optim
from model import Generator2
from util import setup, denorm, get_lastest_ckpt
from data_loader import get_loader

ckpt_path = get_lastest_ckpt()

if torch.cuda.is_available():
    ckpt = torch.load(ckpt_path)

else:
    ckpt = torch.load(ckpt_path, map_location={'cuda:0': 'cpu'})

print("=> ", ckpt_path, " is being loaded")

opt = ckpt['args']

if not torch.cuda.is_available():
    opt.gpu = False


n_epochs = opt.epochs
batch_size = opt.batch_size
z_dim = opt.z_dim
x_dim = opt.x_dim
sample_size = opt.sample_size
lr = opt.lr
log_step = 100
sample_step = 1000
img_size = opt.x_dim

generator = Generator2()

if opt.gpu:
    generator.cuda()

if opt.optim == 'Adam':
    g_optimizer = optim.Adam(generator.parameters(), lr, betas=(opt.beta1, opt.beta2))
elif opt.optim == 'SGD':
    g_optimizer = optim.SGD(generator.parameters(), lr)
else:
    print('optimizer is not set correctly. Adam will be used')
    g_optimizer = optim.Adam(generator.parameters(), lr, betas=(opt.beta1, opt.beta2))

opt.start_epoch = ckpt['epoch']
generator.load_state_dict(ckpt['state_dict'])
z_in_ball = ckpt['latent']
g_optimizer.load_state_dict(ckpt['optimizer'])

sample_path = setup(is_test=True)

image_path = os.path.join(os.getcwd(), 'CelebA', '128_crop')
train_loader = get_loader(image_path=image_path,
                          image_size=opt.x_dim,
                          batch_size=opt.batch_size,
                          num_workers=0)

data_folder = get_loader(image_path=image_path,
                          image_size=opt.x_dim,
                          batch_size=opt.batch_size,
                          image_folder=True
                         )

if opt.gpu:
    learnable_z = z_in_ball.cuda()
else:
    learnable_z = z_in_ball


def to_variable(x, requires_grad=False):
    if opt.gpu:
        x = x.cuda()

    if requires_grad:
        return Variable(x, requires_grad=requires_grad)
    return Variable(x)


def single_item_recon(batch):



    for item_num in range(32):
    # x = train_loader[batch][item_num]
        x = data_folder[opt.batch_size * batch + item_num]
        x = x.view(1, 3, 64, 64)
        z = learnable_z[batch][item_num].view(1, 256, 1, 1)

        x = to_variable(x)
        z = to_variable(z)
        x_hat = generator.forward(z)

    z = learnable_z[batch]



    print("saving recon images in ", sample_path)

    torchvision.utils.save_image(denorm(x.data),
                                 os.path.join(sample_path,
                                              'real_samples-%d-%d.png' % (batch, item_num)), nrow=8)
    # save the generated images
    torchvision.utils.save_image(denorm(x_hat.data),
                                 os.path.join(sample_path,
                                              'recon_test-%d-%d.png' % (batch, item_num)),
                                 nrow=8)


def recon_test(iter_num):
    for i, x in enumerate(train_loader):
        print(i)
        if i == iter_num:
            x = to_variable(x)
            z = to_variable(learnable_z[i])
            x_hat = generator.forward(z)

            print("saving recon images in ", sample_path)

            torchvision.utils.save_image(denorm(x.data),
                                         os.path.join(sample_path,
                                                      'real_samples-%d.png' % iter_num), nrow=8)
            # save the generated images
            torchvision.utils.save_image(denorm(x_hat.data),
                                         os.path.join(sample_path,
                                                      'recon_test-%d.png' % iter_num),
                                         nrow=8)
            break


def generate_test(iter_num):
    for i in range(iter_num):

        z = torch.randn(opt.batch_size, opt.z_dim, 1, 1)
        z = to_variable(z)

        if opt.gpu:
            z.cuda()

        x_hat = generator.forward(z)

        print("saving generated images in ", sample_path)

        # save the generated images
        torchvision.utils.save_image(denorm(x_hat.data),
                                     os.path.join(sample_path,
                                                  'generate_test-%d.png' % i),
                                     nrow=8)


if __name__ == "__main__":
    # single_item_recon(0,0)

    # for i in range(32):
    #     single_item_recon(0,i)

    recon_test(0)
    # recon_test(1)
    # recon_test(2)
    # recon_test(3)
    # generate_test(10)
