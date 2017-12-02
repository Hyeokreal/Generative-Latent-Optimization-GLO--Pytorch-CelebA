import torch
import torchvision
import os
from torch import optim
from model import Generator2
from args import get_opt, print_opt
from util import setup, to_variable, denorm, save_checkpoint, get_lastest_ckpt
import numpy as np
from data_loader import get_loader, process_z
from laplacian_loss import laplacian_loss

opt = get_opt()
print_opt()

# hyper parameters
n_epochs = opt.epochs
batch_size = opt.batch_size
z_dim = opt.z_dim
x_dim = opt.x_dim
sample_size = opt.sample_size
lr = opt.lr
log_step = 100
sample_step = 1000

sample_path, model_path = setup()

image_path = os.path.join(os.getcwd(), 'CelebA', '128_crop')

train_loader = get_loader(image_path=image_path,
                          image_size=opt.x_dim,
                          batch_size=opt.batch_size,
                          num_workers=2)

image_path = os.path.join(os.getcwd(), 'CelebA', '128_crop')

img_size = opt.x_dim

# choose generator type laplacian generator or common dcgan generator
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

z_file_name = 'processed_z'
z_file_name += '_b' + str(opt.batch_size)
z_file_name += '_z' + str(opt.z_dim)
z_file_name += '.pt'

if not os.path.isfile(z_file_name):
    print("no processed z. so create one..")
    process_z(opt.x_dim, opt.z_dim, opt.batch_size, './', './')

if opt.resume == 'auto':
    # auto option load the checkpoint of the lastest models of  the lastest epoch
    ckpt_path = get_lastest_ckpt()
    print("=> ", ckpt_path, " is being loaded")
    ckpt = torch.load(ckpt_path)

    opt.start_epoch = ckpt['epoch']
    generator.load_state_dict(ckpt['state_dict'])
    z_in_ball = ckpt['latent']
    g_optimizer.load_state_dict(ckpt['optimizer'])


elif opt.resume is not None:
    if os.path.isfile(opt.resume):
        ckpt = torch.load(opt.resume)
        opt.start_epoch = ckpt['epoch']
        generator.load_state_dict(ckpt['state_dict'])
        z_in_ball = ckpt['latent']
        g_optimizer.load_state_dict(ckpt['optimizer'])
    else:
        print("No check point file in input path")

else:
    z_in_ball = torch.load(z_file_name)

if opt.gpu:
    learnable_z = z_in_ball.cuda()
else:
    learnable_z = z_in_ball

total_step = len(train_loader)

for epoch in range(opt.start_epoch, n_epochs):
    for i, x in enumerate(train_loader):
        if i == 4880:
            break

        x = to_variable(x)
        z = to_variable(learnable_z[i], requires_grad=True)
        x_hat = generator.forward(z)

        l1_loss = opt.l1_weight * torch.mean(torch.abs(x - x_hat))
        lap_loss = laplacian_loss(x, x_hat, n_levels=-1, cuda=opt.gpu)
        loss = l1_loss + lap_loss

        g_optimizer.zero_grad()
        loss.backward()
        g_optimizer.step()

        if opt.gpu:
            grad = z.grad.data.cuda()
        else:
            grad = z.grad.data

        z_update = learnable_z[i] - opt.alpha * grad
        z_update = z_update.cpu().numpy()
        norm = np.sqrt(np.sum(z_update ** 2, axis=1))
        z_update_norm = z_update / norm[:, np.newaxis]

        if opt.gpu:
            learnable_z[i] = torch.from_numpy(z_update_norm).cuda()
        else:
            learnable_z[i] = torch.from_numpy(z_update_norm).cpu()

        if (i + 1) % log_step == 0:
            print('Epoch [%d/%d], Step[%d/%d], loss: %f, l1: %f, lap: %f'
                  % (epoch + 1, n_epochs, i + 1, total_step, loss.data[0], l1_loss.data[0],
                     lap_loss.data[0]
                     ))

        # save the real images
        if (i + 1) == sample_step:
            torchvision.utils.save_image(denorm(x.data),
                                         os.path.join(sample_path,
                                                      'real_samples-%d-%d.png' % (
                                                          epoch + 1, i + 1)), nrow=4)
        # save the generated images
        if (i + 1) % sample_step == 0:
            torchvision.utils.save_image(denorm(x_hat.data),
                                         os.path.join(sample_path,
                                                      'fake_samples-%d-%d.png' % (
                                                          epoch + 1, i + 1)), nrow=4)

    if (epoch + 1) % opt.ckpt_step == 0:
        print("saving checkpoint ..")
        checkpoint_path = os.path.join(model_path, 'checkpoint_%d.pth.tar' % (epoch + 1))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': generator.state_dict(),
            'latent': learnable_z,
            'optimizer': g_optimizer.state_dict(),
            'args': opt
        }, filename=checkpoint_path)
        print("done.")
