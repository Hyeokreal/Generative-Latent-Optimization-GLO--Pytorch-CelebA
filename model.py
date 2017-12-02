import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from args import get_opt , print_opt

opt = get_opt()


std = 0.01


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def Generator1(latent_dim=opt.z_dim, num_channels=3, image_size=64):
    nz = latent_dim
    ngf = image_size
    nc = num_channels

    # based on DCGAN from pytorch/examples
    generator = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        nn.Tanh(),
        # state size. (nc) x 64 x 64
    )

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    generator.apply(weights_init)

    return generator

def Generator2(latent_dim=opt.z_dim, num_channels=3, image_size=64):
    nz = latent_dim
    ngf = image_size
    nc = num_channels

    # based on DCGAN from pytorch/examples
    generator = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(nz, ngf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        nn.Tanh(),
        # state size. (nc) x 64 x 64
    )

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    generator.apply(weights_init)

    return generator


class Generator(nn.Module):
    """Generator containing 7 deconvolutional layers."""

    def __init__(self, z_dim=opt.z_dim, image_size=64, conv_dim=32):
        super(Generator, self).__init__()
        self.deconv0 = deconv(opt.z_dim, 128, 1, 1, 0)
        self.fc = deconv(128, conv_dim * 8, int(image_size / 16), 1, 0, bn=False)
        self.deconv1 = deconv(conv_dim * 8, conv_dim * 4, 4)
        self.deconv2 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv3 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)
        self.weight_init(mean=0.0, std=0.02)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1) # If image_size is 64, output shape is as below.
        out = self.deconv0(z)
        out = self.fc(out)  # (?, 512, 4, 4)
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.deconv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.deconv3(out), 0.05)  # (?, 64, 32, 32)
        out = F.tanh(self.deconv4(out))  # (?, 3, 64, 64)
        return out

    def print_forward(self, z):
        print("z shape : ", z.size())
        z = z.view(z.size(0), z.size(1), 1, 1)  # If image_size is 64, output shape is as below.
        out = self.deconv0(z)
        print("z shape : ", z.size())
        out = self.fc(out)  # (?, 512, 4, 4)
        print("out shape : ", out.size())
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 256, 8, 8)
        print("out shape : ", out.size())
        out = F.leaky_relu(self.deconv2(out), 0.05)  # (?, 128, 16, 16)
        print("out shape : ", out.size())
        out = F.leaky_relu(self.deconv3(out), 0.05)  # (?, 64, 32, 32)
        print("out shape : ", out.size())
        out = F.tanh(self.deconv4(out))  # (?, 3, 64, 64)
        print("out shape : ", out.size())
        return out


class G(nn.Module):
    def __init__(self, d=16):
        super(G, self).__init__()
        self.deconv0 = nn.ConvTranspose2d(10, 100, 1, 1, 0)
        self.deconv0_bn = nn.BatchNorm2d(100)
        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)

        self.weight_init(mean=0.0, std=0.02)

    def forward(self, input):
        x = F.relu(self.deconv0_bn(self.deconv0(input)))
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(x)))
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)))
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)))
        x = F.sigmoid(self.deconv4(x))
        return x

    def print_forward(self, input):
        x = F.relu(self.deconv0_bn(self.deconv0(input)))
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(x)))
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)))
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)))
        x = F.sigmoid(self.deconv4(x))
        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class lap_G(nn.Module):
    def __init__(self):
        super(lap_G, self).__init__()

        self.bilinear_deconv1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(128)

        self.bilinear_deconv2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.bilinear_deconv3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)

        self.bilinear_deconv4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv4 = nn.Conv2d(32, 16, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(16)

        self.bilinear_deconv5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv5 = nn.Conv2d(16, 8, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(8)

        self.bilinear_deconv6 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv6 = nn.Conv2d(8, 3, 3, 1, 1)

        # self.weight_init(mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, std)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, std)
                m.bias.data.zero_()

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.bn1((self.conv1(self.bilinear_deconv1(input)))))
        x = F.leaky_relu(self.bn2((self.conv2(self.bilinear_deconv2(x)))))
        x = F.leaky_relu(self.bn3((self.conv3(self.bilinear_deconv3(x)))))
        x = F.leaky_relu(self.bn4((self.conv4(self.bilinear_deconv4(x)))))
        x = F.leaky_relu(self.bn5((self.conv5(self.bilinear_deconv5(x)))))
        x = F.tanh(self.conv6(self.bilinear_deconv6(x)))
        return x

    def print_forward(self, input):
        print(input.size())
        x = F.leaky_relu(self.bn1((self.conv1(self.bilinear_deconv1(input)))))
        print(x.size())
        x = F.leaky_relu(self.bn2((self.conv2(self.bilinear_deconv2(x)))))
        print(x.size())
        x = F.leaky_relu(self.bn3((self.conv3(self.bilinear_deconv3(x)))))
        print(x.size())
        x = F.leaky_relu(self.bn4((self.conv4(self.bilinear_deconv4(x)))))
        print(x.size())
        x = F.leaky_relu(self.bn5((self.conv5(self.bilinear_deconv5(x)))))
        print(x.size())
        x = F.tanh(self.conv6(self.bilinear_deconv6(x)))
        print(x.size())
        return x



'''
shape Test
'''

if __name__ == "__main__":
    print_opt()
    batch = opt.batch_size
    x_dim = opt.x_dim
    z_dim = opt.z_dim
    z = torch.zeros([batch, z_dim])
    z = z.view([-1, z_dim, 1, 1])
    z = Variable(z)

    x = torch.zeros([batch, 1, x_dim, x_dim])
    x = Variable(x)

    # g = Generator()
    g = Generator2()
    out = g.forward(z)

    r1 = out.data.numpy()

    print(r1.shape)
