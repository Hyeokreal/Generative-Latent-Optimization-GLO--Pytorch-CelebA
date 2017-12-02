import torch
from torch.autograd import Variable
from time import localtime, strftime
import os
from args import get_opt
import numpy as np
from sklearn.decomposition import PCA
import shutil

opt = get_opt()


def z_init():
    pass


def pca_feature(X, d):
    X = X / 255.
    X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
    # print(X.shape)
    pca = PCA(n_components=d)
    return pca.fit_transform(X)


def setup(is_test=False):
    model_path = opt.model_path

    # check if dir exists
    if not os.path.isdir(opt.sample_path):
        os.mkdir(opt.sample_path)

    # mkdir folder with name by date,time
    folder_sample = strftime("%y-%m-%d-%H-%M-%S", localtime())
    if is_test:
        folder_sample = 'test_' + folder_sample
    sample_path = os.path.join(os.getcwd(), 'samples', folder_sample)
    os.mkdir(sample_path)
    
    # comment text write
    f = open(os.path.join(sample_path, '1-comment.txt'), 'w')
    comment = """ experimental comment here """
    for s in comment:
        f.write(s)
    f.close()

    # comment arg info write
    f = open(os.path.join(sample_path, '1-info.txt'), 'w')
    tuples = vars(opt).items()
    for x in tuples:
        f.write(str(x))
        f.write('\n')
    f.close()
    
    if is_test:
        return sample_path

    if not os.path.isdir(opt.model_path):
        os.mkdir(opt.model_path)

    folder_model = strftime("%y-%m-%d %H-%M-%S", localtime())
    model_path = os.path.join(os.getcwd(), 'models', folder_model)
    os.mkdir(model_path)

    return sample_path, model_path


def to_variable(x, requires_grad=False):
    if opt.gpu:
        x = x.cuda()

    if requires_grad:
        return Variable(x, requires_grad=requires_grad)
    return Variable(x)


def denorm(x):
    """Convert range (-1, 1) to (0, 1)"""
    out = (x + 1) / 2
    return out.clamp(0, 1)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def get_lastest_ckpt():
    model_list = sorted(os.listdir(os.path.join(os.getcwd(), 'models')))[-1]
    lastest_model_path = os.path.join(os.getcwd(), 'models', model_list)
    ckpt_list = sorted(os.listdir(os.path.join(os.getcwd(), 'models', lastest_model_path)))[-1]
    lastest_ckpt = os.path.join(os.getcwd(), 'models', lastest_model_path, ckpt_list)

    return lastest_ckpt




