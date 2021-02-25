"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init


def parse_option():
    '''
    加载默认参数
    '''

    # 获取当前设备的主机名称  用于选择数据集路径、模型保存路径等
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency') # 打印间隔
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency') # 记录间隔
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency') # 保存间隔
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size') # 批次大小
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use') # 数据集加载的线程数
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs') # 总轮次
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods') # 对于两阶段的蒸馏方法，第一阶段的轮次

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate') # 学习率
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list') # 当达到第几轮次时，降低学习率
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate') # 学习率降低率
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay') # 优化器衰减权重
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')  # 优化器动量

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset') #  选择训练集

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2']) # 支持的学生模型
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot') # 教师模型的权重

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst']) # 支持的蒸馏方法
    parser.add_argument('--trial', type=str, default='1', help='trial id') # 区分第几次训练

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification') # 交叉熵损失占总损失的权重
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD') # 蒸馏损失占总损失的权重
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses') # 其余蒸馏损失占总损失的权重

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation') # 常规蒸馏的温度

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension') # 特征维度
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax']) #  提取方式
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE') # 对于NCE，负样本数量
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax') # sofftmax温度参数
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')# 非参数更新的动量

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4]) # 网络中间层，用于FitNets监督中间特征

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']: # 轻量级模型的 学习率较小
        opt.learning_rate = 0.01

    # 根据主机名，设置模型和log保存路径
    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    # 调整学习率
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # 初始化教师模型
    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)
    # tensorboard记录
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    best_acc = 0

    # 加载默认参数
    opt = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader 数据集加载器
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            # 训练集  验证集  训练集样本个数
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        # cifar100是100分类
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model 初始化教师网络 并加载权重，100分类
    model_t = load_teacher(opt.path_t, n_cls)

    # 初始化学生网络
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    # 假数据跑一下，得到 网络中间特征的形状
    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    # 教师网络的中间特征feat_t eg:[f0, f1, f2, f3, f4]，其中f4为特征flatten,接入全连接层   预测结果_      is_feat=True提取中间特征图
    feat_t, _ = model_t(data, is_feat=True)
    # 学生网络的中间特征feat_s   预测结果_
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    if opt.distill == 'kd': # 常规蒸馏
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint': # Fitnets 隐藏层
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention': # AT
        criterion_kd = Attention()
    elif opt.distill == 'nst': # NST
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity() # SP
    elif opt.distill == 'rkd': # RKD
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt': # PKT
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation': # CC
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid': # VID
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound': # AB
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':  # FT
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp': # FSP
        # [ [batch,channel,h,w]  ]    eg:[ [8,32,32,32] [8,64,16,16]   ]
        s_shapes = [s.shape for s in feat_s[:-1]] # 忽略最后flatten的特征
        # [ [batch,channel,h,w]  ]    eg:[ [8,32,32,32] [8,64,16,16]   ]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes) # 初始化FSP 损失
        # init stage training  FSP矩阵仅应用于第一阶段
        # init_trainable_list包含 当前学生对象 参与训练的网络结构
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        # 第一阶段训练  基于FSP，学生学习教师网络关系
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training 第二段是常规训练
        pass
    else:
        raise NotImplementedError(opt.distill)

    # 交叉熵损失
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # 分类损失 classification loss
    criterion_list.append(criterion_div)    # 常规蒸馏损失 KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # 其余蒸馏损失 other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
