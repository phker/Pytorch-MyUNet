import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import cv2

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = 'F:\\dataset\\_ToPaddleUNetDataSet3\\_ToPaddleUNetDataSet\\JPEGImages\\' # 必须 \ 结尾
dir_mask = 'F:\\dataset\\_ToPaddleUNetDataSet3\\_ToPaddleUNetDataSet\\Annotations\\' # 必须 \ 结尾
dir_checkpoint = './checkpoints/'
n_classes = 14 # 总共有多少个分类需要被识别


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              num_workers=6,# 加载数据集的线程数
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss() # 交叉熵损失函数适用于多分类问题. 自带softmax函数
    else:
        criterion = nn.BCEWithLogitsLoss() # 二元交叉熵损失函数适用于2分类问题 ,自带sigmod函数

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image'] 
                # print(imgs[0].shape)
                true_masks = batch['mask']
                # print(true_masks[0].shape)
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                if net.n_classes > 1: 
                    loss = criterion(masks_pred, true_masks.squeeze(1)) # 求损失  # patch 123#bug
                else:
                    loss = criterion(masks_pred, true_masks) # 求损失
                # patch 123#bug end

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300,
                        help='训练轮次', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='每批训练图片个数', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='学习率Learning rate ', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='加载已训练过的模型 Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='下采样缩放图片大小 Downscaling factor of the images')
    parser.add_argument('--input-image-type',  default="*.jpg",  help='file type') #, required=True

    # parser.add_argument('--n-classes', dest='class', type=float, default=14,help='目标识别的个数')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20.0, help='验证集数据占比百分之几 Percent of the data that is used as validation (0-100)')

    return parser.parse_args()
 
#显示图片, 可接受torch的和numpy还有普通的图片文件内容.
def ShowImage(image):    
    from torchvision import transforms

    if(type(image) is np.ndarray):
        # plt.imshow(image) 
        # plt.show()
        cv2.imshow(image)
    elif(torch.is_tensor(image)):
        # plt.imshow(image.numpy()) 
        # plt.show()
        # cv2.imshow(image.numpy())
        unloader = transforms.ToPILImage()
        image = image.cpu().clone()  # clone the tensor
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        image.show()
   

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel # n_classes 是每个像素要获得的概率数
    #   - For 1 class and background, use n_classes=1 # 目标类别为1类和背景时n_class=1
    #   - For 2 classes, use n_classes=1 #  目标类别为2类和背景时n_class=1
    #   - For N > 2 classes, use n_classes=N #  目标类别大于2类时n_class=N
    
    #图片是几通道的
    n_channels = 3
    # imagetype = args.input_image_type
    # if(imagetype=='jpg'):
    #     n_channels = 3
    # elif(imagetype=='png'):
    #     n_channels = 4
    # else:
    #     n_channels = 3


    net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
        # train_net(net=net,
        #           epochs=300,
        #           batch_size=1,
        #           lr=0.001, #0.0001
        #           device=device,
        #           num_workers=2,
        #           img_scale=0.5, #0.5
        #           val_percent= 20 / 100) #用作验证的数据百分比（0-100）
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
