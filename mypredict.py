import argparse
import logging
import os
from os import path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='F:\\project\\AI\\axc.px82.com\\AILabelSystem\\Server\\checkpoints\\CP_epoch5.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', default="C:\\Users\\Administrator\\Desktop\\yff\\tupian2 (11).jpg",
                        metavar='INPUT', nargs='+',  help='filenames of input images') #, required=True
    parser.add_argument('--input-image-type',  default="*.jpg",  help='file type') #, required=True

    parser.add_argument('--output', '-o', default="C:\\Users\\Administrator\\Desktop\\yffout\\",
                       metavar='output', nargs='+', help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=True)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

def get_input_filenames(args):
    in_files = []
    input = args.input
    imagetype = args.input_image_type

    if input == None:
        logging.error("--input params files is required ")
        raise SystemExit()
    # if isinstance(input,str):
    #读取.txt文件中的每一行数据
    if(input.endswith('.txt')):
        f = open(input,"r")         #设置文件对象
        in_files = f.readlines()    #直接将文件中按行读到list里，效果与方法2一样
        f.close()                   #关闭文件 
    # 是目录
    elif(os.path.isdir(input)):
        imagetype = args.input_image_type
        for root, dirs, files in os.walk(input):
            for f in files:
                if f.endswith(imagetype): # jpg或者png
                    in_files.append(os.path.join(root, f))  
    # 是文件
    elif(input.endswith(imagetype)):
        in_files.append(input)
    else:
        in_files.append(input)

    return in_files


def get_output_filenames(in_files,args): 
    out_files = []
    imagetype = args.input_image_type
    out = args.output
    # 没有填这个参数
    if out == None:
        for f in in_files:
            dir,name = os.path.split(f)
            out_files.append(os.path.join(dir,'out_'+name))
    # 输出目录
    elif(os.path.isdir(out)):
        for f in in_files:
            dir,name = os.path.split(f)
            out_files.append(os.path.join(out,'out_'+name)) 
    # 输出到单个文件
    elif(out.endswith(imagetype)):
        out_files.append(out) 
    else:
       for f in in_files:
         dir,name = os.path.split(f)
         out_files.append(os.path.join(dir,'out_'+name))

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = get_input_filenames(args)
    out_files = get_output_filenames(in_files,args)

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
