import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import argparse

import os
import numpy as np

from sklearn.model_selection import train_test_split

from models import ImageToPatches, PatchEmbedding, VisionTransformer, MLP, SelfAttention, OutputProjection, ViTSeg
# from model import UNET
from utils import CustomDataset, train_one_epoch, check_accuracy, save_checkpoint, save_predictions_as_imgs, print_model_parameters




def train(model, loss, optim, dl_train, dl_val, args):

    device = torch.device(args.device)
    model.to(device)

    print(torch.cuda.get_device_properties(0))

    avg_loss = []
    val_loss = []
    lowest_loss = 100000
    for epoch in range(args.NUM_EPOCHS):

        args.epoch = epoch
        l = train_one_epoch(model=model, loss_fn=loss, optim=optim, loader=dl_train, device=device, args=args)

        avg_loss.append(l)
        print("[%d/%d] - epoch end loss: %f"%(epoch,args.NUM_EPOCHS,avg_loss[-1]))

        metrics = check_accuracy(model, dl_val, loss, device, args=args)
        val_loss.append(metrics['ValLoss'])

        print(f"[{epoch}/{args.NUM_EPOCHS}] - Eval metrics -> {metrics}")
        
        if val_loss[-1] < lowest_loss:
            lowest_loss = val_loss[-1]
            print('A better model was found! saving...')

            actual_state = {'optim':optim.state_dict(),'model':model.state_dict(),'epoch':epoch}
            save_checkpoint(actual_state, f"./saved_models/{args.model_name}/", "best_model.pth")

            save_predictions_as_imgs(model, dl_val)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--dataset-dir', type=str, default="../carvana_segmentation/data2/")
    parser.add_argument('--dataset-split-dir', type=str, default="data/ImageSets/Segmentation")
    parser.add_argument('--image-extension', type=str, default='.jpg')
    parser.add_argument('--mask-extension', type=str, default='.gif')

    parser.add_argument('--image-height', type=int, default=512)
    parser.add_argument('--image-width', type=int, default=512)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--embed_size', type=int, default=768)
    parser.add_argument('--num_blocks', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=.2)

    parser.add_argument('--model-name', type=str, default='vit')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--NUM_EPOCHS', type=int, default=50)

    args = parser.parse_args()


    dataset_dir = args.dataset_dir
    split_dir   = args.dataset_split_dir

    LEARNING_RATE = args.learning_rate

    device = torch.device(args.device)

    with open(os.path.join(split_dir, 'train.txt'), 'r') as f:
        X_train = []
        y_train  = []
        for id in f:
            X_train.append(os.path.join(dataset_dir, 'train', id.rstrip()+args.image_extension) )
            y_train.append(os.path.join(dataset_dir, 'train_masks', id.rstrip()+'_mask'+args.mask_extension) )
    

    with open(os.path.join(split_dir, 'val.txt'), 'r') as f:
        X_test = []
        y_test  = []
        for id in f:
            X_test.append(os.path.join(dataset_dir, 'train', id.rstrip()+args.image_extension) )
            y_test.append(os.path.join(dataset_dir, 'train_masks', id.rstrip()+'_mask'+args.mask_extension) )


    train_transform = A.Compose(
        [
            A.Resize(height=args.image_height, width=args.image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=args.image_height, width=args.image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


    train_ds = CustomDataset(
        images=X_train,
        masks=y_train,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_ds = CustomDataset(
        images=X_test,
        masks=y_test,
        transform=val_transforms,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # im, mask = next(iter(train_loader))

    # im = im.detach().cpu().numpy()
    
    # imgps = ImageToPatches(args.image_height, 16)(im)
    # pemb = PatchEmbedding(768, 256)

    # import matplotlib.pyplot as plt

    # plt.imshow((im[0].permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8))
    # plt.show()
    # img0 = imgps[0][0].view(3, 16, 16).permute(1, 2, 0).detach().cpu().numpy()*255
    # for im_patch in imgps[0]:
    #     im_patch = im_patch.view(3, 16, 16).permute(1, 2, 0).detach().cpu().numpy()*255
    #     plt.imshow(im_patch.astype(np.uint8))
    #     plt.show()

    

    # print('image size')
    # print(im.size())
    # print(imgps.size())
    
    # embd = pemb(imgps)
    # print(embd.size())

    # vit = VisionTransformer(args.image_height, 16, 3, 256)
    # vout = vit(im)
    # print(vout.size())

    # mlp = MLP(vout.size(-1), dropout=0.2)
    # mlpout = mlp(vout)
    # print(mlpout.size())

    # att_block = SelfAttention(mlpout.size(-1), 8, dropout=0.2)
    # att_out = att_block(mlpout)
    # print(att_out.size())

    # projection = OutputProjection(im.size(-1), 16, 256, 3)
    # out_proj = projection(att_out)
    # print(out_proj.size())

    # vit = ViTSeg(args.image_height, args.patch_size, args.in_channels, args.out_channels, args.embed_size, args.num_blocks, args.num_heads, args.dropout ).cuda()
    # print_model_parameters(vit)
    # vit_out = vit(im.cuda())
    # print(vit_out.size())

    model = ViTSeg(args.image_height, args.patch_size, args.in_channels, args.out_channels, args.embed_size, args.num_blocks, args.num_heads, args.dropout ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, loss_fn, optimizer, train_loader, val_loader, args=args)

if __name__ == "__main__":
    main()