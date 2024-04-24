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

from models import ImageToPatches, PatchEmbedding, VisionTransformer
# from model import UNET
from utils import CustomDataset, train_one_epoch, check_accuracy, save_checkpoint, save_predictions_as_imgs


def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: x.split('.')[0])


def train(model, loss, optim, dl_train, dl_val, args):

    device = torch.device(args.device)
    model.to(device)

    print(torch.cuda.get_device_properties(0))

    avg_loss = []
    val_loss = []
    lowest_loss = 100000
    for epoch in range(args.NUM_EPOCHS):

        ti = next(iter(dl_train))[0]
        vi = next(iter(dl_val))[0]

        args.epoch = epoch
        l = train_one_epoch(model=model, loss_fn=loss, optim=optim, loader=dl_train, device=device, args=args)

        avg_loss.append(l)
        print("[%d/%d] - epoch end loss: %f"%(epoch,args.NUM_EPOCHS,avg_loss[-1]))

        metrics = check_accuracy(model, dl_val, loss, device, args=args)
        val_loss.append(metrics[3])

        print(f"[{epoch}/{args.NUM_EPOCHS}] - Eval metrics -> acc: {metrics[0]}, Dice: {metrics[1]}, IOU: {metrics[2]}, val loss: {val_loss[-1]}")
        
        if val_loss[-1] < lowest_loss:
            lowest_loss = val_loss[-1]
            print('A better model was found! saving...')

            actual_state = {'optim':optim.state_dict(),'model':model.state_dict(),'epoch':epoch}
            save_checkpoint(actual_state, "./saved_models", "best_model.pth")

            save_predictions_as_imgs(model, dl_val)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--image-dir', type=str, default="data2/train")
    parser.add_argument('--mask-dir', type=str, default="data2/train_masks")

    #image info
    parser.add_argument('--image-height', type=int, default=512)
    parser.add_argument('--image-width', type=int, default=512)

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=2)

    args = parser.parse_args()

    image_dir   = args.image_dir
    mask_dir    = args.mask_dir

    image_ids = np.array([os.path.join(image_dir, x) for x in sorted_fns(image_dir)])
    mask_ids = np.array([os.path.join(mask_dir, x.replace(".jpg", "_mask.gif")) for x in sorted_fns(mask_dir)])

    X_train, X_test, y_train, y_test = train_test_split(image_ids, mask_ids, test_size=0.33, random_state=42)

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

    im, mask = next(iter(train_loader))

    # im = im.detach().cpu().numpy()
    
    imgps = ImageToPatches(args.image_height, 16)(im)
    pemb = PatchEmbedding(768, 256)

    # import matplotlib.pyplot as plt

    # plt.imshow((im[0].permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8))
    # plt.show()
    img0 = imgps[0][0].view(3, 16, 16).permute(1, 2, 0).detach().cpu().numpy()*255
    # for im_patch in imgps[0]:
    #     im_patch = im_patch.view(3, 16, 16).permute(1, 2, 0).detach().cpu().numpy()*255
    #     plt.imshow(im_patch.astype(np.uint8))
    #     plt.show()

    

    print(im.size())
    print(imgps.size())
    
    embd = pemb(imgps)
    print(embd.size())

    vit = VisionTransformer(args.image_height, 16, 3, 256)
    print(vit(im).size())


if __name__ == "__main__":
    main()