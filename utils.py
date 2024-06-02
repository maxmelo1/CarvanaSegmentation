import torch
import torchvision
from dataset import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

PALETTE_PRED = [0,0,0, 255, 0, 0]
PALETTE_GT = [0,0,0, 255, 0, 0]

def save_checkpoint(state, path='./', filename="my_checkpoint.pth.tar"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print("=> Saving checkpoint")
    torch.save(state, os.path.join(path, filename))

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CustomDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CustomDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(model, loader, loss_fn, device="cuda", args=None):
    
    lacc = []
    ldice_score = []
    liou = []
    
    model.eval()

    loss = []

    with torch.no_grad():
        with tqdm(total=len(loader), desc=f'Epoch {args.epoch + 1}/{args.NUM_EPOCHS}', unit='batch') as pbar:
            pbar.set_description(f'Validating epoch {args.epoch}/{args.NUM_EPOCHS}')
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                
                l = loss_fn(preds.squeeze(1), y)
                loss.append(l)

                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).float()

                y = y.unsqueeze(1)

                num_correct = (preds == y).sum()
                num_pixels = torch.numel(preds)
                dice_score = (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
                )
                intersection = (preds.int() & y.int()).sum()
                union = (preds.int() | y.int()).sum()
                
                iou = (intersection + 1e-8) / (union + 1e-8)
                
                acc = num_correct/num_pixels

                lacc.append(acc)
                ldice_score.append(dice_score)
                liou.append(iou)

                pbar.update(1)

                pbar.set_postfix(acc=acc.cpu().item(),
                                 dice=dice_score.cpu().item(),
                                 iou=iou.cpu().item()
                )
                

    # acc = num_correct/num_pixels
    # dice_score /= num_pixels
    # iou /= num_pixels

    metrics = {'Acc': torch.stack(lacc).mean().cpu().item(), 'Dice': torch.stack(ldice_score).mean().cpu().item(), 'IoU': torch.stack(liou).mean().cpu().item(), 'ValLoss': torch.stack(loss).mean().cpu().item()}

    return metrics

def save_predictions_as_imgs(
    model, loader, path="saved_images/", device="cuda"
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    model.eval()

    X, y = next(iter(loader))
    X = X.to(device=device)

    with torch.no_grad():
        preds = torch.sigmoid(model(X))
        preds = (preds > 0.5).float()#.cpu().numpy()

        for idx, (pred, lbl) in enumerate(zip(preds, y)):
            # torchvision.utils.save_image(
            #     pred, os.path.join(path, f"pred_{idx}.png")
            # )
            # torchvision.utils.save_image(lbl.unsqueeze(0), os.path.join(path, f"{idx}.png"))
            im_pred = Image.fromarray(pred.permute(1,2,0).squeeze(-1).cpu().numpy().astype(np.uint8), mode='L')
            im_pred.putpalette(PALETTE_PRED)
            im_pred.save(os.path.join(path, f"pred_{idx}.png"))
            im_lbl = Image.fromarray(lbl.cpu().numpy().astype(np.uint8), mode='L')
            im_lbl.putpalette(PALETTE_GT)
            im_lbl.save(os.path.join(path, f"{idx}.png"))

    model.train()


def train_one_epoch(model, loss_fn, optim, loader, device, args):
    loss = []
    model.train()
    with tqdm(total=len(loader), desc=f'Epoch {args.epoch + 1}/{args.NUM_EPOCHS}', unit='batch') as pbar:
        pbar.set_description(f'Epoch {args.epoch}/{args.NUM_EPOCHS}')
        for X,y in loader:
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            optim.zero_grad()

            pred = model(X)
            l = loss_fn(pred.squeeze(1), y)

            l.backward()
            optim.step()

            loss.append(l)
            pbar.update(1)
            pbar.set_postfix(loss=l.cpu().item())


    return torch.mean(torch.stack(loss))


def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"Parameters: {num_model_parameters/1e6:.2f}M")

    