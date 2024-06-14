import os
import numpy as np

import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--image-dir', type=str, default="../carvana_segmentation/data2/train")
    parser.add_argument('--output-dir', type=str, default="data/ImageSets/Segmentation")
    parser.add_argument('--test-size', type=float, default=.15)
    parser.add_argument('--val-size', type=float, default=.15)

    args = parser.parse_args()

    image_dir   = args.image_dir

    out_dir     = args.output_dir
    test_size   = args.test_size
    val_size    = args.val_size
    train_size  = 1 - (val_size+test_size)

    image_ids = np.array([os.path.join(image_dir, x) for x in os.listdir(image_dir)])
    
    size = len(image_ids)

    indices = np.random.permutation(size)
    train_idx, val_idx, test_idx = indices[:int(size*train_size)], indices[:int(size*val_size)], indices[:int(size*test_size)]
    X_train, X_val, X_test = image_ids[train_idx], image_ids[val_idx], image_ids[test_idx]

    print(f'Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
        for im in X_train:
            name = im.split('/')[-1].split('.')[0]
            f.write(f'{name}\n')

    with open(os.path.join(out_dir, 'val.txt'), 'w') as f:
        for im in X_val:
            name = im.split('/')[-1].split('.')[0]
            f.write(f'{name}\n')

    with open(os.path.join(out_dir, 'test.txt'), 'w') as f:
        for im in X_test:
            name = im.split('/')[-1].split('.')[0]
            f.write(f'{name}\n')

if __name__ == "__main__":
    main()