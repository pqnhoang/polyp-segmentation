import argparse
from .model.model import NeoPolypModel
from data.dataset import NeoPolypDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode, ToPILImage
import torch
import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description='NeoPolyp Inference')
parser.add_argument('--model', type=str, default='model.pth',
                    help='model path')
parser.add_argument('--data_path', type=str, default='data',
                    help='data path')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers')
parser.add_argument('--save_path', type=str, default='/kaggle/working/predicted_masks',
                    help='save path')
parser.add_argument('--csv_path', type=str, default='/kaggle/working/',
                    help='csv path')
args = parser.parse_args()


def mask2rgb(mask):
    color_dict = {0: torch.tensor([0, 0, 0]),
                  1: torch.tensor([1, 0, 0]),
                  2: torch.tensor([0, 1, 0])}
    output = torch.zeros((mask.shape[0], mask.shape[1], 3)).long()
    for k in color_dict.keys():
        output[mask.long() == k] = color_dict[k]
    return output.to(mask.device)


def main():
    print("Loading model...!")
    model = NeoPolypModel.load_from_checkpoint(args.model)
    model.eval()
    all_path = []
    for root, dirs, files in os.walk(os.path.join(args.data_path, "test")):
        for f in files:
            all_path.append(os.path.join(root, f))
    test_dataset = NeoPolypDataset(all_path, session="test")
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    print("Predicting...!")
    for _, (img, file_id, H, W) in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
        with torch.no_grad():
            predicted_mask = model(img.cuda())
        for i in range(args.batch_size):
            filename = file_id[i] + ".png"
            argmax = torch.argmax(predicted_mask[i], 0)
            one_hot = mask2rgb(argmax).float().permute(2, 0, 1)
            mask2img = Resize((H[i].item(), W[i].item()), interpolation=InterpolationMode.NEAREST)(
                ToPILImage()(one_hot))
            mask2img.save(os.path.join(args.save_path, filename))

    def rle_to_string(runs):
        return ' '.join(str(x) for x in runs)

    def rle_encode_one_mask(mask):
        pixels = mask.flatten()
        pixels[pixels > 0] = 255
        use_padding = False
        if pixels[0] or pixels[-1]:
            use_padding = True
            pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
            pixel_padded[1:-1] = pixels
            pixels = pixel_padded

        rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
        if use_padding:
            rle = rle - 1
        rle[1::2] = rle[1::2] - rle[:-1:2]
        return rle_to_string(rle)

    def mask2string(dir):
        # mask --> string
        strings = []
        ids = []
        ws, hs = [[] for i in range(2)]
        for image_id in tqdm(os.listdir(dir), total=len(os.listdir(dir))):
            id = image_id.split('.')[0]
            path = os.path.join(dir, image_id)
            img = cv2.imread(path)[:, :, ::-1]
            h, w = img.shape[0], img.shape[1]
            for channel in range(2):
                ws.append(w)
                hs.append(h)
                ids.append(f'{id}_{channel}')
                string = rle_encode_one_mask(img[:, :, channel])
                strings.append(string)
        r = {
            'ids': ids,
            'strings': strings,
        }
        return r

    print("Converting to csv...!")
    res = mask2string(args.save_path)
    df = pd.DataFrame(columns=['Id', 'Expected'])
    df['Id'] = res['ids']
    df['Expected'] = res['strings']
    if not os.path.isdir(args.csv_path):
        os.mkdir(args.csv_path)
    df.to_csv(os.path.join(args.csv_path, 'output.csv'), index=False)


if __name__ == '__main__':
    main()