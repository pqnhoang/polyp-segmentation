from .transform import Transform
from torch.utils.data import Dataset
import cv2
import numpy as np


class NeoPolypDataset(Dataset):
    def __init__(
        self,
        image_dir: list,
        gt_dir: list | None = None,
        session: str = "train",
        transform: bool = True,
    ) -> None:
        super().__init__()
        self.session = session
        if session == "train":
            self.image_paths = image_dir
            self.gt_paths = gt_dir
            self.length = len(self.image_paths)
        elif session == "val":
            self.image_paths = image_dir
            self.gt_paths = gt_dir
            self.length = len(self.image_paths)
        else:
            self.image_paths = image_dir
            self.length = len(self.image_paths)
        self.transform = Transform(session) if transform else None

    @staticmethod
    def _process_mask(mask_path):
        image = cv2.imread(mask_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define the red color range boundaries (Hue 0-10 and 160-180)
        lower_red1 = np.array([0, 100, 20])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 20])
        upper_red2 = np.array([179, 255, 255])
        
        # Create red masks for both boundaries and combine them
        red_mask_lower = cv2.inRange(image, lower_red1, upper_red1)
        red_mask_upper = cv2.inRange(image, lower_red2, upper_red2)
        red_mask = red_mask_lower + red_mask_upper
        red_mask[red_mask != 0] = 1
        
        # Define green color range boundaries (Hue 36-70)
        green_mask = cv2.inRange(image, (36, 25, 25), (70, 255, 255))
        green_mask[green_mask != 0] = 2
        
        # Combine red and green masks
        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        return combined_mask.astype(np.uint8)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if self.session == "train":
            image = cv2.imread(self.image_paths[index])
            mask = self._process_mask(self.gt_paths[index])
            return self.transform(image, mask) if self.transform else (image, mask)
        elif self.session == "val":
            image = cv2.imread(self.image_paths[index])
            mask = self._process_mask(self.gt_paths[index])
            return self.transform(image, mask) if self.transform else (image, mask)
        else:
            image = cv2.imread(self.image_paths[index])
            height, width, _ = image.shape
            transformed_image = self.transform(image) if self.transform else image
            file_id = self.image_paths[index].split('/')[-1].split('.')[0]
            return transformed_image, file_id, height, width
