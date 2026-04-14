import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class UrduDigitDataset(Dataset):
    """
    Dataset class for Urdu Digit images.
    Organized by subfolders (0-9) for training and a single folder for testing.
    """
    def __init__(self, root_dir, csv_file, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.df = pd.read_csv(csv_file)
        
        self.image_paths = []
        self.labels = []
        
        if not is_test:
            
            for _, row in self.df.iterrows():
                img_id = row['Id']
                label = row['Category']
                
                path = os.path.join(self.root_dir, str(label), f"{int(img_id)}.png")
                self.image_paths.append(path)
                self.labels.append(label)
        else:
            
            for _, row in self.df.iterrows():
                img_id = row['Id']
                path = os.path.join(self.root_dir, f"{int(img_id)}.png")
                self.image_paths.append(path)
                self.labels = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, self.df.iloc[idx]['Id']
        
        return image, self.labels[idx]
