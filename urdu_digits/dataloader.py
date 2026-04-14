from torch.utils.data import DataLoader, random_split
from .dataset import UrduDigitDataset

def get_dataloaders(train_csv, train_root, test_csv, test_root, transform, batch_size=32, val_split=0.2):
    """
    Creates and returns train, validation, and test dataloaders.
    """
    # Create the full training dataset
    full_train_dataset = UrduDigitDataset(
        root_dir=train_root, 
        csv_file=train_csv, 
        transform=transform, 
        is_test=False
    )
    
    # Split training set into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Create the test dataset
    test_dataset = UrduDigitDataset(
        root_dir=test_root, 
        csv_file=test_csv, 
        transform=transform, 
        is_test=True
    )
    
    # Instantiate dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
