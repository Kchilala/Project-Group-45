import torch
from torchvision import transforms
from urdu_digits.dataloader import get_dataloaders

def main():
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),     
        transforms.ToTensor(),            
        transforms.Normalize((0.5,), (0.5,)) 
    ])

   
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv='data/train.csv',
        train_root='data/train/train',
        test_csv='data/test.csv',
        test_root='data/test/test',
        transform=transform,
        batch_size=32
    )

    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    
    images, labels = next(iter(train_loader))
    print(f"\nBatch Images Shape: {images.shape} (Batch, Channel, Height, Width)")
    print(f"Batch Labels Shape: {labels.shape}")
    print(f"Sample Labels: {labels[:10].tolist()}")

if __name__ == "__main__":
    main()