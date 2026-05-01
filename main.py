import torch
import torch.nn as nn
import sys
from pathlib import Path
import pandas as pd
from urdu_digits.model import ResNet18
from torchvision import transforms
from urdu_digits.dataloader import get_dataloaders
from urdu_digits.train_evaluate import evaluate, predict, train_one_epoch

def main():
    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # This transformation is used to transform the images from the dataLoader into 3 input channels for ResNet18.
    # it resizes images to 224x224.
    # Normalizes to the standard ImageNet mean/std values.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
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

    model = ResNet18(num_classes=10, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    num_epochs = 15
    best_val_loss = float("inf")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / "best_resnet18.pth"
    submission_path = Path("submission.csv")

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path} with val loss {best_val_loss:.4f}")

    print(f"\nLoading best model from {best_model_path} for test predictions...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_ids, test_categories = predict(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    sample_submission = pd.read_csv("data/sample_submission.csv")
    submission = pd.DataFrame({
        "Id": test_ids,
        "Category": test_categories,
    })
    submission = submission[sample_submission.columns]
    submission.to_csv(submission_path, index=False)
    print(f"Saved Kaggle submission file to {submission_path}")

if __name__ == "__main__":
    main()
