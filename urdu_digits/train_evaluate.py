import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
"""
Takes one full pass through the training data
and updates the model weights.
 
Takes one full pass through the validation data
but does not update the weights.
 """

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = (
        tqdm(dataloader, total=len(dataloader), desc="Training", leave=False)
        if tqdm is not None
        else dataloader
    )

    for batch_index, (images, labels) in enumerate(progress_bar, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if tqdm is not None:
            progress_bar.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
                batch=f"{batch_index}/{len(dataloader)}",
            )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = (
        tqdm(dataloader, total=len(dataloader), desc="Validation", leave=False)
        if tqdm is not None
        else dataloader
    )

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(progress_bar, start=1):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if tqdm is not None:
                progress_bar.set_postfix(
                    loss=f"{running_loss / total:.4f}",
                    acc=f"{correct / total:.4f}",
                    batch=f"{batch_index}/{len(dataloader)}",
                )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def predict(model, dataloader, device):
    model.eval()

    ids = []
    categories = []

    with torch.no_grad():
        for images, image_ids in dataloader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()

            if hasattr(image_ids, "tolist"):
                image_ids = image_ids.tolist()

            ids.extend(int(image_id) for image_id in image_ids)
            categories.extend(int(prediction) for prediction in predictions)

    return ids, categories
