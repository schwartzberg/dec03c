import torch


def train(model, train_loader, valid_loader, loss_fn, optimizer, metric, device, epochs):
    """Record mean training loss and training and validation accuracy each epoch.

    The loss function must return the mean loss for a batch.
    """
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    model.to(device)
    metric.to(device)

    for _ in range(epochs):
        model.train()
        metric.reset()
        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            metric.update(predictions.detach(), labels)

        history["train_losses"].append(total_loss / total_samples)
        history["train_metrics"].append(metric.compute().item())

        model.eval()
        metric.reset()
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images)
                metric.update(predictions, labels)

        history["valid_metrics"].append(metric.compute().item())
        metric.reset()

    return history
