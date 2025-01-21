import torch
from sklearn.metrics import accuracy_score

def train(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=10):
    train_losses = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        valid_accuracy = evaluate(model, valid_loader, device)
        valid_accuracies.append(valid_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}")

    return train_losses, valid_accuracies


def evaluate(model, valid_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in valid_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # Predicted class labels
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy