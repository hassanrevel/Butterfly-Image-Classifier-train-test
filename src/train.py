import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from src.model import efficientmodel
from src.datasets import get_data_loader
from src.utils import save_model, save_plots

# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs to train our model for', required=False)
    parser.add_argument('--pretrained', action='store_true',
                        help='Whether to use pretrained weights or not', required=False)
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine tune weights or not',required=False)
    parser.add_argument('-lr', '--learning-rate', type=float, dest='learning_rate', default=0.0001,
                        help='Learning rate for training the model', required=False)
    parser.add_argument('--batch_size', type=int, default=16,
                        required=False)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--data_dir', type=str,
                        help="path of directory where the data has been stored", required=True)
    parser.add_argument('--out_dir', type=str, default=None,
                        help="path where artifacts must be stored", required=False)
    parser.add_argument('--version', type=float, default=0.1,
                        help="Version of the model", required=False)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Loading the training and validation dataset
    train_dataloader, valid_dataloader, _ = get_data_loader(args, use_cuda)

    num_classes = len(os.listdir(os.path.join(args.data_dir, 'train')))

    print(f"\n[INFO]: Number of training images: {len(train_dataloader.dataset)}")
    print(f"[INFO]: Number of validation images: {len(valid_dataloader.dataset)}")
    print(f"[INFO]: Number of classes: {num_classes}\n")

    # Learning_parameters.
    lr = args.learning_rate
    epochs = args.epochs
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    model = efficientmodel(pretrained=args.pretrained, fine_tune=args.fine_tune, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimizer
    criterion = nn.CrossEntropyLoss()  # Loss Function

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_dataloader,
                                                  optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_dataloader,
                                                     criterion, device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

        # Save the artifacts
        if args.out_dir is not None:
            save_model(epochs, model, optimizer, criterion, out_dir=args.out_dir, version=args.version)
    if args.out_dir is not None:
        save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir=args.out_dir)
    print("All Done")

if __name__ == '__main__':
    main()