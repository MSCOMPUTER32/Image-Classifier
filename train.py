import os
import argparse  
import json 
import torch 
import torch.nn as nn  
import torch.optim as optim  
from torchvision import models, transforms  
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train a flower classifier')
parser.add_argument('data_directory', type=str, default='flowers', help='Directory of flower data')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Model architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=4096, help='Hidden units in the classifier')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args()

data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),  
        transforms.CenterCrop(224),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),  
        transforms.CenterCrop(224),  
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]),
}

image_datasets = {
    'train': ImageFolder(root=train_dir, transform=data_transforms['train']),
    'valid': ImageFolder(root=valid_dir, transform=data_transforms['valid']),
    'test': ImageFolder(root=test_dir, transform=data_transforms['test']),
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'valid': DataLoader(image_datasets['valid'], batch_size=32, shuffle=False),
    'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False),
}

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
if args.arch == 'vgg16':
    model = models.vgg16(weights='DEFAULT')
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(25088, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden_units, 2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 102),
    )
elif args.arch == 'resnet50':
    model = models.resnet50(weights='DEFAULT')
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden_units, 2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 102),
    )

device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
model.to(device)  

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters() if args.arch == 'vgg16' else model.fc.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_epochs = args.epochs
train_losses = []
valid_losses = []
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  
        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloaders['train'])  
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(dataloaders["train"]):.4f}')

    model.eval()  
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)  
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    valid_losses.append(val_loss / len(dataloaders["valid"]))
    print(f'Validation Loss: {val_loss/len(dataloaders["valid"]):.4f}, Accuracy: {accuracy:.2f}%')

    os.makedirs(args.save_dir, exist_ok=True) 

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        checkpoint = {
            'state_dict': model.state_dict(),
            'class_to_idx': image_datasets['train'].class_to_idx,
            'architecture': args.arch,
            'hidden_units': args.hidden_units
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f'best_model_{args.arch}.pth'))

    scheduler.step()

plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.show()