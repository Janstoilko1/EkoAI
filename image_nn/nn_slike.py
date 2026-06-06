import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt

epoch_num = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),        
    transforms.ColorJitter(0.2, 0.2, 0.2),  
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = ImageFolder(root = "C:/Users/Uporabnik/Desktop/Odpadki/dataset/dataset-resized", transform = val_transform)

train_size = int(0.8 * len(dataset))
validation_size = int(len(dataset) - train_size)

train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

train_dataset.dataset.transform = train_transform
validation_dataset.dataset.transform = val_transform


train_labels = [dataset.targets[i] for i in train_dataset.indices]
class_counts = [train_labels.count(i) for i in range(len(dataset.classes))]
print(f"Slik po razredih v treningu: {dict(zip(dataset.classes, class_counts))}")

sample_weights = [1.0 / class_counts[label] for label in train_labels]
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


train_data_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
validation_data_loader = DataLoader(validation_dataset, batch_size = 32, shuffle = False)

model = timm.create_model("efficientnet_b2", pretrained = True, num_classes = 3, drop_rate = 0.2)

model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

train_losses = []
validation_losses = []
accuracies = []

best_accuracy = 0
patience = 5
epochs_without_improvement = 0

for epoch in range(epoch_num):

    model.train()
    running_train_loss = 0

    for image, class_label in train_data_loader:

        image = image.to(device)
        class_label = class_label.to(device)

        optimizer.zero_grad()

        prediction = model(image)

        image_loss = loss(prediction, class_label)

        image_loss.backward()
        optimizer.step()

        running_train_loss += image_loss.item()

        average_train_loss = running_train_loss / len(train_data_loader)

    train_losses.append(average_train_loss)

    model.eval()
    running_validation_loss = 0

    total = 0
    correct = 0

    with torch.no_grad():
        for image, class_label in validation_data_loader:
            image = image.to(device)
            class_label = class_label.to(device)
            
            prediction = model(image)
            _,predicted_class = torch.max(prediction, 1)

            val_loss = loss(prediction, class_label)
            running_validation_loss += val_loss.item()
            total += class_label.size(0)

            correct += (predicted_class == class_label).sum().item()
        
    average_validation_loss = (
        running_validation_loss /
        len(validation_data_loader)
    )

    validation_losses.append(
        average_validation_loss
    )

    accuracy = 100 * correct / total

    accuracies.append(accuracy)
    scheduler.step(accuracy)

    if accuracy > best_accuracy:

        epochs_without_improvement = 0
        best_accuracy = accuracy

        torch.save({
            'model_state_dict': model.state_dict(),
            'best_accuracy':    best_accuracy,
            'classes':          dataset.classes,
            'class_to_idx':     dataset.class_to_idx,
        }, "best_model.pth")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement > patience:
        print(f"Early stopping, best accuracy: {best_accuracy}%\n")
        break

    print(f"Epoch {epoch+1}, accuracy: {accuracy:.2f}%")

print(dataset.classes)
plt.figure(figsize=(10,6))
plt.plot(
    train_losses,
    label="Train Loss"
)
plt.plot(
    validation_losses,
    label="Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss skozi ucenje")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,5))
plt.plot(accuracies, label="Accuracy")
plt.legend()
plt.title("Accuracy skozi epohe")
plt.show()