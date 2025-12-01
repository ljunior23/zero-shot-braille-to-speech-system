"""
Train Braille character recognition CNN.
Classification model for individual Braille characters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class BrailleDataset(Dataset):
    """Dataset for Braille character images."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Args:
            root_dir: Root directory of dataset
            split: 'train' or 'val'
            transform: Optional transforms
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        
        # Load metadata
        with open(Path(root_dir) / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.characters = metadata['characters']
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Collect all image paths
        self.samples = []
        for char in self.characters:
            char_dir = self.root_dir / char
            for img_path in char_dir.glob('*.png'):
                self.samples.append((str(img_path), self.char_to_idx[char]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to PIL format for transforms
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class BrailleCNN(nn.Module):
    """CNN for Braille character recognition."""
    
    def __init__(self, num_classes: int = 26):
        """
        Args:
            num_classes: Number of characters to classify
        """
        super(BrailleCNN, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 60x40 -> 30x20
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 30x20 -> 15x10
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 15x10 -> 7x5
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {save_path}")
    plt.close()


def main():
    """Main training loop."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Braille CNN')
    parser.add_argument('--data', default='data/braille_dataset',
                       help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output', default='models/braille_cnn.pt',
                       help='Output model path')
    parser.add_argument('--device', default=None,
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print("="*60)
    print("Training Braille Character Recognition CNN")
    print("="*60)
    print(f"Dataset: {args.data}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Datasets
    train_dataset = BrailleDataset(args.data, 'train', train_transform)
    val_dataset = BrailleDataset(args.data, 'val', val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.characters)}")
    print()
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Model
    model = BrailleCNN(num_classes=len(train_dataset.characters))
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    print()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print()
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'char_to_idx': train_dataset.char_to_idx,
                'idx_to_char': train_dataset.idx_to_char,
            }, output_path)
            
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
            print()
    
    # Plot training curves
    plot_path = Path(args.output).parent / 'braille_training_curves.png'
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, plot_path)
    
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()