https://claude.ai/share/89e33cda-cc26-4828-8927-2d6a89e34dbd

I can see this is your CS322P Deep Learning lab exam paper. Let me solve all parts systematically.

## (a) CNN Architecture

```
Input: [B, 1, 32, 32]
        ↓
Conv2D (W1: 16 filters, 3×3, stride 1, pad 1)
z1: [B, 16, 32, 32]  → ReLU → MaxPool(2×2, stride 2)
a1: [B, 16, 16, 16]
        ↓
Conv2D (W2: 32 filters, 3×3, stride 1, pad 1)
z2: [B, 32, 16, 16]  → ReLU → MaxPool(2×2, stride 2)
a2: [B, 32, 8, 8]
        ↓
Flatten → [B, 2048]
        ↓
Fully Connected (W3)
z3: [B, 6]  → Softmax
ŷ:  [B, 6]
```

---

## (b) Parameter Shapes

| Parameter | Shape | Why |
|---|---|---|
| W1 | (16, 1, 3, 3) | 16 filters, 1 input channel, 3×3 kernel |
| b1 | (16,) | one bias per filter |
| W2 | (32, 16, 3, 3) | 32 filters, 16 input channels, 3×3 |
| b2 | (32,) | |
| W3 | (6, 2048) | 6 classes, flattened 32×8×8=2048 |
| b3 | (6,) | |

**Activations with batch size B:**

| Tensor | Shape |
|---|---|
| z1, a1 (before pool) | (B, 16, 32, 32) |
| after MaxPool1 | (B, 16, 16, 16) |
| z2, a2 (before pool) | (B, 32, 16, 16) |
| after MaxPool2 | (B, 32, 8, 8) |
| Flatten | (B, 2048) |
| z3, ŷ | (B, 6) |

---

## (c) Generate Synthetic Dataset

```python
import numpy as np

np.random.seed(42)
n = 6000
num_classes = 6

# Grayscale 32x32 images, normalized to [0, 1]
X = np.random.randint(0, 256, size=(n, 1, 32, 32)).astype(np.float32) / 255.0

# One-hot encoded labels
labels = np.random.randint(0, num_classes, size=n)
Y = np.eye(num_classes)[labels].astype(np.float32)  # (6000, 6)

print("X shape:", X.shape)   # (6000, 1, 32, 32)
print("Y shape:", Y.shape)   # (6000, 6)
```

---

## (d) CNN Implementation in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=1)

        # Fully connected
        self.fc = nn.Linear(32 * 8 * 8, 6)  # 2048 → 6

    def forward(self, x):
        # Layer 1: Conv → ReLU → MaxPool
        z1 = self.conv1(x)
        a1 = self.pool(F.relu(z1))       # [B, 16, 16, 16]

        # Layer 2: Conv → ReLU → MaxPool
        z2 = self.conv2(a1)
        a2 = self.pool(F.relu(z2))       # [B, 32, 8, 8]

        # Flatten
        flat = a2.view(a2.size(0), -1)   # [B, 2048]

        # Fully connected + Softmax
        z3 = self.fc(flat)               # [B, 6]
        y_hat = F.softmax(z3, dim=1)
        return y_hat

model = CNN()
print(model)
```

---

## (e) Batch Gradient Descent Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Prepare data ---
X_tensor = torch.tensor(X)                          # (6000, 1, 32, 32)
Y_tensor = torch.tensor(Y)                          # (6000, 6)
labels_tensor = torch.argmax(Y_tensor, dim=1)       # (6000,) for CrossEntropyLoss

dataset = TensorDataset(X_tensor, labels_tensor)
loader = DataLoader(dataset, batch_size=6000, shuffle=True)  # full-batch GD

# --- Model, loss, optimizer ---
model = CNN()
criterion = nn.CrossEntropyLoss()                   # cross-entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.005) # α = 0.005

# --- Training loop ---
for epoch in range(1, 11):  # 10 epochs
    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        # Forward pass (use z3 directly, not softmax — CrossEntropyLoss applies it internally)
        z3 = model.fc(
            model.pool(F.relu(model.conv2(
                model.pool(F.relu(model.conv1(X_batch)))
            ))).view(X_batch.size(0), -1)
        )

        loss = criterion(z3, y_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch:2d}/10 | Loss: {loss.item():.4f}")
```

> **Tip:** `nn.CrossEntropyLoss` internally applies log-softmax, so pass raw `z3` logits to it — not the softmax output. This matches the loss formula in the paper exactly.

---

**Quick revision checklist before your practical:**
- Know the formula: output size = `(W - F + 2P) / S + 1`
- After MaxPool(2×2): spatial dims halve
- `32→16→8` after two pools → flatten = `32 × 8 × 8 = 2048`
- CrossEntropyLoss = NLLLoss + LogSoftmax combined

This is the 2023-24 paper — a full MNIST pipeline. Let me write the complete solution covering all 6 sections.

---

## 1. Data Preparation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Normalize to [0, 1]
transform = transforms.Compose([
    transforms.ToTensor()  # scales pixel values to [0.0, 1.0] automatically
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

print(f"Train samples: {len(train_dataset)}")  # 60,000
print(f"Test samples:  {len(test_dataset)}")   # 10,000
```

---

## 2. Network Architecture

```
Input (784) → FC(128) → ReLU → FC(64) → ReLU → FC(10) → Softmax
```

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)   # Input → Hidden 1
        self.fc2 = nn.Linear(128, 64)    # Hidden 1 → Hidden 2
        self.fc3 = nn.Linear(64, 10)     # Hidden 2 → Output

    def forward(self, x):
        x = x.view(-1, 784)             # Flatten 28×28 → 784
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)                 # Raw logits (softmax inside loss)
        return x

model = MNISTNet()
print(model)
# Total params: 784×128 + 128×64 + 64×10 = ~109K
```

---

## 3. Training

```python
criterion = nn.CrossEntropyLoss()              # categorical cross-entropy (logits input)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

for epoch in range(1, 11):
    # --- Training ---
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc  = correct / total

    # --- Validation (using test set here) ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss    += loss.item()
            preds        = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_loss /= len(test_loader)
    val_acc   = val_correct / val_total

    train_losses.append(train_loss); val_losses.append(val_loss)
    train_accs.append(train_acc);    val_accs.append(val_acc)

    print(f"Epoch {epoch:2d}/10 | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
```

---

## 4. Evaluation

```python
model.eval()
test_loss, correct, total = 0, 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        test_loss += criterion(outputs, labels).item()
        preds      = outputs.argmax(dim=1)
        correct   += (preds == labels).sum().item()
        total     += labels.size(0)

print(f"Test Loss:     {test_loss / len(test_loader):.4f}")
print(f"Test Accuracy: {correct / total * 100:.2f}%")
# Expected: ~97-98% accuracy
```

---

## 5. Hyperparameter Tuning

```python
import pandas as pd

configs = [
    {"lr": 0.01,   "batch_size": 32},
    {"lr": 0.001,  "batch_size": 32},
    {"lr": 0.0001, "batch_size": 32},
    {"lr": 0.001,  "batch_size": 64},
    {"lr": 0.001,  "batch_size": 128},
]

results = []

for cfg in configs:
    tr_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    te_loader = DataLoader(test_dataset,  batch_size=cfg["batch_size"], shuffle=False)

    m   = MNISTNet()
    opt = optim.Adam(m.parameters(), lr=cfg["lr"])

    # Train 5 epochs for quick comparison
    for _ in range(5):
        m.train()
        for imgs, lbls in tr_loader:
            opt.zero_grad()
            loss = criterion(m(imgs), lbls)
            loss.backward(); opt.step()

    # Evaluate
    m.eval(); correct = 0; total = 0
    with torch.no_grad():
        for imgs, lbls in te_loader:
            preds   = m(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)

    acc = correct / total * 100
    results.append({**cfg, "accuracy": f"{acc:.2f}%"})
    print(f"lr={cfg['lr']}, batch={cfg['batch_size']} → Acc: {acc:.2f}%")

print(pd.DataFrame(results))
```

**Expected observations:**

| lr | batch | Accuracy |
|---|---|---|
| 0.01 | 32 | ~96% (unstable) |
| 0.001 | 32 | ~98% (best) |
| 0.0001 | 32 | ~95% (slow) |
| 0.001 | 64 | ~97.5% |
| 0.001 | 128 | ~97% |

---

## 6. Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# --- Plot 1: Loss & Accuracy curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses,   label='Val Loss')
axes[0].set_title('Loss over Epochs')
axes[0].set_xlabel('Epoch'); axes[0].legend()

axes[1].plot(train_accs, label='Train Acc')
axes[1].plot(val_accs,   label='Val Acc')
axes[1].set_title('Accuracy over Epochs')
axes[1].set_xlabel('Epoch'); axes[1].legend()

plt.tight_layout(); plt.savefig('loss_accuracy.png'); plt.show()

# --- Plot 2: Sample predictions ---
model.eval()
images, labels = next(iter(test_loader))

with torch.no_grad():
    preds = model(images).argmax(dim=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img = images[i].squeeze().numpy()
    ax.imshow(img, cmap='gray')
    color = 'green' if preds[i] == labels[i] else 'red'
    ax.set_title(f"Pred: {preds[i].item()} | True: {labels[i].item()}", color=color)
    ax.axis('off')

plt.suptitle('Test Predictions (green=correct, red=wrong)')
plt.tight_layout(); plt.savefig('predictions.png'); plt.show()
```

---

**Key numbers to remember for your practical:**

| | Value |
|---|---|
| Input size | 784 (28×28 flattened) |
| Architecture | 784 → 128 → 64 → 10 |
| Total params | ~109,386 |
| Optimizer | Adam, lr=0.001 |
| Loss | CrossEntropyLoss |
| Expected accuracy | ~97–98% |

Okay, crunch time! Since you need to *understand* not just memorize, I'll teach you the mental model behind everything — so you can reconstruct the code even if you forget a line.

---

# The Big Picture — What is a Neural Network doing?

Think of it like a **series of transformations** on data:

```
Raw Data → [Layer 1] → [Layer 2] → ... → Prediction → Compare with truth → Fix weights → Repeat
```

Every layer just does: **multiply → add bias → apply activation**. That's it.

---

# Concept 1 — The Two Networks in your papers

| | 2023-24 Paper | 2024-25 Paper |
|---|---|---|
| Name | MLP (Fully Connected) | CNN (Convolutional) |
| Input | 28×28 image → **flatten to 784** | 32×32 image → **keep 2D shape** |
| Good at | Simple patterns | Spatial patterns (edges, shapes) |
| Layers | Linear → Linear → Linear | Conv → Conv → Linear |

---

# Concept 2 — Why Flatten for MLP, but NOT for CNN?

**MLP** doesn't understand "pixels next to each other are related." So you smash the whole image into one long list:
```
28×28 image → 784 numbers in a row → feed to neurons
```

**CNN** *does* understand spatial relationships. It slides a small filter (like a 3×3 magnifying glass) across the image looking for edges, curves, etc. So you keep the 2D shape.

---

# Concept 3 — Layers, simply explained

### Linear Layer (both papers)
```python
nn.Linear(in, out)
# Does: output = input × W + b
# W shape: (out, in)   ← remember this!
```

### Conv2D Layer (2024-25 paper)
```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
# Slides a filter across the image
# Output size = (W - F + 2P) / S + 1
```

For your exam: input 32×32, filter 3×3, stride 1, padding 1:
```
(32 - 3 + 2×1) / 1 + 1 = 32   ← size stays same with padding=1!
```

### MaxPool
```python
nn.MaxPool2d(2, 2)
# Just takes the max in each 2×2 block
# Effect: HALVES the spatial dimensions → 32→16→8
```

### ReLU
```python
F.relu(x)
# Does: if x < 0 → 0, else keep x
# Why: adds non-linearity so network learns complex patterns
```

### Softmax
```python
F.softmax(x, dim=1)
# Converts raw scores to probabilities that sum to 1
# e.g. [2.1, 0.3, 1.5] → [0.6, 0.1, 0.3]
```

---

# Concept 4 — The Training Loop (same structure ALWAYS)

This is the most important thing to memorize. Every training loop is these **5 steps**:

```python
for epoch in range(num_epochs):
    for batch in dataloader:

        # 1. ZERO the gradients (clear previous step)
        optimizer.zero_grad()

        # 2. FORWARD pass (get predictions)
        outputs = model(inputs)

        # 3. COMPUTE loss (how wrong are we?)
        loss = criterion(outputs, labels)

        # 4. BACKWARD pass (calculate gradients)
        loss.backward()

        # 5. UPDATE weights
        optimizer.step()
```

**Analogy:** You guess an answer → check how wrong → figure out which way to fix → take a small step in that direction. Repeat 10,000 times.

---

# Concept 5 — Loss Functions (which one and why)

| Loss | When to use | In PyTorch |
|---|---|---|
| CrossEntropyLoss | Multi-class (your exams) | `nn.CrossEntropyLoss()` |

**⚠️ Critical gotcha:** `CrossEntropyLoss` applies softmax *internally*. So pass **raw logits** (output of last Linear layer), NOT softmax output. Both your papers need this.

---

# Concept 6 — Optimizers

```python
# Adam — smart, adapts learning rate, almost always works
optim.Adam(model.parameters(), lr=0.001)

# SGD — simple gradient descent
optim.SGD(model.parameters(), lr=0.005)
```

2023-24 asks for **Adam**, 2024-25 asks for **SGD (lr=0.005)**.

---

# Concept 7 — Shape Tracking (the skill that gets you marks)

Train yourself to track shape at every step. For the CNN paper:

```
Input:          [B, 1, 32, 32]   ← B=batch, 1=grayscale, 32×32
After Conv1:    [B, 16, 32, 32]  ← 16 filters, size same (padding=1)
After MaxPool1: [B, 16, 16, 16]  ← halved
After Conv2:    [B, 32, 16, 16]  ← 32 filters
After MaxPool2: [B, 32,  8,  8]  ← halved again
After Flatten:  [B, 2048]        ← 32×8×8 = 2048
After FC:       [B, 6]           ← 6 classes
```

For MLP:
```
Input:     [B, 784]   ← 28×28 flattened
After FC1: [B, 128]
After FC2: [B, 64]
After FC3: [B, 10]    ← 10 digit classes
```

---

# Quick Reference Card

```python
# ─── MLP skeleton (2023-24) ────────────────────────────
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)          # flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)           # raw logits, NO softmax here


# ─── CNN skeleton (2024-25) ────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc    = nn.Linear(32*8*8, 6)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [B,16,16,16]
        x = self.pool(torch.relu(self.conv2(x)))  # [B,32,8,8]
        x = x.view(-1, 32*8*8)                   # flatten
        return self.fc(x)                         # raw logits


# ─── Training loop (same for both!) ────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)   # or SGD

for epoch in range(10):
    for images, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# ─── Evaluation ────────────────────────────────────────
model.eval()
with torch.no_grad():          # no gradient needed for eval
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
```

---

# What to focus on in the next few hours

1. **Understand the 5-step training loop** — you'll write this from memory
2. **Know the shape at every layer** — examiners love asking this
3. **Remember the 2 gotchas:**
   - `x.view(-1, 784)` to flatten for MLP
   - Pass raw logits to `CrossEntropyLoss`, not softmax output
4. **Know which paper uses what:** Adam vs SGD, MLP vs CNN, 10 classes vs 6 classes

Want me to do a **mock Q&A** where I ask you questions like an examiner, or go deeper on any specific concept?
