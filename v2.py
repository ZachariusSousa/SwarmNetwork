import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

###############################################################################
#                          HYPERPARAMETERS
###############################################################################
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 5
TIMESTEPS = 3       # How many recurrent steps in the "mass"
IMG_SIZE = 28       # We'll assume MNIST's 28x28
KERNEL_SIZE = 3     # Local adjacency via a 3x3 kernel
PADDING = 1         # So output stays 28x28
NUM_OUTPUTS = 10    # MNIST has 10 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

###############################################################################
#                          DATASET & DATALOADERS
###############################################################################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # standard for MNIST
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

###############################################################################
#                    MODEL: BrainMassNet w/ Multiple Subspaces
###############################################################################
class BrainMassNet(nn.Module):
    """
    A 2D 'mass' of neurons, each cell in a (28 x 28) grid.
    Local adjacency is defined by a 3x3 convolution kernel (1 -> 1 channel).
    We unroll for TIMESTEPS.

    For the outputs: we have NUM_OUTPUTS trainable subregions (circles),
    each producing one scalar from the final activation, for a total of 10 scalars
    used directly as the class logits.
    """
    def __init__(self, img_size, timesteps, num_outputs):
        super().__init__()
        self.img_size = img_size   # e.g. 28
        self.timesteps = timesteps
        self.num_outputs = num_outputs

        # Local adjacency as a single-channel Conv2d
        # in_channels=1, out_channels=1 => shape: (1,1,kernel_size,kernel_size)
        # This is effectively a "shared adjacency" for the entire 2D grid
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=KERNEL_SIZE,
            padding=PADDING,
            bias=True
        )

        # We'll define a simple scale for how we add the input at each step
        # This is a learned parameter that merges the raw input image into the activation.
        self.input_scale = nn.Parameter(torch.ones(1))

        # For each of the num_outputs classes, define a (mu_x, mu_y, log_radius).
        # We'll keep them in [0,1] for the grid, then map to 28x28.
        # We'll store them in two separate parameters for clarity:
        #   subspace_centers: shape (num_outputs, 2) for (mu_x, mu_y)
        #   subspace_log_r:   shape (num_outputs,) for log(radius)
        self.subspace_centers = nn.Parameter(
            torch.rand(num_outputs, 2)  # each in [0,1], initially random
        )
        self.subspace_log_r = nn.Parameter(
            torch.full((num_outputs,), -1.0)  # log(r), so r ~ e^-1 = 0.367 initially
        )

        # Precompute a coordinate grid for each neuron in [0,1] x [0,1].
        # coords_y[x], coords_x[x] => the normalized position of each pixel center
        y_coords = torch.linspace(0.5/img_size, 1.0 - 0.5/img_size, steps=img_size)
        x_coords = torch.linspace(0.5/img_size, 1.0 - 0.5/img_size, steps=img_size)
        # shape (img_size,) each
        # We might do a meshgrid to get shape (img_size, img_size)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # We'll store them as buffers so they're on the correct device
        self.register_buffer("grid_y", yy)  # (28, 28)
        self.register_buffer("grid_x", xx)  # (28, 28)

    def forward(self, x):
        """
        x: shape (batch_size, 1, 28, 28) for MNIST
        Returns: logits of shape (batch_size, 10)
        """
        batch_size = x.size(0)

        # Initialize the mass activation as zeros
        a = torch.zeros_like(x, device=x.device)  # shape (batch_size,1,28,28)

        # Recurrent updates
        for _ in range(self.timesteps):
            # Weighted sum of local neighbors
            # shape => (batch_size,1,28,28)
            local_update = self.conv(a)

            # Add scaled input at each step (you could do it just once at t=0 if you prefer)
            update = local_update + self.input_scale * x

            # Tanh nonlinearity
            a = torch.tanh(update)

        # Now 'a' is our final activation map after T timesteps
        # shape (batch_size,1,28,28)

        # For each subspace i in [0..num_outputs-1], compute a circular mask
        # based on center (mu_x, mu_y), radius = exp(log_r).
        # Then we do a weighted average of 'a' under that mask => a scalar for that output.

        # Let's build an array to hold these 10 scalars for each batch
        # shape (batch_size, num_outputs)
        outputs = []

        # grid_x, grid_y each shape (28,28)
        # we'll broadcast them against subspace_centers[i]
        for i in range(self.num_outputs):
            mu_x = torch.clamp(self.subspace_centers[i, 0], 0.0, 1.0)
            mu_y = torch.clamp(self.subspace_centers[i, 1], 0.0, 1.0)
            radius = torch.exp(self.subspace_log_r[i])  # > 0

            # dist^2 = (x - mu_x)^2 + (y - mu_y)^2
            dist_sq = (self.grid_x - mu_x)**2 + (self.grid_y - mu_y)**2

            # a soft circle mask => exp( -dist_sq / (2*r^2) )
            mask_2d = torch.exp(-dist_sq / (2*radius*radius + 1e-9))
            # shape (28, 28)

            # We'll multiply a by this mask and sum
            # a is (batch_size,1,28,28)
            # mask_2d is (28,28)
            # => broadcast to (batch_size,1,28,28)
            masked_a = a * mask_2d.unsqueeze(0).unsqueeze(1)  # shape (batch_size,1,28,28)

            # sum or average?
            # We'll sum them and then divide by total mask sum for a "mean activation"
            mask_sum = mask_2d.sum() + 1e-9
            scalar_per_batch = masked_a.view(batch_size, -1).sum(dim=1) / mask_sum

            # shape => (batch_size,)
            outputs.append(scalar_per_batch.unsqueeze(1))

        # Concatenate the 10 outputs => (batch_size, 10)
        logits = torch.cat(outputs, dim=1)

        return logits


###############################################################################
#                           INSTANTIATE MODEL
###############################################################################
model = BrainMassNet(
    img_size=IMG_SIZE,
    timesteps=TIMESTEPS,
    num_outputs=NUM_OUTPUTS
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

###############################################################################
#                           TRAINING / EVAL LOOP
###############################################################################
train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # images.shape => (batch_size, 1, 28, 28)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # metrics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # Evaluate
    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_test = criterion(logits, labels)
            running_test_loss += loss_test.item() * images.size(0)
            _, preds_test = torch.max(logits, dim=1)
            correct_test += (preds_test == labels).sum().item()
            total_test += labels.size(0)

    epoch_test_loss = running_test_loss / total_test
    epoch_test_acc = 100.0 * correct_test / total_test
    test_losses.append(epoch_test_loss)
    test_accs.append(epoch_test_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
          f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")

###############################################################################
#                           PLOT RESULTS
###############################################################################
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.show()

plt.figure()
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Epoch')
plt.legend()
plt.show()
