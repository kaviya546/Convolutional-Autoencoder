# Convolutional Autoencoder for Image Denoising
## AIM
To develop a convolutional autoencoder for image denoising application.
## Problem Statement and Dataset
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0–9), often used for image processing tasks. The goal of this experiment is image denoising using autoencoders, a neural network designed to learn efficient representations. By introducing noise to images, the model is trained to reconstruct clean versions.
## DESIGN STEPS
## STEP 1:
Load MNIST dataset and convert to tensors.
### STEP 2:
Apply Gaussian noise to images for training.
### STEP 3:
Design encoder-decoder architecture for reconstruction.
### STEP 4:
Use MSE loss to measure reconstruction quality.
### STEP 5:
Train autoencoder using Adam optimizer efficiently.
### STEP 6:
Evaluate model on noisy and clean images.
### STEP 7:
Visualize results comparing original, noisy, denoised versions.
### STEP 8:
Improve performance by tuning hyperparameters carefully.
## PROGRAM
### Name: KAVIYA SNEKA M
### Register Number: 212223040091
```py
# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x

```
```py
# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```
```py
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(loader):.4f}")
```
## OUTPUT

### Model Summary


<img width="639" height="387" alt="image" src="https://github.com/user-attachments/assets/70ae7678-ce01-4154-9bb6-a256ebc8eb2d" />




### Original vs Noisy Vs Reconstructed Image

<img width="1182" height="454" alt="image" src="https://github.com/user-attachments/assets/2eace665-55a4-4b0f-a728-a9353f182298" />



## RESULT
A convolutional autoencoder for image denoising application is developed successfully.
