# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the Generator for VanillaGAN
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Define the Discriminator for VanillaGAN
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# VanillaGAN class
class VanillaGAN:
    def __init__(self, latent_dim, output_dim, lr=0.0002, beta1=0.5):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, output_dim)
        self.discriminator = Discriminator(output_dim)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.BCELoss()

    def train_step(self, real_data):
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        self.d_optimizer.zero_grad()
        d_real_output = self.discriminator(real_data)
        d_real_loss = self.criterion(d_real_output, real_labels)
        
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z)
        d_fake_output = self.discriminator(fake_data.detach())
        d_fake_loss = self.criterion(d_fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        g_fake_output = self.discriminator(fake_data)
        g_loss = self.criterion(g_fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss.item(), g_loss.item()

# Define the Generator for Conditional VanillaGAN
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, output_dim):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), self.latent_dim)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        return self.model(x)

# Define the Discriminator for Conditional VanillaGAN
class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ConditionalDiscriminator, self).__init__()
        self.num_classes = num_classes
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim + num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        return self.model(x)

# Conditional VanillaGAN class
class ConditionalVanillaGAN:
    def __init__(self, latent_dim, num_classes, output_dim, lr=0.0002, beta1=0.5):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.generator = ConditionalGenerator(latent_dim, num_classes, output_dim)
        self.discriminator = ConditionalDiscriminator(output_dim, num_classes)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.BCELoss()

    def train_step(self, real_data, labels):
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        self.d_optimizer.zero_grad()
        d_real_output = self.discriminator(real_data, labels)
        d_real_loss = self.criterion(d_real_output, real_labels)
        
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z, labels)
        d_fake_output = self.discriminator(fake_data.detach(), labels)
        d_fake_loss = self.criterion(d_fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        g_fake_output = self.discriminator(fake_data, labels)
        g_loss = self.criterion(g_fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss.item(), g_loss.item()

# Training function for both GANs
def train_gan(gan, dataloader, num_epochs, device, conditional=False):
    gan.generator.to(device)
    gan.discriminator.to(device)

    for epoch in range(num_epochs):
        for i, (real_data, labels) in enumerate(dataloader):
            real_data = real_data.view(real_data.size(0), -1).to(device)
            
            if conditional:
                labels = labels.to(device)
                d_loss, g_loss = gan.train_step(real_data, labels)
            else:
                d_loss, g_loss = gan.train_step(real_data)
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')

# Function to generate samples
def generate_samples(gan, num_samples, latent_dim, device, conditional=False, num_classes=10):
    gan.generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        if conditional:
            labels = torch.arange(10).repeat(num_samples // 10 + 1)[:num_samples].to(device)
            samples = gan.generator(z, labels)
        else:
            samples = gan.generator(z)
    return samples.view(num_samples, 1, 28, 28)

# Function to visualize samples
def visualize_samples(samples, title):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].cpu().squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()

    # save the generated samples
    plt.savefig(f'{title}.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 100
    output_dim = 784  # 28x28
    num_classes = 10
    batch_size = 64
    num_epochs = 50
    lr = 0.0002
    beta1 = 0.5

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

    # Train VanillaGAN
    print("Training VanillaGAN...")
    vanilla_gan = VanillaGAN(latent_dim, output_dim, lr, beta1)
    train_gan(vanilla_gan, dataloader, num_epochs, device)

    # Generate and visualize samples from VanillaGAN
    vanilla_samples = generate_samples(vanilla_gan, 100, latent_dim, device)
    visualize_samples(vanilla_samples, "VanillaGAN Generated Samples")

    # Train Conditional VanillaGAN
    print("Training Conditional VanillaGAN...")
    cond_vanilla_gan = ConditionalVanillaGAN(latent_dim, num_classes, output_dim, lr, beta1)
    train_gan(cond_vanilla_gan, dataloader, num_epochs, device, conditional=True)

    # Generate and visualize samples from Conditional VanillaGAN
    cond_vanilla_samples = generate_samples(cond_vanilla_gan, 100, latent_dim, device, conditional=True)
    visualize_samples(cond_vanilla_samples, "Conditional VanillaGAN Generated Samples")

    print("Training and visualization complete.")


