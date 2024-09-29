# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the Generator for WGAN
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

# Define the Critic for WGAN
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

# WGAN class
class WGAN:
    def __init__(self, latent_dim, output_dim, critic_iterations=5, weight_clip=0.01, lr=5e-5):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, output_dim)
        self.critic = Critic(output_dim)
        self.critic_iterations = critic_iterations
        self.weight_clip = weight_clip
        self.g_optimizer = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.c_optimizer = optim.RMSprop(self.critic.parameters(), lr=lr)

    def train_step(self, real_data):
        batch_size = real_data.size(0)

        # Train Critic
        for _ in range(self.critic_iterations):
            self.c_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.latent_dim)
            fake_data = self.generator(z).detach()
            c_real = self.critic(real_data).reshape(-1)
            c_fake = self.critic(fake_data).reshape(-1)
            
            loss_critic = -(torch.mean(c_real) - torch.mean(c_fake))
            loss_critic.backward()
            self.c_optimizer.step()

            # Weight clipping
            for p in self.critic.parameters():
                p.data.clamp_(-self.weight_clip, self.weight_clip)

        # Train Generator
        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z)
        c_fake = self.critic(fake_data).reshape(-1)
        loss_gen = -torch.mean(c_fake)
        loss_gen.backward()
        self.g_optimizer.step()

        return loss_critic.item(), loss_gen.item()

# Define the Conditional Generator for CWGAN
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, output_dim):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
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

# Define the Conditional Critic for CWGAN
class ConditionalCritic(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ConditionalCritic, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim + num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        return self.model(x)

# Conditional WGAN class
class ConditionalWGAN:
    def __init__(self, latent_dim, num_classes, output_dim, critic_iterations=5, weight_clip=0.01, lr=5e-5):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.generator = ConditionalGenerator(latent_dim, num_classes, output_dim)
        self.critic = ConditionalCritic(output_dim, num_classes)
        self.critic_iterations = critic_iterations
        self.weight_clip = weight_clip
        self.g_optimizer = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.c_optimizer = optim.RMSprop(self.critic.parameters(), lr=lr)

    def train_step(self, real_data, labels):
        batch_size = real_data.size(0)

        # Train Critic
        for _ in range(self.critic_iterations):
            self.c_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.latent_dim)
            fake_data = self.generator(z, labels).detach()
            c_real = self.critic(real_data, labels).reshape(-1)
            c_fake = self.critic(fake_data, labels).reshape(-1)
            
            loss_critic = -(torch.mean(c_real) - torch.mean(c_fake))
            loss_critic.backward()
            self.c_optimizer.step()

            # Weight clipping
            for p in self.critic.parameters():
                p.data.clamp_(-self.weight_clip, self.weight_clip)

        # Train Generator
        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z, labels)
        c_fake = self.critic(fake_data, labels).reshape(-1)
        loss_gen = -torch.mean(c_fake)
        loss_gen.backward()
        self.g_optimizer.step()

        return loss_critic.item(), loss_gen.item()

# Training function for both GANs
def train_gan(gan, dataloader, num_epochs, device, conditional=False):
    gan.generator.to(device)
    gan.critic.to(device)

    for epoch in range(num_epochs):
        for i, (real_data, labels) in enumerate(dataloader):
            real_data = real_data.view(real_data.size(0), -1).to(device)
            labels = labels.to(device)
            
            if conditional:
                c_loss, g_loss = gan.train_step(real_data, labels)
            else:
                c_loss, g_loss = gan.train_step(real_data)
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], c_loss: {c_loss:.4f}, g_loss: {g_loss:.4f}')

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
    plt.show()

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 100
    output_dim = 784  # 28x28
    num_classes = 10
    batch_size = 64
    num_epochs = 50
    lr = 5e-5
    critic_iterations = 5
    weight_clip = 0.01

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

    # Train WGAN
    print("Training WGAN...")
    wgan = WGAN(latent_dim, output_dim, critic_iterations, weight_clip, lr)
    train_gan(wgan, dataloader, num_epochs, device)

    # Generate and visualize samples from WGAN
    wgan_samples = generate_samples(wgan, 100, latent_dim, device)
    visualize_samples(wgan_samples, "WGAN Generated Samples")

    # Train Conditional WGAN
    print("Training Conditional WGAN...")
    cwgan = ConditionalWGAN(latent_dim, num_classes, output_dim, critic_iterations, weight_clip, lr)
    train_gan(cwgan, dataloader, num_epochs, device, conditional=True)

    # Generate and visualize samples from Conditional WGAN
    cwgan_samples = generate_samples(cwgan, 100, latent_dim, device, conditional=True)
    visualize_samples(cwgan_samples, "Conditional WGAN Generated Samples")

    print("Training and visualization complete.")


