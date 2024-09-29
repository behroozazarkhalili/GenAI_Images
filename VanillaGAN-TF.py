# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Define the Generator for VanillaGAN
class Generator(keras.Model):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = keras.Sequential([
            keras.layers.Dense(256, input_dim=latent_dim),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(1024),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(output_dim, activation='tanh')
        ])

    def call(self, z):
        return self.model(z)

# Define the Discriminator for VanillaGAN
class Discriminator(keras.Model):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = keras.Sequential([
            keras.layers.Dense(1024, input_dim=input_dim),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        return self.model(x)

# VanillaGAN class
class VanillaGAN:
    def __init__(self, latent_dim, output_dim, lr=0.0002, beta1=0.5):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, output_dim)
        self.discriminator = Discriminator(output_dim)
        self.g_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
        self.loss_fn = keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # Train Discriminator
        with tf.GradientTape() as tape:
            z = tf.random.normal((batch_size, self.latent_dim))
            fake_data = self.generator(z, training=True)
            d_real_output = self.discriminator(real_data, training=True)
            d_fake_output = self.discriminator(fake_data, training=True)
            d_real_loss = self.loss_fn(real_labels, d_real_output)
            d_fake_loss = self.loss_fn(fake_labels, d_fake_output)
            d_loss = d_real_loss + d_fake_loss

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as tape:
            z = tf.random.normal((batch_size, self.latent_dim))
            fake_data = self.generator(z, training=True)
            g_fake_output = self.discriminator(fake_data, training=True)
            g_loss = self.loss_fn(real_labels, g_fake_output)

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return d_loss, g_loss

# Define the Generator for Conditional VanillaGAN
class ConditionalGenerator(keras.Model):
    def __init__(self, latent_dim, num_classes, output_dim):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.label_emb = keras.layers.Embedding(num_classes, num_classes)
        
        self.model = keras.Sequential([
            keras.layers.Dense(256, input_dim=latent_dim + num_classes),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(1024),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dense(output_dim, activation='tanh')
        ])

    def call(self, z, labels):
        z = tf.reshape(z, (-1, self.latent_dim))
        c = self.label_emb(labels)
        x = tf.concat([z, c], axis=1)
        return self.model(x)

# Define the Discriminator for Conditional VanillaGAN
class ConditionalDiscriminator(keras.Model):
    def __init__(self, input_dim, num_classes):
        super(ConditionalDiscriminator, self).__init__()
        self.num_classes = num_classes
        
        self.label_emb = keras.layers.Embedding(num_classes, num_classes)
        
        self.model = keras.Sequential([
            keras.layers.Dense(1024, input_dim=input_dim + num_classes),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, x, labels):
        x = tf.reshape(x, (-1, 784))
        c = self.label_emb(labels)
        x = tf.concat([x, c], axis=1)
        return self.model(x)

# Conditional VanillaGAN class
class ConditionalVanillaGAN:
    def __init__(self, latent_dim, num_classes, output_dim, lr=0.0002, beta1=0.5):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.generator = ConditionalGenerator(latent_dim, num_classes, output_dim)
        self.discriminator = ConditionalDiscriminator(output_dim, num_classes)
        self.g_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
        self.loss_fn = keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(self, real_data, labels):
        batch_size = tf.shape(real_data)[0]
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # Train Discriminator
        with tf.GradientTape() as tape:
            z = tf.random.normal((batch_size, self.latent_dim))
            fake_data = self.generator(z, labels, training=True)
            d_real_output = self.discriminator(real_data, labels, training=True)
            d_fake_output = self.discriminator(fake_data, labels, training=True)
            d_real_loss = self.loss_fn(real_labels, d_real_output)
            d_fake_loss = self.loss_fn(fake_labels, d_fake_output)
            d_loss = d_real_loss + d_fake_loss

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as tape:
            z = tf.random.normal((batch_size, self.latent_dim))
            fake_data = self.generator(z, labels, training=True)
            g_fake_output = self.discriminator(fake_data, labels, training=True)
            g_loss = self.loss_fn(real_labels, g_fake_output)

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return d_loss, g_loss

# Training function for both GANs
def train_gan(gan, dataset, num_epochs, conditional=False):
    for epoch in range(num_epochs):
        for i, (real_data, labels) in enumerate(dataset):
            real_data = tf.reshape(real_data, (-1, 784))
            
            if conditional:
                d_loss, g_loss = gan.train_step(real_data, labels)
            else:
                d_loss, g_loss = gan.train_step(real_data)
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')

# Function to generate samples
def generate_samples(gan, num_samples, latent_dim, conditional=False, num_classes=10):
    z = tf.random.normal((num_samples, latent_dim))
    if conditional:
        labels = tf.repeat(tf.range(10), num_samples // 10 + 1)[:num_samples]
        samples = gan.generator(z, labels, training=False)
    else:
        samples = gan.generator(z, training=False)
    return tf.reshape(samples, (num_samples, 28, 28, 1))

# Function to visualize samples
def visualize_samples(samples, title):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, :, :, 0], cmap='gray')
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
    lr = 0.0002
    beta1 = 0.5

    # Load MNIST dataset
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)

    # Train VanillaGAN
    print("Training VanillaGAN...")
    vanilla_gan = VanillaGAN(latent_dim, output_dim, lr, beta1)
    train_gan(vanilla_gan, dataset, num_epochs)

    # Generate and visualize samples from VanillaGAN
    vanilla_samples = generate_samples(vanilla_gan, 100, latent_dim)
    visualize_samples(vanilla_samples, "VanillaGAN Generated Samples")

    # Train Conditional VanillaGAN
    print("Training Conditional VanillaGAN...")
    cond_vanilla_gan = ConditionalVanillaGAN(latent_dim, num_classes, output_dim, lr, beta1)
    train_gan(cond_vanilla_gan, dataset, num_epochs, conditional=True)

    # Generate and visualize samples from Conditional VanillaGAN
    cond_vanilla_samples = generate_samples(cond_vanilla_gan, 100, latent_dim, conditional=True)
    visualize_samples(cond_vanilla_samples, "Conditional VanillaGAN Generated Samples")

    print("Training and visualization complete.")


