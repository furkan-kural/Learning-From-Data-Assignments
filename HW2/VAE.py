# VAE.py
# Template for implementing a Variational Autoencoder (VAE) in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        """
        Initialize the VAE model.

        Parameters:
        - input_dim: dimensionality of input (e.g., 784 for MNIST)
        - hidden_dim: number of units in the hidden layer
        - latent_dim: dimensionality of the latent space
        """
        super(VAE, self).__init__()

        # ===== Encoder layers =====
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ===== Decoder layers =====
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encode input x into latent mean and log-variance.

        Input:
        - x: tensor of shape (batch_size, input_dim)

        Returns:
        - mu: mean of latent distribution
        - logvar: log-variance of latent distribution
        """
        # Encoder: 784 -> hidden_dim -> (mu, logvar)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Apply the reparameterization trick to sample z ~ N(mu, sigma^2).
        z = mu + sigma * epsilon, where epsilon ~ N(0, I)

        Input:
        - mu: mean tensor of shape (batch_size, latent_dim)
        - logvar: log-variance tensor of shape (batch_size, latent_dim)

        Returns:
        - z: sampled latent vector of shape (batch_size, latent_dim)
        """
        # Compute sigma from logvar: sigma = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        # Sample epsilon from standard normal distribution
        eps = torch.randn_like(std)
        # Reparameterization: z = mu + sigma * epsilon
        z = mu + std * eps
        return z

    def decode(self, z):
        """
        Decode latent vector z to reconstruct input x_hat.

        Input:
        - z: latent vector tensor of shape (batch_size, latent_dim)

        Returns:
        - x_hat: reconstructed input of shape (batch_size, input_dim)
        """
        # Decoder: latent_dim -> hidden_dim -> input_dim
        h = F.relu(self.fc2(z))
        x_hat = torch.sigmoid(self.fc3(h))
        return x_hat

    def forward(self, x):
        """
        Forward pass: encode -> reparameterize -> decode.

        Input:
        - x: input tensor of shape (batch_size, input_dim)

        Returns:
        - x_hat: reconstructed input of shape (batch_size, input_dim)
        - mu: latent mean of shape (batch_size, latent_dim)
        - logvar: latent log-variance of shape (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def vae_loss(x, x_hat, mu, logvar):
    """
    Compute the VAE loss (ELBO) = reconstruction loss + KL divergence.
    
    Loss = -ELBO = -E[log p(x|z)] + D_KL(q(z|x)||p(z))
    
    For MNIST, we use Binary Cross-Entropy (BCE) for reconstruction loss.
    KL divergence: D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Inputs:
    - x: original input tensor of shape (batch_size, input_dim)
    - x_hat: reconstructed input tensor of shape (batch_size, input_dim)
    - mu: latent mean tensor of shape (batch_size, latent_dim)
    - logvar: latent log-variance tensor of shape (batch_size, latent_dim)

    Returns:
    - loss: scalar tensor (total VAE loss)
    """
    # Reconstruction loss: Binary Cross-Entropy
    # BCE = -sum(x * log(x_hat) + (1-x) * log(1-x_hat))
    # We use reduction='sum' to sum over all elements, then divide by batch size
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
    
    # KL divergence: D_KL(q(z|x)||p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # where p(z) = N(0, I) is the prior
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD = torch.mean(KLD)  # Average over batch
    
    # Total loss = reconstruction loss + KL divergence
    loss = BCE + KLD
    
    return loss
