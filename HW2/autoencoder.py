import numpy as np

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01):
        """
        Initialize weights and biases for a 1-layer encoder and 1-layer decoder.

        Parameters:
            input_dim  -- dimensionality of input (e.g., 784 for MNIST)
            hidden_dim -- dimensionality of bottleneck (e.g., 16, 32, or 64)
            learning_rate -- gradient descent step size
        """
        # Initialize weights with small random values
        self.W_e = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_e = np.zeros((hidden_dim, 1))
        self.W_d = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_d = np.zeros((input_dim, 1))

        self.lr = learning_rate

    def encoder(self, x):
        """
        Apply encoder transformation: z = ReLU(W_e·x + b_e)
        
        Parameters:
            x -- input data of shape (input_dim, batch_size) or (batch_size, input_dim)
        
        Returns:
            z -- encoded representation of shape (hidden_dim, batch_size)
        """
        # Handle both (input_dim, batch_size) and (batch_size, input_dim) formats
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.shape[0] != self.W_e.shape[1] and x.shape[-1] == self.W_e.shape[1]:
            # If input is (batch_size, input_dim), transpose to (input_dim, batch_size)
            x = x.T
        
        # Compute: z = ReLU(W_e·x + b_e)
        z = np.dot(self.W_e, x) + self.b_e
        z = np.maximum(0, z)  # ReLU activation
        return z

    def decoder(self, z):
        """
        Apply decoder transformation: x_hat = sigmoid(W_d·z + b_d)
        
        Parameters:
            z -- encoded representation of shape (hidden_dim, batch_size)
        
        Returns:
            x_hat -- reconstructed input of shape (input_dim, batch_size)
        """
        # Compute: x_hat = sigmoid(W_d·z + b_d)
        x_hat = np.dot(self.W_d, z) + self.b_d
        # Sigmoid activation
        x_hat = 1 / (1 + np.exp(-np.clip(x_hat, -500, 500)))  # Clip for numerical stability
        return x_hat

    def compute_loss(self, x, x_hat):
        """
        Compute Mean Squared Error loss: L = (1/m) * sum((x - x_hat)^2)
        
        Parameters:
            x -- original input of shape (input_dim, batch_size) or (batch_size, input_dim)
            x_hat -- reconstructed input of shape (input_dim, batch_size)
        
        Returns:
            loss -- scalar MSE loss
        """
        # Handle input format
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.shape[0] != x_hat.shape[0]:
            x = x.T
        
        # Compute MSE: (1/m) * sum((x - x_hat)^2)
        m = x.shape[1]
        loss = np.mean((x - x_hat) ** 2)
        return loss

    def backward(self, x, z, x_hat):
        """
        Compute gradients using manual backpropagation.
        
        Parameters:
            x -- original input of shape (input_dim, batch_size) or (batch_size, input_dim)
            z -- encoded representation of shape (hidden_dim, batch_size)
            x_hat -- reconstructed input of shape (input_dim, batch_size)
        
        Returns:
            grads -- dictionary containing gradients for W_e, b_e, W_d, b_d
        """
        # Handle input format
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.shape[0] != x_hat.shape[0]:
            x = x.T
        
        m = x.shape[1]
        
        # Loss: L = (1/m) * sum((x - x_hat)^2)
        # Gradient w.r.t. x_hat: ∂L/∂x_hat = (2/m) * (x_hat - x)
        dx_hat = (2.0 / m) * (x_hat - x)
        
        # Decoder gradients
        # x_hat = sigmoid(W_d·z + b_d)
        # Let a_d = W_d·z + b_d, then x_hat = sigmoid(a_d)
        # ∂L/∂a_d = dx_hat * sigmoid'(a_d) = dx_hat * sigmoid(a_d) * (1 - sigmoid(a_d))
        #          = dx_hat * x_hat * (1 - x_hat)
        a_d = np.dot(self.W_d, z) + self.b_d
        sigmoid_derivative = x_hat * (1 - x_hat)
        da_d = dx_hat * sigmoid_derivative
        
        # ∂L/∂W_d = da_d · z^T
        dW_d = np.dot(da_d, z.T)
        # ∂L/∂b_d = sum(da_d, axis=1, keepdims=True)
        db_d = np.sum(da_d, axis=1, keepdims=True)
        
        # Encoder gradients (backprop through decoder)
        # ∂L/∂z = W_d^T · da_d
        dz = np.dot(self.W_d.T, da_d)
        
        # z = ReLU(W_e·x + b_e)
        # Let a_e = W_e·x + b_e, then z = ReLU(a_e)
        # ∂L/∂a_e = dz * ReLU'(a_e) = dz * (a_e > 0)
        a_e = np.dot(self.W_e, x) + self.b_e
        relu_derivative = (a_e > 0).astype(float)
        da_e = dz * relu_derivative
        
        # ∂L/∂W_e = da_e · x^T
        dW_e = np.dot(da_e, x.T)
        # ∂L/∂b_e = sum(da_e, axis=1, keepdims=True)
        db_e = np.sum(da_e, axis=1, keepdims=True)
        
        grads = {
            'W_e': dW_e,
            'b_e': db_e,
            'W_d': dW_d,
            'b_d': db_d
        }
        
        return grads

    def step(self, grads):
        """
        Update weights using gradient descent.
        
        Parameters:
            grads -- dictionary containing gradients for W_e, b_e, W_d, b_d
        """
        self.W_e -= self.lr * grads['W_e']
        self.b_e -= self.lr * grads['b_e']
        self.W_d -= self.lr * grads['W_d']
        self.b_d -= self.lr * grads['b_d']

    def train(self, X, epochs=20, batch_size=128):
        """
        Train the autoencoder using batch gradient descent.
        
        Parameters:
            X -- training data of shape (n_samples, input_dim)
            epochs -- number of training epochs
            batch_size -- size of each batch
        
        Returns:
            losses -- list of average loss per epoch
        """
        n_samples = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            # Process in batches
            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i:i+batch_size]
                batch = batch.T  # Convert to (input_dim, batch_size)
                
                # Forward pass
                z = self.encoder(batch)
                x_hat = self.decoder(z)
                
                # Compute loss
                loss = self.compute_loss(batch, x_hat)
                epoch_losses.append(loss)
                
                # Backward pass
                grads = self.backward(batch, z, x_hat)
                
                # Update parameters
                self.step(grads)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def reconstruct(self, x):
        """
        Reconstruct input data (helper method for inference).
        
        Parameters:
            x -- input data of shape (n_samples, input_dim) or (input_dim, n_samples)
        
        Returns:
            x_hat -- reconstructed data of shape (n_samples, input_dim)
        """
        # Handle input format
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        original_shape = x.shape
        if x.shape[0] != self.W_e.shape[1]:
            # Input is (n_samples, input_dim)
            x = x.T  # Convert to (input_dim, n_samples)
        
        # Forward pass
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        # Convert back to (n_samples, input_dim)
        if original_shape[0] != self.W_e.shape[1]:
            x_hat = x_hat.T
        
        return x_hat
