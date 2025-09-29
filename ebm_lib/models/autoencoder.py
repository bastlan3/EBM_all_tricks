import torch.nn as nn

class SimpleConvAutoencoder(nn.Module):
    """
    A simple convolutional autoencoder. This can be pre-trained and then
    used to provide a latent space for an EBM.
    """
    def __init__(self, input_shape=(3, 32, 32), latent_dim=64):
        super().__init__()
        c, h, w = input_shape

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 16, 3, stride=2, padding=1),  # -> 16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> 8x8
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (h // 8) * (w // 8), latent_dim)
        )

        # --- Decoder ---
        self.decoder_fc = nn.Linear(latent_dim, 64 * (h // 8) * (w // 8))
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, h // 8, w // 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, c, 3, stride=2, padding=1, output_padding=1),  # -> 32x32
            nn.Sigmoid() # Output values between 0 and 1
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        z = self.decoder_fc(z)
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)