import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, n_feats=256):
        super(ResidualBlock, self).__init__()

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feats, n_feats, 3),
            nn.InstanceNorm2d(n_feats),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feats, n_feats, 3),
            nn.InstanceNorm2d(n_feats)
        ]

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, n_feats=64, n_downsample=2, n_res_blocks=3, shared_block=None):
        super(Encoder, self).__init__()

        # Input Conv Block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, n_feats, 7),
            nn.InstanceNorm2d(n_feats),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Downsampling
        in_feats = n_feats
        out_feats = in_feats*2
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(in_feats, out_feats, 4, 2, 1),
                nn.InstanceNorm2d(out_feats),
                nn.ReLU(inplace=True)
            ]
            in_feats = out_feats
            out_feats = in_feats*2

        # Residual Blocks
        for _ in range(n_res_blocks):
            layers += [ResidualBlock(in_feats)]

        self.body = nn.Sequential(*layers)
        self.shared_block = shared_block

    def reparameterization(self, mu):
        noise = torch.zeros_like(mu).normal_(mean=0, std=1).to(mu.device)
        return noise + mu

    def forward(self, x):
        x = self.body(x)
        mu = self.shared_block(x)
        z = self.reparameterization(mu)
        return {'mu': mu, 'z': z}


class Generator(nn.Module):
    def __init__(self, out_channels=3, n_feats=64, n_upsample=2, n_res_blocks=3, shared_block=None):
        super(Generator, self).__init__()

        self.shared_block = shared_block

        layers = []
        in_feats = n_feats * (2 ** n_upsample)
        # Residual Blocks
        for _ in range(n_res_blocks):
            layers += [ResidualBlock(in_feats)]

        # Upsampling
        out_feats = in_feats//2
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(in_feats, out_feats, 4, 2, 1),
                nn.InstanceNorm2d(out_feats),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_feats = out_feats
            out_feats = in_feats//2

        # Output Conv Block
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_feats, out_channels, 7),
            nn.Tanh()
        ]

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_block(x)
        x = self.body(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, n_feats=64):
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv2d(in_channels, n_feats, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for _ in range(3):
            layers += [
                nn.Conv2d(n_feats, n_feats*2, 4, 2, 1),
                nn.InstanceNorm2d(n_feats*2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            n_feats = n_feats*2

        layers += [
            nn.Conv2d(n_feats, 1, 3, padding=1)
        ]

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        return x


if __name__ == "__main__":
    shared_E = ResidualBlock()
    shared_G = ResidualBlock()
    encoder = Encoder(shared_block=shared_E)
    generator = Generator(shared_block=shared_G)
    discriminator = Discriminator()

    input = torch.randn(10, 3, 128, 128)
    print('input:{}'.format(input.shape))

    latent = encoder(input)['z']
    print('latent:{}'.format(latent.shape))

    output = generator(latent)
    print('output:{}'.format(output.shape))

    prediction = discriminator(output)
    print('prediction:{}'.format(prediction.shape))
