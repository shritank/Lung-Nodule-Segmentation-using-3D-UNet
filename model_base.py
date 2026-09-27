import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two consecutive 3D convolutions with Instance Norm and ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels,       base_channels)
        self.enc2 = ConvBlock(base_channels,      base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2,  base_channels * 4)

        # Downsampling
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # Upsampling
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels,     kernel_size=2, stride=2)

        # Decoder
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Output
        self.output_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        b  = self.bottleneck(self.down(e3))

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output_conv(d1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet3D().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    dummy_input = torch.randn(2, 1, 64, 64, 64).to(device)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: {output.min():.3f} to {output.max():.3f}")