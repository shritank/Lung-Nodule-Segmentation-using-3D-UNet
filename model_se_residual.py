import torch
import torch.nn as nn


# ─── Squeeze and Excitation Block ─────────────────────────────────────────────
class SEBlock(nn.Module):
    """
    3D Squeeze-and-Excitation block (Hu et al., 2018).
    Recalibrates channel-wise feature responses adaptively.

    Squeeze: global average pooling → single value per channel
    Excitation: two FC layers → per-channel weights between 0 and 1
    Scale: multiply original features by learned channel weights
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.squeeze    = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        scale = self.squeeze(x).view(b, c)
        scale = self.excitation(scale).view(b, c, 1, 1, 1)
        return x * scale


# ─── Residual Conv Block with SE ──────────────────────────────────────────────
class ResidualSEBlock(nn.Module):
    """
    Residual block with SE channel attention.

    Main path:
        Conv3d → IN → ReLU → Conv3d → IN → SE → + residual → ReLU

    Residual path:
        1x1 Conv3d + IN (if in_channels != out_channels)
        Identity (if channels match)
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
        )

        self.se = SEBlock(out_channels, reduction=reduction)

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.InstanceNorm3d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main(x)
        out = self.se(out)
        out = out + self.residual(x)
        out = self.relu(out)
        return out


# ─── SE Residual U-Net ────────────────────────────────────────────────────────
class UNet3D(nn.Module):
    """
    3D U-Net with ResidualSEBlock in every encoder, bottleneck, and decoder block.
    No attention gates — pure residual + SE architecture.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, se_reduction=16):
        super().__init__()

        # Encoder
        self.enc1 = ResidualSEBlock(in_channels,       base_channels,     reduction=se_reduction)
        self.enc2 = ResidualSEBlock(base_channels,      base_channels * 2, reduction=se_reduction)
        self.enc3 = ResidualSEBlock(base_channels * 2,  base_channels * 4, reduction=se_reduction)

        # Downsampling
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResidualSEBlock(base_channels * 4, base_channels * 8, reduction=se_reduction)

        # Upsampling
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels,     kernel_size=2, stride=2)

        # Decoder
        self.dec3 = ResidualSEBlock(base_channels * 8, base_channels * 4, reduction=se_reduction)
        self.dec2 = ResidualSEBlock(base_channels * 4, base_channels * 2, reduction=se_reduction)
        self.dec1 = ResidualSEBlock(base_channels * 2, base_channels,     reduction=se_reduction)

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