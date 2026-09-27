import torch
import torch.nn as nn


# ─── Conv Block (Baseline) ────────────────────────────────────────────────────
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


# ─── Attention Gate ───────────────────────────────────────────────────────────
class AttentionGate(nn.Module):
    """
    3D Attention Gate inspired by the Attention U-Net paper (Oktay et al., 2018),
    adapted to use InstanceNorm3d instead of BatchNorm.

    This variant's skip placement (deeper two skips gated, shallowest left
    unfiltered — see UNet3D.forward below) matches the original paper's choice.

    One departure from the original paper remains: this gate ends with an extra
    output projection (W, 1x1 Conv + IN) after x * attention. The original
    paper's gate output is just x * attention, with no projection afterward.

    Flow:
        g ──► Wg ──┐
                   + ──► ReLU ──► psi (1x1 Conv + Sigmoid) ──► attention map
        x ──► Wx ──┘
                                        x * attention ──► W (1x1 Conv + IN) ──► return
    """
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv3d(g_channels, inter_channels, kernel_size=1),
            nn.InstanceNorm3d(inter_channels)
        )
        self.Wx = nn.Sequential(
            nn.Conv3d(x_channels, inter_channels, kernel_size=1),
            nn.InstanceNorm3d(inter_channels)
        )
        # No InstanceNorm before sigmoid — matches original paper
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Output projection — extra refinement step from original paper
        self.W = nn.Sequential(
            nn.Conv3d(x_channels, x_channels, kernel_size=1),
            nn.InstanceNorm3d(x_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        att = self.relu(self.Wg(g) + self.Wx(x))
        att = self.psi(att)
        out = x * att.expand_as(x)
        out = self.W(out)
        return out


# ─── Selective Attention U-Net ────────────────────────────────────────────────
class UNet3D(nn.Module):
    """
    Baseline 3D U-Net with Selective Attention Gates.

    Attention applied only on deeper skip connections (e2, e3) where
    higher-level semantic features benefit from filtering. The shallowest
    skip (e1) passes through unfiltered — low-level edges and textures
    are always useful for precise boundary delineation.

    Encoder and decoder use plain ConvBlocks identical to the baseline.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()

        # Encoder — plain ConvBlocks (baseline)
        self.enc1 = ConvBlock(in_channels,       base_channels)
        self.enc2 = ConvBlock(base_channels,      base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2,  base_channels * 4)

        # Downsampling
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck — plain ConvBlock (baseline)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # Upsampling
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels,     kernel_size=2, stride=2)

        # Attention Gates on deeper skips only (e2, e3)
        # e1 passes through unfiltered
        self.att3 = AttentionGate(
            g_channels=base_channels * 4,
            x_channels=base_channels * 4,
            inter_channels=base_channels * 2
        )
        self.att2 = AttentionGate(
            g_channels=base_channels * 2,
            x_channels=base_channels * 2,
            inter_channels=base_channels
        )

        # Decoder — plain ConvBlocks (baseline)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Output
        self.output_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        b  = self.bottleneck(self.down(e3))

        # Decoder — attention on e3 and e2, e1 unfiltered
        g3 = self.up3(b)
        e3 = self.att3(g3, e3)
        d3 = self.dec3(torch.cat([g3, e3], dim=1))

        g2 = self.up2(d3)
        e2 = self.att2(g2, e2)
        d2 = self.dec2(torch.cat([g2, e2], dim=1))

        # e1 passes through unfiltered
        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, e1], dim=1))

        return self.output_conv(d1)


# Alias
AttentionUNet3D = UNet3D


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