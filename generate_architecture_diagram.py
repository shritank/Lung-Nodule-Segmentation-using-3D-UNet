"""
Draws the shared 3D U-Net skeleton (base_channels=32, input 64^3) directly
from model_base.py -- every channel count and spatial dimension used in the
figure is introspected from the actual instantiated model and a real forward
pass, not hand-typed.
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from model_base import UNet3D

PROJECT_DIR = Path(r"C:\Users\ramar\OneDrive\Desktop\Lung Nodule Analysis")
OUT_PATH = PROJECT_DIR / "paper" / "figures" / "unet_architecture.png"

# ─── Introspect real layer channel counts from the instantiated model ────────
model = UNet3D()
convs = dict(model.named_modules())


def io_channels(name):
    mod = convs[name]
    return mod.in_channels, mod.out_channels


enc_out = {
    'enc1': io_channels('enc1.block.3')[1],
    'enc2': io_channels('enc2.block.3')[1],
    'enc3': io_channels('enc3.block.3')[1],
}
bottleneck_out = io_channels('bottleneck.block.3')[1]
dec_out = {
    'dec3': io_channels('dec3.block.3')[1],
    'dec2': io_channels('dec2.block.3')[1],
    'dec1': io_channels('dec1.block.3')[1],
}
out_channels = io_channels('output_conv')[1]

# ─── Confirm real spatial dimensions via an actual forward pass ──────────────
x = torch.randn(1, 1, 64, 64, 64)
shapes = {}


def hook(name):
    def fn(mod, inp, out):
        shapes[name] = tuple(out.shape)
    return fn


for name in ['enc1', 'enc2', 'enc3', 'bottleneck', 'dec3', 'dec2', 'dec1', 'output_conv']:
    convs[name].register_forward_hook(hook(name))

with torch.no_grad():
    model(x)

spatial = {k: v[2] for k, v in shapes.items()}  # cube side length

print("Introspected from model_base.UNet3D (base_channels=32, input 64^3):")
for k in ['enc1', 'enc2', 'enc3', 'bottleneck', 'dec3', 'dec2', 'dec1', 'output_conv']:
    ch = {'enc1': enc_out['enc1'], 'enc2': enc_out['enc2'], 'enc3': enc_out['enc3'],
          'bottleneck': bottleneck_out, 'dec3': dec_out['dec3'], 'dec2': dec_out['dec2'],
          'dec1': dec_out['dec1'], 'output_conv': out_channels}[k]
    print(f"  {k:14s} {ch:>4d} ch  @ {spatial[k]}^3")

# ─── Draw ──────────────────────────────────────────────────────────────────
# NOTE: this figure renders single-column in the paper (width=\columnwidth,
# ~3.5in). Text size in the final PDF is governed by the ratio of fontsize
# (fixed, in points) to figsize (fixed, in inches): LaTeX scales the whole
# raster to the column width, so the on-page point size of a label is
#     fontsize * (3.5in / figsize_width_in).
# At the previous 10.5in-wide canvas that factor was 1/3, rendering the 17pt
# block labels at only ~5.7pt -- hence "too small on the page". Dropping the
# canvas to 7.5in raises the factor to ~0.47, putting those labels near 8pt
# against 10pt body text, and simultaneously makes the figure TALLER on the
# page (7.5in * 0.47 = 3.5in, versus 2.33in before) because the aspect ratio
# is now square rather than 1.5:1. Both effects are what "make it bigger"
# actually requires at a width that is already capped at \columnwidth.
# Shrinking the canvas does compress the boxes horizontally, so box widths
# below were re-checked against their label lengths after this change.
fig, ax = plt.subplots(figsize=(7.5, 7.5))

BLUE = '#4C72B0'    # encoder conv blocks
GREEN = '#55A868'   # bottleneck
ORANGE = '#DD8452'  # decoder conv blocks
GREY = '#888888'    # pooling / transpose conv
SKIP = '#333333'    # skip connections

box_h = 1.6


def draw_block(x0, y0, w, h, color, label, sublabel):
    ax.add_patch(patches.Rectangle((x0, y0), w, h, facecolor=color, edgecolor='black',
                                   linewidth=1.4, alpha=0.85, zorder=3))
    ax.text(x0 + w / 2, y0 + h * 0.62, label, ha='center', va='center',
            fontsize=19, fontweight='bold', color='white', zorder=4)
    ax.text(x0 + w / 2, y0 + h * 0.28, sublabel, ha='center', va='center',
            fontsize=15, color='white', zorder=4)


def draw_arrow(x0, y0, x1, y1, color='black', style='-', lw=2.0, label=None, label_dy=0.15):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, linestyle=style))
    if label:
        # va='center' matters for these two-line labels: without it matplotlib
        # anchors multi-line text on the BASELINE of the last line, so the block
        # grows upward from the anchor and the first line ("MaxPool3d") rode up
        # into the bottom edge of the box above. Centring the whole block on the
        # midpoint of the arrow keeps it inside the inter-row gap.
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + label_dy, label,
                fontsize=14, color=color, ha='center', va='center')


# Column x-positions: encoder | bottleneck | decoder.
# These were retuned when the canvas narrowed to 7.5in. What matters for
# whether a label fits its box is the box's width as a FRACTION of the total
# x-range, because that fraction times the on-page figure width (\columnwidth,
# ~3.5in) is the box's real width in inches, while the label's size is fixed in
# points. The old geometry (w=2.5 over a 13.3-unit range) gave each box only
# ~19% of the width and spent ~40% on the empty skip-connection gap, so at the
# narrower canvas every label overflowed. Pulling the columns together and
# widening the boxes raises that to ~30% each, which fits the longest encoder
# and decoder labels with margin to spare.
xL, xR = 0.5, 5.8
w = 4.0
xM = (xL + xR + w) / 2.0   # bottleneck centred between the two columns

# Vertical positions (top = full resolution, going down = downsampled).
# row_spacing must clear box_h (1.3) PLUS enough room for a two-line arrow
# label at the current fontsize -- with the smaller figsize above, the same
# point-sized text now occupies more data units than it used to, so this
# spacing was widened from the original 2.5 to avoid the arrow labels
# overlapping the boxes above/below them (caught by rendering and inspecting
# the actual PNG, not by calculation alone).
row_spacing = 3.6
y_bott = 2.0
y_enc3 = y_bott + row_spacing
y_enc2 = y_enc3 + row_spacing
y_enc1 = y_enc2 + row_spacing

io_y = y_enc1 + box_h + 1.0
# io_h must be tall enough for TWO lines of text at the enlarged fontsizes;
# at 0.85 the "input"/"output_conv" label collided with the "1 ch, 64^3" line
# beneath it. draw_block places its two lines at 0.62h and 0.28h, so the box
# height has to scale with the font, not stay fixed.
io_h = 1.4

draw_block(xL, y_enc1, w, box_h, BLUE, f"enc1: {enc_out['enc1']} ch", f"{spatial['enc1']}\u00b3 voxels")
draw_block(xL, y_enc2, w, box_h, BLUE, f"enc2: {enc_out['enc2']} ch", f"{spatial['enc2']}\u00b3 voxels")
draw_block(xL, y_enc3, w, box_h, BLUE, f"enc3: {enc_out['enc3']} ch", f"{spatial['enc3']}\u00b3 voxels")

# The "bottleneck: NNN ch" label is longer than the other boxes' labels, so it
# gets its own wider box, kept centered on the same midpoint (xM + w/2) as
# before so the connecting arrows (which reference that midpoint directly)
# still land correctly.
bottleneck_w = 5.6
bottleneck_x0 = xM - bottleneck_w / 2
draw_block(bottleneck_x0, y_bott, bottleneck_w, box_h, GREEN, f"bottleneck: {bottleneck_out} ch", f"{spatial['bottleneck']}\u00b3 voxels")

draw_block(xR, y_enc3, w, box_h, ORANGE, f"dec3: {dec_out['dec3']} ch", f"{spatial['dec3']}\u00b3 voxels")
draw_block(xR, y_enc2, w, box_h, ORANGE, f"dec2: {dec_out['dec2']} ch", f"{spatial['dec2']}\u00b3 voxels")
draw_block(xR, y_enc1, w, box_h, ORANGE, f"dec1: {dec_out['dec1']} ch", f"{spatial['dec1']}\u00b3 voxels")

# Input / output
draw_block(xL, io_y, w, io_h, '#555555', "input", "1 ch, 64\u00b3")
draw_block(xR, io_y, w, io_h, '#555555', "output_conv", f"{out_channels} ch, 64\u00b3")

# Input -> enc1
draw_arrow(xL + w / 2, io_y, xL + w / 2, y_enc1 + box_h, color='black')
# Encoder downsampling (MaxPool3d k2 s2)
draw_arrow(xL + w / 2, y_enc1, xL + w / 2, y_enc2 + box_h, color=GREY, label='MaxPool3d\n(2,2,2)')
draw_arrow(xL + w / 2, y_enc2, xL + w / 2, y_enc3 + box_h, color=GREY, label='MaxPool3d\n(2,2,2)')
draw_arrow(xL + w / 2, y_enc3, xM - 1.3, y_bott + box_h, color=GREY, label='MaxPool3d\n(2,2,2)')

# Bottleneck -> up3 -> dec3 (ConvTranspose3d k2 s2)
draw_arrow(xM + 1.3, y_bott + box_h, xR + w / 2, y_enc3, color=GREY, label='ConvTranspose3d\n(2,2,2)')
# Decoder upsampling
draw_arrow(xR + w / 2, y_enc3 + box_h, xR + w / 2, y_enc2, color=GREY, label='ConvTranspose3d\n(2,2,2)')
draw_arrow(xR + w / 2, y_enc2 + box_h, xR + w / 2, y_enc1, color=GREY, label='ConvTranspose3d\n(2,2,2)')
# dec1 -> output
draw_arrow(xR + w / 2, y_enc1 + box_h, xR + w / 2, io_y, color='black')

# Skip connections (concatenation), dashed horizontal.
# No in-figure caption for these: the gap between the columns was deliberately
# narrowed to buy width for the boxes (so their labels could be set larger),
# and no legible label fits there any more. The paper's caption already states
# "Dashed lines indicate skip connections (channel-wise concatenation before
# each decoder block)", so an in-figure label would be redundant anyway --
# better to spend the space on type that is actually readable at 3.5in wide.
for y in [y_enc1, y_enc2, y_enc3]:
    draw_arrow(xL + w, y + box_h / 2, xR, y + box_h / 2, color=SKIP, style='--', lw=1.8)

ax.set_xlim(-0.3, 10.1)
ax.set_ylim(1.2, 17.2)
ax.axis('off')
# In-figure title. CAREFUL: this title is what previously shrank the whole
# diagram. At 19pt its second line was wider than the axes, and
# bbox_inches='tight' crops to the WIDEST artist -- so the saved PNG came out
# 9.7in wide instead of the declared 7.5in, and LaTeX then scaled that whole
# 9.7in down to the target width, shrinking every label by ~23%.
#
# The fix is to keep the title narrower than the diagram itself so it never
# drives the crop. The binding line is the second one (~68 chars); at ~0.5em
# average glyph width it fits the ~7.4in axes only up to about 15pt. Hence 14pt
# below, with a little margin.
#
# RULE: after changing this title's text or size, re-check the saved PNG width.
# If it exceeds ~7.4in the title is overhanging again and everything in the
# figure will silently render smaller on the page.
ax.set_title("3D U-Net skeleton shared by all four architecture variants\n"
             "(introspected from model_base.py; base_channels=32, input patch 64³)",
             fontsize=13)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f"\nSaved: {OUT_PATH}")
