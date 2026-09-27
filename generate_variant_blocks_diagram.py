"""
Compact 4-panel figure showing what distinguishes the four architecture
variants. Every channel count / hyperparameter is introspected from the
instantiated models, not hand-typed.

Panels:
  (a) ConvBlock        -- used by base, attention, selective_attention
  (b) ResidualSEBlock  -- used by se_residual
  (c) AttentionGate    -- used by attention, selective_attention
  (d) Skip-connection treatment matrix across all four variants
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from model_base import UNet3D as Base
from model_att import UNet3D as Att
from model_selectiveatt import UNet3D as Sel
from model_se_residual import UNet3D as SER

OUT = Path(r"C:\Users\ramar\OneDrive\Desktop\Lung Nodule Analysis\paper\figures\variant_blocks.png")

# ─── Introspect ──────────────────────────────────────────────────────────────
models = {'base': Base(), 'attention': Att(), 'selective_attention': Sel(), 'se_residual': SER()}
mods = {k: dict(m.named_modules()) for k, m in models.items()}

gates = {k: [g for g in ('att1', 'att2', 'att3') if g in mods[k]] for k in models}

ag = mods['attention']['att3']
ag_x_in, ag_inter = ag.Wx[0].in_channels, ag.Wx[0].out_channels
ag_psi_out = ag.psi[0].out_channels
ag_W_in, ag_W_out = ag.W[0].in_channels, ag.W[0].out_channels
ag_ratio = ag_x_in // ag_inter

se_blk = mods['se_residual']['enc1']
se_fc = [l for l in se_blk.se.excitation if hasattr(l, 'in_features')]
se_c, se_hidden = se_fc[0].in_features, se_fc[0].out_features
se_reduction = se_c // se_hidden

enc_skips = [(mods['base']['enc1'].block[3].out_channels),
             (mods['base']['enc2'].block[3].out_channels),
             (mods['base']['enc3'].block[3].out_channels)]

print("Introspected:")
for k in models:
    print(f"  {k:20s} params={sum(p.numel() for p in models[k].parameters()):,}  gates={gates[k]}")
print(f"  AttentionGate: Wx C->C/{ag_ratio} (e.g. {ag_x_in}->{ag_inter}), psi ->{ag_psi_out}, "
      f"W {ag_W_in}->{ag_W_out}")
print(f"  SE: reduction r={se_reduction} (e.g. {se_c}->{se_hidden}->{se_c})")
print(f"  skip channel widths: {enc_skips}")

# ─── Draw ────────────────────────────────────────────────────────────────────
# Canvas width sets how large this figure renders on the page. The figure is
# already 	extwidth (a figure*), so width cannot grow further; instead the
# on-page size of everything is fontsize * (7.16in / crop_width_in), so a
# NARROWER canvas at the same font sizes renders the whole diagram larger.
# Trimmed 19.1 -> 17.6in for a modest ~9% enlargement. Narrowing compresses
# each panel horizontally, so the render was inspected for label overflow
# afterwards rather than assumed safe.
fig = plt.figure(figsize=(17.6, 4.3))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.15, 1.35, 1.5], wspace=0.28)

CONV = '#4C72B0'
NORM = '#9BB7D4'
ACT = '#C7C7C7'
SE = '#55A868'
GATE = '#DD8452'
ADD = '#B04C4C'

def box(ax, y, txt, color, h=0.72, x=0.5, w=2.4, fs=10.7, tc='white', bold=False):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='black', lw=1.1, zorder=3))
    ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=fs,
            color=tc, zorder=4, fontweight='bold' if bold else 'normal')

def arr(ax, x, y0, y1, color='black', lw=1.5, style='-'):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, linestyle=style))

# ---- (a) ConvBlock -----------------------------------------------------------
ax = fig.add_subplot(gs[0, 0]); ax.axis('off'); ax.set_xlim(0, 3.4); ax.set_ylim(-0.4, 7.4)
ys = [6.2, 5.2, 4.2, 2.9, 1.9, 0.9]
labels = [f'Conv3d 3\u00b3', 'InstanceNorm3d', 'ReLU', f'Conv3d 3\u00b3', 'InstanceNorm3d', 'ReLU']
cols = [CONV, NORM, ACT, CONV, NORM, ACT]
tcs = ['white', 'black', 'black', 'white', 'black', 'black']
for y, l, c, tc in zip(ys, labels, cols, tcs):
    box(ax, y, l, c, tc=tc)
for i in range(len(ys) - 1):
    arr(ax, 1.7, ys[i], ys[i+1] + 0.72)
ax.set_title('(a) ConvBlock\nbase, attention, selective', fontsize=12.5, fontweight='bold', pad=4)
ax.text(1.7, -0.3, 'skips: plain concatenation', fontsize=10.2, ha='center', style='italic')

# ---- (b) ResidualSEBlock -----------------------------------------------------
ax = fig.add_subplot(gs[0, 1]); ax.axis('off'); ax.set_xlim(0, 3.9); ax.set_ylim(-0.4, 7.4)
ys = [6.4, 5.55, 4.7, 3.85, 3.0, 2.15]
labels = ['Conv3d 3\u00b3', 'InstanceNorm3d', 'ReLU', 'Conv3d 3\u00b3', 'InstanceNorm3d',
          f'SE (r={se_reduction})']
cols = [CONV, NORM, ACT, CONV, NORM, SE]
tcs = ['white', 'black', 'black', 'white', 'black', 'white']
for y, l, c, tc in zip(ys, labels, cols, tcs):
    box(ax, y, l, c, h=0.62, w=2.3, fs=10.4, tc=tc)
for i in range(len(ys) - 1):
    arr(ax, 1.65, ys[i], ys[i+1] + 0.62)
# add node
ax.add_patch(patches.Circle((1.65, 1.55), 0.22, facecolor=ADD, edgecolor='black', lw=1.1, zorder=4))
ax.text(1.65, 1.55, '+', ha='center', va='center', fontsize=15.9, color='white',
        fontweight='bold', zorder=5)
arr(ax, 1.65, 2.15, 1.79)
box(ax, 0.5, 'ReLU', ACT, h=0.6, w=2.3, fs=10.4, tc='black')
arr(ax, 1.65, 1.33, 1.12)
# shortcut
ax.annotate('', xy=(1.87, 1.55), xytext=(3.55, 1.55),
            arrowprops=dict(arrowstyle='-|>', color=ADD, lw=1.5))
ax.plot([3.55, 3.55], [1.55, 7.18], color=ADD, lw=1.5)
ax.plot([1.65, 3.55], [7.18, 7.18], color=ADD, lw=1.5)
ax.add_patch(patches.Circle((1.65, 7.18), 0.07, facecolor=ADD, edgecolor=ADD, zorder=5))
ax.annotate('', xy=(1.65, 7.02), xytext=(1.65, 7.30),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1.5))
ax.text(3.62, 4.0, 'shortcut\n1\u00b3 conv + IN', fontsize=9.4, color=ADD,
        rotation=90, va='center', ha='left')
ax.set_title('(b) ResidualSEBlock\nse_residual', fontsize=12.5, fontweight='bold', pad=4)
ax.text(1.65, -0.3, 'skips: plain concatenation', fontsize=10.2, ha='center', style='italic')

# ---- (c) AttentionGate -------------------------------------------------------
ax = fig.add_subplot(gs[0, 2]); ax.axis('off'); ax.set_xlim(0, 4.6); ax.set_ylim(-0.4, 7.4)
box(ax, 6.4, f'$W_g$ 1\u00b3: C\u2192C/{ag_ratio}', CONV, h=0.62, x=0.15, w=1.9, fs=10.2)
box(ax, 6.4, f'$W_x$ 1\u00b3: C\u2192C/{ag_ratio}', CONV, h=0.62, x=2.4, w=1.9, fs=10.2)
ax.text(1.1, 7.2, 'g (decoder)', fontsize=9.9, ha='center')
ax.text(3.35, 7.2, 'x (encoder skip)', fontsize=9.9, ha='center')
ax.add_patch(patches.Circle((2.22, 5.75), 0.2, facecolor=ADD, edgecolor='black', lw=1.1, zorder=4))
ax.text(2.22, 5.75, '+', ha='center', va='center', fontsize=14.5, color='white',
        fontweight='bold', zorder=5)
ax.annotate('', xy=(2.05, 5.9), xytext=(1.1, 6.4), arrowprops=dict(arrowstyle='-|>', color='black', lw=1.4))
ax.annotate('', xy=(2.39, 5.9), xytext=(3.35, 6.4), arrowprops=dict(arrowstyle='-|>', color='black', lw=1.4))
for y, l, c, tc in [(4.75, 'ReLU', ACT, 'black'),
                    (3.75, f'$\\psi$ 1\u00b3: C/{ag_ratio}\u2192{ag_psi_out}', GATE, 'white'),
                    (2.75, '$\\sigma$ (sigmoid) \u2192 $\\alpha$', GATE, 'white')]:
    box(ax, y, l, c, h=0.62, x=1.05, w=2.35, fs=10.2, tc=tc)
arr(ax, 2.22, 5.55, 5.37); arr(ax, 2.22, 4.75, 4.37); arr(ax, 2.22, 3.75, 3.37)
ax.add_patch(patches.Circle((2.22, 2.1), 0.2, facecolor=ADD, edgecolor='black', lw=1.1, zorder=4))
ax.text(2.22, 2.1, '\u00d7', ha='center', va='center', fontsize=14.5, color='white',
        fontweight='bold', zorder=5)
arr(ax, 2.22, 2.75, 2.32)
ax.annotate('', xy=(2.42, 2.1), xytext=(4.35, 2.1), arrowprops=dict(arrowstyle='-|>', color='black', lw=1.4))
ax.plot([4.35, 4.35], [2.1, 6.71], color='black', lw=1.4, linestyle=':')
box(ax, 0.85, f'$W$ 1\u00b3 + IN', CONV, h=0.62, x=1.05, w=2.35, fs=10.2)
arr(ax, 2.22, 1.9, 1.49)
ax.text(2.22, 0.35, r'output $\hat{x}$', fontsize=10.2, ha='center')
ax.set_title('(c) AttentionGate\nattention, selective', fontsize=12.5, fontweight='bold', pad=4)
ax.text(2.3, -0.3, 'additive gate on skip connection', fontsize=10.2, ha='center', style='italic')

# ---- (d) Skip treatment matrix ----------------------------------------------
ax = fig.add_subplot(gs[0, 3]); ax.axis('off'); ax.set_xlim(-0.2, 5.0); ax.set_ylim(-0.4, 7.4)
order = ['base', 'attention', 'selective_attention', 'se_residual']
disp = {'base': 'base', 'attention': 'attention',
        'selective_attention': 'selective', 'se_residual': 'se_residual'}
skip_names = [f'skip1\n{enc_skips[0]} ch', f'skip2\n{enc_skips[1]} ch', f'skip3\n{enc_skips[2]} ch']
x0, colw, rowh = 1.45, 1.05, 0.95
for j, sn in enumerate(skip_names):
    ax.text(x0 + colw * j + colw / 2, 6.35, sn, fontsize=9.9, ha='center', va='center')
for i, k in enumerate(order):
    y = 5.5 - i * rowh
    ax.text(1.35, y + rowh / 2 - 0.12, disp[k], fontsize=10.7, ha='right', va='center',
            family='monospace')
    for j, gname in enumerate(['att1', 'att2', 'att3']):
        gated = gname in gates[k]
        cx = x0 + colw * j
        ax.add_patch(patches.Rectangle((cx + 0.08, y - 0.28), colw - 0.16, rowh - 0.28,
                                       facecolor=GATE if gated else '#E8E8E8',
                                       edgecolor='black', lw=1.0, zorder=3))
        ax.text(cx + colw / 2, y + rowh / 2 - 0.28, 'AG' if gated else 'concat',
                fontsize=9.6, ha='center', va='center',
                color='white' if gated else 'black', fontweight='bold' if gated else 'normal',
                zorder=4)
ax.set_title('(d) Skip-connection treatment\nper variant', fontsize=12.5, fontweight='bold', pad=4)
ax.text(2.8, 0.75, 'AG = attention gate applied', fontsize=9.9, ha='center', color=GATE)
ax.text(2.8, 0.3, 'concat = ungated concatenation', fontsize=9.9, ha='center', color='#555555')

fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"\nSaved: {OUT}")
