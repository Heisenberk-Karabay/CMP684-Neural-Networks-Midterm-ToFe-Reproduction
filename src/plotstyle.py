"""Shared matplotlib style for all report figures (light, print-friendly)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('seq_blue', [SURFACE, '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#104281', '#0d366b'])

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.prop_cycle': matplotlib.cycler(color=SERIES), 'lines.linewidth': 2, 'lines.markersize': 6,
    'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})


def image_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
