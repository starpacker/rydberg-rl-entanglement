"""Shared plotting configuration."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'blue': '#4E79A7', 'orange': '#F28E2B', 'green': '#59A14F',
    'red': '#E15759', 'purple': '#B07AA1', 'brown': '#9C755F',
}
FIGURE_DIR = 'figures'
SINGLE_COL = (3.5, 2.8)
DOUBLE_COL = (7.0, 4.5)
