import matplotlib as mpl
import numpy as np


mpl.rcParams.update({
    'font.family': 'normal',
    'font.serif': [],
    'font.sans-serif': [],
    'font.monospace': [],
    'font.size': 12,
    'text.usetex': False,
})


import matplotlib.cm as cm
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


sns.set_context('paper', font_scale=2.5)
sns.set_style('white')
sns.mpl.rcParams['legend.frameon'] = 'False'


def plot_deps(data, colors=None, fpath=None, name_spec=None,
              xlabel='Number of requests', ylabel=None, title=None,
              lim_x=None, lim_y=None):

    if colors is None:
        colors = [
            '#8b1d1d', '#000099', '#558000', '#ffbf00', '#00FFFF' ,
            '#CE0071', '#485536', '#FFF800', '#66ffcc', '#5f91ac',
            '#ff66ff', '#6699ff', '#cc0000', '#333300', '#804000']

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    y_max = -np.infty
    y_min = +np.infty

    for i, (name, item) in enumerate(data.items()):
        if item.get('skip') == True:
            continue

        x = np.arange(len(item['avg'])) + 1

        ax.plot(x, item['avg'], label=name, color=colors[i],
            marker='o', markersize=0,
            linestyle='-' if i < 5 else '--',
            linewidth=3 if name_spec == name else 1)

        ax.fill_between(x, item['min'], item['max'], alpha=0.4, color=colors[i])

        y_max = max(y_max, np.max(item['max']))
        y_min = min(y_min, np.min(item['min']))

    _prep_ax(ax, xlog=True, ylog=(y_max - y_min > 300), leg=True)

    if lim_x is not None:
        ax.set_xlim(*lim_x)
    if lim_y is not None:
        ax.set_ylim(*lim_y)

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _prep_ax(ax, xlog=False, ylog=False, leg=False, xint=False, xticks=None):
    if xlog:
        ax.semilogx()
    if ylog:
        ax.set_yscale('symlog')

    if leg:
        ax.legend(loc='upper left', frameon=True)

    ax.grid(ls=":")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if xint:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if xticks is not None:
        ax.set(xticks=xticks, xticklabels=xticks)
