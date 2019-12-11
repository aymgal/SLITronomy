__author__ = 'aymgal'

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def nice_colorbar(mappable, position='right', pad=0.1, size='5%', **divider_kwargs):
    divider_kwargs.update({'position': position, 'pad': pad, 'size': size})
    ax = mappable.axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**divider_kwargs)
    return plt.colorbar(mappable, cax=cax)
    