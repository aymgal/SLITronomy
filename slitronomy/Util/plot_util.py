__author__ = 'aymgal'

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def nice_colorbar(mappable, position='right', pad=0.1, size='5%', **divider_kwargs):
    divider_kwargs.update({'position': position, 'pad': pad, 'size': size})
    ax = mappable.axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**divider_kwargs)
    return plt.colorbar(mappable, cax=cax)

def log_cmap(cmap_name, vmin, vmax):
    base_cmap = plt.get_cmap(cmap_name)
    return ReNormColormapAdaptor(base_cmap, mpl.colors.LogNorm(vmin, vmax))

class ReNormColormapAdaptor(mpl.colors.Colormap):
    """
    Colormap adaptor that uses another Normalize instance
    for the colormap than applied to the mappable.
    
    credits : https://stackoverflow.com/questions/26253947/logarithmic-colormap-in-matplotlib
    """
    def __init__(self, base, cmap_norm, orig_norm=None):
        if orig_norm is None:
            if isinstance(base, mpl.cm.ScalarMappable):
                orig_norm = base.norm
                base = base.cmap
            else:
                orig_norm = mpl.colors.Normalize(0,1)
        base.set_bad(color=base(0))
        base.set_under(base(0))
        self._base = base
        if (
            isinstance(cmap_norm,type(mpl.colors.Normalize))
            and issubclass(cmap_norm,mpl.colors.Normalize)
        ):
            # a class was provided instead of an instance. create an instance
            # with the same limits.
            cmap_norm = cmap_norm(orig_norm.vmin,orig_norm.vmax)
        self._cmap_norm = cmap_norm
        self._orig_norm = orig_norm

    def __call__(self, X, **kwargs):
        """ Re-normalise the values before applying the colormap. """
        return self._base(self._cmap_norm(self._orig_norm.inverse(X)),**kwargs)

    def __getattr__(self,attr):
        """ Any other attribute, we simply dispatch to the underlying cmap. """
        return getattr(self._base,attr)