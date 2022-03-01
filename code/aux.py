import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import matplotlib as mpl


def log_interp1d(xx, yy, kind='linear', fill_value=np.nan):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = interp1d(logx, logy, kind=kind, bounds_error=False, fill_value=fill_value)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def data_on_grid(x, y, x_grid, interpolation_kind='linear', unit='TeV', loglog=True, fill_value=np.nan):
    y_unit = y.unit
    if loglog==True:
        interpol_model=log_interp1d(x.to('TeV').value, y.value, kind=interpolation_kind)
        ys_loginterp=interpol_model(x_grid.to('TeV').value)
    ys_loginterp[np.isnan(ys_loginterp)]=fill_value
    return ys_loginterp * y_unit

def make_grid(xmin, xmax, npoints=300, unit='TeV', log=True):
    xmin_val = xmin.to('TeV').value
    xmax_val = xmax.to('TeV').value
    if log==True:
        x_grid = np.logspace(np.log10(xmin_val), np.log10(xmax_val), npoints)
    else:
        x_grid = np.linspace(xmin_val, xmax_val, npoints)
    return (x_grid *u.TeV).to(unit)

def table_to_dict(table, keycolumn_name, valuecolumn_name):
    dict = {}
    for column in range(len(table)):
        dict[table[keycolumn_name][column]] = table[valuecolumn_name][column]
    return dict

def get_key_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def get_names_str(name_tuple, name_dict):
    if isinstance(name_tuple,str):
        return name_dict[name_tuple]
    elif isinstance(name_tuple,list):
        namestring = ''
        for name in name_tuple:
            if name.isdigit():
                namestring = namestring[:-1]
                namestring += (' (' + name + ')/')
            else:
                namestring += (name_dict[name] + '/')
        namestring = namestring[:-1]
        return namestring
    else:
        print("Error")

#def read_files():
    
    
class PlottingStyle:
    def __init__(self, stylename, figshape='widerect', legend='side', energy_unit = 'TeV'):
        
        if stylename == 'antique':
            from palettable.cartocolors.qualitative import Antique_10 as colormap
            mpl.rcParams['image.cmap'] = mpl.colors.Colormap(colormap)
            self.colors = colormap.mpl_colors
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=self.colors)
            mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathpazo}']
            
            self.frameon = False
            pgf_with_rc_fonts = {                      # setup matplotlib to use latex for output
            "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
            "text.usetex": True,                # use LaTeX to write all text
            "font.family": "Palatino",
            "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
            "font.sans-serif": [],
            "font.monospace": [],
            "axes.labelsize": 16,               # LaTeX default is 10pt font.
            "font.size": 14,
            "legend.fontsize": 14,               # Make the legend/label fonts a little smaller
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pgf.preamble": [
                r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
                r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
                r"\usepackage{mathpazo}",
                ]
            }
            
            
            
        if figshape == 'widerect':
            self.ratio = 0.65
            self.image_resolution = 5000
            self.figwidth  = 30/ 2.54 
            self.figheight  = self.ratio * self.figwidth

        mpl.rcParams.update(pgf_with_rc_fonts)

        self.energy_unit = energy_unit
        self.legend = legend
