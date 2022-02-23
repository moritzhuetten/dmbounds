import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u

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

#def read_files():
    