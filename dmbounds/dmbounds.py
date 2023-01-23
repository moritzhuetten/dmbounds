#!/usr/bin/env python3

# %Part of https://github.com/moritzhuetten/DMbounds under the
# %Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License,
# %see LICENSE.rst

import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import matplotlib as mpl
from matplotlib import pyplot as plt
import random
from astropy.io import ascii
import glob
import pandas as pd
import logging
import ipywidgets as wid
from IPython.display import Markdown, clear_output, HTML
import os
from math import nan
import itertools

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _check_duplicates(elem_list):
    """Check if given list elem_list contains any duplicates"""

    elem_set = set()
    for elem in elem_list:
        if elem in elem_set:
            return True, elem
        else:
            elem_set.add(elem)
    return False, None


def _data_on_grid(
        x,
        y,
        x_grid,
        interpolation_kind='linear',
        unit='TeV',
        loglog=True,
        fill_value=np.nan):
    """Transfer the the function given by the node points x,y on a grid given
       by the node points x_grid and using log-log interpolation.

    Parameters
    ----------
    x : (N,) array_like
        A 1-D dimensionful array of with dimension matching 'unit'. Default:
        energy
    y : (N,) array_like
        A 1-D dimensionful array. The length must be equal to xx
    x_grid: (N,) array_like
        A 1-D dimensionful array of with dimension matching 'unit'. Default:
        energy
    interpolation_kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer speci-
        fying the order of the spline interpolator to use. The string has to be
        one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
        ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’,
        ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth,
        first, second or third order; ‘previous’ and ‘next’ simply return the
        previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ
        when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’
        rounds up and ‘nearest’ rounds down. Default is ‘linear’.
    unit : str, optional
        astropy unit (default: TeV -> energy dimension) needed to specify the
        dimension of the x and x_grid (energy, length, time,...). The exact
        unit chosen in that dimension does not matter.
    loglog : bool, optional
        Choose loglog interpolation if True, linlin interpolation otherwise.
    fill_value : array-like or (array-like, array_like) or “extrapolate”, optional
        if a ndarray (or float), this value will be used to fill in for
        requested points outside of the data range. If not provided, then the
        default is NaN. The array- like must broadcast properly to the dimen-
        sions of the non-interpolation axes.

    Returns
    -------
    (N,) array_like
        The dimensionful values of the quantity y sampled on x_grid.
    """

    y_unit = y.unit
    if loglog:
        interpol_model = _log_interp1d(
            x.to(unit).value, y.value, kind=interpolation_kind)
        ys_interp = interpol_model(x_grid.to(unit).value)
    else:
        interpol_model = interp1d(
            x.to(unit).value,
            y.value,
            kind=interpolation_kind)
        ys_interp = interpol_model(x_grid.to(unit).value)
    ys_interp[np.isnan(ys_interp)] = fill_value
    return ys_interp * y_unit


def _filter_dataframe(metadata_df, mode, instrument, channel):
    """Filter the data frame to all rows which fullfill the condition on mode,
       instrument, channel.

    Parameters
    ----------
    metadata_df : pandas dataframe
        Metadata frame to filter
    mode : str
        has to start with 'ann' or 'dec' for annihilation or decay.
    instrument : str
        a short name string being present in legend_instruments.ecsv, or 'all'
    channel : str
        a short name string being present in legend_channel.ecsv, or 'all'

    Returns
    -------
    list
        List with row indices of the input dataframe matching the conditions on
        mode, instrument, channel.
    """

    mode_list = metadata_df.index[metadata_df['Mode'] == mode[:3]].tolist()

    if instrument == 'all':
        inst_list = metadata_df.index.tolist()
    else:
        inst_key = _get_key_from_value(INSTRUMENT_DICT, instrument)[0]
        if inst_key == 'multi-inst':
            inst_list = metadata_df.index[metadata_df['Instrument'].apply(
                type) == list].tolist()
        else:
            inst_list = metadata_df.index[metadata_df['Instrument'] == inst_key].tolist(
            )

    if channel == 'all':
        channel_list = metadata_df.index.tolist()
    else:
        channel_list = metadata_df.index[metadata_df['Channel'] == channel].tolist(
        )

    return _intersection3(mode_list, inst_list, channel_list)


def _get_key_from_value(dictionary, value):
    """Do the reverse query of searching the key name(s) of a value.

    Parameters
    ----------
    dictionary : dict
        The dictionary to search for.
    value : str
        The value whose key(s) to search to have this value

    Returns
    -------
    list
        A list of all the keys having that value. Returns empty list if no key
        has searched value.
    """

    return [k for k, v in dictionary.items() if v == value]


def _get_names_str(name_list, name_dict):
    """Get the long names of an instrument/channel/target from the short name
       definitions

    Parameters
    ----------
    name_list : string or list
        A string or list of strings containing the short name definitions (used
        as dictionary keys). If a list entry contains a number, this entry is
        not treated as a short name key, but directly printed into the output
        string
    name_dict : dict
        The dictionary to search for the short name definitions as keys

    Returns
    -------
    str
        A string containing all the long names corresponding to the short names
        in name_list.
    """

    if isinstance(name_list, str):
        return name_dict[name_list]
    elif isinstance(name_list, list):
        namestring = ''
        for name in name_list:
            if name.isdigit():
                namestring = namestring[:-1]
                namestring += (' (' + name + ')/')
            else:
                namestring += (name_dict[name] + '/')
        namestring = namestring[:-1]
        return namestring
    else:
        logging.error("Name not found")


def _init_metadata():
    """Initialize the pandas dataframe with the meta data from all bounds
       present in the data base.

        Returns
    -------
    pandas dataframe
        The dataframe with all the meta data.
    """

    files_all = []
    for name in INSTRUMENT_DICT.keys():
        files_all.append(glob.glob(MODULE_DIR + "/bounds/" + name + "/*.ecsv"))
    files_all = [x for row in files_all for x in row]

    metadata_df = pd.DataFrame(
        columns=(
            'Instrument',
            'Target',
            'Mode',
            'Channel',
            'Year',
            'Observation time',
            'Title',
            'DOI',
            'Arxiv',
            'Comment',
            'File name'))

    for i, file in enumerate(files_all):
        filename = file.split("/")[-1][:-5]

        file_inst_name = filename.split("_")[0]
        file_year = filename.split("_")[1]
        file_target = filename.split("_")[2]
        file_mode = filename.split("_")[3]
        file_channel = filename.split("_")[4]

        metadata = ascii.read(file).meta

        if metadata['instrument'][:10] == 'multi-inst':
            instruments = metadata['instrument'].split("-")[2:]
            meta_inst_name = 'multi-inst'
            file_inst_name = meta_inst_name
        else:
            meta_inst_name = metadata['instrument']
            instruments = metadata['instrument']

        if metadata['source'][:5] == 'multi':
            target_info = metadata['source'].split("-")
            meta_target = target_info[0]
        else:
            meta_target = metadata['source']
            target_info = metadata['source']

        try:
            assert meta_inst_name == file_inst_name
        except BaseException:
            logging.warning("Instrument name not consistent in " + str(file))
        try:
            assert metadata['year'] == file_year
        except BaseException:
            logging.warning("Year not consistent in " + str(file))
        try:
            assert meta_target == file_target
        except BaseException:
            logging.warning("Target name not consistent in " + str(file))
        try:
            assert metadata['channel'] == file_channel
        except BaseException:
            logging.warning("Channel name not consistent in " + str(file))

        metadata_df.loc[i] = [
            instruments,
            target_info,
            file_mode,
            metadata['channel'],
            metadata['year'],
            metadata['obs_time'],
            metadata['reference'],
            metadata['doi'],
            metadata['arxiv'],
            metadata['comment'],
            file]

    return metadata_df


def _intersection2(lst1, lst2):
    """Returns the intersection elements of two lists.

    Parameters
    ----------
    lst1 : list
        First list
    lst2 : list
        Second list

    Returns
    -------
    str
        List with intersecting elements. N.B.: Duplicate list elements are
        reduced.
    """

    # should be identical to list(set(lst1).intersection(set(lst2)))
    return list(set(lst1) & set(lst2))


def _intersection3(lst1, lst2, lst3):
    """Returns the intersection elements of three lists.

    Parameters
    ----------
    lst1 : list
        First list
    lst2 : list
        Second list
    lst3 : list
        Third list

    Returns
    -------
    list
        List with intersecting elements. N.B.: Duplicate list elements are
        reduced.
    """

    return list(set(lst1) & set(lst2) & set(lst3))


def _labels4dropdown(metadata_df):
    """Create the label strings for the dropdown menu in the interactive
       selection.

    Parameters
    ----------
    metadata_df : pandas data frame

    Returns
    -------
    list
        List with same lengths as rows in the input data frame containing the
        label texts as strings
    """

    labels = []
    for index, row in metadata_df.iterrows():
        label_str = ''
        label_str += _get_names_str(row['Instrument'], INSTRUMENT_DICT)
        label_str += ' (' + row['Year'] + '): ' + _get_names_str(row['Target'],
                                                                 TARGET_DICT) \
        + ', ' + row['Channel'] + ' (' + row['Mode'] + '.)'  # + str(index)
        if row['Comment'] != '':
            label_str += ', ' + row['Comment']
        labels.append(label_str)
    return labels


def _log_interp1d(xx, yy, kind='linear', fill_value=np.nan):
    """Log-log interpolation function based on scipy.interpolate.interp1d()

    Parameters
    ----------
    xx : (N,) array_like
        A 1-D array of real values.
    yy : (N,) array_like
        A 1-D array of real values. The length must be equal to xx
    kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer speci-
        fying the order of the spline interpolator to use. The string has to be
        one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
        ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’,
        ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth,
        first, second or third order; ‘previous’ and ‘next’ simply return the
        previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ
        when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’
        rounds up and ‘nearest’ rounds down. Default is ‘linear’.
    fill_value : array-like or (array-like, array_like) or “extrapolate”, optional
        if a ndarray (or float), this value will be used to fill in for
        requested points outside of the data range. If not provided, then the
        default is NaN. The array- like must broadcast properly to the
        dimensions of the non-interpolation axes.

    Returns
    -------
    function
        a function to call to evaluate the interpolated value at desired x
        coordinate.
    """

    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = interp1d(
        logx,
        logy,
        kind=kind,
        bounds_error=False,
        fill_value=fill_value)

    def log_interp(zz): return np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp


def _make_grid(xmin, xmax, npoints=300, unit='TeV', log=True):
    """Create a grid of dimensionful values.

    Parameters
    ----------
    xmin : dimensionful quantity
        A dimensionful number defining the lower bound of the grid.
        Default dimension: energy
    ymin : dimensionful quantity
        A dimensionful number defining the upper bound of the grid.
        Default dimension: energy
    npoints: int, optional
        Number of grid points
    unit : str, optional
        astropy unit (default: TeV -> energy dimension) in which the grid shall
        be sampled. Must match the dimension of xmin and xmax
    log : boolean, optional
        needed to specify the dimension of the x and x_grid (energy, length,
        time,...). The exact unit chosen in that dimension does not matter.

    Returns
    -------
    (N,) array_like
        The dimensionful values of the quantity y sampled on x_grid.
    """

    xmin_val = xmin.to(unit).value
    xmax_val = xmax.to(unit).value
    if log:
        x_grid = np.logspace(np.log10(xmin_val), np.log10(xmax_val), npoints)
    else:
        x_grid = np.linspace(xmin_val, xmax_val, npoints)
    return x_grid * u.Unit(unit)


def _table_to_dict(table, keycolumn_name, valuecolumn_name):
    """Transform an astropy table with two specified columns into a dictionary

    Parameters
    ----------
    table : astropy table
        The astropy table must have at least two, but can have more than two
        columns.
    keycolumn_name : str
        Name of the column which should represent the dictionary keys
    valuecolumn_name : str
        Name of the column which should represent the dictionary values

    Returns
    -------
    dict
        A dictionary in which {'keycolumn_entry': 'valuecolumn_entry',...} or
        dict[keycolumn_entry] = valuecolumn_entry
    """

    dictionary = {}
    for column in range(len(table)):
        dictionary[table[keycolumn_name][column]
                   ] = table[valuecolumn_name][column]
    return dictionary


class PlottingStyle:
    """
    Class to define the dark matter limits plots' style

    ...

    Attributes
    ----------
    stylename : str
        Plotting style font and color scheme. Choose between
          -  'antique'
          -  'standard'
    figshape : str, optional
        Figure shape style. So far, only 'widerect' implemented
    legend : str, optional
        The legend style. Choose between
          - 'side'
          - 'fancy'
    energy_unit : str, optional
        energy unit in which the plot will be displayed. Must be an energy unit.
        Default: 'TeV'
    mode : str, optional
        Choose between 'ann' (annihilation) or 'dec' (dacay). Default: 'ann'

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(
            self,
            stylename,
            figshape='widerect',
            legend='side',
            energy_unit='TeV',
            mode='ann'):

        self.style = stylename

        if self.style == 'antique':
            from palettable.cartocolors.qualitative import Antique_10 as colormap
            mpl.rcParams['image.cmap'] = mpl.colors.Colormap(colormap)
            self.colors = colormap.mpl_colors
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=self.colors)
            mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
            # mpl.rcParams['lines.linewidth'] = 1.0
            self.frameon = False
            pgf_with_rc_fonts = {   # setup matplotlib to use latex for output
                "pgf.texsystem": "pdflatex",  # change if using xetex or lautex
                "text.usetex": True,         # use LaTeX to write all text
                "font.family": "Palatino",
                "font.serif": [],            # blank entries should cause plots to inherit fonts from the document
                "font.sans-serif": [],
                "font.monospace": [],
                "axes.labelsize": 16,        # LaTeX default is 10pt font.
                "font.size": 14,
                "legend.fontsize": 14,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.top": True,
                "ytick.right": True,
                "pgf.preamble":
                r"\usepackage[utf8x]{inputenc} \
                \usepackage[T1]{fontenc} \
                \usepackage{mathpazo}"

            }

        elif self.style == 'standard':
            # colormap = plt.get_cmap('tab10')
            # self.colors = [colormap(k) for k in np.linspace(0, 1, 10)]
            from palettable.cartocolors.qualitative import Pastel_10 as colormap
            mpl.rcParams['image.cmap'] = mpl.colors.Colormap(colormap)
            self.colors = colormap.mpl_colors
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=self.colors)
            mpl.rcParams['text.latex.preamble'] = [
                r'\usepackage{siunitx}',
                # i need upright \micro symbols, but you need...
                r'\sisetup{detect-all}',
                # ...this to force siunitx to actually use your fonts
                r'\usepackage{helvet}',    # set the normal font here
                r'\usepackage{sansmath}',
                # load up the sansmath so that math -> helvet
                r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
            ]
            mpl.rcParams['lines.linewidth'] = 2.5
            self.frameon = False
            pgf_with_rc_fonts = {
                "pgf.texsystem": "pdflatex",
                "text.usetex": False,
                "font.family": "arial",
                "font.serif": [],
                "font.sans-serif": ['arial'],
                "font.monospace": ['arial'],
                "axes.labelsize": 16,
                "font.size": 14,
                "legend.fontsize": 14,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.top": True,
                "ytick.right": True,
                "pgf.preamble":
                r"\usepackage[utf8x]{inputenc} \
                \usepackage[T1]{fontenc}"
            }
        else:
            logging.error("unknow style name %s" % self.style)

        if figshape == 'widerect':
            self.ratio = 0.65
            self.image_resolution = 5000
            self.figwidth = 30 / 2.54
            self.figheight = self.ratio * self.figwidth
        else:
            logging.error("unknow figure shape style name %s" % self.style)

        mpl.rcParams.update(pgf_with_rc_fonts)

        self.energy_unit = energy_unit
        self.legend = legend
        self.mode = mode[:3]

        self.ymin = None
        self.ymax = None
        self.dynamic_plotrange = False

        self.color_cycle = 'random'

        if mode[:3] == 'ann':
            self.ylabel = r'$\langle\sigma v \rangle$ $[\mathrm{cm^3\,s^{-1}}]$'
        elif mode[:3] == 'dec':
            self.ylabel = r'$\tau$ $[\mathrm{s}]$'
        else:
            logging.error(
                "Unknown mode name %s. Must be either 'ann' or 'dec'." %
                self.style)


def plot(metadata_df, style=None):
    """Creates a limits plot from the metadata specified by a pandas dataframe.

    Parameters
    ----------
    metadata_df : pandas dataframe
        Metadata frame of the limits to include in the figure
    style : PlottingStyle() instance, optional
        Specify the plotting style of the limits plot

    Returns
    -------
    (2,) matplotlib figure object, matplotlib axis object
        matplotlib objects which can be further manipulated.
    """

    try:
        metadata_df.iloc[0]["Mode"]
    except BaseException:
        logging.error("Dataframe is empty. Must select at least one limit.")
        return None, None

    if style is None:
        style = PlottingStyle('antique', mode=metadata_df.iloc[0]["Mode"])

    data_raw_vec = []
    labels_plot = []
    interpol_style = []
    labels_plot_short = []
    xmin = 1e6 * u.TeV
    xmax = 1e-6 * u.TeV

    if style.mode == 'ann':
        blindval = 1e40
        if style.style == 'standard':
            blindval = np.nan
        if style.ymin is None:
            style.ymin = 1e-27
            style.dynamic_plotrange = True
        if style.ymax is None:
            style.ymax = 5e-20
    else:
        blindval = 1e-40
        if style.style == 'standard':
            blindval = np.nan
        if style.ymin is None:
            style.ymin = 1e20
        if style.ymax is None:
            style.ymax = 5e27
            style.dynamic_plotrange = True

    is_band = []
    for index, row in metadata_df.iterrows():
        data_raw = ascii.read(row['File name'])
        xaxis = data_raw.colnames[0]
        yaxis = data_raw.colnames[1]
        if yaxis == 'sigmav_lo' and style.legend != 'fancy':
            is_band.append(True)
            data_raw_y = [data_raw[yaxis], data_raw['sigmav_hi']]
        else:
            if yaxis == 'sigmav_lo':
                yaxis == 'sigmav'
            is_band.append(False)
            data_raw_y = data_raw[yaxis]
        data_raw_vec.append([data_raw[xaxis], data_raw_y])
        if min(data_raw[xaxis].to('TeV')) < xmin:
            xmin = min(data_raw[xaxis].to('TeV'))
        if max(data_raw[xaxis].to('TeV')) > xmax:
            xmax = max(data_raw[xaxis].to('TeV'))

        label_str = r''
        label_str += _get_names_str(row['Instrument'], INSTRUMENT_DICT)
        label_str += ' (' + row['Year'] + ')'
        labels_plot_short.append(label_str)
        label_str += ': ' + _get_names_str(row['Target'],
                                           TARGET_DICT) \
        + ', $' + _get_names_str(row['Channel'],
                                 CHANNEL_DICT) + '$'
        labels_plot.append(label_str)
        if 'gammagamma' in row['Channel'] or 'Sommerfeld' in row['Comment']:
            interpol_style.append('linear')
        else:
            interpol_style.append('quadratic')

    xmin /= 2
    xmax *= 2
    x_grid = _make_grid(
        xmin,
        xmax,
        npoints=2000,
        unit=style.energy_unit,
        log=True)
    n_plot = len(data_raw_vec)

    data_to_plot = []
    ymin_data = style.ymin
    ymax_data = style.ymax
    plot_minvalues = np.zeros(n_plot)
    plot_maxvalues = np.zeros(n_plot)

    for index in range(n_plot):
        if not is_band[index]:
            data_gridded = _data_on_grid(
                data_raw_vec[index][0],
                data_raw_vec[index][1],
                x_grid,
                interpolation_kind=interpol_style[index],
                fill_value=blindval)
            raw_vals = data_raw_vec[index][1]
        else:
            data_gridded1 = _data_on_grid(
                data_raw_vec[index][0],
                data_raw_vec[index][1][0],
                x_grid,
                interpolation_kind=interpol_style[index],
                fill_value=blindval)
            data_gridded2 = _data_on_grid(
                data_raw_vec[index][0],
                data_raw_vec[index][1][1],
                x_grid,
                interpolation_kind=interpol_style[index],
                fill_value=blindval)
            data_gridded = [data_gridded1, data_gridded2]
            raw_vals = list(itertools.chain(*data_raw_vec[index][1]))
        data_to_plot.append(data_gridded)
        plot_minvalues[index] = min(raw_vals)
        plot_maxvalues[index] = max(raw_vals)

        if style.mode == 'ann' and style.dynamic_plotrange:
            if min(raw_vals) < ymin_data:
                ymin_data = 0.5 * min(raw_vals)
        elif style.mode == 'dec' and style.dynamic_plotrange:
            if max(raw_vals) > ymax_data:
                ymax_data = 2 * max(raw_vals)

    if len(plot_minvalues) > 1:
        if style.mode == 'ann':
            plot_ranking_ids = [
                i for i in reversed(
                    np.argsort(plot_minvalues))]
        else:
            plot_ranking_ids = [
                i for i in reversed(
                    np.argsort(plot_maxvalues))]
    else:
        plot_ranking_ids = np.argsort(plot_minvalues)

    envelope = []
    for i in range(len(x_grid)):
        minvals = []
        for j in range(len(data_to_plot)):
            if not is_band[j]:
                val = data_to_plot[j][i].value
            else:
                if style.mode == 'ann':
                    val = data_to_plot[j][0][i].value
                else:
                    val = data_to_plot[j][1][i].value
            if np.isnan(val):
                val = blindval
            minvals.append(val)
        if style.mode == 'ann':
            envelope.append(0.95 * min(minvals))
        else:
            envelope.append(1.05 * max(minvals))

    plot_limits = plt.figure(figsize=(style.figwidth, style.figheight))
    ax = plt.gca()

    if style.color_cycle == 'random':
        random_order = list(range(n_plot))
        random.shuffle(random_order)
        order = random_order
    elif style.color_cycle == 'standard':
        order = list(range(n_plot))
    else:
        if len(style.color_cycle) < n_plot:
            logging.error("Color cycle must be as long as # of plotted limits")
        order = style.color_cycle

    for j in plot_ranking_ids:

        try:
            color = style.colors[order[j]]
        except BaseException:
            color = order[j]
        if not is_band[j]:
            plt.plot(
                x_grid,
                data_to_plot[j],
                label=labels_plot[j],
                color=color)
        else:
            plt.fill_between(
                x_grid.value,
                data_to_plot[j][0].value,
                data_to_plot[j][1].value,
                label=labels_plot[j],
                color=color,
                alpha=0.6)
        if style.style == 'antique':
            if not is_band[j]:
                if style.mode == 'ann':
                    plt.fill_between(
                        x_grid.value, data_to_plot[j].value, np.ones(
                            len(x_grid)), alpha=0.1, color=color)
                else:
                    plt.fill_between(
                        x_grid.value,
                        np.ones(
                            len(x_grid)),
                        data_to_plot[j].value,
                        alpha=0.1,
                        color=color)
            else:
                if style.mode == 'ann':
                    plt.fill_between(
                        x_grid.value, data_to_plot[j][1].value, np.ones(
                            len(x_grid)), alpha=0.1, color=color)
                else:
                    plt.fill_between(
                        x_grid.value,
                        np.ones(
                            len(x_grid)),
                        data_to_plot[j][0].value,
                        alpha=0.1,
                        color=color)

        if style.legend == 'fancy':
            if style.mode == 'ann':
                try:
                    i_legend = next(
                        x[0] for x in enumerate(
                            data_to_plot[j].value) if x[1] < 1e40)
                except BaseException:
                    i_legend = next(
                        x[0] for x in enumerate(
                            data_to_plot[j][0].value) if x[1] < 1e40)
                valign = 'top'
                vpad = 0.9 * style.ymax
            else:
                i_legend = next(
                    x[0] for x in enumerate(
                        data_to_plot[j].value) if x[1] > 1e-40)
                valign = 'bottom'
                vpad = 1.2 * style.ymin

            plt.text(
                x_grid[i_legend].value,
                vpad,
                labels_plot_short[j],
                horizontalalignment='right',
                verticalalignment=valign,
                snap=True,
                color=color,
                rotation=90)

    if style.style == 'antique':
        plt.plot(x_grid, envelope, linewidth=5, color='k', alpha=0.2)

    if style.legend == 'side':
        plt.legend(
            bbox_to_anchor=(
                1.02,
                1),
            loc="upper left",
            frameon=style.frameon)

    if style.mode == 'ann':
        wimp_model = ascii.read(
            MODULE_DIR +
            "/modelpredictions/wimp_huetten2017_analytical_omega012.ecsv")
        xtext = 0.9 * \
            min(wimp_model["mass"].to(style.energy_unit)[-1].value, plt.gca().get_xlim()[1])
        plt.text(xtext,
                 wimp_model["sigmav"][-1],
                 "Thermal WIMP prediction",
                 horizontalalignment='right',
                 verticalalignment='bottom',
                 snap=True,
                 color='k',
                 alpha=0.3)
        wimp_model_gridded = _data_on_grid(
            wimp_model["mass"],
            wimp_model["sigmav"],
            x_grid,
            interpolation_kind='quadratic',
            fill_value=1 / blindval)
        # if style.style == 'antique':
        plt.fill_between(
            x_grid.value,
            np.zeros(2000),
            wimp_model_gridded.value,
            color='k',
            alpha=0.1)

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([ymin_data, ymax_data])
    plt.xlim([xmin.to(style.energy_unit).value,
             xmax.to(style.energy_unit).value])

    plt.xlabel(r'$m_{\mathrm{DM}}$ $\mathrm{[' + style.energy_unit + ']}$')
    plt.ylabel(style.ylabel)

    plt.text(1, 1, 'made with github.com/moritzhuetten/dmbounds',
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax.transAxes, snap=True, color='k', alpha=0.3, size=10)

    return plot_limits, ax


def metadata():
    """Load all metadata present in the database into a Pandas dataframe.

    Returns
    -------
    pandas dataframe
        dataframe with the meta data.
    """

    return METADATA_DF


def interactive_selection():
    """Create an IPython widget to filter the limits interactively, and to
       display a figure on the fly.

    Returns
    -------
    dict
        A dictionary returning the selection state of all database entries.
        Can be used as input for the filter_metadata() method.
    """

    inst_list = list(INSTRUMENT_DICT.values())
    inst_list.insert(0, 'all')

    channel_list = list(CHANNEL_DICT.keys())
    channel_list.insert(0, 'all')

    metadata_df = metadata()
    labels = _labels4dropdown(metadata_df)

    style = PlottingStyle('antique')

    def _multi_checkbox_widget(options_dict):
        """ Widget with a search field and lots of checkboxes """

        # adapted from
        # https://gist.github.com/MattJBritton/9dc26109acb4dfe17820cf72d82f1e6f

        style_widget = wid.Dropdown(
            options=[
                'antique',
                'standard',
                'fancy'],
            description='Plotting style')

        mode_widget = wid.Dropdown(
            options=[
                'annihilation',
                'decay'],
            description='Mode')
        instrument_widget = wid.Dropdown(
            options=inst_list, description='Instrument')
        channel_widget = wid.Dropdown(
            options=channel_list, description='Channel')

        output_widget = wid.Output()
        options = [x for x in options_dict.values()]

        start_index_list = _filter_dataframe(
            metadata_df, 'annihilation', 'all', 'all')
        start_options = []
        for index in start_index_list:
            start_options.append(options[index])
        start_options = sorted([x for x in start_options],
                               key=lambda x: x.description, reverse=False)
        style.__init__('antique', mode='ann')

        options_layout = wid.Layout(
            overflow='auto',
            border='1px solid black',
            width='950px',
            height='200px',
            flex_flow='row wrap',
            display='flex',
        )

        # selected_widget = wid.Box(children=[options[0]])
        options_widget = wid.VBox(options, layout=options_layout)
        options_widget.children = start_options
        # print(options_widget.children)
        # left_widget = wid.VBox(search_widget, selected_widget)
        multi_select = wid.VBox(
            [style_widget, mode_widget, instrument_widget, channel_widget, options_widget])

        @output_widget.capture()
        def on_checkbox_change(change):

            # selected_recipe = change["owner"].description
            # print(options_widget.children)
            # selected_item = wid.Button(description = change["new"])
            # selected_widget.children = [] #selected_widget.children +
            # [selected_item]
            options_widget.children = sorted(
                [x for x in options_widget.children], key=lambda x: x.description, reverse=False)
            options_widget.children = sorted(
                [x for x in options_widget.children], key=lambda x: x.value, reverse=True)
            # options_widget.children = [x for x in options_widget.children]

        for checkbox in options:
            checkbox.observe(on_checkbox_change, names="value")

        # Wire the search field to the checkboxes
        @output_widget.capture()
        def dropdown_change(*args):

            index_list = _filter_dataframe(
                metadata_df,
                mode_widget.value,
                instrument_widget.value,
                channel_widget.value)
            new_options = []
            for index in index_list:
                new_options.append(options[index])
            new_options = sorted([x for x in new_options],
                                 key=lambda x: x.description, reverse=False)
            options_widget.children = sorted(
                [x for x in new_options], key=lambda x: x.value, reverse=True)

            clear_output()
            display(Markdown('Filtered %d data sets.' % len(index_list)))
            if args[0].owner.description == 'Mode':
                clear_output()
                display(
                    Markdown('Attention when switching between annihilation and decay'))
            if style_widget.value == 'fancy':
                style.__init__(
                    'antique',
                    legend='fancy',
                    mode=mode_widget.value)
            else:
                style.__init__(style_widget.value, mode=mode_widget.value)

        mode_widget.observe(dropdown_change)
        instrument_widget.observe(dropdown_change)
        channel_widget.observe(dropdown_change)
        style_widget.observe(dropdown_change)

        display(output_widget)
        return multi_select

    a, b = _check_duplicates(labels)
    if a:
        logging.error(
            "Duplicate label entry found for %s. Please make it unique in the file header." %
            b)

    options_dict = {
        x: wid.Checkbox(
            description=x,
            value=False,
            style={"description_width": "0px"},
            layout=wid.Layout(width='100%', flex_flow='wrap')
        ) for x in labels
    }

    def _interactive_output(**args):

        i_vec = [
            index for index,
            (key,
             value) in enumerate(
                args.items()) if value]
        if len(i_vec) > 0:
            metadata_filtered_df = metadata_df.loc[i_vec]
            # display(Markdown('Selected %d data sets.'%len(i_vec)))
            figure, ax = plot(metadata_filtered_df, style)
            # display(metadata_df.loc[i_vec])
            return figure
        else:
            display(Markdown('Nothing selected'))

        # display(results)

    ui = _multi_checkbox_widget(options_dict)
    out = wid.interactive_output(_interactive_output, options_dict)
    display(wid.VBox([ui, out]))

    return options_dict


def filter_metadata(options_dict):
    """Create a Pandas data frame with the entries filtered by the
       interactive_selection() method

    Parameters
    ----------
    options_dict : dict
        A dictionary with the selection state of all database entries created
        by the interactive_selection() method.

    Returns
    -------
    pandas dataframe
        dataframe with the filtered meta data.
    """

    i_vec = [
        index for index,
        (key,
         value) in enumerate(
            options_dict.items()) if value.value]
    return METADATA_DF.loc[i_vec]


def get_data(metadata_df):
    """Return the actual limits data from a metadata frame.

    Parameters
    ----------
    metadata_df : pandas dataframe
        dataframe with containing the limits whose data shall be returned.

    Returns
    -------
    list
        A list of same lenght as rows in the input data frame, each entry
        containing an astropy table with the data.
    """

    data = []
    for index, row in metadata_df.iterrows():
        data.append(ascii.read(row['File name']))
    return data


def show_metadata(metadata_df):
    """Display the metadata frame as pretty HTML table with clickable hyperlinks.

    Parameters
    ----------
    metadata_df : pandas dataframe
        dataframe with the metadata to display

    Returns
    -------
    HTML rendered table object
    """

    metadata_disp = metadata_df.copy()
    metadata_disp['DOI'] = metadata_disp['DOI'].apply(
        lambda x: f'<a href="https://doi.org/{x}" target=_blank>{x}</a>')
    metadata_disp['Arxiv'] = metadata_disp['Arxiv'].apply(
        lambda x: f'<a href="https://arxiv.org/abs/{x}" target=_blank>{x}</a>')
    return HTML(metadata_disp.to_html(render_links=True, escape=False))


INSTRUMENT_DICT = _table_to_dict(
    ascii.read(
        MODULE_DIR +
        "/legends/legend_instruments.ecsv"),
    'shortname',
    'longname')
CHANNEL_DICT = _table_to_dict(
    ascii.read(
        MODULE_DIR +
        "/legends/legend_channels.ecsv"),
    'shortname',
    'latex')
TARGET_DICT = _table_to_dict(
    ascii.read(
        MODULE_DIR +
        "/legends/legend_targets.ecsv"),
    'shortname',
    'longname')
METADATA_DF = _init_metadata()
