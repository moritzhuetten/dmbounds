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

module_dir = os.path.dirname(os.path.abspath(__file__))

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
        logging.error("Name not found")

#def read_files():
def intersection2(lst1, lst2):
    return list(set(lst1) & set(lst2))    

def intersection3(lst1, lst2, lst3):
    return list(set(lst1) & set(lst2) & set(lst3))  
    
class PlottingStyle:
    def __init__(self, stylename, figshape='widerect', legend='side', energy_unit = 'TeV', mode = 'ann'):
        
        self.style = stylename
        
        if self.style == 'antique':
            from palettable.cartocolors.qualitative import Antique_10 as colormap
            mpl.rcParams['image.cmap'] = mpl.colors.Colormap(colormap)
            self.colors = colormap.mpl_colors
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=self.colors)
            mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
            #mpl.rcParams['lines.linewidth'] = 1.0
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
            "pgf.preamble": 
                r"\usepackage[utf8x]{inputenc} \
                \usepackage[T1]{fontenc} \
                \usepackage{mathpazo}"
                
            }

        elif self.style == 'standard':
            #colormap = plt.get_cmap('tab10')
            #self.colors = [colormap(k) for k in np.linspace(0, 1, 10)]
            from palettable.cartocolors.qualitative import Pastel_10 as colormap
            mpl.rcParams['image.cmap'] = mpl.colors.Colormap(colormap)
            self.colors = colormap.mpl_colors
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=self.colors)
            mpl.rcParams['text.latex.preamble'] = [
                   r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
                   r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
                   r'\usepackage{helvet}',    # set the normal font here
                   r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
                   r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
            ]
            mpl.rcParams['lines.linewidth'] = 2.5
            self.frameon = False
            pgf_with_rc_fonts = {                      # setup matplotlib to use latex for output
            "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
            "text.usetex": False,                # use LaTeX to write all text
            "font.family": "arial",
            "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
            "font.sans-serif": ['arial'],
            "font.monospace": ['arial'],
            "axes.labelsize": 16,               # LaTeX default is 10pt font.
            "font.size": 14,
            "legend.fontsize": 14,               # Make the legend/label fonts a little smaller
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
            logging.error("unknow style name %s"%self.style)    
            
        if figshape == 'widerect':
            self.ratio = 0.65
            self.image_resolution = 5000
            self.figwidth  = 30/ 2.54 
            self.figheight  = self.ratio * self.figwidth

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
            
def plot(df, style=None):

    try:
        df.iloc[0]["Mode"]
    except:
        logging.error("Dataframe is empty. Must select at least one limit.")

    if style==None:
        style = PlottingStyle('antique', mode=df.iloc[0]["Mode"])

    data_raw_vec = []
    labels_plot = []
    interpol_style = []
    labels_plot_short = []
    xmin = 1e6*u.TeV
    xmax = 1e-6*u.TeV
    
    if style.mode == 'ann':
        blindval = 1e40
        if style.style == 'standard': blindval = np.nan
        if style.ymin == None:
            style.ymin = 1e-27
            style.dynamic_plotrange = True
        if style.ymax == None:
            style.ymax = 5e-20
    else:
        blindval = 1e-40
        if style.style == 'standard': blindval = np.nan
        if style.ymin == None:
            style.ymin = 1e20
        if style.ymax == None:
            style.ymax = 5e27
            style.dynamic_plotrange = True 

    is_band = []
    for index, row in df.iterrows():
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
        if min(data_raw[xaxis].to('TeV')) < xmin: xmin = min(data_raw[xaxis].to('TeV'))
        if max(data_raw[xaxis].to('TeV')) > xmax: xmax = max(data_raw[xaxis].to('TeV'))

        label_str = r''
        label_str += get_names_str(row['Instrument'], instrument_dict)
        label_str += ' (' + row['Year'] + ')'
        labels_plot_short.append(label_str)
        label_str += ': ' + get_names_str(row['Target'], target_dict) + ', $' + get_names_str(row['Channel'], channel_dict) + '$'
        labels_plot.append(label_str)
        if 'gammagamma' in row['Channel'] or 'Sommerfeld' in row['Comment']:
            interpol_style.append('linear')
        else:
            interpol_style.append('quadratic')

    xmin /= 2
    xmax *= 2
    x_grid = make_grid(xmin, xmax, npoints=2000, unit=style.energy_unit, log=True)
    n_plot =  len(data_raw_vec)

    data_to_plot = []
    ymin_data = style.ymin
    ymax_data = style.ymax
    plot_minvalues = np.zeros(n_plot)
    plot_maxvalues = np.zeros(n_plot)
    
    for index in range(n_plot):
        if not is_band[index]:
            data_gridded = data_on_grid(data_raw_vec[index][0], data_raw_vec[index][1], x_grid, interpolation_kind=interpol_style[index], fill_value = blindval)
            raw_vals = data_raw_vec[index][1]
        else:
            data_gridded1 = data_on_grid(data_raw_vec[index][0], data_raw_vec[index][1][0], x_grid, interpolation_kind=interpol_style[index], fill_value = blindval)
            data_gridded2 = data_on_grid(data_raw_vec[index][0], data_raw_vec[index][1][1], x_grid, interpolation_kind=interpol_style[index], fill_value = blindval)
            data_gridded = [data_gridded1,data_gridded2]
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
            plot_ranking_ids = [i for i in reversed(np.argsort(plot_minvalues))]
        else:
            plot_ranking_ids = [i for i in reversed(np.argsort(plot_maxvalues))]
    else:            
        plot_ranking_ids = np.argsort(plot_minvalues)
      
    
    envelope= []
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
            if np.isnan(val): val = blindval
            minvals.append(val)
        if style.mode == 'ann':
            envelope.append(0.95*min(minvals))
        else:
            envelope.append(1.05*max(minvals))

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
        except:
            color = order[j]
        if not is_band[j]:
            plt.plot(x_grid, data_to_plot[j], label=labels_plot[j], color=color)
        else:
            plt.fill_between(x_grid.value, data_to_plot[j][0].value, data_to_plot[j][1].value, label=labels_plot[j], color=color, alpha=0.6)
        if style.style == 'antique':
            if not is_band[j]:
                if style.mode == 'ann':
                    plt.fill_between(x_grid.value, data_to_plot[j].value, np.ones(len(x_grid)), alpha=0.1, color=color)
                else:
                    plt.fill_between(x_grid.value, np.ones(len(x_grid)), data_to_plot[j].value, alpha=0.1, color=color)
            else:
                if style.mode == 'ann':
                    plt.fill_between(x_grid.value, data_to_plot[j][1].value, np.ones(len(x_grid)), alpha=0.1, color=color)
                else:
                    plt.fill_between(x_grid.value, np.ones(len(x_grid)), data_to_plot[j][0].value, alpha=0.1, color=color)
 
        if style.legend == 'fancy':
            if style.mode == 'ann':
                try:
                    i_legend = next(x[0] for x in enumerate(data_to_plot[j].value) if x[1] < 1e40)
                except:
                    i_legend = next(x[0] for x in enumerate(data_to_plot[j][0].value) if x[1] < 1e40)
                valign = 'top'
                vpad = 0.9*style.ymax
            else:
                i_legend = next(x[0] for x in enumerate(data_to_plot[j].value) if x[1] > 1e-40)
                valign = 'bottom'
                vpad = 1.2*style.ymin
                
            plt.text(x_grid[i_legend].value, vpad,  labels_plot_short[j], horizontalalignment='right', verticalalignment=valign, snap=True, color=color, rotation=90)

    if style.style == 'antique':
        plt.plot(x_grid, envelope, linewidth=5, color='k', alpha=0.2)

    if style.legend == 'side':
        plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", frameon=style.frameon)

    if style.mode == 'ann':
        wimp_model = ascii.read(module_dir + "/modelpredictions/wimp_huetten2017_analytical_omega012.ecsv")
        xtext = 0.9*min(wimp_model["mass"].to(style.energy_unit)[-1].value, plt.gca().get_xlim()[1])
        plt.text(xtext, wimp_model["sigmav"][-1],  "Thermal WIMP prediction", horizontalalignment='right', verticalalignment='bottom', snap=True, color='k', alpha=0.3)
        wimp_model_gridded = data_on_grid(wimp_model["mass"], wimp_model["sigmav"], x_grid, interpolation_kind='quadratic', fill_value = 1/blindval)
        #if style.style == 'antique':
        plt.fill_between(x_grid.value, np.zeros(2000), wimp_model_gridded.value, color='k', alpha=0.1)
        #else:
        #    plt.plot(x_grid.value, wimp_model_gridded.value, color='k', alpha=0.1, linewidth=5)


    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([ymin_data, ymax_data])
    plt.xlim([xmin.to(style.energy_unit).value, xmax.to(style.energy_unit).value]);

    plt.xlabel(r'$m_{\mathrm{DM}}$ $\mathrm{[' +style.energy_unit + ']}$');
    plt.ylabel(style.ylabel);
    
    return plot_limits, ax
    
def filter_dataframe(metadata_df, Mode, Instrument, Channel):
        
    mode_list = metadata_df.index[metadata_df['Mode'] == Mode[:3]].tolist()

    if Instrument == 'all':
        inst_list = metadata_df.index.tolist()
    else:
        inst_key = get_key_from_value(instrument_dict, Instrument)[0]
        if inst_key == 'multi-inst':
            inst_list = metadata_df.index[metadata_df['Instrument'].apply(type) == list].tolist()
        else:
            inst_list = metadata_df.index[metadata_df['Instrument'] == inst_key].tolist()

    if Channel == 'all':
        channel_list = metadata_df.index.tolist()
    else:
        channel_list = metadata_df.index[metadata_df['Channel'] == Channel].tolist()

    return intersection3(mode_list, inst_list, channel_list)

def init_metadata():

    files_all = []
    for name in instrument_dict.keys():
        files_all.append(glob.glob(module_dir + "/bounds/"+ name +"/*.ecsv"))
    files_all = [x for row in files_all for x in row]

    metadata_df = pd.DataFrame(columns=('Instrument', 'Target', 'Mode', 'Channel', 'Year', 'Observation time','Title', 'DOI', 'Arxiv', 'Comment', 'File name'))

    for i,file in enumerate(files_all):
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
        except:
            logging.warning("Instrument name not consistent in " + str(file))
        try:
            assert metadata['year'] == file_year
        except:
            logging.warning("Year not consistent in " + str(file))
        try:
            assert meta_target == file_target
        except:
            logging.warning("Target name not consistent in " + str(file))
        try:
            assert metadata['channel'] == file_channel
        except:
            logging.warning("Channel name not consistent in " + str(file))

        metadata_df.loc[i] = [instruments, target_info, file_mode, metadata['channel'], metadata['year'], 
                     metadata['obs_time'], metadata['reference'], metadata['doi'], metadata['arxiv'], metadata['comment'], file]
        
    return metadata_df

def metadata():
    return metadata_df

def labels4dropdown(metadata_df):
    labels = []
    for index, row in metadata_df.iterrows():    
        label_str = ''
        label_str += get_names_str(row['Instrument'], instrument_dict)
        label_str += ' (' + row['Year'] + '): ' + get_names_str(row['Target'], target_dict) + ', ' + row['Channel'] + ' (' + row['Mode'] + '.)'# + str(index)
        if row['Comment'] != '':
            label_str += ', ' + row['Comment']
        labels.append(label_str)
    return labels

def interactive_selection():

    inst_list = list(instrument_dict.values())
    inst_list.insert(0, 'all')

    channel_list = list(channel_dict.keys())
    channel_list.insert(0,'all')

    metadata_df = metadata()
    labels = labels4dropdown(metadata_df)


    def multi_checkbox_widget(options_dict):
        """ Widget with a search field and lots of checkboxes """

        style_widget = wid.Dropdown(options = ['antique', 'standard', 'fancy'], description='Plotting style')

        mode_widget = wid.Dropdown(options = ['annihilation', 'decay'], description='Mode')
        instrument_widget = wid.Dropdown(options = inst_list, description='Instrument')
        channel_widget = wid.Dropdown(options = channel_list, description='Channel')

        output_widget = wid.Output()
        options = [x for x in options_dict.values()]

        start_index_list = filter_dataframe(metadata_df, 'annihilation', 'all', 'all')
        start_options = []
        for index in start_index_list: start_options.append(options[index])
        start_options = sorted([x for x in start_options], key = lambda x: x.description, reverse = False)
        style.__init__('antique', mode='ann')

        options_layout = wid.Layout(
            overflow='auto',
            border='1px solid black',
            width='950px',
            height='200px',
            flex_flow='row wrap',
            display='flex',
        )

        #selected_widget = wid.Box(children=[options[0]])
        options_widget = wid.VBox(options, layout=options_layout)
        options_widget.children = start_options    
        #print(options_widget.children)
        #left_widget = wid.VBox(search_widget, selected_widget)
        multi_select = wid.VBox([style_widget,mode_widget, instrument_widget, channel_widget, options_widget])

        @output_widget.capture()
        def on_checkbox_change(change):

            selected_recipe = change["owner"].description
            #print(options_widget.children)
            #selected_item = wid.Button(description = change["new"])
            #selected_widget.children = [] #selected_widget.children + [selected_item]
            options_widget.children = sorted([x for x in options_widget.children], key = lambda x: x.description, reverse = False)
            options_widget.children = sorted([x for x in options_widget.children], key = lambda x: x.value, reverse = True)
            #options_widget.children = [x for x in options_widget.children]

        for checkbox in options:
            checkbox.observe(on_checkbox_change, names="value")

        # Wire the search field to the checkboxes
        @output_widget.capture()
        def dropdown_change(*args):

            index_list = filter_dataframe(metadata_df, 
                                          mode_widget.value, 
                                          instrument_widget.value, channel_widget.value)
            new_options = []
            for index in index_list: new_options.append(options[index])
            new_options = sorted([x for x in new_options], key = lambda x: x.description, reverse = False)
            options_widget.children = sorted([x for x in new_options], key = lambda x: x.value, reverse = True)

            clear_output()
            display(Markdown('Filtered %d data sets.'%len(index_list)))
            if args[0].owner.description == 'Mode':
                clear_output()
                display(Markdown('Attention when switching between annihilation and decay'))
            if style_widget.value == 'fancy':
                style.__init__('antique', legend='fancy', mode=mode_widget.value)
            else:
                style.__init__(style_widget.value, mode=mode_widget.value)

        mode_widget.observe(dropdown_change)
        instrument_widget.observe(dropdown_change)
        channel_widget.observe(dropdown_change)
        style_widget.observe(dropdown_change)

        display(output_widget)
        return multi_select

    style = PlottingStyle('antique')
    
    a,b = checkIfDuplicates_2(labels)
    if a:
        logging.error("Duplicate label entry found for %s. Please make it unique in the file header."%b)

    options_dict = {
        x: wid.Checkbox(
            description=x, 
            value=False,
            style={"description_width":"0px"},
            layout=wid.Layout(width='100%', flex_flow='wrap')
        ) for x in labels
    }

    def f(**args): 

        i_vec = [index for index, (key, value) in enumerate(args.items()) if value]
        if len(i_vec) > 0:
            metadata_filtered_df = metadata_df.loc[i_vec]
            #display(Markdown('Selected %d data sets.'%len(i_vec)))
            figure, ax = plot(metadata_filtered_df, style)
            #display(metadata_df.loc[i_vec])
            return figure
        else:
            display(Markdown('Nothing selected'))

        #display(results)

    ui = multi_checkbox_widget(options_dict)
    out = wid.interactive_output(f, options_dict)
    display(wid.VBox([ui, out]))
    
    return options_dict, metadata_df

def filter_metadata(options_dict, metadata_df):
    i_vec = [index for index, (key, value) in enumerate(options_dict.items()) if value.value]
    return metadata_df.loc[i_vec]

def get_data(df):
    data = []
    for index, row in df.iterrows():
        data.append(ascii.read(row['File name']))
    return data

def show_metadata(metadata):
    metadata_disp = metadata.copy()  
    metadata_disp['DOI'] = metadata_disp['DOI'].apply(lambda x: f'<a href="https://doi.org/{x}" target=_blank>{x}</a>')
    metadata_disp['Arxiv'] = metadata_disp['Arxiv'].apply(lambda x: f'<a href="https://arxiv.org/abs/{x}" target=_blank>{x}</a>')
    return HTML(metadata_disp.to_html(render_links=True, escape=False))

def checkIfDuplicates_2(listOfElems):
    ''' Check if given list contains any duplicates '''    
    setOfElems = set()
    for elem in listOfElems:
        if elem in setOfElems:
            return True, elem
        else:
            setOfElems.add(elem)         
    return False, None

instrument_dict = table_to_dict(ascii.read(module_dir + "/legends/legend_instruments.ecsv"),'shortname', 'longname')
channel_dict = table_to_dict(ascii.read(module_dir + "/legends/legend_channels.ecsv"),'shortname', 'latex')
target_dict = table_to_dict(ascii.read(module_dir + "/legends/legend_targets.ecsv"),'shortname', 'longname')
metadata_df = init_metadata()

