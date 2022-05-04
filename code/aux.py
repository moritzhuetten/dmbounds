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
from IPython.display import Markdown, clear_output

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
def intersection2(lst1, lst2):
    return list(set(lst1) & set(lst2))    

def intersection3(lst1, lst2, lst3):
    return list(set(lst1) & set(lst2) & set(lst3))  
    
class PlottingStyle:
    def __init__(self, stylename, figshape='widerect', legend='side', energy_unit = 'TeV', mode = 'ann'):
        
        if stylename == 'antique':
            from palettable.cartocolors.qualitative import Antique_10 as colormap
            mpl.rcParams['image.cmap'] = mpl.colors.Colormap(colormap)
            self.colors = colormap.mpl_colors
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=self.colors)
            mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
            
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

        elif stylename == 'standard':
            from palettable.cartocolors.qualitative import Antique_10 as colormap
            mpl.rcParams['image.cmap'] = mpl.colors.Colormap(colormap)
            self.colors = colormap.mpl_colors
            mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=self.colors)
            mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
            
            self.frameon = False
            pgf_with_rc_fonts = {                      # setup matplotlib to use latex for output
            "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
            "text.usetex": True,                # use LaTeX to write all text
            "font.family": "sans-serif",
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
        else:
            print("unknow style name",stylename)    
            
        if figshape == 'widerect':
            self.ratio = 0.65
            self.image_resolution = 5000
            self.figwidth  = 30/ 2.54 
            self.figheight  = self.ratio * self.figwidth

        mpl.rcParams.update(pgf_with_rc_fonts)

        self.energy_unit = energy_unit
        self.legend = legend
        self.mode = mode[:3]
        
        if mode[:3] == 'ann':
            self.ymin = 1e-27
            self.ymax = 5e-20
            self.ylabel = r'$\langle\sigma v \rangle$ $[\mathrm{cm^3\,s^{-1}}]$'
        elif mode[:3] == 'dec':
            self.ymin = 1e20
            self.ymax = 5e27
            self.ylabel = r'$\tau$ $[\mathrm{s}]$'           
            
def plotting(df_full, i_vec, style, instrument_dict, target_dict, channel_dict):
    metadata_filtered_df = df_full.loc[i_vec]

    data_raw_vec = []
    labels_plot = []
    interpol_style = []
    labels_plot_short = []
    xmin = 1e6*u.TeV
    xmax = 1e-6*u.TeV
    
    if style.mode == 'ann':
        blindval = 1e40
    else:
        blindval = 1e-40

    for index, row in metadata_filtered_df.iterrows():
        data_raw = ascii.read(row['File name'])
        if row['Mode'] == 'ann':
            yaxis = 'sigmav'
        else:
            yaxis = 'tau'
        data_raw_vec.append([data_raw['mass'], data_raw[yaxis]])
        if min(data_raw['mass'].to('TeV')) < xmin: xmin = min(data_raw['mass'].to('TeV'))
        if max(data_raw['mass'].to('TeV')) > xmax: xmax = max(data_raw['mass'].to('TeV'))

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

        

        data_gridded = data_on_grid(data_raw_vec[index][0], data_raw_vec[index][1], x_grid, interpolation_kind=interpol_style[index], fill_value = blindval)
        data_to_plot.append(data_gridded)
        plot_minvalues[index] = min(data_raw_vec[index][1])
        plot_maxvalues[index] = max(data_raw_vec[index][1])
        
        if style.mode == 'ann':
            if min(data_gridded.value) < ymin_data:
                ymin_data = 0.5 * min(data_gridded.value)
        else:
            if max(data_gridded.value) > ymax_data:
                ymax_data = 2 * max(data_gridded.value)
                           
    
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
            val = data_to_plot[j][i].value
            if np.isnan(val): val = blindval
            minvals.append(val)
        if style.mode == 'ann':
            envelope.append(0.95*min(minvals))
        else:
            envelope.append(1.05*max(minvals))

    plot_limits = plt.figure(figsize=(style.figwidth, style.figheight))


    random_order = list(range(n_plot))
    random.shuffle(random_order)

    for j in plot_ranking_ids:


        plt.plot(x_grid, data_to_plot[j], label=labels_plot[j], color=style.colors[random_order[j]])
        if style.mode == 'ann':
            plt.fill_between(x_grid.value, data_to_plot[j].value, np.ones(len(x_grid)), alpha=0.1, color=style.colors[random_order[j]])
        else:
            plt.fill_between(x_grid.value, np.ones(len(x_grid)), data_to_plot[j].value, alpha=0.1, color=style.colors[random_order[j]])
            
        if style.legend == 'fancy':
            if style.mode == 'ann': 
                i_legend = next(x[0] for x in enumerate(data_to_plot[j].value) if x[1] < 1e40)
                valign = 'top'
                vpad = 0.9*style.ymax
            else:
                i_legend = next(x[0] for x in enumerate(data_to_plot[j].value) if x[1] > 1e-40)
                valign = 'bottom'
                vpad = 1.2*style.ymin
                
            plt.text(x_grid[i_legend].value, vpad,  labels_plot_short[j], horizontalalignment='right', verticalalignment=valign, snap=True, color=style.colors[random_order[j]], rotation=90)

    plt.plot(x_grid, envelope, linewidth=5, color='k', alpha=0.2)

    if style.legend == 'side':
        plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", frameon=style.frameon)

    if style.mode == 'ann':
        wimp_model = ascii.read("modelpredictions/wimp_steigman2012_numerical.ecsv")
        plt.text(wimp_model["mass"].to(style.energy_unit)[-1].value, wimp_model["sigmav"][-1],  "Steigman et al. (2012) thermal WIMP prediction", horizontalalignment='right', verticalalignment='bottom', snap=True, color='k', alpha=0.3)
        wimp_model_gridded = data_on_grid(wimp_model["mass"], wimp_model["sigmav"], x_grid, interpolation_kind='quadratic', fill_value = 1e-40)
        plt.fill_between(x_grid.value, np.zeros(2000), wimp_model_gridded.value, color='k', alpha=0.1)


    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([ymin_data, ymax_data])
    plt.xlim([xmin.to(style.energy_unit).value, xmax.to(style.energy_unit).value]);

    plt.xlabel(r'$m_{\mathrm{DM}}$ $\mathrm{[' +style.energy_unit + ']}$');
    plt.ylabel(style.ylabel);
    
    return plot_limits
    
def filter_dataframe(metadata_df, Mode, Instrument, Channel, instrument_dict):
        
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

def load_metadata_df(instrument_dict):

    files_all = []
    for name in instrument_dict.keys():
        files_all.append(glob.glob("bounds/"+ name +"/*.ecsv"))
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

def labels4dropdown(metadata_df, instrument_dict, target_dict):
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
    instrument_dict = table_to_dict(ascii.read("./legend_instruments.ecsv"),'shortname', 'longname')
    channel_dict = table_to_dict(ascii.read("./legend_channels.ecsv"),'shortname', 'latex')
    target_dict = table_to_dict(ascii.read("./legend_targets.ecsv"),'shortname', 'longname')

    inst_list = list(instrument_dict.values())
    inst_list.insert(0, 'all')

    channel_list = list(channel_dict.keys())
    channel_list.insert(0,'all')

    metadata_df = load_metadata_df(instrument_dict)
    labels = labels4dropdown(metadata_df, instrument_dict, target_dict)


    def multi_checkbox_widget(options_dict):
        """ Widget with a search field and lots of checkboxes """

        style_widget = wid.Dropdown(options = ['antique', 'standard', 'fancy'], description='Plotting style')

        mode_widget = wid.Dropdown(options = ['annihilation', 'decay'], description='Mode')
        instrument_widget = wid.Dropdown(options = inst_list, description='Instrument')
        channel_widget = wid.Dropdown(options = channel_list, description='Channel')

        output_widget = wid.Output()
        options = [x for x in options_dict.values()]

        start_index_list = filter_dataframe(metadata_df, 'annihilation', 'all', 'all', instrument_dict)
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
                                          instrument_widget.value, channel_widget.value, instrument_dict)
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
            #display(Markdown('Selected %d data sets.'%len(i_vec)))
            figure = plotting(metadata_df, i_vec, style, instrument_dict, target_dict, channel_dict)
            #display(metadata_df.loc[i_vec])
            return figure
        else:
            display(Markdown('Nothing selected'))

        #display(results)

    ui = multi_checkbox_widget(options_dict)
    out = wid.interactive_output(f, options_dict)
    display(wid.VBox([ui, out]))
    
    return options_dict, metadata_df

def selected_metadata(options_dict, metadata_df):
    i_vec = [index for index, (key, value) in enumerate(options_dict.items()) if value.value]
    return metadata_df.loc[i_vec]
