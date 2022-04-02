import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.events import MenuItemClick
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColorBar, ColumnDataSource, Div, LinearColorMapper
from bokeh.models.widgets import Select
from bokeh.plotting import figure
from bokeh.transform import transform

class DisplayData:
  def __init__(self, cur_page):
    self.cur_page = cur_page
    self.path = os.getcwd()

    # initialize colors
    self.colors = ['#B3A369', '#EAAA00', '#F5D580', '#003057', '#335161', '#004F9F', '#1879DB', '#545454', '#E5E5E5', '#8E8B76']

    # read data
    self.raw_icu_df = pd.read_csv(self.path + '/visualization/data/cleaned_with_icu_preprocessed_no_noise_corrected.csv')
    self.raw_hosp_df = pd.read_csv(self.path + '/visualization/data/cleaned_with_hosp_modified.csv')
    self.columns_info = pd.read_csv(self.path + '/visualization/data_details_for_plotting.csv')

    # initialize variables
    self.x_names = []
    self.y_values = []
    self.plot_data = {}
    self.TOOLTIPS = [
        ("label", "@category"),
        ("value", "@counts")
    ]

    self.cur_icu_plot = None
    self.cur_hosp_plot = None
    self.cur_icu_source = None
    self.cur_hosp_source = None
  
  # define callback methods
  def draw_graph(self, column_name, is_icu):
    if is_icu:    
      freq = self.raw_icu_df[column_name].value_counts()
      df = freq.rename_axis('unique_values').to_frame('counts')
      info = self.columns_info.loc[self.columns_info['Category'] == column_name]

      if info.empty:
        sorted_df = df.sort_values('unique_values')
        x_names = list(map(str, sorted_df.index.values))
        plot_data = dict(category=x_names, counts=list(sorted_df['counts'].values))
        self.cur_icu_plot.x_range.factors = x_names
        if column_name == 'age':
          self.cur_icu_plot.xaxis.ticker = [0,10,20,30,40,50,60,70,80,90,100]
          self.cur_icu_plot.xaxis.axis_label = "Age"
      else:
        x_names = [info.iloc[0]['0'], info.iloc[0]['1']]
        y_values = list(df['counts'].values)
        plot_data = dict(category=x_names, counts=y_values)
        self.cur_icu_plot.x_range.factors = x_names

      self.cur_icu_source.data = plot_data
      self.cur_icu_plot.title.text = "Raw data distribution for col: " + column_name
      self.cur_icu_plot.sizing_mode = "stretch_both"
    else:
      freq = self.raw_hosp_df[column_name].value_counts()
      df = freq.rename_axis('unique_values').to_frame('counts')
      info = self.columns_info.loc[self.columns_info['Category'] == column_name]
      if info.empty:
        sorted_df = df.sort_values('unique_values')
        x_names = list(map(str, sorted_df.index.values))
        plot_data = dict(category=x_names, counts=list(sorted_df['counts'].values))
        self.cur_hosp_plot.x_range.factors = x_names
        if column_name == 'age':
          self.cur_hosp_plot.xaxis.ticker = [0,10,20,30,40,50,60,70,80,90,100]
          self.cur_hosp_plot.xaxis.axis_label = "Age"
      else:  
        x_names = [info.iloc[0]['0'], info.iloc[0]['1']]
        y_values = list(df['counts'].values)
        plot_data = dict(category=x_names, counts=y_values)
        self.cur_hosp_plot.x_range.factors = x_names

      self.cur_hosp_source.data = plot_data
      self.cur_hosp_plot.title.text = "Raw data distribution for col: " + column_name
      self.cur_icu_plot.sizing_mode = "stretch_both"

  def dataShowICUCallback(self, attr, old, new):
    self.draw_graph(new, True)

  def dataShowHospCallback(self, attr, old, new):
    self.draw_graph(new, False)

  def display(self):
    # add text
    div = Div(text="\n\n<h2>Choose feature to see its distribution in the available data</h2>\n\n", width=1500, height=50)

    self.cur_icu_source = ColumnDataSource(data=self.plot_data)
    self.cur_icu_plot = figure(x_range=self.x_names, tools=[], sizing_mode="scale_width", height=350, tooltips=self.TOOLTIPS, margin=(0,0,0,20))
    self.cur_icu_plot.vbar(x='category', top='counts', source=self.cur_icu_source, width=0.9, bottom=0, fill_color=self.colors[0])
    self.cur_icu_plot.yaxis.formatter.use_scientific = False
    self.cur_icu_plot.yaxis.axis_label = 'Number of patients'

    self.cur_hosp_source = ColumnDataSource(data=self.plot_data)
    self.cur_hosp_plot = figure(x_range=self.x_names, tools=[], sizing_mode="scale_width", height=350, tooltips=self.TOOLTIPS, margin=(0,0,0,20))
    self.cur_hosp_plot.vbar(x='category', top='counts', source=self.cur_hosp_source, width=0.9, bottom=0, fill_color=self.colors[0])
    self.cur_hosp_plot.yaxis.formatter.use_scientific = False
    self.cur_hosp_plot.yaxis.axis_label = 'Number of patients'

    # prepare dropdown for ICU
    icu_columns = self.raw_icu_df.columns.tolist()
    selectICUFeature = Select(title="For ICU predictions:", value=icu_columns[0], options=icu_columns)

    # prepare dropdown for Hospitalization
    hosp_columns = self.raw_hosp_df.columns.tolist()
    selectHospFeature = Select(title="For Hospitalization predictions:", value=hosp_columns[0], options=hosp_columns)

    self.draw_graph(icu_columns[0], True)
    self.draw_graph(hosp_columns[0], False)

    # set callback on dropdown for ICU
    selectICUFeature.on_change("value", self.dataShowICUCallback)

    # set callback on dropdown for Hospitalization
    selectHospFeature.on_change("value", self.dataShowHospCallback)

    # display text div
    self.cur_page.add_root(row(div, width=1500))

    # display drop downs and first graphs for ICU and Hospitalization
    self.cur_page.add_root(row(column(selectICUFeature, self.cur_icu_plot, margin=(10,50,50,0)), column(selectHospFeature, self.cur_hosp_plot, margin=(10,50,50,0)), width=1500))

    # add text
    div2 = Div(text="\n\n<h2>Feature correlation graphs for ICU and Hospitalization</h2>\n\n", width=1500, height=50)

    # display text div
    self.cur_page.add_root(row(div2, width=1500))

    # draw correlation matrix for ICU
    icu_df_corr = self.raw_icu_df
    corr_matrix_icu = icu_df_corr.drop(columns = ['icu']).corr()

    corr_matrix_icu.index.name = 'IcuFeatures1'
    corr_matrix_icu.columns.name = 'IcuFeatures2'

    corr_matrix_icu = corr_matrix_icu.stack().rename("value").reset_index()

    mapper = LinearColorMapper(palette=self.colors, low=corr_matrix_icu.value.min(), high=corr_matrix_icu.value.max())

    corr_matrix_icu_tooltip = [
        ("x", "@IcuFeatures1"),
        ("y", "@IcuFeatures2"),
        ("value", "@value{1.111}")
    ]

    # create heatmap figure for ICU
    corr_heatmap_icu = figure(tools=[], width=700, height=500, title="ICU features", x_range=list(corr_matrix_icu.IcuFeatures1.drop_duplicates()),
        y_range=list(corr_matrix_icu.IcuFeatures2.drop_duplicates()), tooltips = corr_matrix_icu_tooltip)

    corr_heatmap_icu.rect(x="IcuFeatures1", y="IcuFeatures2", source=ColumnDataSource(corr_matrix_icu), width=1, height=1, line_color=None,
        fill_color=transform('value', mapper))

    color_bar = ColorBar(color_mapper=mapper, width=8,  location=(0,0))

    corr_heatmap_icu.add_layout(color_bar, 'right')
    corr_heatmap_icu.xaxis.major_label_orientation = 1.2

    # draw correlation matrix for Hospitalization
    hosp_df_corr = self.raw_hosp_df
    corr_matrix_hosp = hosp_df_corr.drop(columns = ['patient_type']).corr()

    corr_matrix_hosp.index.name = 'HospFeatures1'
    corr_matrix_hosp.columns.name = 'HospFeatures2'

    corr_matrix_hosp = corr_matrix_hosp.stack().rename("value").reset_index()

    mapper = LinearColorMapper(palette=self.colors, low=corr_matrix_hosp.value.min(), high=corr_matrix_hosp.value.max())

    corr_matrix_hosp_tooltip = [
        ("x", "@HospFeatures1"),
        ("y", "@HospFeatures2"),
        ("value", "@value{1.111}")
    ]

    # create heatmap figure for Hospitalization
    corr_heatmap_hosp = figure(tools=[], width=700, height=500, title="Hospitalization features", x_range=list(corr_matrix_hosp.HospFeatures1.drop_duplicates()),
        y_range=list(corr_matrix_hosp.HospFeatures2.drop_duplicates()), tooltips = corr_matrix_hosp_tooltip)

    corr_heatmap_hosp.rect(x="HospFeatures1", y="HospFeatures2", source=ColumnDataSource(corr_matrix_hosp), width=1, height=1, line_color=None,
        fill_color=transform('value', mapper))

    color_bar = ColorBar(color_mapper=mapper, width=8,  location=(0,0))

    corr_heatmap_hosp.add_layout(color_bar, 'right')
    corr_heatmap_hosp.xaxis.major_label_orientation = 1.2

    self.cur_page.add_root(row(corr_heatmap_icu, corr_heatmap_hosp, width=1500))
