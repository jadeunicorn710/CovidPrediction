import os
import numpy as np
import pandas as pd
from bokeh.events import MenuItemClick
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Legend, BasicTicker, ColorBar, LogColorMapper, LinearColorMapper, PrintfTickFormatter, Div
from bokeh.models.widgets import Select, DataTable, TableColumn
from bokeh.plotting import figure, gridplot, curdoc
from bokeh.transform import transform

global TOOLS
TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select,hover"

colors = ['#b3a369', '#eaaa00', '#f5d580', '#003057', '#335161', '#004f9f', '#1879db', '#545454', '#E5E5E5', '#8E8B76']

path = os.getcwd()
global snn_icu_df
global snn_hosp_df
global lr_icu_df
global lr_hosp_df
global icu_plot
global hosp_plot
global icu_cds
global hosp_cds
global draw_comp_model_graphs
global cur_plot
cur_plot = None

snn_icu_df = pd.read_csv(path + '/visualization/data/icu_neural_network_8_train_history.csv')
snn_hosp_df = pd.read_csv(path + '/visualization/data/hospitalization_neural_network_8_train_history.csv')
lr_icu_df = pd.read_csv(path + '/visualization/data/icu_logistic_regression_8_train_history.csv')
lr_hosp_df = pd.read_csv(path + '/visualization/data/hospitalization_logistic_regression_8_train_history.csv').head(100)

div_header = Div(text="\n\n<h1>Model Analysis</h1>\n\n", width=1500, height=50)
# display text div
curdoc().add_root(row(div_header, width=1500))

div_subheader = Div(text="<h3>Compare our Standard Neural Network and Logistic Regression models uding different comparative metrics for ICU Admission and Hospitalization.</h3>", width=1500, height=50)
# display text div
curdoc().add_root(row(div_subheader, width=1500))

df_columns = ['loss', 'auc', 'precision', 'recall']
metric_selector = Select(title="Select a Metric to Compare:", value=df_columns[0], options=df_columns)


#SNN vs LR ICU
snn_icu_columned = snn_icu_df.rename(columns={df_columns[0]:'ycol_snn'})
lr_icu_columned = lr_icu_df.rename(columns={df_columns[0]:'ycol_lr'})
icu_important_columns = pd.concat([snn_icu_columned['epoch'], snn_icu_columned['ycol_snn'], lr_icu_columned['ycol_lr']], axis=1)
icu_cds = ColumnDataSource(icu_important_columns)
icu_plot = figure(tools=TOOLS, width = 600, height = 300, title="SNN vs. LR ICU Results by Metric: " + df_columns[0])
si = icu_plot.line(x='epoch',y='ycol_snn',source=icu_cds, line_color=colors[1])
li = icu_plot.line(x='epoch',y='ycol_lr',source=icu_cds, line_color=colors[6])
icu_legend = Legend(items=[("SNN",[si]),("LR",[li])])
icu_plot.add_layout(icu_legend, 'right')
icu_plot.xaxis.axis_label = 'epoch'
icu_plot.yaxis.axis_label = df_columns[0]
#SNN vs LR ICU

#SNN vs LR HOSP
snn_hosp_columned = snn_hosp_df.rename(columns={df_columns[0]:'ycol_snn'})
lr_hosp_columned = lr_hosp_df.rename(columns={df_columns[0]:'ycol_lr'})
hosp_important_columns = pd.concat([snn_hosp_columned['epoch'], snn_hosp_columned['ycol_snn'], lr_hosp_columned['ycol_lr']], axis=1)
hosp_cds = ColumnDataSource(hosp_important_columns)
hosp_plot = figure(tools=TOOLS, width = 600, height = 300, title="SNN vs. LR Hospitalization Results by Metric: " + df_columns[0])
sh = hosp_plot.line(x='epoch',y='ycol_snn',source=hosp_cds, line_color=colors[1])
lh = hosp_plot.line(x='epoch',y='ycol_lr',source=hosp_cds, line_color=colors[6])
hosp_legend = Legend(items=[("SNN",[sh]),("LR",[lh])])
hosp_plot.add_layout(hosp_legend, 'right')
hosp_plot.xaxis.axis_label = 'epoch'
hosp_plot.yaxis.axis_label = df_columns[0]
#SNN vs LR HOSP

#SNN vs LR ICU ROC
sitest = pd.read_csv(path + '/visualization/data/icu_neural_network_8_test_roc.csv')
litest = pd.read_csv(path + '/visualization/data/icu_logistic_regression_8_test_roc.csv')
icu_roc = figure(tools=TOOLS, width = 600, height = 300, title="SNN vs. LR ICU ROC")
sir = icu_roc.line(x=sitest['fp'],y=sitest['tp'], line_color=colors[1])
lir = icu_roc.line(x=litest['fp'],y=litest['tp'], line_color=colors[6])
icu_roc_legend = Legend(items=[("SNN",[sir]),("LR", [lir])])
icu_roc.add_layout(icu_roc_legend, 'right')
icu_roc.xaxis.axis_label = 'False Positive'
icu_roc.yaxis.axis_label = 'True Positive'
#SNN vs LR ICU ROC

#SNN vs LR HOSP ROC
shtest = pd.read_csv(path + '/visualization/data/hospitalization_neural_network_8_test_roc.csv')
lhtest = pd.read_csv(path + '/visualization/data/hospitalization_logistic_regression_8_test_roc.csv')
hosp_roc = figure(tools=TOOLS, width = 600, height = 300, title="SNN vs. LR Hospitalization ROC")
shr = hosp_roc.line(x=shtest['fp'],y=shtest['tp'], line_color=colors[1])
lhr = hosp_roc.line(x=lhtest['fp'],y=lhtest['tp'], line_color=colors[6])
hosp_roc_legend = Legend(items=[("SNN",[shr]),("LR", [lhr])])
hosp_roc.add_layout(hosp_roc_legend, 'right')
hosp_roc.xaxis.axis_label = 'False Positive'
hosp_roc.yaxis.axis_label = 'True Positive'
#SNN vs LR HOSP ROC

div0 = Div(text="<h2>Confusion Matrix of SNN and LR for ICU Admission</h2>", width=1500, height=50)

#SNN ICU CM
snn_icu_results = pd.read_csv(path + '/visualization/data/icu_neural_network_8_test_results.csv', usecols=['tp', 'fp', 'tn', 'fn'])
sicmlist1 = [snn_icu_results['tp'][0], snn_icu_results['fp'][0]]
sicmlist2 = [snn_icu_results['fn'][0], snn_icu_results['tn'][0]]
sicm_df = pd.DataFrame(
  {
    'ICU': sicmlist1,
    'No ICU': sicmlist2
  }, index=['ICU', 'No ICU']
)
sicm_columns = [
    TableColumn(field='index', title='Prediction'),
    TableColumn(field='ICU', title='ICU'),
    TableColumn(field='No ICU', title='No ICU')
]
sicm = DataTable(source=ColumnDataSource(sicm_df), columns=sicm_columns, width=300, height=120)
#SNN ICU CM

#LR ICU CM
lr_icu_results = pd.read_csv(path + '/visualization/data/icu_logistic_regression_8_test_results.csv', usecols=['tp', 'fp', 'tn', 'fn'])
licmlist1 = [lr_icu_results['tp'][0], lr_icu_results['fp'][0]]
licmlist2 = [lr_icu_results['fn'][0], lr_icu_results['tn'][0]]
licm_df = pd.DataFrame(
  {
    'ICU': licmlist1,
    'No ICU': licmlist2
  }, index=['ICU', 'No ICU']
)
licm_columns = [
    TableColumn(field='index', title='Prediction'),
    TableColumn(field='ICU', title='ICU'),
    TableColumn(field='No ICU', title='No ICU')
]
licm = DataTable(source=ColumnDataSource(licm_df), columns=licm_columns, width=300, height=120)
#LR ICU CM

div1 = Div(text="<h2>Confusion Matrix of SNN and LR for Hosptalization</h2>", width=1500, height=50)

#SNN HOSP CM
snn_hosp_results = pd.read_csv(path + '/visualization/data/hospitalization_neural_network_8_test_results.csv', usecols=['tp', 'fp', 'tn', 'fn'])
shcmlist1 = [snn_hosp_results['tp'][0], snn_hosp_results['fp'][0]]
shcmlist2 = [snn_hosp_results['fn'][0], snn_hosp_results['tn'][0]]
shcm_df = pd.DataFrame(
  {
    'Hospitalization': shcmlist1,
    'No Hospitalization': shcmlist2
  }, index=['Hospitlization', 'No Hospitalization']
)
shcm_columns = [
    TableColumn(field='index', title='Prediction'),
    TableColumn(field='Hospitalization', title='Hospitalization'),
    TableColumn(field='No Hospitalization', title='No Hospitalization')
]
shcm = DataTable(source=ColumnDataSource(shcm_df), columns=shcm_columns, width=300, height=120)
#SNN HOSP CM

#LR HOSP CM
lr_hosp_results = pd.read_csv(path + '/visualization/data/hospitalization_neural_network_8_test_results.csv', usecols=['tp', 'fp', 'tn', 'fn'])
lhcmlist1 = [lr_hosp_results['tp'][0], lr_hosp_results['fp'][0]]
lhcmlist2 = [lr_hosp_results['fn'][0], lr_hosp_results['tn'][0]]
lhcm_df = pd.DataFrame(
  {
    'Hospitalization': lhcmlist1,
    'No Hospitalization': lhcmlist2
  }, index=['Hospitlization', 'No Hospitalization']
)
lhcm_columns = [
    TableColumn(field='index', title='Prediction'),
    TableColumn(field='Hospitalization', title='Hospitalization'),
    TableColumn(field='No Hospitalization', title='No Hospitalization')
]
lhcm = DataTable(source=ColumnDataSource(lhcm_df), columns=lhcm_columns, width=300, height=120)
#LR HOSP CM

div2 = Div(text="<h2>Coefficient Breakdown of Logistic Regression for ICU Admission and Hospitalization</h2>", width=1500, height=50)

#LR ICU COEFF
lr_icu_coeffs = pd.read_csv(path + '/visualization/data/lr_icu_coefficients.csv')
lr_icu_coeffs_list = [lr_icu_coeffs['age'][0], lr_icu_coeffs['sex'][0], lr_icu_coeffs['pneumonia'][0], lr_icu_coeffs['diabetes'][0], lr_icu_coeffs['copd'][0], lr_icu_coeffs['asthma'][0],
    lr_icu_coeffs['inmsupr'][0], lr_icu_coeffs['other_disease'][0], lr_icu_coeffs['cardiovascular'][0], lr_icu_coeffs['obesity'][0], lr_icu_coeffs['renal_chronic'][0], lr_icu_coeffs['tobacco'][0], lr_icu_coeffs['bias'][0]]
lr_icu_coeffs_df = pd.DataFrame(
  {
    'Coefficient': lr_icu_coeffs_list
  }, index =['Age', 'Sex', 'Pneumonia', 'Diabetes', 'COPD', 'Asthma', 'Immunosuppression', 'Other Disease', 'Cardiovascular', 'Obesity', 'Renal Chronic', 'Tobacco', 'Bias']
)
licoeff_columns = [
    TableColumn(field='index', title='Predictor'),
    TableColumn(field='Coefficient', title='Coefficient')
]
licoeff = DataTable(source=ColumnDataSource(lr_icu_coeffs_df), columns=licoeff_columns, height=300)
#LR ICU COEFF

#LR HOSP COEFF
lr_hosp_coeffs = pd.read_csv(path + '/visualization/data/lr_hospital_coefficients.csv')
lr_hosp_coeffs_list = [lr_hosp_coeffs['age'][0], lr_hosp_coeffs['sex'][0], lr_hosp_coeffs['pneumonia'][0], lr_hosp_coeffs['diabetes'][0], lr_hosp_coeffs['copd'][0], lr_hosp_coeffs['asthma'][0],
    lr_hosp_coeffs['inmsupr'][0], lr_hosp_coeffs['other_disease'][0], lr_hosp_coeffs['cardiovascular'][0], lr_hosp_coeffs['obesity'][0], lr_hosp_coeffs['renal_chronic'][0], lr_hosp_coeffs['tobacco'][0], lr_hosp_coeffs['bias'][0]]
lr_hosp_coeffs_df = pd.DataFrame(
  {
    'Coefficient': lr_hosp_coeffs_list
  }, index =['Age', 'Sex', 'Pneumonia', 'Diabetes', 'COPD', 'Asthma', 'Immunosuppression', 'Other Disease', 'Cardiovascular', 'Obesity', 'Renal Chronic', 'Tobacco', 'Bias']
)
lhcoeff_columns = [
    TableColumn(field='index', title='Predictor'),
    TableColumn(field='Coefficient', title='Coefficient')
]
lhcoeff = DataTable(source=ColumnDataSource(lr_hosp_coeffs_df), columns=lhcoeff_columns, height=300)
#LR HOSP COEFF

def draw_comp_model_graphs(column_name):
  import pandas as pd

  icu_plot.title.text = "SNN vs. LR ICU Results by Metric: " + column_name
  snn_icu_columned = snn_icu_df.rename(columns={column_name:'ycol_snn'})
  lr_icu_columned = lr_icu_df.rename(columns={column_name:'ycol_lr'})
  icu_important_columns = pd.concat([snn_icu_columned['epoch'], snn_icu_columned['ycol_snn'], lr_icu_columned['ycol_lr']], axis=1)
  icu_plot.yaxis.axis_label = column_name
  icu_cds.data = icu_important_columns

  hosp_plot.title.text = "SNN vs LR Hospitalization Results by Metric: " + column_name
  snn_hosp_columned = snn_hosp_df.rename(columns={column_name:'ycol_snn'})
  lr_hosp_columned = lr_hosp_df.rename(columns={column_name:'ycol_lr'})
  hosp_important_columns = pd.concat([snn_hosp_columned['epoch'], snn_hosp_columned['ycol_snn'], lr_hosp_columned['ycol_lr']], axis=1)
  hosp_plot.yaxis.axis_label = column_name
  hosp_cds.data = hosp_important_columns


p = gridplot([[icu_plot, hosp_plot], [icu_roc, hosp_roc], [row(div0, width=600)], [sicm, licm], [row(div1, width=600)], [shcm, lhcm], [row(div2, width=600)], [licoeff, lhcoeff]])
curdoc().add_root(row(metric_selector, width=300))
curdoc().add_root(p)

def dataShowCallback(attr, old, new):
  draw_comp_model_graphs(new)

metric_selector.on_change("value", dataShowCallback)

# display graphs
draw_comp_model_graphs(df_columns[0])

