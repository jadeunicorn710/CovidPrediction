import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from bokeh.events import MenuItemClick
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Legend, BasicTicker, ColorBar, LinearColorMapper, LogColorMapper, PrintfTickFormatter, Div, Slider
from bokeh.models.widgets import Select, DataTable, DateFormatter, TableColumn, TextInput
from bokeh.plotting import figure, gridplot, curdoc
from bokeh.transform import transform


global pred_path
path = os.getcwd()
pred_path = path + '/visualization/data/'

global predictor, model, scaler_age, onehot, metrics, dpt_source, update_prediction, update_threshold, gender, yes_no, df
global dataShowCallback, textUpdateCallback, batch_size, X, x, y, evaluation, opt, loss, cm_source, bar_source, slider_input, threshold
global factor_chart, df_with_predictions, colors, predictor_columns, slider_result, predictions

datasets = ['Test Set 1', 'Test Set 2', 'Test Set 3', 'Test Set 4', 'Test Set 5']
metric_selector = Select(title="Please Select a Test Set to Use:", value=datasets[0], options=datasets)

predictor = ['age','sex','pneumonia','diabetes','copd','asthma','inmsupr','hypertension','other_disease','cardiovascular','obesity','renal_chronic','tobacco']
metrics = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

gender = {0:'Female',1:'Male'}
yes_no = {0:'N',1:'Y'}

div_header = Div(text="<h1>Hospitalization Predictor</h1>", width=1500, height=50)
# display text div
curdoc().add_root(row(div_header, width=1500))

div_subheader = Div(text="<h3>Use our SNN Model to predict Hospitalizations for a given dataset.</h3>", width=1500, height=50)
# display text div
curdoc().add_root(row(div_subheader, width=1500))

# Load SNN with ADAM Optimizer
fread = open(pred_path + 'snn_hospital.json')
model_json = fread.read()
fread.close()
model = keras.models.model_from_json(model_json)
model.load_weights(pred_path + 'snn_hospital.h5')
scaler_age = pickle.load(open(pred_path + 'snn_hospital_age_scaler.pkl', "rb"))
onehot = pickle.load(open(pred_path + 'snn_hospital_encoder.pkl', "rb"))

df = pd.read_csv(pred_path + "hospitalization_sample_1.csv")

x = df[predictor].values
y = df[['patient_type']].values

# time to make predictions!
x_age = scaler_age.transform(x[0:, 0:1])
x_remaining = onehot.transform(x[0:, 1:])

X = np.concatenate((x_age, x_remaining), axis=1)            
predictions = model.predict(x=X)
#define evaluation parameters
batch_size=4096
learning_rate = 1e-3
beta_1 = 0.9
beta_2=0.999
epsilon=1e-07
amsgrad=False
opt = keras.optimizers.Adam(lr=learning_rate, 
                                            beta_1=beta_1,
                                            beta_2=beta_2,
                                            epsilon=epsilon,
                                            amsgrad=amsgrad)
loss = keras.losses.BinaryCrossentropy(from_logits=False)

model.compile(optimizer=opt, #'adam',
            loss=loss,
            metrics=metrics)

evaluation = model.evaluate(x=X, y=y, batch_size=batch_size, verbose=0)

curdoc().add_root(row(metric_selector, width=300))

df_with_predictions = df.copy()
predictions_in_percentage = []
for prediction in predictions:
    val = '{:.4f}'.format(100 * prediction[0])
    predictions_in_percentage.append(val)
df_with_predictions['predict_percent'] = np.array(predictions_in_percentage)
df_with_predictions_mapped = df_with_predictions.copy().replace({'sex':gender,'pneumonia':yes_no,'diabetes':yes_no,'copd':yes_no,'asthma':yes_no,'inmsupr':yes_no,
    'hypertension':yes_no,'other_disease':yes_no,'cardiovascular':yes_no,'obesity':yes_no,'renal_chronic':yes_no,'tobacco':yes_no,'patient_type':yes_no})

dpt_source = ColumnDataSource(df_with_predictions_mapped)
dpt_columns = [
    TableColumn(field='sex', title='Sex'),
    TableColumn(field='pneumonia', title='Pneumonia'),
    TableColumn(field='age', title='Age'),
    TableColumn(field='diabetes', title='Diabetes'),
    TableColumn(field='copd', title='COPD'),
    TableColumn(field='asthma', title='Asthma'),
    TableColumn(field='inmsupr', title='Immunosuppression'),
    TableColumn(field='hypertension', title='Hypertension'),
    TableColumn(field='other_disease', title='Other Disease'),
    TableColumn(field='cardiovascular', title='Cardiovascular'),
    TableColumn(field='obesity', title='Obesity'),
    TableColumn(field='renal_chronic', title='Renal Chronic'),
    TableColumn(field='tobacco', title='Tobacco'),
    TableColumn(field='patient_type', title='Hospitalization'),
    TableColumn(field='predict_percent', title='Predicted')
]
data_predicted_table = DataTable(source=dpt_source, columns=dpt_columns, width = 900, height = 150)

results_header = Div(text="<h2>Predictive Results</h2>", width=900, height=30)
# display text div
curdoc().add_root(row(results_header))

curdoc().add_root(data_predicted_table)

results_subheader_text = "<p>This is data set along with the actual outcome (Hospitalization) and our predicted need for hospitalization (Predicted) as a percentage value.</p>"
results_subheader = Div(text=results_subheader_text, width=900, height=50)
# display text div
curdoc().add_root(row(results_subheader))

slider_header = Div(text="<h2>Interactive Classification</h2>", width=900, height=50)
# display text div
curdoc().add_root(row(slider_header))

# slider
slider_input = Slider(start=0.00, end=100.00, value=50.00, step=0.01, title='Use the threshold slider to forecast how many hospitalizations to expect. The current threshold is', width=900)
curdoc().add_root(slider_input)
# slider

# predictors + threshold
threshold = 50.0000
predictor_cols = list(df_with_predictions.columns[2:-1].delete(1))
predictor_counts = []
df_hosp = df_with_predictions.loc[pd.to_numeric(df_with_predictions['predict_percent']) > threshold]
for predictor_val in predictor_cols:
    predictor_counts.append(int(df_hosp[predictor_val].value_counts()[1]))
# predictors + threshold
slider_result = Div(text="<h3>With a threshold of {:.2f}, you can expect {} hospitalizations out of {} cases with an accuracy of {:.4f}%.</h3>"
    .format(threshold, df_hosp.shape[0], df.shape[0], 100*evaluation[5]), width=900)
# display text div


colors = ['#B3A369','#EAAA00','#F5D580','#003057','#335161','#004F9F','#1879DB','#545454','#808080','#E5E5E5','#8E8B76']
predictor_columns = ['Pneumonia','Diabetes','COPD','Asthma','Immunosuppression','Hypertension','Other Disease','Cardiovascular','Obesity','Renal Chronic','Tobacco']

bar_source = ColumnDataSource(data=dict(predictors=predictor_columns, counts=predictor_counts, color=colors))
factor_chart_tooltip = [
    ("Predictor: ", "@predictors"),
    ("Number: ", "@counts{1}")
]
factor_chart = figure(x_range=predictor_columns, y_range=(0,max(predictor_counts)), title='Predictor Counts: Symptoms Your Hospitalized Patients Have', width=900, height=600, tooltips=factor_chart_tooltip)
factor_chart.vbar(x='predictors',top='counts',width=0.8,legend_field='predictors',source=bar_source, color='color')
factor_chart.xaxis.major_label_orientation = 1
factor_chart.yaxis.axis_label = 'Number of Occurrences'
# curdoc().add_root(factor_chart)


# CM
model_results = dict(zip(model.metrics_names, evaluation))

cmlist1 = [model_results['tp'], model_results['fn']]
cmlist2 = [model_results['fp'], model_results['tn']]
cm_df = pd.DataFrame(
  {
    'Hospitalization': cmlist1,
    'No Hospitalization': cmlist2
  }, index=['Hospitalization', 'No Hospitalization']
)
cm_columns = [
    TableColumn(field='index', title='Prediction'),
    TableColumn(field='Hospitalization', title='Hospitalization'),
    TableColumn(field='No Hospitalization', title='No Hospitalization')
]
cm_source = ColumnDataSource(cm_df)
cm = DataTable(source=cm_source, columns=cm_columns, width=400, height=90)
# CM



div3_text = "<p>The Confusion Matrix shows the distribution of True and False Positives and Negatives. Keep in mind a lower threshold will result in more false positives, which in turn will result in higher resource allocation.</p>"
div3 = Div(text=div3_text, width=500, height=50)
# display text div
curdoc().add_root(row(slider_result))

p = gridplot([[row(div3),cm]])
curdoc().add_root(p)

div4_text = "<h3>The Factor Distribution Chart shows which factors are most commonly found in hospitalized patients.</h3>"
div4 = Div(text=div4_text, width=900)
curdoc().add_root(div4)

curdoc().add_root(factor_chart)
# curdoc().add_root(cm)



def update_prediction(table_name):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix
    
    if table_name == 'Test Set 1':
        filepath = pred_path + 'hospitalization_sample_1.csv'
    elif table_name == 'Test Set 2':
        filepath = pred_path + 'hospitalization_sample_2.csv'
    elif table_name == 'Test Set 3':
        filepath = pred_path + 'hospitalization_sample_3.csv'
    elif table_name == 'Test Set 4':
        filepath = pred_path + 'hospitalization_sample_4.csv'
    elif table_name == 'Test Set 5':
        filepath = pred_path + 'hospitalization_sample_5.csv'
    df = pd.read_csv(filepath)

    x = df[predictor].values
    y = df[['patient_type']].values

    # time to make predictions!
    x_age = scaler_age.transform(x[0:, 0:1])
    x_remaining = onehot.transform(x[0:, 1:])
    # re-generate predictions
    X = np.concatenate((x_age, x_remaining), axis=1)  
    # update data table          
    predictions = model.predict(x=X)
    predictions_in_percentage = []
    for prediction in predictions:
        val = '{:.4f}'.format(100 * prediction[0])
        predictions_in_percentage.append(val)
    df_with_predictions = df.copy()
    df_with_predictions['predict_percent'] = np.array(predictions_in_percentage)
    df_with_predictions_mapped = df_with_predictions.copy().replace({'sex':gender,'pneumonia':yes_no,'diabetes':yes_no,'copd':yes_no,'asthma':yes_no,'inmsupr':yes_no,
    'hypertension':yes_no,'other_disease':yes_no,'cardiovascular':yes_no,'obesity':yes_no,'renal_chronic':yes_no,'tobacco':yes_no,'patient_type':yes_no})
    dpt_source.data = df_with_predictions_mapped
    # update data table

    # update confusion matrix
    threshold = 50.0000
    conf_mat = confusion_matrix(y, predictions > (threshold/100))
    cmlist1 = [conf_mat[1][1], conf_mat[1][0]]
    cmlist2 = [conf_mat[0][1], conf_mat[0][0]]
    cm_df = pd.DataFrame(
    {
        'Hospitalization': cmlist1,
        'No Hospitalization': cmlist2
    }, index=['Hospitalization', 'No Hospitalization']
    )
    cm_source.data = cm_df
    # update confusion matrix
    
    # regenerate slider and barplot stuff at threshold
    predictor_cols = list(df_with_predictions.columns[2:-1].delete(1))
    predictor_counts = []
    df_hosp = df_with_predictions.loc[pd.to_numeric(df_with_predictions['predict_percent']) > threshold]
    for predictor_val in predictor_cols:
        predictor_counts.append(int(df_hosp[predictor_val].value_counts()[1]))
    bar_source.data = dict(predictors=predictor_columns, counts=predictor_counts, color=colors)
    slider_result.text = "<h3>With a threshold of {:.2f}, you can expect {} hospitalizations out of {} cases with an accuracy of {:.4f}%.</h3>".format(threshold, df_hosp.shape[0], df.shape[0], 100*evaluation[5])
    slider_input.value = 50.0000
    # regenerate slider and barplot stuff at threshold

def update_threshold(threshold_val):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix

    threshold = threshold_val
    # regenerate CM and barplot
    conf_mat = confusion_matrix(y, predictions > (threshold/100))
    cmlist1 = [conf_mat[1][1], conf_mat[1][0]]
    cmlist2 = [conf_mat[0][1], conf_mat[0][0]]
    cm_df = pd.DataFrame(
    {
        'Hospitalization': cmlist1,
        'No Hospitalization': cmlist2
    }, index=['Hospitalization', 'No Hospitalization']
    )
    cm_source.data = cm_df
    predictor_cols = list(df_with_predictions.columns[2:-1].delete(1))
    predictor_counts = []
    df_hosp = df_with_predictions.loc[pd.to_numeric(df_with_predictions['predict_percent']) > threshold]
    for predictor_val in predictor_cols:
        if df_hosp[predictor_val].value_counts().size > 1:
            predictor_counts.append(int(df_hosp[predictor_val].value_counts()[1]))
        else:
            predictor_counts.append(0)
    bar_source.data = dict(predictors=predictor_columns, counts=predictor_counts, color=colors)
    # regenerate CM and barplot
    predicted_vals = []
    compared_vals = []
    for prediction in predictions:
        if prediction[0] > threshold/100:
            predicted_vals.append(1)
        else:
            predicted_vals.append(0)
    for index, value in enumerate(y.tolist()):
        if value[0] == predicted_vals[index]:
            compared_vals.append(1)
        else:
            compared_vals.append(0)
    acc = sum(compared_vals) / len(compared_vals)
    slider_result.text = "<h3>With a threshold of {:.2f}, you can expect {} hospitalizations out of {} cases with an accuracy of {:.4f}%.</h3>".format(threshold, df_hosp.shape[0], df.shape[0], 100*acc)

def dataShowCallback(attr, old, new):
    update_prediction(new)

def sliderUpdateCallback(attr, old, new):
    update_threshold(new)

metric_selector.on_change('value', dataShowCallback)

slider_input.on_change('value', sliderUpdateCallback)