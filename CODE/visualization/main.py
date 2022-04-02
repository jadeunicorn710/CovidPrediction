import os

from bokeh.events import ButtonClick
from bokeh.layouts import row
from bokeh.models.widgets import Button, Div
from bokeh.io import output_file, show
from bokeh.plotting import curdoc
from bokeh.core.enums import Align
from display_data import DisplayData

global title_div
global dataButton
global modelButton
global predButton

path = os.path.dirname(os.path.abspath(__file__))

# initialize colors
colors = ['#B3A369', '#EAAA00', '#F5D580', '#003057', '#335161', '#004F9F', '#1879DB', '#545454', '#E5E5E5', '#8E8B76']

def addTitleButtons():
    curdoc().clear()
    curdoc().add_root(row(title_div,width=620))
    curdoc().add_root(row(dataButton, modelButton, predButton, width=900))

def dataShowCallback(event):
    addTitleButtons()
    DisplayData(curdoc()).display()

def modelShowCallback(event):
    addTitleButtons()
    exec(open((path + "/display_model.py")).read())

def predShowCallback(event):
    addTitleButtons()
    exec(open((path + "/display_predictor.py")).read())

title_div = Div(text="""<h1>COVID-19 Hospitalization and ICU Prediction</h1>\n<h3>An Interactive Visualization Based on Patients' Preconditions</h3>""", width=620)
title_div.background = colors[8]

dataButton = Button(label="Show data distribution", button_type="primary", width=200, background=colors[3])
dataButton.on_event(ButtonClick, dataShowCallback)

modelButton = Button(label="Show model results", button_type="primary", width=200, background=colors[3])
modelButton.on_event(ButtonClick, modelShowCallback)

predButton = Button(label="Show model predictor", button_type="primary", width=200, background=colors[3])
predButton.on_event(ButtonClick, predShowCallback)

addTitleButtons()
