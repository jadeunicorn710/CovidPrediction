Video Presentation [here](https://www.dropbox.com/s/eqoamtecf6ucet3/poster%20presentation%20yqi.mov?dl=0)

Webapp [here](https://young-bastion-46390.herokuapp.com/visualization)


DESCRIPTION:

The CODE section consists of three parts: a) raw data and feature preprocessing module (feature selection), b) Logistic Regression and Neural Network training modules (models) and c) interactive visualization web application (visualization).

The feature preprocessing module will load raw data, clean any inaccurate and irrelevant information and select features for the model training. The model training section has multiple modules (logistic regression and neural network), each of which reads data generated from the feature selection module, builds a model, encodes data to be fed to the model, and finally, trains and evaluates the model using charts and graphs. The visualization module consists of web pages that reads data generated from the trained model and display charts and graphs with interactive elements.


INSTALLATION:

1. You need to install Python and data science packages to run the experiment on your machine.  To install Python (python 3.8 or higher), visit this site (https://wiki.python.org/moin/BeginnersGuide/Download) and follow instructions for your operating system. 
For example, if you have an Ubuntu 20.x system, you can install the Python package using the command below. Make sure to include pip while installing the Python package.

	~$ sudo apt-get update
	~$ sudo apt install -y python3-pip 
	~$ sudo apt install python3-testresources

2. Go to the "CODE" folder and you will find requirements.txt file. Run the pip command to install the remaining dependencies as shown below. 

	~/CODE$ sudo pip3 install -r requirements.txt

3. If your installation is complete, move to the EXECUTION section. 


EXECUTION:

1. Open your shell and move to the "CODE" folder. This folder will be your root folder to run the experiment and visualize the results.

	~/CODE$ _

2. Run Jupyter notbook

	~/CODE$ jupyter notebook

3. If your browser does not open the notebook automatically, copy the URL on the shell screen and paste it on your browser’s address bar.

4. From your jupyter notebook interface, open the "feature selection" folder.

5. Open data_preprocessing.ipynb. Click on "Kernel" from the menu and click on "Restart & Run All". The module will preprocess original data and generate multiple files for the experiment.

6. Go back to the root folder of your jupyter notebook and open the "models" folder.

7. Open hospitalization_logistic_regression_8.ipynb. Click on "Kernel" from the menu and click on "Restart & Run All". It will take some time (up to 5 min) to complete.

8. Open hospitalization_neural_network_8.ipynb. Click on "Kernel" from the menu and click on "Restart & Run All". It will take some time (up to 5 min) to complete.

9. Open icu_logistic_regression_8.ipynb. Click on "Kernel" from the menu and click on "Restart & Run All". It will take some time (up to 5 min) to complete.

10. Open icu_neural_network_8.ipynb. Click on "Kernel" from the menu and click on "Restart & Run All". It will take some time (up to 5 min) to complete.

11. Open snn_hospitalisation_prediction.ipynb. Click on "Kernel" from the menu and click on "Restart & Run All". You can verify the functionality of prediction class (python) which will be used in the web application. 

12. Open another shell and go to the "CODE" folder. Run the web application by entering the command shown below:

	~/CODE$ bokeh serve visualization

13. Open new broswer window (or tab) and copy the URL on the shell screen and paste it on your browser’s address bar.

14. Click "Show data distribution", "Show model results" and "Show model predictor" to evaluate data distribution, trained model and prediction performance.


DEMO VIDEO:

https://youtu.be/dhDOvUl8Gvg

