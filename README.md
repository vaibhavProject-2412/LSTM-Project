# LSTM Electricity Consumption Prediction Project

This project uses an LSTM model to predict electricity consumption based on geodemographic data. Follow the steps below to set up the project, download necessary datasets, and execute the scripts in the correct order.

## Project Setup

1. **Clone the Repository**:
   Clone the project repository to your local machine:
   ```
   git clone https://github.com/vaibhavProject-2412/LSTM-Project.git
   cd LSTM-Project
   ```
2. **Install Dependencies: Install the required Python packages**:

```
pip install -r requirements.txt
```
3. **Download Datasets: Download the necessary datasets from Kaggle and place them in the data/ folder**:

- [daily_dataset.csv](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london?select=daily_dataset.csv)

- [information_households.csv](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london?select=informations_households.csv)

- [acorn_details.csv](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london?select=acorn_details.csv)

4. **Create Folder Structure: Set up the following folders in the project directory**:

```
mkdir data models results
```
- data/: Store the downloaded datasets here.<br>
- models/: Save the trained model and related outputs here.<br>
- results/: Store the visual output of model predictions, accuracy metrics, and other results.<br>

**Running the Project** <br>
**Data Preprocessing: First, run the data_preprocessing.py script to load and process the datasets. This will prepare the data for model training:**

```
python scripts/data_preprocessing.py
```

**Correlation Analysis: After preprocessing the data, run correlation_analysis.py to perform and save correlation visualizations:**

```
python scripts/correlation_analysis.py
```

**Model Training: Finally, run the lstm_forecasting_model.py script to train the LSTM model. The trained model will be saved in the models/ folder:**

```
python scripts/lstm_forecasting_model.py
```
**Folder Overview**
- data/: Contains the datasets (daily_dataset.csv, information_households.csv, acorn_details.csv).<br>
- models/: Stores the trained model file, such as lstm_model.h5.<br>
- results/: Contains output results like correlation plots, prediction vs. actual graphs, and loss curves.<br>

**Project Files**
- scripts/data_preprocessing.py: Script to load and preprocess the datasets.<br>
- scripts/correlation_analysis.py: Script to analyze and visualize correlations in the data.<br>
- scripts/lstm_forecasting_model.py: Script to build, train, and evaluate the LSTM model.<br>
