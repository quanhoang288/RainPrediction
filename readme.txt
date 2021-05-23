Requirement: Python 3.7+
Install requirements: pip install -r requirements.txt 

Data is provided in data folder. You can also download data from the following google drive link: 
https://drive.google.com/drive/folders/1B9636KXkRa8PgkefMBfvJYQIuPgJUJID?usp=sharing

All of our work including data exploration, feature engineering, modeling and evaluation are conducted in the python notebooks in notebooks directory.
You can navigate to that directory to see all the work and results. It include following notebooks: 
+ data_exploration.ipynb: perform exploratory data analysis 
+ feature_engineering.ipynb: perform feature engineering 
+ modeling.ipynb: perform modeling and hyperparameter tuning for 3 models 
+ evaluate.ipynb: evaluate best results of 3 models and evaluate on resampled datasets for balancing the dataset 
+ data_resample.ipynb: contains code for resample the dataset to handle unbalanced data problem 
+ subdata_generate.ipynb: contains code for generate different datasets with different sizes and number of features
+ rain_in_australia_full.ipynb: contains the full code for all the previous parts

Besides, we also provided some scripts to reproduce the experiments we conducted. Instructions on how to use those scripts are provided below

Preprocess data using preprocess.py: python preprocess.py (Make sure the CSV data file is included in data folder)

Perform data splitting using data_split.py: python data_split.py
Required arguments: 
	--feature_file: input feature vectors file location
	-- label_file: input label file location
Optional arguments:
	--test_size: Size of test set. Default 0.2
	--do_split_val: Whether to split the original data into train, validation and test set
	--val_size: Size of validation. Used when apply argument do_split_val. Default 0.2
	--output_dir: Output directory to store splitted data

Train model using train.py: python train.py 
Required arguments: 
	--model_name: name of model to train 
	--feature_file: input feature vectors file location
	-- label_file: input label file location
Optional arguments:
	--kernel: kernel used for SVM. Default linear
	--C: value for regularization term. Default 1.0
	--gamma: gamma value for rbf, sigmoid and polynomial kernel SVM. Default 'scale'
	--degree: degree value for polynomial kernel
	--output_file: name of file to save model as 
	--output_dir: Output directory to store trained model

Evaluate model using evaluate.py: python evaluate.py
Required arguments: 
	--model_name: name of model to train 
	--feature_file: input feature vectors file location to evaluate
	-- label_file: input label file location to evaluate
	--checkpoint_file: file name of model checkpoint
	--checkpoint_dir: directory where model checkpoint is stored
Optional arguments:
	--do_plot_result: Whether to plot evaluation result 

Resample data to handle unbalanced data problem: python data_resample.py
Arguments: 
	--sampling_strategy: sampling method to use. Available options: ros (Random Oversampling), smote (SMOTE), rus (Random Undersampling), tomek (TomekLinks), enn (EdittedNearestNeighbor)
	--ratio: Sampling ratio (after resampled) (<= 1.0). Default 1.0
	--feature_file: Input feature file
        --label_file: Input label file
	--output_dir: directory to store resampled data


