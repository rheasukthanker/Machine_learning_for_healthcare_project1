# Machine Learning for Healthcare (Project 1)

Our pre-trained models are uploaded at https://polybox.ethz.ch/index.php/s/9dLB08VfdqrdEH0
### Installing packages :
``` pip install -r requirements.txt ```

### Jupyter notebooks 
1. ```Data_exploration_project1.ipynb``` this notebook plots the different time series, the class distributions and the kmeans clustering of the time series
2. ```PCA-Visualisation.ipynb``` this notebook performs pca on the datasets and plots them color coded in a class-wise fashion
3. ``` Ensemble_ptb.ipynb``` Here we present 2 types of ensemble learning methods (majority vote and logistic regression based) on the models trained. This notebook also contains the AUROC and AUPRC scores for the PTB dataset
4. ``` Ensemble_mit.ipynb``` Here we present 2 types of ensemble learning methods (majority vote and logistic regression based) on the models trained. 

### General instructions to run .py files on Leonhard cluster
1. ```module load python_gpu/3.7.1```
2. ```module load hdf5/1.10.1```
3. ```bsub -n 3 -R "rusage[ngpus_excl_p=1]" -W 6:00 python abc.py ``` where abc.py is the corresponding file name. To run the files below replace abc with the appropriate file name

### Constituents of .py files
1. ```bidir_rnn_mit.py``` and ```bidir_rnn_ptb.py``` contain bidirectional lstm models for the two datasets.
2. ```gru_rnn_simple_mit.py``` and ```gru_rnn_simple_ptb.py``` contain gru network models for the two datasets.
3. ```lstm_rnn_simple_mit.py``` and ```lstm_rnn_simple_ptb.py``` contain lstm network models for the two datasets.
4. ```residual_connections_mitb.py``` and ```residual_connections_ptb.py``` contain cnn with residual connections for the two datasets.
5. ```rnn_simple_mit.py``` and ```rnn_simple_ptb.py``` contain simple rnn network models for the two datasets
6. ```transfer_learning_option2.py``` contains code for transfering pre-trained model from mitbh dataset to ptb dataset