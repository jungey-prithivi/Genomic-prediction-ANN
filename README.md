# Genomic-prediction-ANN

This code perfroms Genomic prediction using the *Tensorflow* and *Keras* libraries. 
Two types of input files are required: 1) **Genotype file** and **Phenotype File**. The example files are in the input folders. Genotype files consist of SNP information in the header and is  a binary matrix, while phenotype file consists of the measured phenotypic values alond with the accession (Plant line) in the row index.
The script uses **Hyperband** tuner from keras to find optimal hyperparameters. The range for the hyperparameter search can be adjusted as well in the script.
The results are stored in an output folder. 
