# CloudAnomaly
This repository contains the files for the Cloud Anomaly Detection project. The primary code for the project can be found in the DGAD folder and run with "python main.py". Code for replicating previous experiments is also included and can be used as follows: 

 - clustergcn.py: create_partition accepts an adjacency matrix and feature matrix and uses the partitioning method described in the ClusterGCN paper to create a subgraph and returns the corresponding adjacency and feature matrices. It also accepts two hyperparameters, p (how many partitions to create) and q (how many of those partitions to combine into a subgraph).
 - nstep_neighbor.py: adj_to_bias accepts an adjacency matrix and nhood parameter and returns the n-step neighbor version of the adjacency matrix.
 - UNSW_preprocess.py: preprocess_UNSW accepts a csv path and test (whether to return test set or train set) parameter. The csv path should specify a location with the four csv files comprising the dataset (UNSW-NB15_1.csv, UNSW-NB15_2.csv, etc.). It returns the adjacency and feature matrices for the graph at every time step in the dataset, the ground truth list of anomalies, the total number of graphs, and the size of each graph.

Additionally, the repository contains results from the experiments conducted so far (Experiments.md) as well as summaries of the two datasets introduced in Spring 2020 (UNSW.md and DARPA.md).
