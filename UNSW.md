# UNSW-NB15 Dataset

This [dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/) consists of network data based on a hybrid of real modern normal activities and synthetic contemporary attack behaviors provided in four .csv files. We combine the four .csv files into a single dataframe and use the first 80% of records as the training set and the remainder as the test set. Packets are labeled with either a “0” or a “1” depending on whether or not they were part of an attack, and detailed information about each attack including start time, end time, attack category, and source ip are available in the provided ground truth table separate from the dataset. There are about 2.5 million packets and 47 unique IP addresses in total, and about 12.6% of the packets are part of an attack.

The current method of creating graphs from the packet data involves treating all unique source and destination IP addresses as nodes and creating a directed edge from the source IP of a packet to its destination IP within the respective time interval (currently, a period of one hour). Seven features are used from the dataset:

1.  ct_dst_sport_ltm: No. of connections of the same destination address and the source port in 100 connections according to the last time.
    
2.  tcprtt: TCP connection setup round-trip time, the sum of ’synack’ and ’ackdat’.
    
3.  dwin: Destination TCP window advertisement value
    
4.  ct_src_dport_ltm: No. of connections of the same source address and the destination port in 100 connections according to the last time.
    
5.  ct_dst_src_ltm: No. of connections of the same source and the destination address in 100 connections according to the last time.
    
6.  ct_dst_ltm: No. of connections of the same destination address in 100 connections according to the last time.
    
7.  smeansz: Mean of the flow packet size transmitted by the src
    

For each packet, features 1, 3, 5, and 6 are assigned to the destination IP, features 4 and 7 are assigned to the source IP, and feature 2 is assigned to both. Finally, a node is considered an anomaly if any packet originating from the IP address within the time interval is part of an attack (i.e. Label of “1”).

Possibilities for further experiments with this dataset include visualization of the graph at different time steps; analyzing distribution of features among true positives, false positives, true negatives, and false negatives; exploring alternative methods of feature assignment (e.g. aggregating features across all packets in a single time interval); and optimizing the preprocessing stage for better runtimes.
