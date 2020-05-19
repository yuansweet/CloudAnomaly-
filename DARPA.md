
# DARPA-1999 Intrusion Detection


This [dataset](https://archive.ll.mit.edu/ideval/data/1999data.html) consists of network traffic data originally provided in a .tcpdump format. Some of the packets included in the packet sniffing data (both outside and inside sniffing data are provided) are part of attacks on the network; the dates, start times and destinations of these attacks are available in a [detection truth list](https://archive.ll.mit.edu/ideval/docs/detections_1999.html) on the dataset website.

Due to the difficulty of parsing .tcpdump files directly in Python, the process we came up with for converting the dataset into graph data first involves using Wireshark (or another suitable protocol analysis tool) to read the .tcpdump files downloaded from the site and export them as .pcap files. Once in this format, the files can be read using Python’s pyshark library, which gives access to any and all information associated with individual packets in the packet capture files.

One proposed approach to creating dynamic graphs from the packet data is to treat each unique IP address that appears as a source or destination in the .pcap file as a node, and create a directed edge between nodes if any packets have been sent from the first node to the second in the time interval represented by the graph. Initially, from scanning a sample of the dataset, it appeared that there would be insufficient unique nodes to allow this approach to be effective at training the model; however, a scan of the entirety of one of the days in the set shows that roughly 850 unique IP addresses appear in a day’s worth of the network traffic data. This number of nodes is unlikely to test the scaling capabilities of the model, especially with a sparse graph, but it should provide a sufficiently large graph to be useful for anomaly detection.

In order to be effective, this network graph data will require useful, relevant node features that might help to indicate attacks: three features we have already implemented are:

-   Number of data packets sent by a node
    
-   Average length of packets sent by the node
    
-   Maximum length of any packet sent by the node.
    

Other ideas for potential features include:

-   Average time in between groups of packets sent by the node
    
-   Number of unique addresses to which the node sent data packets
    
-   Largest number of data packets sent by the node to any single address.
    

Expanding this list of node features with further ideas will likely be necessary for experiments involving this approach in order to provide the most information for the detection.

  

Another approach to creating dynamic graphs from the DARPA-1999 data is to represent the data as a heterogeneous graph. If we consider the network and host parts of the IP address separately, we can treat each network as a subgraph and treat the host machines within the network as nodes within each subgraph. Since the packet capture files, particularly the inside sniffing data files, have a substantial portion of intra-network traffic, this could be an effective way to structure the graph and take advantage of the heterogenous graph capabilities of the model. Other possible subnodes include email addresses and TCP ports that communicate within each network.

  

The data files themselves are from different weeks, and each week has a different number of attacks associated with it. Weeks 1 and 3 contain no attacks, and as such, files from these weeks are unlikely to be useful in training an anomaly detection system. Despite being from the “training” portion of the dataset, Week 2 contains a small number of attacks (12 or fewer per day, across both inside and outside data), so if data from this week is used, the attacks will be very sparse. Weeks 4 and 5 contain a slightly higher proportion of attacks (201 attacks total across both weeks), and as such, would be the most likely to be effective at training our model by providing more attack examples.
