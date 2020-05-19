def preprocess_UNSW(csv_path, test=False):
    # Dict of all features in the dataset and their datatypes
    my_col = {
    'srcip':str,
    'sport':np.int64,
    'dstip':str,
    'dsport':np.int64,
    'proto':str,
    'state':str,
    'dur':float,
    'sbytes':np.int64,
    'dbytes':np.int64,
    'sttl':np.int64,
    'dttl':np.int64,
    'sloss':np.int64,
    'dloss':np.int64,
    'service':str,
    'Sload':float,
    'Dload':float,
    'Spkts':np.int64,
    'Dpkts':np.int64,
    'swin':np.int64,
    'dwin':np.int64,
    'stcpb':np.int64,
    'dtcpb':np.int64,
    'smeansz':np.int64,
    'dmeansz':np.int64,
    'trans_depth':np.int64,
    'res_bdy_len':np.int64,
    'Sjit':float,
    'Djit':float,
    'Stime':np.datetime64,
    'Dtime':np.datetime64,
    'Sintpkt':float,
    'Dintpkt':float,
    'tcprtt':float,
    'synack':float,
    'ackdat':float,
    'is_sm_ips_ports':bool,
    'ct_state_ttl':np.int64,
    'ct_flw_http_mthd':np.int64,
    'is_ftp_login':bool,
    'ct_ftp_cmd':np.int64,
    'ct_srv_src':np.int64,
    'ct_srv_dst':np.int64,
    'ct_dst_ltm':np.int64,
    'ct_src_ltm':np.int64,
    'ct_src_dport_ltm':np.int64,
    'ct_dst_sport_ltm':np.int64,
    'ct_dst_src_ltm':np.int64,
    'attack_cat':str,
    'Label':bool
    }

    # Read and combine csv files into dataframe, then choose train or test set
    df1 = pd.read_csv(csv_path + "UNSW-NB15_1.csv", names=my_col.keys())
    df2 = pd.read_csv(csv_path + "UNSW-NB15_2.csv", names=my_col.keys())
    df3 = pd.read_csv(csv_path + "UNSW-NB15_3.csv", names=my_col.keys())
    df4 = pd.read_csv(csv_path + "UNSW-NB15_4.csv", names=my_col.keys())
    df = pd.concat([df1, df2, df3, df4])
    if test:
      df = df.tail(len(df)//5)
    else:
      df = df.head(4 * len(df)//5)

    # Get all unique IP addresses and assign a node to each one
    graph_size = len(pd.unique(df[['srcip', 'dstip']].values.ravel('K')))
    ips = dict()
    count = 0
    for ip in pd.unique(df[['srcip', 'dstip']].values.ravel('K')):
      ips[ip] = count
      count += 1
    time_max = int(df.describe()['Dtime'][7])
    time_min = int(df.describe()['Stime'][3])
    feature_names = ['ct_dst_sport_ltm',"tcprtt","dwin","ct_src_dport_ltm","ct_dst_src_ltm","ct_dst_ltm","smeansz"]

    # Assign a number to each feature
    count = 0
    features = dict()
    for feature in feature_names:
      features[feature] = count
      count += 1
    num_features = len(features.keys())
    total_clips = int(time_max - time_min) // 3600 + 1 # Total number of graphs to create from the dataset based on time interval
    node_feature = torch.zeros((total_clips, graph_size, num_features))
    edge = torch.zeros((total_clips, graph_size, graph_size))
    abnormal = torch.zeros((total_clips, graph_size))
    total_features = ['srcip', 'dstip','Stime'] + list(features.keys()) +['Label']

    for row in df[total_features].values:
      srcip = row[0]
      dstip = row[1]
      Stime = row[2]
      src_idx = ips[srcip]
      dst_idx = ips[dstip]
      time_idx = int(Stime - time_min) // 3600
      edge[time_idx][src_idx][dst_idx] = 1

      node_feature[time_idx][dst_idx][0] = row[3]
      node_feature[time_idx][src_idx][1] = row[4]
      node_feature[time_idx][dst_idx][1] = row[4]
      node_feature[time_idx][dst_idx][2] = row[5]
      node_feature[time_idx][src_idx][3] = row[6]
      node_feature[time_idx][dst_idx][4] = row[7]
      node_feature[time_idx][dst_idx][5] = row[8]
      node_feature[time_idx][src_idx][6] = row[9]

      abnormal[time_idx][src_idx] = max(row[10], abnormal[time_idx][src_idx])
    return edge, node_feature, abnormal, total_clips, graph_size
