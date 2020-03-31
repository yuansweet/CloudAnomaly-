import numpy as np
import csv
import time
import datetime
import torch

def csv_read(path):
    data = []
    with open(path,'r',encoding='utf-8') as f:
        reader = csv.reader(f,dialect='excel')
        next(reader)
        for row in reader:
            data.append(row)
    return np.array(data)#np.transpose(data).astype(np.float64)

def Caltime(date1, date2):
    date1 = time.strptime(date1, "%Y/%m/%d %H:%M")
    date2 = time.strptime(date2, "%Y/%m/%d %H:%M")

    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])

    return (date2 - date1).days

def save_graph(feature_path, link_path):
    feature = csv_read(feature_path)
    link = csv_read(link_path)

    start_data = link[0,3]
    new_lable = []
    print('start!')

    node_dict = {}
    for i in range(feature.shape[0]):
        node_dict[feature[i][0]] = i

    for l in link:
        if l[0] not in node_dict.keys() or l[1] not in node_dict.keys():
            continue
        date = Caltime(start_data, l[3])
        if date< 0:
            continue
        if date > 100:
            break

        new_lable.append(node_dict[l[0]])
        new_lable.append(node_dict[l[1]])

    new_lable = sorted(list(set(new_lable)))
    new_feature = []

    for i in new_lable:
        new_feature.append(feature[i])
    new_feature = np.array(new_feature)
    print(new_feature.shape)

    new_node_dict = {}
    for i in range(new_feature.shape[0]):
        new_node_dict[feature[i][0]] = i

    graph = np.zeros((new_feature.shape[0], new_feature.shape[0]), dtype=np.float16)
    node = np.zeros((new_feature.shape[0], new_feature.shape[1] - 1), dtype=np.float16)
    day = 0

    print('start!')
    for l in link:
        if l[0] not in new_node_dict.keys() or l[1] not in new_node_dict.keys():
            continue
        date = Caltime(start_data, l[3])
        if date < 0:
            continue
        if date > 100:
            break

        if date > day:
            day = date

            np.save('reddit_data2/node/node' + str(day) + '.npy', node)
            np.save('reddit_data2/graph/graph' + str(day) + '.npy', graph)
            np.save('reddit_data2/dict/node_dict' + str(day) + '.npy', new_node_dict)

            graph = np.zeros((new_feature.shape[0], new_feature.shape[0]), dtype=np.float16)
            node = np.zeros((new_feature.shape[0], new_feature.shape[1] - 1), dtype=np.float16)

        graph[new_node_dict[l[0]], new_node_dict[l[1]]] = float(l[4])
        graph[new_node_dict[l[0]], new_node_dict[l[0]]] = 1.0
        graph[new_node_dict[l[1]], new_node_dict[l[1]]] = 1.0
        node[new_node_dict[l[0]]] = np.asarray(feature[new_node_dict[l[0]]][1:], dtype=np.float16)
        node[new_node_dict[l[1]]] = np.asarray(feature[new_node_dict[l[1]]][1:], dtype=np.float16)

    np.save('reddit_data2/node/node' + str(day) + '.npy', node)
    np.save('reddit_data2/graph/graph' + str(day) + '.npy', graph)
    np.save('reddit_data2/dict/node_dict' + str(day) + '.npy', new_node_dict)


def load_graph(feature_path, link_path, dict_path=None):
    node = np.load(feature_path).astype(np.float64)
    graph = np.load(link_path).astype(np.float64)
    dict = None
    if dict_path is not None:
        d = np.load(dict_path)
        dict = d.item()
    return torch.tensor(node), torch.tensor(graph), dict



#def delete(edge, node)

if __name__ == '__main__':
    save_graph('node.csv', 'link.csv')
    a,_,_=load_graph('reddit_data2/node/node101.npy','reddit_data/graph/graph101.npy')
    print(a)
    print(torch.sum(a))