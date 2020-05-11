from ordered_set import OrderedSet
from collections import defaultdict as ddict
import numpy as np
import torch
from tqdm import tqdm
import queue


def create_graph(edge_index, edge_type):
    graph = {}
    all_triples = torch.cat([edge_index.transpose(
        0, 1), edge_type.unsqueeze(1)], dim=1)
    print("Graph creating...")
    for data in tqdm(all_triples):
        source = data[0].data.item()
        target = data[1].data.item()
        value = data[2].data.item()

        if(source not in graph.keys()):
            graph[source] = {}
            graph[source][target] = value
        else:
            graph[source][target] = value
    print('Done')
    return graph


def bfs(graph, source, nbd_size=2):
    visit = {}
    distance = {}
    parent = {}
    distance_lengths = {}

    visit[source] = 1
    distance[source] = 0
    parent[source] = (-1, -1)

    q = queue.Queue()
    q.put((source, -1))

    while(not q.empty()):
        top = q.get()
        if top[0] in graph.keys():
            for target in graph[top[0]].keys():
                if(target in visit.keys()):
                    continue
                else:
                    q.put((target, graph[top[0]][target]))

                    distance[target] = distance[top[0]] + 1

                    visit[target] = 1
                    if distance[target] > 2:
                        continue
                    parent[target] = (top[0], graph[top[0]][target])

                    if distance[target] not in distance_lengths.keys():
                        distance_lengths[distance[target]] = 1

    neighbors = {}
    for target in visit.keys():
        if(distance[target] != nbd_size):
            continue
        edges = [-1, parent[target][1]]
        relations = []
        entities = [target]
        temp = target
        while(parent[temp] != (-1, -1)):
            relations.append(parent[temp][1])
            entities.append(parent[temp][0])
            temp = parent[temp][0]

        if(distance[target] in neighbors.keys()):
            neighbors[distance[target]].append(
                (tuple(relations), tuple(entities[:-1])))
        else:
            neighbors[distance[target]] = [
                (tuple(relations), tuple(entities[:-1]))]

    return neighbors


def get_further_neighbors(graph, nbd_size=2):
    print('Getting 2hop neighbors')
    neighbors = {}
    for source in tqdm(graph.keys()):
        temp_neighbors = bfs(graph, source, nbd_size)
        for distance in temp_neighbors.keys():
            if(source in neighbors.keys()):
                if(distance in neighbors[source].keys()):
                    neighbors[source][distance].append(
                        temp_neighbors[distance])
                else:
                    neighbors[source][distance] = temp_neighbors[distance]
            else:
                neighbors[source] = {}
                neighbors[source][distance] = temp_neighbors[distance]
    print('Done')
    print("length of neighbors dict is ", len(neighbors))
    return neighbors


def get_batch_nhop_neighbors_all(batch_sources, node_neighbors, num_rels, device, partial_2hop=True, nbd_size=2):
    train_edge = batch_sources.cpu().numpy()
    unique_train_entity = list(set(train_edge[0]).union(train_edge[1]))
    batch_source_triples = []
    hop_edge_index, hop_edge_type = [], []
    print("length of unique_entities ", len(unique_train_entity))
    for source in unique_train_entity:
        # randomly select from the list of neighbors
        if source in node_neighbors.keys():
            nhop_list = node_neighbors[source][nbd_size]

            for i, tup in enumerate(nhop_list):
                if(partial_2hop and i >= 10):
                    break
                hop_edge_index.append((source, nhop_list[i][1][0]))
                hop_edge_type.append((nhop_list[i][0][-1], nhop_list[i][0][0]))
    for source in unique_train_entity:
        if source in node_neighbors.keys():
            nhop_list = node_neighbors[source][nbd_size]
            for i, tup in enumerate(nhop_list):
                if(partial_2hop and i >= 10):
                    break
                hop_edge_index.append((nhop_list[i][1][0],source))
                hop_edge_type.append((nhop_list[i][0][-1]+num_rels, nhop_list[i][0][0]+num_rels))
    hop_edge_index	= torch.LongTensor(hop_edge_index).to(device).t()
    hop_edge_type	= torch.LongTensor(hop_edge_type). to(device)

    return hop_edge_index, hop_edge_type
