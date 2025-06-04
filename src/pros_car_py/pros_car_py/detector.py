import cv2
import numpy as np
import random
import math
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
import time

def angle_between(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return dy / dx 

def group_parallel_lines(edges, spatial_eps=20, angle_eps=np.radians(10), min_samples=2):
    if not edges:
        return []
    
    features = []
    
    edge_dict = {}
    for i in range(0, len(edges), 4):
        edge_index = i // 4
        x1, y1, x2, y2 = edges[i:i+4]
        edge_dict[edge_index] = {
            'start': (x1, y1),
            'end': (x2, y2)
        }
    print(edge_dict)
    lines = list(edge_dict.items())

    for edge_id, line in lines:
        x0, y0 = line["start"]
        x1, y1 = line["end"]
        midx = (x0 + x1) / 2
        midy = (y0 + y1) / 2
        angle = angle_between(line["start"], line["end"])
        print(f'edge id: {edge_id}, line: {line}, angle: {angle}')
        # features.append([midx, midy, math.cos(angle), math.sin(angle)])
        features.append([midx, midy, angle])
    # print(lines)
    
    X = np.array(features)
    # spatial_scale = 1.0 / spatial_eps
    # angle_scale = 1.0 / angle_eps

    # X[:, 0:2] *= spatial_scale
    # X[:, 2:4] *= angle_scale

    clustering = DBSCAN(eps=3.0, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    print(labels)
    groups = []
    for label in set(labels):
        if label == -1:
            continue
        group = {
            "edges": [],
            "edge_ids": []
        }
        for idx, line_label in enumerate(labels):
            if line_label == label:
                edge_id, edge = lines[idx]
                group["edges"].append(edge)
                group["edge_ids"].append(edge_id)
        groups.append(group)
    print(groups)
    return groups