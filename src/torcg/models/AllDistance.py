import torch.nn as nn
import torch
import torch.nn.functional as F
# from sklearn.neighbors.dist_metrics import DistanceMetric, MahalanobisDistance, SEuclideanDistance, WMinkowskiDistance, \
#     MinkowskiDistance, ChebyshevDistance, ManhattanDistance, EuclideanDistance
from scipy.spatial.distance import *
import numpy as np


class AllDistance(nn.Module):
    def __init__(self, device):
        super(AllDistance, self).__init__()
        self.device = device
        # self.haversine = DistanceMetric.get_metric('haversine')
        # self.haversine = DistanceMetric.get_metric('haversine')

    def forward(self, out1, out2):
        all_distance = []

        for i, out1_line in enumerate(out1.split(1, dim=0)):
            out2_line = out2[i].cpu().detach().numpy()
            out1_line = out1_line.squeeze(0).cpu().detach().numpy()
            # pair = torch.cat(out2[i], out1_line.squeeze(0)).tolist()

            dis_array = [
                 braycurtis(out1_line, out2_line),
                 canberra(out1_line, out2_line),
                 chebyshev(out1_line, out2_line),
                 cityblock(out1_line, out2_line),
                 correlation(out1_line, out2_line),
                 cosine(out1_line, out2_line),
                 dice(out1_line, out2_line),
                 euclidean(out1_line, out2_line),
                 hamming(out1_line, out2_line),
                 # torch.FloatTensor(jaccard(out1_line, out2_line), device=self.device),
                 # torch.FloatTensor(kulsinski(out1_line, out2_line), device=self.device),
                 # torch.FloatTensor(mahalanobis(out1_line, out2_line), device=self.device),
                 # torch.FloatTensor(matching(out1_line, out2_line), device=self.device),
                 minkowski(out1_line, out2_line),
                 # torch.FloatTensor(rogerstanimoto(out1_line, out2_line), device=self.device),
                 # torch.FloatTensor(russellrao(out1_line, out2_line), device=self.device),
                 # torch.FloatTensor(seuclidean(out1_line, out2_line), device=self.device),
                 # torch.FloatTensor(sokalmichener(out1_line, out2_line), device=self.device),
                 # torch.FloatTensor(sokalsneath(out1_line, out2_line), device=self.device),
                 sqeuclidean(out1_line, out2_line),
                 # torch.FloatTensor(wminkowski(out1_line, out2_line), device=self.device),
                 yule(out1_line, out2_line)
            ]
            all_distance.append(dis_array)
        tendis = torch.FloatTensor(all_distance, device=self.device).to(self.device)
        # tendis = torch.cat(all_distance, dim=0).to(self.device)
        return tendis
