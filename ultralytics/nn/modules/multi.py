"""
multi modules for language and coord
"""
import copy

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from .conv import Conv
import math
import matplotlib.pyplot as plt

__all__ = {
    "Pass",
    "Get_Img",
    "EuclideanRSM",
    "ManhattanRSM",
    "DotRSM",
    "PearsonRSM",
    "W_Concat_two",
    "W_Add",
    "Maxpool2_Add",
    "W_Add_Tanh",
    "Ripple_Attention_Linear",
    "Laplace_RA",
    "Inverse_Distance_RA",
    "Softmax_RA",
    "Maha_RA",
}


def get_avg_flang(layer, word_mask):
    flang_list = []
    for i in range(layer.size(0)):
        flang_with_cls_sep = layer[i][word_mask[i] == 1][:]  # (n+2,768)
        flang_without_cls_sep = flang_with_cls_sep[1:-1][:]  # (n,768)
        flang_mean = flang_without_cls_sep.mean(dim=0, keepdim=True)  # (1,768)
        flang_list.append(flang_mean)
    flang = torch.stack(flang_list).squeeze(1)  # (32,768)
    return flang


def get_max_flang(layer, word_mask):
    flang_list = []
    for i in range(layer.size(0)):
        flang_with_cls_sep = layer[i][word_mask[i] == 1][:]  # (n+2,768)
        flang_without_cls_sep = flang_with_cls_sep[1:-1][:]  # (n,768)
        flang_max, max_idx = flang_without_cls_sep.max(dim=0, keepdim=True)  # (1,768)
        flang_list.append(flang_max)
    flang = torch.stack(flang_list).squeeze(1)  # (32,768)
    return flang


class Pass(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Get_Img(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if isinstance(x, list):
            return x[0]
        else:
            return x


class EuclideanRSM(nn.Module):
    def __init__(self, c1, c2, text_dim, i):
        super().__init__()
        self.flang_layer_num = i
        self.emb_size = int((c1 + text_dim) / 2)
        self.mapping_lang = nn.Linear(text_dim, self.emb_size)
        self.mapping_fusion = torch.nn.Sequential(
            Conv(c1, c1, 1),
            Conv(c1, c1, 3),
            nn.Conv2d(c1, c2, 1)
        )
        self.mapping_visu = Conv(c1, c1, 1)
        self.gamma = nn.Linear(self.emb_size, c1)
        self.beta = nn.Linear(self.emb_size, c1)

    def forward(self, x):
        img = x[0]
        flang = x[1][1][self.flang_layer_num]  # (32, 20, 768)
        word_mask = x[1][2]
        batch_size = img.shape[0]

        flang_map = self.mapping_lang(flang)  # (32,20,448)
        ga = torch.tanh(self.gamma(flang_map))
        be = torch.tanh(self.beta(flang_map))
        gamma = get_avg_flang(ga, word_mask)  # (32,channel)
        beta = get_max_flang(be, word_mask)

        v_ori = torch.tanh(self.mapping_visu(img))

        g = gamma.view(batch_size, -1, 1, 1).expand_as(v_ori)
        b = beta.view(batch_size, -1, 1, 1).expand_as(v_ori)
        v_film = F.normalize((g * v_ori + b), p=2, dim=1)
        v_film = self.mapping_fusion(v_film)

        v_film_avg = torch.tanh(v_film).mean(dim=(2, 3), keepdim=False)  # (32,channel)
        img_avg = torch.tanh(img).mean(dim=(2, 3), keepdim=False)

        v_film_dist = torch.norm(v_film_avg - gamma, dim=1)
        img_dist = torch.norm(img_avg - gamma, dim=1)
        v_film_weight = 1 / (1 + v_film_dist)
        img_weight = 1 / (1 + img_dist)
        distance_weight = torch.tanh(v_film_weight - img_weight)

        img_mask = torch.ones_like(v_film_weight)
        v_film_mask = torch.ones_like(v_film_weight)
        v_film_mask = v_film_mask + distance_weight
        img_mask = img_mask - distance_weight

        img_mask = img_mask.view(batch_size, -1, 1, 1).expand_as(img)
        v_film_mask = v_film_mask.view(batch_size, -1, 1, 1).expand_as(v_film)
        v_add = torch.mul(img, img_mask) + torch.mul(v_film, v_film_mask)
        v_final = F.normalize(v_add, p=2, dim=1)
        return v_final


class ManhattanRSM(nn.Module):
    def __init__(self, c1, c2, text_dim, i):
        super().__init__()
        self.flang_layer_num = i
        self.emb_size = int((c1 + text_dim) / 2)
        self.mapping_lang = nn.Linear(text_dim, self.emb_size)
        self.mapping_fusion = torch.nn.Sequential(
            Conv(c1, c1, 1),
            Conv(c1, c1, 3),
            nn.Conv2d(c1, c2, 1)
        )
        self.mapping_visu = Conv(c1, c1, 1)
        self.gamma = nn.Linear(self.emb_size, c1)
        self.beta = nn.Linear(self.emb_size, c1)

    def forward(self, x):
        img = x[0]
        flang = x[1][1][self.flang_layer_num]  # (32, 20, 768)
        word_mask = x[1][2]
        batch_size = img.shape[0]

        flang_map = self.mapping_lang(flang)  # (32,20,448)
        ga = torch.tanh(self.gamma(flang_map))
        be = torch.tanh(self.beta(flang_map))
        gamma = get_avg_flang(ga, word_mask)  # (32,channel)
        beta = get_max_flang(be, word_mask)

        v_ori = torch.tanh(self.mapping_visu(img))

        g = gamma.view(batch_size, -1, 1, 1).expand_as(v_ori)
        b = beta.view(batch_size, -1, 1, 1).expand_as(v_ori)
        v_film = F.normalize((g * v_ori + b), p=2, dim=1)
        v_film = self.mapping_fusion(v_film)

        v_film_avg = torch.tanh(v_film).mean(dim=(2, 3), keepdim=False)  # (32,channel)
        img_avg = torch.tanh(img).mean(dim=(2, 3), keepdim=False)

        v_film_dist = torch.sum(torch.abs(v_film_avg - gamma), dim=1)
        img_dist = torch.sum(torch.abs(img_avg - gamma), dim=1)
        v_film_weight = 1 / (1 + v_film_dist)
        img_weight = 1 / (1 + img_dist)
        distance_weight = torch.tanh(v_film_weight - img_weight)

        img_mask = torch.ones_like(v_film_weight)
        v_film_mask = torch.ones_like(v_film_weight)
        v_film_mask = v_film_mask + distance_weight
        img_mask = img_mask - distance_weight

        img_mask = img_mask.view(batch_size, -1, 1, 1).expand_as(img)
        v_film_mask = v_film_mask.view(batch_size, -1, 1, 1).expand_as(v_film)
        v_add = torch.mul(img, img_mask) + torch.mul(v_film, v_film_mask)
        v_final = F.normalize(v_add, p=2, dim=1)
        return v_final


class DotRSM(nn.Module):
    def __init__(self, c1, c2, text_dim, i):
        super().__init__()
        self.flang_layer_num = i
        self.emb_size = int((c1 + text_dim) / 2)
        self.mapping_lang = nn.Linear(text_dim, self.emb_size)
        self.mapping_fusion = torch.nn.Sequential(
            Conv(c1, c1, 1),
            Conv(c1, c1, 3),
            nn.Conv2d(c1, c2, 1)
        )
        self.mapping_visu = Conv(c1, c1, 1)
        self.gamma = nn.Linear(self.emb_size, c1)
        self.beta = nn.Linear(self.emb_size, c1)

    def forward(self, x):
        img = x[0]
        flang = x[1][1][self.flang_layer_num]  # (32, 20, 768)
        word_mask = x[1][2]
        batch_size = img.shape[0]

        flang_map = self.mapping_lang(flang)  # (32,20,448)
        ga = torch.tanh(self.gamma(flang_map))
        be = torch.tanh(self.beta(flang_map))
        gamma = get_avg_flang(ga, word_mask)  # (32,channel)
        beta = get_max_flang(be, word_mask)

        v_ori = torch.tanh(self.mapping_visu(img))

        g = gamma.view(batch_size, -1, 1, 1).expand_as(v_ori)
        b = beta.view(batch_size, -1, 1, 1).expand_as(v_ori)
        v_film = F.normalize((g * v_ori + b), p=2, dim=1)
        v_film = self.mapping_fusion(v_film)

        v_film_avg = torch.tanh(v_film).mean(dim=(2, 3), keepdim=False)  # (32,channel)
        img_avg = torch.tanh(img).mean(dim=(2, 3), keepdim=False)

        v_film_weight = torch.sum(v_film_avg * gamma, dim=1)
        img_weight = torch.sum(img_avg * gamma, dim=1)
        v_film_weight = torch.tanh(v_film_weight)
        img_weight = torch.tanh(img_weight)
        distance_weight = torch.tanh(v_film_weight - img_weight)

        img_mask = torch.ones_like(v_film_weight)
        v_film_mask = torch.ones_like(v_film_weight)
        v_film_mask = v_film_mask + distance_weight
        img_mask = img_mask - distance_weight

        img_mask = img_mask.view(batch_size, -1, 1, 1).expand_as(img)
        v_film_mask = v_film_mask.view(batch_size, -1, 1, 1).expand_as(v_film)
        v_add = torch.mul(img, img_mask) + torch.mul(v_film, v_film_mask)
        v_final = F.normalize(v_add, p=2, dim=1)
        return v_final


def pearson_corr(x, y):
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)
    vx = x - x_mean
    vy = y - y_mean
    corr = torch.sum(vx * vy, dim=1) / (torch.norm(vx, dim=1) * torch.norm(vy, dim=1) + 1e-8)
    return corr


class PearsonRSM(nn.Module):
    def __init__(self, c1, c2, text_dim, i):
        super().__init__()
        self.flang_layer_num = i
        self.emb_size = int((c1 + text_dim) / 2)
        self.mapping_lang = nn.Linear(text_dim, self.emb_size)
        self.mapping_fusion = torch.nn.Sequential(
            Conv(c1, c1, 1),
            Conv(c1, c1, 3),
            nn.Conv2d(c1, c2, 1)
        )
        self.mapping_visu = Conv(c1, c1, 1)
        self.gamma = nn.Linear(self.emb_size, c1)
        self.beta = nn.Linear(self.emb_size, c1)

    def forward(self, x):
        img = x[0]
        flang = x[1][1][self.flang_layer_num]  # (32, 20, 768)
        word_mask = x[1][2]
        batch_size = img.shape[0]

        flang_map = self.mapping_lang(flang)  # (32,20,448)
        ga = torch.tanh(self.gamma(flang_map))
        be = torch.tanh(self.beta(flang_map))
        gamma = get_avg_flang(ga, word_mask)  # (32,channel)
        beta = get_max_flang(be, word_mask)

        v_ori = torch.tanh(self.mapping_visu(img))

        g = gamma.view(batch_size, -1, 1, 1).expand_as(v_ori)
        b = beta.view(batch_size, -1, 1, 1).expand_as(v_ori)
        v_film = F.normalize((g * v_ori + b), p=2, dim=1)
        v_film = self.mapping_fusion(v_film)

        v_film_avg = torch.tanh(v_film).mean(dim=(2, 3), keepdim=False)  # (32,channel)
        img_avg = torch.tanh(img).mean(dim=(2, 3), keepdim=False)

        v_film_weight = pearson_corr(v_film_avg, gamma)
        img_weight = pearson_corr(img_avg, gamma)
        distance_weight = v_film_weight - img_weight

        img_mask = torch.ones_like(v_film_weight)
        v_film_mask = torch.ones_like(v_film_weight)
        v_film_mask = v_film_mask + distance_weight
        img_mask = img_mask - distance_weight

        img_mask = img_mask.view(batch_size, -1, 1, 1).expand_as(img)
        v_film_mask = v_film_mask.view(batch_size, -1, 1, 1).expand_as(v_film)
        v_add = torch.mul(img, img_mask) + torch.mul(v_film, v_film_mask)
        v_final = F.normalize(v_add, p=2, dim=1)
        return v_final


class W_Concat_two(nn.Module):
    def __init__(self, dim=1):
        super(W_Concat_two, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.zeros(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.00001

    def forward(self, x):
        weight = torch.tanh(self.weight) + 1
        x_cat = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x_cat, self.dim)


class W_Add(nn.Module):
    def __init__(self, num_of_in):
        super(W_Add, self).__init__()
        self.num_of_in = num_of_in
        self.weight = nn.Parameter(torch.zeros(num_of_in, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        weight = F.tanh(self.weight) + 1.0/self.num_of_in
        x_weighted = [feat * weight for feat, weight in zip(x, weight)]
        x_sum = torch.sum(torch.stack(x_weighted), dim=0)
        return x_sum


class Maxpool2_Add(nn.Module):
    def __init__(self):
        super(Maxpool2_Add, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.maxpool(x)


class W_Add_Tanh(nn.Module):
    def __init__(self, num_of_in, w):
        super(W_Add_Tanh, self).__init__()
        self.num_of_in = num_of_in
        self.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        weight = F.tanh(self.weight) + 1.0
        x_weighted = [feat * weight for feat, weight in zip(x, weight)]
        x_sum = torch.sum(torch.stack(x_weighted), dim=0)
        return x_sum


class Ripple_Attention_Linear(nn.Module):
    def __init__(self, c1, c2, text_dim, num_of_center, sigma_w, i):
        super(Ripple_Attention_Linear, self).__init__()
        self.flang_layer_num = i
        self.num_of_center = num_of_center
        self.w = sigma_w
        self.emb_size = int((text_dim + num_of_center * 2) / 2)
        self.mapping_lang = nn.Sequential(
            nn.Linear(text_dim, self.emb_size),
            nn.Linear(self.emb_size, num_of_center * 2)
        )
        self.mapping_visu = Conv(c1, c2, 1, 1)
        self.gamma = Conv(num_of_center, num_of_center, 1, 1)
        self.beta = Conv(num_of_center, num_of_center, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.mapping_fusion = torch.nn.Sequential(
            Conv(c1, c1, 1),
            Conv(c1, c1, 3),
            nn.Conv2d(c1, c2, 1)
        )
        self.add_weight = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        img = x[0]
        flang = x[1][1][self.flang_layer_num]  # (32, 20, 768)
        word_mask = x[1][2]
        batch_size, height = img.shape[0], img.shape[2]

        map_visu = torch.tanh(self.mapping_visu(img))
        flang_avg = get_avg_flang(flang, word_mask)  # (32, 768)
        map_flang = self.sigmoid(self.mapping_lang(flang_avg))  # (32, num_of_center * 2)
        maps = map_flang.view(batch_size, self.num_of_center, 2)

        sigma = self.w * height
        y = (maps[:, :, 0] * height).int().unsqueeze(-1).unsqueeze(-1)
        x = (maps[:, :, 1] * height).int().unsqueeze(-1).unsqueeze(-1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=maps.device),
            torch.arange(height, device=maps.device),
            indexing='ij'
        )
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)

        weights = torch.exp(-((grid_y - y) ** 2 + (grid_x - x) ** 2) / (2 * sigma ** 2))

        gamma = torch.tanh(self.gamma(weights))
        beta = torch.tanh(self.beta(weights))
        g = torch.mean(gamma, dim=1, keepdim=True)
        b, _ = torch.max(beta, dim=1, keepdim=True)
        g = g.expand_as(map_visu)
        b = b.expand_as(map_visu)

        v_film = F.normalize((g * map_visu + b), p=2, dim=1)
        v_film = self.mapping_fusion(v_film)

        add_w = torch.tanh(self.add_weight)
        v_add = (1 + add_w) * img + (1 - add_w) * v_film
        v_final = F.normalize(v_add, p=2, dim=1)

        return v_final


class Laplace_RA(nn.Module):
    def __init__(self, c1, c2, text_dim, num_of_center, sigma_w, i):
        super(Laplace_RA, self).__init__()
        self.flang_layer_num = i
        self.num_of_center = num_of_center
        self.w = sigma_w
        self.emb_size = int((text_dim + num_of_center * 2) / 2)
        self.mapping_lang = nn.Sequential(
            nn.Linear(text_dim, self.emb_size),
            nn.Linear(self.emb_size, num_of_center * 2)
        )
        self.mapping_visu = Conv(c1, c2, 1, 1)
        self.gamma = Conv(num_of_center, num_of_center, 1, 1)
        self.beta = Conv(num_of_center, num_of_center, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.mapping_fusion = torch.nn.Sequential(
            Conv(c1, c1, 1),
            Conv(c1, c1, 3),
            nn.Conv2d(c1, c2, 1)
        )
        self.add_weight = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        img = x[0]
        flang = x[1][1][self.flang_layer_num]  # (32, 20, 768)
        word_mask = x[1][2]
        batch_size, height = img.shape[0], img.shape[2]

        map_visu = torch.tanh(self.mapping_visu(img))
        flang_avg = get_avg_flang(flang, word_mask)  # (32, 768)
        map_flang = self.sigmoid(self.mapping_lang(flang_avg))  # (32, num_of_center * 2)
        maps = map_flang.view(batch_size, self.num_of_center, 2)

        sigma = self.w * height
        y = (maps[:, :, 0] * height).int().unsqueeze(-1).unsqueeze(-1)
        x = (maps[:, :, 1] * height).int().unsqueeze(-1).unsqueeze(-1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=maps.device),
            torch.arange(height, device=maps.device),
            indexing='ij'
        )
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)

        dist = torch.abs(grid_y - y) + torch.abs(grid_x - x)
        weights = torch.exp(-dist / (sigma + 1e-8))

        gamma = torch.tanh(self.gamma(weights))
        beta = torch.tanh(self.beta(weights))
        g = torch.mean(gamma, dim=1, keepdim=True)
        b, _ = torch.max(beta, dim=1, keepdim=True)
        g = g.expand_as(map_visu)
        b = b.expand_as(map_visu)

        v_film = F.normalize((g * map_visu + b), p=2, dim=1)
        v_film = self.mapping_fusion(v_film)

        add_w = torch.tanh(self.add_weight)
        v_add = (1 + add_w) * img + (1 - add_w) * v_film
        v_final = F.normalize(v_add, p=2, dim=1)

        return v_final


class Inverse_Distance_RA(nn.Module):
    def __init__(self, c1, c2, text_dim, num_of_center, sigma_w, i):
        super(Inverse_Distance_RA, self).__init__()
        self.flang_layer_num = i
        self.num_of_center = num_of_center
        self.w = sigma_w
        self.emb_size = int((text_dim + num_of_center * 2) / 2)
        self.mapping_lang = nn.Sequential(
            nn.Linear(text_dim, self.emb_size),
            nn.Linear(self.emb_size, num_of_center * 2)
        )
        self.mapping_visu = Conv(c1, c2, 1, 1)
        self.gamma = Conv(num_of_center, num_of_center, 1, 1)
        self.beta = Conv(num_of_center, num_of_center, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.mapping_fusion = torch.nn.Sequential(
            Conv(c1, c1, 1),
            Conv(c1, c1, 3),
            nn.Conv2d(c1, c2, 1)
        )
        self.add_weight = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        img = x[0]
        flang = x[1][1][self.flang_layer_num]  # (32, 20, 768)
        word_mask = x[1][2]
        batch_size, height = img.shape[0], img.shape[2]

        map_visu = torch.tanh(self.mapping_visu(img))
        flang_avg = get_avg_flang(flang, word_mask)  # (32, 768)
        map_flang = self.sigmoid(self.mapping_lang(flang_avg))  # (32, num_of_center * 2)
        maps = map_flang.view(batch_size, self.num_of_center, 2)

        # sigma = self.w * height
        y = (maps[:, :, 0] * height).int().unsqueeze(-1).unsqueeze(-1)
        x = (maps[:, :, 1] * height).int().unsqueeze(-1).unsqueeze(-1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=maps.device),
            torch.arange(height, device=maps.device),
            indexing='ij'
        )
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)

        dist_squared = (grid_y - y) ** 2 + (grid_x - x) ** 2
        dist_squared = dist_squared / (torch.mean(dist_squared.float()) + 1e-8)
        weights = 1 / (1 + dist_squared)

        gamma = torch.tanh(self.gamma(weights))
        beta = torch.tanh(self.beta(weights))
        g = torch.mean(gamma, dim=1, keepdim=True)
        b, _ = torch.max(beta, dim=1, keepdim=True)
        g = g.expand_as(map_visu)
        b = b.expand_as(map_visu)

        v_film = F.normalize((g * map_visu + b), p=2, dim=1)
        v_film = self.mapping_fusion(v_film)

        add_w = torch.tanh(self.add_weight)
        v_add = (1 + add_w) * img + (1 - add_w) * v_film
        v_final = F.normalize(v_add, p=2, dim=1)

        return v_final


class Softmax_RA(nn.Module):
    def __init__(self, c1, c2, text_dim, num_of_center, sigma_w, i):
        super(Softmax_RA, self).__init__()
        self.flang_layer_num = i
        self.num_of_center = num_of_center
        self.w = sigma_w
        self.emb_size = int((text_dim + num_of_center * 2) / 2)
        self.mapping_lang = nn.Sequential(
            nn.Linear(text_dim, self.emb_size),
            nn.Linear(self.emb_size, num_of_center * 2)
        )
        self.mapping_visu = Conv(c1, c2, 1, 1)
        self.gamma = Conv(num_of_center, num_of_center, 1, 1)
        self.beta = Conv(num_of_center, num_of_center, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.mapping_fusion = torch.nn.Sequential(
            Conv(c1, c1, 1),
            Conv(c1, c1, 3),
            nn.Conv2d(c1, c2, 1)
        )
        self.add_weight = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        img = x[0]
        flang = x[1][1][self.flang_layer_num]  # (32, 20, 768)
        word_mask = x[1][2]
        batch_size, height = img.shape[0], img.shape[2]

        map_visu = torch.tanh(self.mapping_visu(img))
        flang_avg = get_avg_flang(flang, word_mask)  # (32, 768)
        map_flang = self.sigmoid(self.mapping_lang(flang_avg))  # (32, num_of_center * 2)
        maps = map_flang.view(batch_size, self.num_of_center, 2)

        # sigma = self.w * height
        y = (maps[:, :, 0] * height).int().unsqueeze(-1).unsqueeze(-1)
        x = (maps[:, :, 1] * height).int().unsqueeze(-1).unsqueeze(-1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=maps.device),
            torch.arange(height, device=maps.device),
            indexing='ij'
        )
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)

        dist_squared = (grid_y - y) ** 2 + (grid_x - x) ** 2
        dist_squared = dist_squared / (torch.mean(dist_squared.float()) + 1e-8)
        weights = torch.softmax(-dist_squared.view(batch_size, self.num_of_center, -1), dim=-1)
        weights = weights.view(batch_size, self.num_of_center, height, height)

        gamma = torch.tanh(self.gamma(weights))
        beta = torch.tanh(self.beta(weights))
        g = torch.mean(gamma, dim=1, keepdim=True)
        b, _ = torch.max(beta, dim=1, keepdim=True)
        g = g.expand_as(map_visu)
        b = b.expand_as(map_visu)

        v_film = F.normalize((g * map_visu + b), p=2, dim=1)
        v_film = self.mapping_fusion(v_film)

        add_w = torch.tanh(self.add_weight)
        v_add = (1 + add_w) * img + (1 - add_w) * v_film
        v_final = F.normalize(v_add, p=2, dim=1)

        return v_final


class Maha_RA(nn.Module):
    def __init__(self, c1, c2, text_dim, num_of_center, sigma_w, i):
        super(Maha_RA, self).__init__()
        self.flang_layer_num = i
        self.num_of_center = num_of_center
        self.w = sigma_w
        self.emb_size = int((text_dim + num_of_center * 2) / 2)
        self.mapping_lang = nn.Sequential(
            nn.Linear(text_dim, self.emb_size),
            nn.Linear(self.emb_size, num_of_center * 2)
        )
        self.mapping_visu = Conv(c1, c2, 1, 1)
        self.gamma = Conv(num_of_center, num_of_center, 1, 1)
        self.beta = Conv(num_of_center, num_of_center, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.mapping_fusion = torch.nn.Sequential(
            Conv(c1, c1, 1),
            Conv(c1, c1, 3),
            nn.Conv2d(c1, c2, 1)
        )
        self.add_weight = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.get_sigma = nn.Conv2d(c1, num_of_center, 3, 1)

    def forward(self, x):
        img = x[0]
        flang = x[1][1][self.flang_layer_num]  # (32, 20, 768)
        word_mask = x[1][2]
        batch_size, height = img.shape[0], img.shape[2]

        map_visu = torch.tanh(self.mapping_visu(img))
        flang_avg = get_avg_flang(flang, word_mask)  # (32, 768)
        map_flang = self.sigmoid(self.mapping_lang(flang_avg))  # (32, num_of_center * 2)
        maps = map_flang.view(batch_size, self.num_of_center, 2)

        sigma_x = self.w * height
        sigma_weight = self.sigmoid(torch.mean(self.get_sigma(img), dim=(2, 3), keepdim=True))
        sigma_y = sigma_x * sigma_weight

        y = (maps[:, :, 0] * height).int().unsqueeze(-1).unsqueeze(-1)
        x = (maps[:, :, 1] * height).int().unsqueeze(-1).unsqueeze(-1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=maps.device),
            torch.arange(height, device=maps.device),
            indexing='ij'
        )
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)

        weights = torch.exp(
            -(
                ((grid_y - y) ** 2) / (2 * sigma_y ** 2) +
                ((grid_x - x) ** 2) / (2 * sigma_x ** 2)
            )
        )

        gamma = torch.tanh(self.gamma(weights))
        beta = torch.tanh(self.beta(weights))
        g = torch.mean(gamma, dim=1, keepdim=True)
        b, _ = torch.max(beta, dim=1, keepdim=True)
        g = g.expand_as(map_visu)
        b = b.expand_as(map_visu)

        v_film = F.normalize((g * map_visu + b), p=2, dim=1)
        v_film = self.mapping_fusion(v_film)

        add_w = torch.tanh(self.add_weight)
        v_add = (1 + add_w) * img + (1 - add_w) * v_film
        v_final = F.normalize(v_add, p=2, dim=1)

        return v_final
