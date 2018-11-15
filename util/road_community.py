# -*- coding: utf-8 -*-

from util.const import *
import json
import numpy as np
from util.coord_transform_util import wgs84_to_bd09
import os
import matplotlib.pyplot as plt


def load_roads(city):
    city_geo_path = GEO_PATH % (city, city)
    with open(city_geo_path, 'r') as f:
        roads = json.load(f)

    return roads


def grid_connect_statistic(block, filter_type=False, save=True, size=500):
    """
    get city block grid neighborhood information
    :param block: city block
    :param filter_type: whether filter road type
    :param save: weather save the data or not
    :param size: grid size
    :return:
    """
    lat_steps, lng_steps = block.lat_steps, block.lng_steps
    roads = load_roads(block.city.value)

    city_neighbors = np.zeros((lat_steps * lng_steps, lat_steps * lng_steps))

    for road in roads['features']:
        geo_type = road['properties']['type']

        if filter_type and (geo_type not in FILTER_STREET_TYPE):
            continue

        bd_coord = [wgs84_to_bd09(*coord) for coord in road['geometry']['coordinates']]
        road_neighbors = set([])

        for coord in bd_coord:
            if (block.left_lower.lng <= coord[0] <= block.right_upper.lng) and (
                            block.left_lower.lat <= coord[1] <= block.right_upper.lat):
                row = int(math.floor((coord[1] - block.left_lower.lat) / block.height))
                col = int(math.floor((coord[0] - block.left_lower.lng) / block.width))
                index = row * lng_steps + col
                road_neighbors.add(index)

        road_neighbors = list(road_neighbors)
        for i in range(len(road_neighbors)):
            for j in range(i + 1, len(road_neighbors)):
                city_neighbors[road_neighbors[i], road_neighbors[j]] += 1
                city_neighbors[road_neighbors[j], road_neighbors[i]] += 1

    print('City %s, gird size: %s, connect count: %s' % (block.city.value, pow((lat_steps * lng_steps), 2),
                                                         (city_neighbors > 0).sum()))

    if save:
        np.save(ROAD_SAVE_PATH % (block.city.value, size), city_neighbors)


def plot_grid_connect(block, path):
    if os.path.exists(path):
        city_neighbors = np.load(path)
        size = city_neighbors.shape[0]

        plt.figure(figsize=(8, 8))
        for i in range(size):
            for j in range(i + 1, size):
                if city_neighbors[i, j] > 0:
                    x1, y1 = i // block.lng_steps, i % block.lng_steps
                    x2, y2 = j // block.lng_steps, j % block.lng_steps
                    plt.plot([x1, x2], [y1, y2], linewidth=city_neighbors[i, j] / 10.0)
        plt.show()


if __name__ == '__main__':
    _, _, sh_block = get_sh_range(width=0.005, height=0.005)
    grid_connect_statistic(sh_block)

    _, _, bj_block = get_bj_range(width=0.005, height=0.005)
    grid_connect_statistic(bj_block)

    _, _, nb_block = get_nb_range(width=0.005, height=0.005)
    grid_connect_statistic(nb_block)
