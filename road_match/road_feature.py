# -*- coding: utf-8 -*-

import geopandas as gpd
import numpy as np
from util.db_helper import connect_to_db, query_area_poi_vector, query_area_light_value, close_db
import pandas as pd
from util.const import POI_FIRST_CLASS_EN, city_block_dict, City, NEW_FEATURE_NAMES, WINDOW_SIZE_FEATURES
from util.coord_transform_util import wgs84_to_bd09, distance
import logging
from road_match.road_util import get_centroid_location

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

logger = logging.getLogger(__name__)


def load_road(road_path, expand=0.0005):
    road_df = gpd.read_file(road_path)
    # road_df = (road_df[road_df['hot'] == 1])
    # road_df = road_df.reset_index(drop=True)

    # expand road boundary
    road_bound = road_df.bounds
    road_bound[['minx', 'miny']] -= expand
    road_bound[['maxx', 'maxy']] += expand

    road_centroid = road_df.centroid

    road_bound_area = road_bound.apply(
        lambda row: (row['maxx'] - row['minx']) * 1000 * (row['maxy'] - row['miny']) * 1000, axis=1)

    return road_df, road_bound, road_centroid, road_bound_area


def get_poi_entropy(row):
    """
    compute road vector entropy
    :param row:
    :return:
    """
    pois = row[0:-1]
    sum_poi = row[-1]

    pois = pois[pois > 0]
    if len(pois) == 0:
        return 0
    else:
        prob_pois = pois / float(sum_poi)
        log_prob_pois = np.log(prob_pois)
        return -1 * np.sum(prob_pois * log_prob_pois)


def match_poi(city, road_bound, road_bound_area, road_centroid, mode='centroid', size=0.0025, to_vector=False):
    """
    match poi vector with road segment
    :param city:
    :param road_bound:
    :param road_bound_area
    :param road_centroid:
    :param mode: feature extraction mode (bound / centroid)
    :param size: valid when mode == centroid
    :param to_vector whether transform poi count to poi vector
    :return:
    """
    poi_features = list()
    conn = connect_to_db()

    # transform wgs84 coordinates to bd09 coordinates, and match Baidu POI
    if mode == 'bound':
        for index, row in road_bound.iterrows():
            min_x, min_y = wgs84_to_bd09(row['minx'], row['miny'])
            max_x, max_y = wgs84_to_bd09(row['maxx'], row['maxy'])

            poi_vector = query_area_poi_vector(conn,
                                               left_lower_lng=min_x,
                                               left_lower_lat=min_y,
                                               right_upper_lng=max_x,
                                               right_upper_lat=max_y,
                                               city=city)
            if road_bound_area is not None:
                poi_vector = [x / road_bound_area[index] for x in poi_vector]
            logger.info('Match road %s, poi vector %s' % (index, poi_vector))
            poi_features.append(poi_vector)
    else:
        for index, row in road_centroid.iterrows():
            lng, lat = row['lng'], row['lat']

            min_x, min_y = wgs84_to_bd09(lng - size, lat - size)
            max_x, max_y = wgs84_to_bd09(lng + size, lat + size)

            poi_vector = query_area_poi_vector(conn,
                                               left_lower_lng=min_x,
                                               left_lower_lat=min_y,
                                               right_upper_lng=max_x,
                                               right_upper_lat=max_y,
                                               city=city)
            logger.info('Match road %s, poi vector %s' % (index, poi_vector))
            poi_features.append(poi_vector)

    close_db(conn)

    poi_df = pd.DataFrame(np.array(poi_features), columns=POI_FIRST_CLASS_EN)
    poi_df['num_pois'] = poi_df.apply(lambda x: np.sum(x), axis=1, raw=True)
    poi_df['poi_entropy'] = poi_df.apply(get_poi_entropy, axis=1, raw=True)

    if to_vector:
        for col in POI_FIRST_CLASS_EN:
            poi_df[col] /= poi_df['num_pois'] + 1.0

    return poi_df


def fill_nan_light(light_series):
    light_series[light_series == 0] = np.nan
    light_series.fillna(method='ffill', inplace=True)
    light_series.fillna(method='bfill', inplace=True)
    return light_series


def match_light(city, road_bound, road_centroid, mode='centroid', size=0.0025, expand=0.0005):
    """
    match light intensity with road segment
    :param city:
    :param road_bound:
    :param road_centroid:
    :param mode: feature extraction mode (bound / centroid)
    :param size: valid when mode == centroid
    :param expand road bound expand size
    :param fillna: whether fill nan value of light
    :return:
    """
    light_values = list()
    conn = connect_to_db()

    if mode == 'bound':
        for index, row in road_bound.iterrows():
            light_intensity = query_area_light_value(conn,
                                                     left_lower_lat=row['miny'] - expand,
                                                     left_lower_lng=row['minx'] - expand,
                                                     right_upper_lat=row['maxy'] + expand,
                                                     right_upper_lng=row['maxx'] + expand,
                                                     city=city)
            light_values.append(light_intensity)
            logger.info('Match road %s, light intensity %s' % (index, light_intensity))
    else:
        for index, row in road_centroid.iterrows():
            lng, lat = row['lng'], row['lat']
            light_intensity = query_area_light_value(conn,
                                                     left_lower_lat=lat - size,
                                                     left_lower_lng=lng - size,
                                                     right_upper_lat=lat + size,
                                                     right_upper_lng=lng + size,
                                                     city=city)
            light_values.append(light_intensity)
            logger.info('Match road %s, light intensity %s' % (index, light_intensity))

    close_db(conn)
    light_series = pd.Series(np.array(light_values), name='light')
    return light_series


def match_light_dis(city, road_centroid):
    light_df = pd.read_csv('../data/meta_data/%s_light_cluster.csv' % city)
    logger.info('Match light distance feature')

    def min_light_distance(lng, lat):
        dis = light_df.apply(lambda row: distance(row['lng'], row['lat'], lng, lat), axis=1)
        return np.min(dis)

    light_dis = pd.Series(road_centroid.apply(
        lambda x: min_light_distance(x['lng'], x['lat']), axis=1), name='light_dis')
    return light_dis


def match_subway(city, road_centroid):
    subway_df = pd.read_csv('../data/meta_data/%s_subway_wgs.csv' % city)
    logger.info('Match subway distance feature')

    def min_subway_distance(lng, lat):
        dis = subway_df.apply(lambda row: distance(row['lng'], row['lat'], lng, lat), axis=1)
        return np.min(dis)

    subway_dis = pd.Series(road_centroid.apply(
        lambda x: min_subway_distance(x['lng'], x['lat']), axis=1), name='subway_dis')
    return subway_dis


def match_business_center(city, road_centroid):
    center_df = pd.read_csv('../data/meta_data/%s_city_center.csv' % city)
    logger.info('Match business center feature')

    def min_shop_center_distance(lng, lat):
        dis = center_df.apply(lambda row: distance(row['lng'], row['lat'], lng, lat), axis=1)
        return np.min(dis)

    def min_shop_center_level(lng, lat):
        dis = center_df.apply(lambda row: distance(row['lng'], row['lat'], lng, lat), axis=1)
        return center_df.iat[np.argmin(dis), -1]

    center_dis = pd.Series(road_centroid.apply(
        lambda x: min_shop_center_distance(x['lng'], x['lat']), axis=1), name='business_dis')
    center_level = pd.Series(road_centroid.apply(
        lambda x: min_shop_center_level(x['lng'], x['lat']), axis=1), name='business_level')

    center_feature_df = pd.concat([center_dis, center_level], axis=1)
    return center_feature_df


def window_size_data(road_feature_df, window_size_columns=WINDOW_SIZE_FEATURES):
    for feature in window_size_columns:
        threshold = np.percentile(road_feature_df[feature], 99)
        road_feature_df.loc[road_feature_df[feature] > threshold, feature] = threshold


def road_based_feature_extract(city_block, road_path, save_path, mode='centroid', size=0.0025, expand=0.0005,
                               light_fillna=False, window_size=True):
    """
    feature extraction from poi, light, etc.
    :param city_block:
    :param road_path:
    :param save_path:
    :param mode:
    :param size: when mode='centroid', extract area size -> (size * 2, size * 2) square
    :param expand: when mode='bound', extract bound area and expand size in four direction
    :param light_fillna: whether fill nan value of light intensity
    :param window_size: whether window size the dimension of the data
    :return:
    """
    city_name = city_block.city.value
    road_df, road_bound, road_centroid, road_bound_area = load_road(road_path, expand=expand)

    road_feature_df = pd.DataFrame(road_df[['mobike', 'hot', 'id', 'merge_id']])
    road_location_df = get_centroid_location(road_centroid)

    road_type_dummy = pd.get_dummies(road_df['type'], prefix_sep='road_')
    poi_df = match_poi(city_name, road_bound, None, road_location_df, mode=mode, size=size)

    light_series = match_light(city_name, road_bound, road_location_df, mode=mode, size=size)
    light_dis_series = match_light_dis(city_name, road_location_df)

    subway_series = match_subway(city_name, road_location_df)
    business_df = match_business_center(city_name, road_location_df)

    road_feature_df = pd.concat(
        [road_location_df, road_feature_df, road_type_dummy, poi_df, light_series,
         light_dis_series, subway_series, business_df], axis=1)

    if light_fillna:
        road_feature_df.sort_values(by=['lng', 'lat'], inplace=True)
        road_light = fill_nan_light(road_feature_df['light'].copy())
        road_feature_df['light'] = road_light

    if window_size:
        window_size_data(road_feature_df)

    road_feature_df.to_csv(save_path, index=False)


def junction_based_feature_extract(city_block, road_path, save_path, size=0.0025):
    """
    feature extraction from poi, light, etc.
    :param city_block:
    :param road_path:
    :param save_path:
    :param size: extract area size -> (size * 2, size * 2) square
    :return:
    """
    city_name = city_block.city.value
    road_feature_df = pd.read_csv(road_path)
    road_feature_df.rename_axis({"count": "junction_count"}, axis="columns", inplace=True)

    road_location_df = road_feature_df[['lng', 'lat']]

    poi_df = match_poi(city_name, None, None, road_location_df, mode='centroid', size=size, to_vector=False)

    light_series = match_light(city_name, None, road_location_df, mode='centroid', size=size)
    light_dis_series = match_light_dis(city_name, road_location_df)

    subway_series = match_subway(city_name, road_location_df)
    business_df = match_business_center(city_name, road_location_df)

    road_feature_df = pd.concat(
        [road_feature_df, poi_df, light_series,
         light_dis_series, subway_series, business_df], axis=1)

    road_feature_df.to_csv(save_path, index=False)


def road_neighbor_based_feature_extraction(feature_path, save_path, distance_threshold=1, window=5,
                                           preprocess=True):
    feature_df = pd.read_csv(feature_path)
    feature_df.sort_values(by=['id', 'merge_id'], inplace=True)

    if preprocess:
        city_feature = feature_df.loc[:, 'food': 'business_level']
        city_feature['num_pois'] = city_feature['num_pois'].apply(np.log1p)
        city_feature['light'] = city_feature['light'].apply(np.log1p)
        city_feature['subway_dis'] = city_feature['subway_dis'].apply(np.log1p)
        city_feature = (city_feature - city_feature.mean()) / city_feature.std()
        feature_df.loc[:, 'food': 'business_level'] = city_feature.values

    new_feature_dfs = []
    for index, row in feature_df.iterrows():
        org_row = row.copy()
        org_row = org_row[['lng', 'lat', 'mobike', 'hot', 'id', 'merge_id']]
        lng, lat = org_row['lng'], org_row['lat']

        filter_feature_dfs = feature_df[
            (feature_df['lng'] > lng - 0.02) &
            (feature_df['lng'] < lng + 0.02) &
            (feature_df['lat'] > lat - 0.02) &
            (feature_df['lat'] < lat + 0.02)
        ]
        distances = filter_feature_dfs.apply(lambda x: distance(lng, lat, x[0], x[1]), axis=1, raw=True)
        distances = distances[distances < distance_threshold]
        # assert len(distances) >= window
        if len(distances) < window:
            logger.info('{} can not find {} nearest roads'.format(str(org_row), window))
            continue

        distances.sort_values(ascending=True, inplace=True)
        distances = distances.head(window)

        new_features = list()
        new_features.append(org_row)
        for i, r_id in enumerate(distances.index):
            merge_row = feature_df.loc[r_id, NEW_FEATURE_NAMES].copy()
            merge_row.rename_axis(lambda x: (x + '_{}').format(i), inplace=True)
            new_features.append(merge_row)
        new_feature_row = pd.concat(new_features)

        new_feature_dfs.append(new_feature_row)

    new_feature_df = pd.concat(new_feature_dfs, axis=1).T
    new_feature_df.to_csv(save_path, index=False)
    return new_feature_df


if __name__ == '__main__':
    extract_mode = 'bound'
    extract_size = 0.005
    block = city_block_dict[City.NB]

    # road_based_feature_extract(
    #     block, '../data/road/hot_statistic_%s_unique_new_type_kernel/%s_500_week.geojson' % (
    #         extract_mode, block.city.value),
    #     save_path='../data/road/train_%s_unique_new_type_kernel/%s_500_week.csv' % (
    #         extract_mode, block.city.value),
    #     mode=extract_mode, size=extract_size, expand=0.0005, light_fillna=True, window_size=False
    # )

    road_neighbor_based_feature_extraction(
        feature_path='../data/road/train_bound_unique_new_type_kernel_merge_new2/%s_500_week.csv' % block.city.value,
        save_path='../data/road/train_bound_unique_new_type_kernel_merge_cnn_new2/%s_500_week.csv' % block.city.value
    )

    # junction_based_feature_extract(block, './data/junction/hot_statistic/%s_500.csv' % block.city.value,
    #                                save_path='./data/junction/train/%s_500_500.csv' % block.city.value,
    #                                size=extract_size)
    # feature_extract(city_block_dict[City.BJ], './data/hot_res/bj_500.geojson', save_path='data/train/bj_500.csv')
    # feature_extract(city_block_dict[City.NB], './data/hot_res/nb_500.geojson', save_path='data/train/nb_500.csv')

