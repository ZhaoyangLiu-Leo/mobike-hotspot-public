# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from util.const import GENERAL_FEATURE_NAMES, NEW_FEATURE_NAMES
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

logger = logging.getLogger(__name__)


def preprocess_data(city_df, percentile, down_sampling, down_ratio):
    hot_threshold = np.percentile(city_df.mobike, percentile)
    city_df['hot'] = 0
    city_df.loc[city_df.mobike >= hot_threshold, 'hot'] = 1
    logger.info('Percentile {}, mobike count {}'.format(percentile, hot_threshold))

    if down_sampling:
        hot_df = city_df[city_df.mobike >= hot_threshold].copy()
        un_hot_df = city_df[city_df.mobike < hot_threshold].copy()
        un_hot_df = un_hot_df.sample(len(hot_df) * down_ratio, random_state=0)
        city_df = pd.concat([hot_df, un_hot_df])

    return city_df


def load_city_pair_data(source_city, target_city, path_pattern, percent, task_type='reg', data_mode='org',
                        log_scale=False, road_type_feature=('cycleway', 'subway'), scaler='std'):
    source_df = pd.read_csv(path_pattern % source_city)
    target_df = pd.read_csv(path_pattern % target_city)
    source_df = source_df[list(target_df)]

    logger.info('Source city size {}'.format(source_df.shape))
    logger.info('Target city size {}'.format(target_df.shape))

    if data_mode == 'org':
        source_x, source_y = get_city_xy(city_df=source_df, percentile=percent,
                                         task_type=task_type, down_sampling=False,
                                         road_type_feature=road_type_feature)
        target_x, target_y = get_city_xy(city_df=target_df, percentile=percent,
                                         task_type=task_type, down_sampling=False,
                                         road_type_feature=road_type_feature)

        source_x = StandardScaler().fit_transform(source_x)
        target_x = StandardScaler().fit_transform(target_x)
    else:
        source_x, source_y = get_neighbor_city_xy(city_df=source_df, percentile=percent,
                                                  task_type=task_type, down_sampling=False,
                                                  merge_road_type=False)
        target_x, target_y = get_neighbor_city_xy(city_df=target_df, percentile=percent,
                                                  task_type=task_type,
                                                  down_sampling=False,
                                                  merge_road_type=False)

    if task_type == 'reg':
        if log_scale:
            source_y = np.round(np.log2(source_y + 1))
            target_y = np.round(np.log2(target_y + 1))
            source_y = source_y[:, np.newaxis]
            target_y = target_y[:, np.newaxis]
        else:
            if scaler == 'std':
                source_y = StandardScaler().fit_transform(source_y[:, np.newaxis])
                target_y = StandardScaler().fit_transform(target_y[:, np.newaxis])
            else:
                source_y = MinMaxScaler().fit_transform(source_y[:, np.newaxis])
                target_y = MinMaxScaler().fit_transform(target_y[:, np.newaxis])
            # source_y, target_y, y_scaler = scale_x(source_y, target_y, scaler=StandardScaler())
    else:
        source_y = source_y[:, np.newaxis]
        target_y = target_y[:, np.newaxis]

    return source_x, source_y, target_x, target_y


def get_city_xy(city_df, percentile, task_type='cls', down_sampling=True, down_ratio=2, merge_road_type=True,
                road_type_feature=('cycleway', 'unclassified')):
    city_df = preprocess_data(city_df, percentile, down_sampling, down_ratio)

    city_feature = city_df.loc[:, 'food': 'business_level']
    city_feature['num_pois'] = city_feature['num_pois'].apply(np.log1p)
    city_feature['light'] = city_feature['light'].apply(np.log1p)
    city_feature['subway_dis'] = city_feature['subway_dis'].apply(np.log1p)

    city_feature = (city_feature - city_feature.mean()) / city_feature.std()
    if merge_road_type:
        city_feature = pd.concat([city_df.loc[:, road_type_feature[0]: road_type_feature[1]], city_feature], axis=1)

    if task_type == 'cls':
        return city_feature.values, city_df.hot.values
    else:
        return city_feature.values, city_df.mobike.values


def get_neighbor_city_xy(city_df, percentile, task_type='cls', down_sampling=True, down_ratio=2, window=5,
                         merge_road_type=True):
    city_df = preprocess_data(city_df, percentile, down_sampling, down_ratio)

    feature_names = []
    for i in range(window):
        if merge_road_type:
            feature_names += [(x + '_{}').format(i) for x in NEW_FEATURE_NAMES]
        else:
            feature_names += [(x + '_{}').format(i) for x in GENERAL_FEATURE_NAMES]
    city_feature = city_df[feature_names]

    if task_type == 'cls':
        return city_feature.values.reshape((len(city_df), window, -1)), city_df.hot.values
    else:
        return city_feature.values.reshape((len(city_df), window, -1)), city_df.mobike.values

