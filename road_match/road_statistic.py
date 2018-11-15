# -*- coding: utf-8 -*-

import os
import geopandas as gpd
import numpy as np
import logging
import pandas as pd
import datetime


"""
Hotspots roads jaccard similarity evaluation.
Author: Zhaoyang Liu
Created time: 2017-11-10
"""


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

logger = logging.getLogger(__name__)


class JaccardMatchRes:

    def __init__(self, start_time, end_time, jaccard_val):
        self.start_time = start_time
        self.end_time = end_time
        self.jaccard_val = jaccard_val

    def __str__(self):
        return '"%s", %.4f' % (self.start_time[:19], self.jaccard_val)


def jaccard_similarity(hot1, hot2):
    hot1_set = set(hot1)
    hot2_set = set(hot2)
    int_size = len(hot1_set.intersection(hot2_set))
    union_size = len(hot1_set.union(hot2_set))
    return float(int_size) / union_size


def get_hot_road_ids(file_path, hot_percentile):
    """
    get mobike top hot_percentile road ids
    :return:
    """
    if file_path.endswith('geojson'):
        road_df = gpd.read_file(file_path)
    else:
        road_df = pd.read_csv(file_path)
    hot_value = np.percentile(road_df['mobike'], hot_percentile)
    road_df = road_df[road_df['mobike'] >= hot_value]

    logger.info('Match file: %s, max value: %s' % (file_path, np.max(road_df['mobike'])))
    logger.info('Match file: %s, hot threshold: %s' % (file_path, hot_value))
    logger.info('Match file: %s, hot road count: %s' % (file_path, road_df.shape[0]))
    return road_df.index.values


def get_match_files(match_res_path):
    days = os.listdir(match_res_path)
    if '.DS_Store' in days:
        days.remove('.DS_Store')

    match_file_list = list()

    for day in days:
        day_dir = os.path.join(match_res_path, day)
        match_file_list += [(day, x) for x in os.listdir(day_dir) if x != '.DS_Store']

    match_file_list = sorted(match_file_list, key=lambda item: datetime.datetime.strptime(
        item[0] + ' ' + item[1].split('.')[0], '%Y-%m-%d %H_%M_%S'))
    return match_file_list


def get_mobike_road(match_res_path, file_paths, hot_mode, hot_percentile, hot_count, save_path, mode='mean'):
    """
    statistic time period mobike average number and mark hot road segment
    :return:
    """
    assert len(file_paths) > 0

    first_path = os.path.join(match_res_path, *file_paths[0])
    if first_path.endswith('geojson'):
        first_road_df = gpd.read_file(first_path)
    else:
        first_road_df = pd.read_csv(first_path)

    for i, (day, hour) in enumerate(file_paths):
        if i > 0:
            road_path = os.path.join(match_res_path, day, hour)

            if road_path.endswith('geojson'):
                road_df = gpd.read_file(road_path)
            else:
                road_df = pd.read_csv(road_path)
            if mode == 'mean':
                first_road_df['mobike'] = first_road_df['mobike'].values + road_df['mobike'].values
            else:
                first_road_df['mobike'] = np.maximum(first_road_df['mobike'].values, road_df['mobike'].values)

    if mode == 'mean':
        first_road_df['mobike'] /= float(len(file_paths))
    if hot_mode == 'percentile':
        hot_value = np.percentile(first_road_df['mobike'], hot_percentile)
    else:
        hot_value = first_road_df['mobike'].sort_values().iloc[-hot_count]
    first_road_df['hot'] = 0
    first_road_df.loc[first_road_df['mobike'] >= hot_value, 'hot'] = 1

    if save_path is not None and not os.path.exists(save_path):
        if save_path.endswith('geojson'):
            first_road_df.to_file(save_path, driver='GeoJSON', encoding='utf-8')
        else:
            first_road_df.to_csv(save_path, index=False)

    return first_road_df, hot_value


def get_group_hot_road_ids(match_res_path, file_paths, hot_mode, hot_percentile, hot_count, filter_hot=True):
    """
    average period time mobikes, and find hot roads
    :return:
    """
    first_road_df, hot_value = get_mobike_road(match_res_path, file_paths, hot_mode, hot_percentile, hot_count, None)

    if filter_hot:
        first_road_df = first_road_df[first_road_df['hot'] == 1]

    start_day, start_hour = file_paths[0][0], file_paths[0][1].split('.')[0]
    end_day, end_hour = file_paths[-1][0], file_paths[-1][1].split('.')[0]
    time_range = '%s %s-%s %s' % (start_day, start_hour, end_day, end_hour)
    logger.info('Match time: %s, max value: %s' % (time_range, np.max(first_road_df['mobike'])))
    logger.info('Match time: %s, hot threshold: %s' % (time_range, hot_value))
    logger.info('Match time: %s, hot road count: %s' % (time_range, first_road_df.shape[0]))
    return time_range, first_road_df.index.values


def road_hot_statistic(match_res_path, time_unit=1, hot_mode='count', hot_percentile=99, hot_count=100,
                       save_path=None):
    """
    hours unit hot road jaccard similarity statistic
    :param match_res_path:
    :param time_unit: hour unit, eg: 1 3, 6 hours
    :param hot_mode: measure the hotspots by count or percentile
    :param hot_percentile: hot percentile threshold
    :param hot_count: hotspots count
    :param save_path: jaccarrd measure save_path
    :return:
    """
    match_file_list = get_match_files(match_res_path)

    jaccard_res = list()
    group_count = len(match_file_list) // time_unit

    last_time, last_hot_ids = get_group_hot_road_ids(
        match_res_path, match_file_list[:time_unit], hot_mode=hot_mode,
        hot_percentile=hot_percentile, hot_count=hot_count
    )

    for i in range(1, group_count - 1):
        cur_time, cur_hot_ids = get_group_hot_road_ids(
            match_res_path,
            match_file_list[i * time_unit: (i + 1) * time_unit],
            hot_mode=hot_mode,
            hot_percentile=hot_percentile,
            hot_count=hot_count
        )
        jaccard_val = jaccard_similarity(last_hot_ids, cur_hot_ids)
        jaccard_res.append(JaccardMatchRes(last_time, cur_time, jaccard_val))

        logger.info('%s, %s, %s' % (jaccard_res[-1].start_time, jaccard_res[-1].end_time, jaccard_res[-1].jaccard_val))
        last_hot_ids = cur_hot_ids
        last_time = cur_time

    for res in jaccard_res:
        print(str(res))

    if save_path is not None:
        jaccard_list = [x.__dict__ for x in jaccard_res]
        jaccard_df = pd.DataFrame(jaccard_list)
        jaccard_df.to_csv(save_path, index=None)


def road_hot_statistic_by_day(match_res_path, hot_mode, hot_percentile=99, hot_count=100,
                              save_path=None):
    """
    day unit hot road jaccard similarity statistic
    :param match_res_path:
    :param hot_mode
    :param hot_percentile:
    :param hot_count
    :param save_path
    :return:
    """
    days = os.listdir(match_res_path)
    if '.DS_Store' in days:
        days.remove('.DS_Store')

    jaccard_res = list()
    if len(days) > 2:
        first_day_file_list = [(days[0], x)
                               for x in os.listdir(os.path.join(match_res_path, days[0]))
                               if x != '.DS_Store']
        last_time, last_hot_ids = get_group_hot_road_ids(
            match_res_path, first_day_file_list, hot_mode=hot_mode,
            hot_percentile=hot_percentile, hot_count=hot_count)

        for i in range(1, len(days)):
            day_dir = os.path.join(match_res_path, days[i])
            day_file_list = [(days[i], x) for x in os.listdir(day_dir) if x != '.DS_Store']
            cur_time, cur_hot_ids = get_group_hot_road_ids(
                match_res_path, day_file_list, hot_mode=hot_mode,
                hot_percentile=hot_percentile, hot_count=hot_count
            )

            jaccard_val = jaccard_similarity(last_hot_ids, cur_hot_ids)
            jaccard_res.append(JaccardMatchRes(last_time, cur_time, jaccard_val))

            logger.info(str(jaccard_res[-1]))
            last_hot_ids = cur_hot_ids
            last_time = cur_time

    for res in jaccard_res:
        print(str(res))

    if save_path is not None:
        jaccard_list = [x.__dict__ for x in jaccard_res]
        jaccard_df = pd.DataFrame(jaccard_list)
        jaccard_df.to_csv(save_path, index=None)


def road_statistic(match_res_path, save_path, hot_mode, hot_percentile=99, hot_count=100, statistic_mode='mean'):
    """
    average every time step mobike number, and mark the hot road
    :return:
    """
    logger.info('Statistic data version: ' + match_res_path)
    match_file_list = get_match_files(match_res_path)
    get_mobike_road(match_res_path, match_file_list, hot_mode, hot_percentile, hot_count, save_path, mode=statistic_mode)


def remove_bom(file):
    f = open(file, 'r')
    if f.read(3) == b'\xef\xbb\xbf':
        f_body = f.read()
        f.close()
        with open(file, 'w') as f:
            f.write(f_body)


def copy_files(input_dir):
    match_file_list = get_match_files(input_dir)
    for path in match_file_list:
        file_path = os.path.join(input_dir, *path)
        remove_bom(file_path)
        # with open(file_path, 'r') as f:
        #     file_content = f.read()
        #     if file_content[:3] == codecs.BOM_UTF8:
        #         file_content = file_content[:3]
        # with open(file_path, 'w') as f:
        #     f.write(file_content)
        logger.info('Finish file:' + file_path)

if __name__ == '__main__':
    city_name = 'sh'
    road_size = 500
    hot_count = 100
    time_unit = 6
    logger.info(city_name)

    # road_hot_statistic(
    #     '../data/road/match_res_bound_unique_new_type_bike/%s_%s' % (city_name, road_size),
    #     time_unit=time_unit, hot_mode='count', hot_percentile=99, hot_count=hot_count,
    #     save_path='../results/%s_%s_%s_jaccard.csv' % (city_name, hot_count, time_unit)
    # )
    road_hot_statistic_by_day('../data/road/match_res_bound_unique_new_type_bike/%s_%s' % (city_name, road_size),
                              hot_mode='count', hot_percentile=99, hot_count=hot_count,
                              save_path='../results/%s_%s_day_jaccard.csv' % (city_name, hot_count))
    # copy_files('../data/road/match_res_dump/%s_1000' % city_name)

    # road_statistic(
    #     '../data/road/match_res_bound_unique_new_type_kernel/%s_%s' % (city_name, road_size),
    #     save_path='../data/road/hot_statistic_bound_unique_new_type_kernel/%s_%s_week.geojson' % (city_name, road_size),
    #     statistic_mode='mean'
    # )
    # road_statistic('./match_res/sh_500', save_path='./data/hot_res/sh_500.geojson')
    # road_statistic('./match_res/nb_500', save_path='./data/hot_res/nb_500.geojson')
