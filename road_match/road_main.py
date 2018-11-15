# -*- coding: utf-8 -*-

from road_match.road_segment import match_city_mobike_with_road
from road_match.road_statistic import road_statistic
from road_match.road_feature import road_based_feature_extract
from util.const import *
import os


if __name__ == '__main__':
    extract_size = 0.005
    match_dis = int(extract_size * 1000 * 100)
    block = city_block_dict[City.SH]
    city_name = block.city.value

    extract_mode = 'bound'
    for i, kernel_bandwidth in enumerate([0.1, 0.01, 0.005, 0.001, 0.0001]):
        if kernel_bandwidth > 0.0001:
            data_version = '%s_unique_new_type_kernel_euc_bandwidth_%s' % (extract_mode, i)

            match_dir = '../data/road/match_res_%s' % data_version
            if not os.path.exists(match_dir):
                os.mkdir(match_dir)

            match_city_mobike_with_road(
                road_map_path='../data/road/road_map/merge_segment_%s/%s.geojson' % (match_dis, city_name),
                day_range=('2017-07-17', '2017-07-21'),
                city_block=block,
                output_dir=os.path.join(match_dir, '%s_%s' % (city_name, match_dis)),
                filter_type=True,
                is_bound_road=True,
                is_weekday=False,
                parallel=True,
                match_mode='kernel',
                kernel_grid_search=False,
                kernel_bandwidth=kernel_bandwidth,
                boundary_mode='unique',
                size=extract_size/2,
                expand=0.0005
            )

            hot_path = '../data/road/hot_statistic_%s' % data_version
            if not os.path.exists(hot_path):
                os.mkdir(hot_path)
            road_statistic(
                '../data/road/match_res_%s/%s_%s' % (data_version, city_name, match_dis),
                save_path=os.path.join(hot_path, '%s_%s_week.geojson' % (city_name, match_dis)),
                hot_mode='count', hot_count=100 if city_name == 'nb' else 800,
                statistic_mode='mean'
            )

            feature_path = '../data/road/train_%s' % data_version
            if not os.path.exists(feature_path):
                os.mkdir(feature_path)

            road_based_feature_extract(
                block, os.path.join(hot_path, '%s_%s_week.geojson' % (city_name, match_dis)),
                save_path=os.path.join(feature_path, '%s_%s_week.csv' % (city_name, match_dis)),
                mode=extract_mode, size=extract_size, expand=0.0005, light_fillna=True, window_size=False
            )
