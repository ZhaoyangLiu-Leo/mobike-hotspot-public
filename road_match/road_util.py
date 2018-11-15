# -*- coding: utf-8 -*-

import os
from util.db_helper import connect_to_db, query_mobike_day_spider_times, close_db, query_mobike_by_spider_time
import logging
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from util.coord_transform_util import gcj02_to_wgs84, bd09_to_wgs84
import numpy as np


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

logger = logging.getLogger(__name__)


def mobike_lng_lat_transform(mobike_data, coord='bd09'):
    if coord == 'gcj':
        wgs84_location = [gcj02_to_wgs84(x[0], x[1]) for x in mobike_data[:, -2:]]
    else:
        wgs84_location = [bd09_to_wgs84(x[0], x[1]) for x in mobike_data[:, -2:]]
    mobike_data[:, -2:] = np.array(wgs84_location)


def make_output_dirs(output_dir, start_date, end_date, day_delta, is_weekday):
    """
    make output directories
    :return:
    """
    while start_date <= end_date:
        if is_weekday and 6 <= start_date.weekday() + 1 <= 7:
            start_date += day_delta
            continue

        spider_day = start_date.strftime('%Y-%m-%d')

        day_output_dir = os.path.join(output_dir, spider_day)
        if not os.path.exists(day_output_dir):
            os.mkdir(day_output_dir)
        start_date += day_delta


def get_mobike_day_spider_times(city, spider_day):
    """
    get mobike spider times during the given spider day
    """
    conn = connect_to_db()
    spider_times = query_mobike_day_spider_times(conn, city, spider_day)
    close_db(conn)
    return spider_times


def get_epoch_mobike(city_name, spider_time, transform_geo_series=True):
    """
    get mobike geo series location during the spider time
    :param city_name:
    :param spider_time:
    :param transform_geo_series:
    :return:
    """
    conn = connect_to_db()
    mobike_data = query_mobike_by_spider_time(conn, city_name, spider_time)
    logger.info('Spider time %s, mobike data %s' % (spider_time, len(mobike_data)))
    close_db(conn)
    mobike_lng_lat_transform(mobike_data, coord='gcj')

    if transform_geo_series:
        mobike_series = gpd.GeoSeries([Point(x, y) for x, y in zip(mobike_data[:, -2], mobike_data[:, -1])])
        return mobike_series
    else:
        return pd.DataFrame(mobike_data[:, -2:], columns=['lng', 'lat'])


def match_bike_with_centroid(lng, lat, size, mobike_df):
    index = (mobike_df['lng'] >= lng - size) & (
        mobike_df['lng'] <= lng + size) & (
        mobike_df['lat'] >= lat - size) & (
        mobike_df['lat'] <= lat + size)

    match_count = len(mobike_df[index])
    logger.info('Road: (%s, %s), match %s mobike' % (lng, lat, match_count))
    return match_count


def get_centroid_location(road_centroid):
    lng = pd.Series(road_centroid.apply(lambda x: x.coords[0][0]), name='lng')
    lat = pd.Series(road_centroid.apply(lambda x: x.coords[0][1]), name='lat')

    return pd.concat([lng, lat], axis=1)

