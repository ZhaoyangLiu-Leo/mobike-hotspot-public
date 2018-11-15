# -*- coding: utf-8 -*-

# try:
from xml.etree import cElementTree as ET
# except ImportError, e:
#     from xml.etree import ElementTree as ET

import pandas as pd
import logging
import os
import multiprocessing
import datetime
from road_match.road_util import make_output_dirs, get_mobike_day_spider_times, get_epoch_mobike
from road_match.road_util import match_bike_with_centroid
from util.const import City, city_block_dict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

logger = logging.getLogger(__name__)


def mobikes_junction_single_match(road_junction, city, spider_time, output_dir, size):
    road_junction = road_junction.copy()
    mobike_location = get_epoch_mobike(city, spider_time, transform_geo_series=False)

    road_junction['mobike'] = road_junction.apply(lambda row: match_bike_with_centroid(
        row['lng'], row['lat'], size, mobike_location), axis=1)
    road_junction['hot'] = 0
    logger.info('Total mobike %s, match junction %s' % (len(mobike_location), (road_junction['mobike'] > 0).sum()))

    output_path = os.path.join(output_dir, spider_time.strftime('%H_%M_%S') + '.csv')
    if not os.path.exists(output_path):
        road_junction.to_csv(output_path, index=False)


def extract_intersections(osm, save_path=None):
    """
    This function takes an osm file as an input. It then goes through each xml
    element and searches for nodes that are shared by two or more ways.
    Parameter:
    :param osm: An xml file that contains OpenStreetMap's map information
    :param save_path: save intersection path
    """
    tree = ET.parse(osm)
    counter = {}
    for child in tree.iter(tag='way'):
        for item in child:
            if item.tag == 'nd':
                nd_ref = item.attrib['ref']
                if nd_ref not in counter:
                    counter[nd_ref] = 0
                counter[nd_ref] += 1

    # Find nodes that are shared with more than one way, which might correspond to intersections
    intersections = filter(lambda x: counter[x] > 1,  counter)
    logger.info('Intersection node match finish, find %s point' % len(intersections))

    # Extract intersection coordinates, You can plot the result using this url. http://www.darrinward.com/lat-long/
    intersection_coordinates = []
    for elem in tree.iter(tag='node'):
        node_id = elem.attrib['id']

        if node_id in intersections:
            coordinate = (node_id, float(elem.attrib['lon']), float(elem.attrib['lat']),
                          counter[node_id])
            logger.info(str(coordinate))
            intersection_coordinates.append(coordinate)

    if save_path is not None:
        intersection_df = pd.DataFrame(intersection_coordinates, columns=['node_id', 'lng', 'lat', 'count'])
        intersection_df.drop_duplicates(subset=['node_id'], inplace=True)
        intersection_df.to_csv(save_path, index=False)
    return intersection_coordinates


def load_road_junction(junction_path, city_block, bound=True):
    road_junction = pd.read_csv(junction_path)

    if bound:
        filter_index = (road_junction['lng'] >= city_block.left_lower.lng) & (
            road_junction['lat'] >= city_block.left_lower.lat) & (
                           road_junction['lng'] <= city_block.right_upper.lng) & (
                           road_junction['lat'] <= city_block.right_upper.lat)
        road_junction = road_junction[filter_index].copy()
        road_junction = road_junction.reset_index(drop=True)

    return road_junction


def match_city_mobike_with_junction(junction_path, day_range, city_block, output_dir, size=0.0025,
                                    is_bound_road=True, is_weekday=True, parallel=True):
    """
    match mobike with road segment during range data
    :return:
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pool = multiprocessing.Pool(processes=4)

    road_junction = load_road_junction(junction_path, city_block, is_bound_road)

    start_date = datetime.datetime.strptime(day_range[0], '%Y-%m-%d')
    end_date = datetime.datetime.strptime(day_range[1], '%Y-%m-%d')
    day_delta = datetime.timedelta(days=1)
    city = city_block.city.value + '_all'

    make_output_dirs(output_dir, start_date, end_date, day_delta, is_weekday)

    while start_date <= end_date:
        if is_weekday and 6 <= start_date.weekday() + 1 <= 7:
            start_date += day_delta
            continue

        spider_day = start_date.strftime('%Y-%m-%d')
        spider_times = get_mobike_day_spider_times(city, spider_day)

        day_output_dir = os.path.join(output_dir, spider_day)

        hour_flag = [True] * 24
        for spider_time in spider_times:
            if hour_flag[spider_time.hour]:
                hour_flag[spider_time.hour] = False

                if parallel:
                    pool.apply_async(
                        mobikes_junction_single_match,
                        args=(
                            road_junction,
                            city, spider_time, day_output_dir, size
                        )
                    )
                else:
                    mobikes_junction_single_match(
                        road_junction,
                        city, spider_time, day_output_dir, size
                    )

        start_date += day_delta

    if parallel:
        pool.close()
        pool.join()


if __name__ == '__main__':
    # extract_intersections('/Users/towardsun/Downloads/ningbo_china.osm',
    #                       save_path='../data/road_junction/nb.csv')
    run_city = City.SH
    match_size = 0.0025

    match_city_mobike_with_junction(junction_path='../data/junction/road_junction/%s.csv' % run_city.value,
                                    day_range=('2017-07-16', '2017-07-16'),
                                    city_block=city_block_dict[run_city],
                                    output_dir='../data/junction/match_res/%s_%s' % (run_city.value,
                                                                                     int(match_size * 2 * 1000 * 100)),
                                    size=match_size,
                                    is_bound_road=True,
                                    is_weekday=False,
                                    parallel=True)
