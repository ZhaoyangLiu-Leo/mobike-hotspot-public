# -*- coding: utf-8 -*-


from shapely.geometry import LineString
from util.coord_transform_util import distance
from util.const import *
import datetime
import multiprocessing
from road_match.road_util import *
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import time


"""
Description:
    Match mobikes with the road segments. We split the road segments according to the OpenStreetMap roads data.
    The average length of road segments are about 0.5km.
Author: Zhaoyang Liu
Created time: 2017-11-10
"""

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

logger = logging.getLogger(__name__)

DIS_THRESHOLD = 0.005


def coord_distance(coord1, coord2):
    lng1, lat1 = coord1
    lng2, lat2 = coord2
    return distance(lng1, lat1, lng2, lat2)


def get_road_unit(city_road_df):
    """
    获取最细粒度路段信息
    :param city_road_df:
    :return:
    """
    road_segments = []
    for index, road_item in city_road_df.iterrows():
        coordinates = road_item['geometry'].coords
        logger.info('Match %s road, road name %s' % (index, road_item['name']))
        for j in range(len(coordinates) - 1):
            road_segment = road_item.copy()
            road_segment['geometry'] = LineString([coordinates[j], coordinates[j + 1]])
            road_segment['sub_road'] = j
            road_segment['distance'] = coord_distance(coordinates[j], coordinates[j + 1])
            road_segments.append(road_segment)

    return road_segments


def get_road_segments(city_block, bound=True, filter_type=None, output_path=None):
    """
    抽取路段信息
    :param city_block:
    :param bound:
    :param filter_type:
    :param output_path:
    :return:
    """
    city_name = city_block.city.value
    city_geo_path = GEO_PATH % (city_name, city_name)
    city_road_df = gpd.read_file(city_geo_path)
    logger.info('Road Size: %s' % city_road_df.shape[0])

    if filter_type is not None:
        city_road_df = city_road_df[city_road_df['type'].isin(filter_type)]
        city_road_df.reset_index(inplace=True)
        logger.info('Filter road size: %s' % city_road_df.shape[0])

    if bound:
        city_road_df, road_segment_bound = bound_road(city_road_df, city_block)
        logger.info('Bound road size: %s' % city_road_df.shape[0])

    road_segments = get_road_unit(city_road_df)

    new_city_road_df = gpd.GeoDataFrame(road_segments, columns=list(city_road_df) + ['sub_road', 'distance'])
    new_city_road_df.reset_index(inplace=True)
    new_city_road_df.set_geometry('geometry')

    if output_path is not None:
        if os.path.exists(output_path):
            os.remove(output_path)
        new_city_road_df.to_file(output_path, driver='GeoJSON', encoding='utf-8')


def construct_merge_segment(segment_item, coords, dis, merge_id):
    """
    构造合并路段后的Geo Series
    :return:
    """
    merge_item = segment_item.copy()
    coords.append(segment_item['geometry'].coords[1])
    merge_item['geometry'] = LineString(coords)
    merge_item['distance'] = dis
    merge_item['merge_id'] = merge_id

    return merge_item


def merge_road_segments(input_path, output_path, distance_threshold=0.5):
    """
    合并基础路段成为长度更大的路段
    :param input_path:
    :param output_path:
    :param distance_threshold:
    :return:
    """
    segment_df = gpd.read_file(input_path)
    # segment_df.sort_values(by=['id', 'sub_road'], ascending=True, inplace=True)
    logger.info('Load road segment size: %s' % segment_df.shape[0])

    segment_df['merge_id'] = 0
    road_ids = segment_df['id'].unique()
    merge_segments = []

    for r_id in road_ids:
        road_df = segment_df[segment_df['id'] == r_id]
        merge_id = 0
        dis, coords, merge_segment_ids = 0, list(), list()

        for index, segment_item in road_df.iterrows():
            dis += segment_item['distance']
            coords.append(segment_item['geometry'].coords[0])
            merge_segment_ids.append(segment_item['sub_road'])

            if dis >= distance_threshold:
                merge_item = construct_merge_segment(segment_item, coords, dis, merge_id)
                merge_segments.append(merge_item)

                logger.info('Merge road %s, merge sub segment %s, dis %s' % (
                    segment_item['name'], merge_segment_ids, dis))
                merge_id += 1
                dis, coords, merge_segment_ids = 0, list(), list()

        # 合并剩余长度不足够的路段
        if dis > 0:
            merge_item = construct_merge_segment(road_df.iloc[-1, :], coords, dis, merge_id)
            merge_segments.append(merge_item)

    merge_df = gpd.GeoDataFrame(merge_segments, columns=list(segment_df) + ['merge_id'])
    logger.info('Merge road size: %s' % merge_df.shape[0])
    merge_df.set_geometry('geometry')

    if os.path.exists(output_path):
        os.remove(output_path)
    merge_df.to_file(output_path, driver='GeoJSON', encoding='utf-8')


def match_bike_with_segments(road_segment_df, road_segment_bound, location, expand=0.0, mode='unique'):
    """
    match single mobike location with road segment
    :param road_segment_df: road segment geo pandas DataFrame
    :param road_segment_bound: road segment bound
    :param location:
    :param expand: road bound expand size when matching mobike
    :param mode: unique/dump whether mobikes are matched with the only road segment
    :return:
    """
    lng, lat = list(location.coords)[0]
    candidate = (road_segment_bound.minx - expand <= lng) & (lng <= road_segment_bound.maxx + expand) & (
        road_segment_bound.miny - expand <= lat) & (lat <= road_segment_bound.maxy + expand)

    local_segment_df = road_segment_df[candidate]
    min_road = []

    if len(local_segment_df) > 0:
        dis = local_segment_df.distance(location)
        if mode == 'unique':
            min_dis = np.min(dis)
            if min_dis < DIS_THRESHOLD:
                min_road = [np.argmin(dis)]
                logger.info('Mobike location %s match %s road, distance %s' % (
                    location, local_segment_df.loc[min_road[0]]['name'], min_dis))
        else:
            min_road = dis[dis < DIS_THRESHOLD].index.values
            logger.info('Mobike location %s match %s road' % (
                location, local_segment_df.loc[min_road]['name']))

    return min_road


def bound_road(road_df, city_block):
    road_bound = road_df.bounds
    filter_index = (road_bound['minx'] >= city_block.left_lower.lng) & (
        road_bound['miny'] >= city_block.left_lower.lat) & (
                       road_bound['maxx'] <= city_block.right_upper.lng) & (
                       road_bound['maxy'] <= city_block.right_upper.lat
                   )
    road_df = road_df[filter_index].copy()
    road_bound = road_bound[filter_index].copy()

    road_df = road_df.reset_index(drop=True)
    road_bound = road_bound.reset_index(drop=True)

    return road_df, road_bound


def filter_road_length(road_segment_df, length_threshold):
    filter_index = (road_segment_df['distance'] > length_threshold[0]) & (
        road_segment_df['distance'] < length_threshold[1])
    road_segment_df = road_segment_df[filter_index]
    return road_segment_df.reset_index(drop=True)


def load_road_segment(road_segment_path, city_block, is_filter_type=True, is_filter_bound=True, is_filter_length=True,
                      bound_area_threshold=50, length_threshold=(0.3, 1)):
    """
    load road segment and compute the road bound
    :param road_segment_path:
    :param city_block:
    :param is_filter_type: filter road type, exclude the noise road type effect
    :param is_filter_bound: filter bound area, exclude the over size road bound
    :param is_filter_length: filter road length, exclude the over long or short road segment
    :param bound_area_threshold:
    :param length_threshold
    :return:
    """
    road_segment_df = gpd.read_file(road_segment_path)
    road_segment_df['mobike'] = 0
    road_segment_df['hot'] = 0
    logger.info('Road segment Size: %s' % road_segment_df.shape[0])

    if is_filter_type:
        road_segment_df = road_segment_df[road_segment_df['type'].isin(FILTER_STREET_TYPE)]
        logger.info('Filter type segment Size: %s' % road_segment_df.shape[0])

    if is_filter_length:
        road_segment_df = filter_road_length(road_segment_df, length_threshold)
        logger.info('Final length Size: %s' % road_segment_df.shape[0])

    road_segment_df, road_segment_bound = bound_road(road_segment_df, city_block)

    if is_filter_bound:
        road_bound_area = road_segment_bound.apply(
            lambda row: (row['maxx'] - row['minx']) * 1000 * (row['maxy'] - row['miny']) * 1000, axis=1)
        filter_index = road_bound_area < bound_area_threshold
        road_segment_df = road_segment_df[filter_index]
        road_segment_bound = road_segment_bound[filter_index]

        road_segment_df.reset_index(drop=True, inplace=True)
        road_segment_bound.reset_index(drop=True, inplace=True)

    logger.info('Final segment Size: %s' % road_segment_df.shape[0])

    return road_segment_df, road_segment_bound, road_segment_df.centroid


def mobikes_road_single_match(road_segment, road_segment_bound, city, spider_time, output_dir,
                              expand=0.0005, mode='unique'):
    """
    distribute the mobike to the only one road segment
    :return:
    """
    road_segment_df = road_segment.copy()

    road_segment_df['mobike'] = 0
    road_segment_df['hot'] = 0

    mobike_series = get_epoch_mobike(city, spider_time)

    match_count = 0
    for location in mobike_series.values:
        road_ids = match_bike_with_segments(road_segment_df, road_segment_bound, location, expand, mode)
        if len(road_ids) > 0:
            road_segment_df.loc[road_ids, 'mobike'] += 1
            match_count += 1

    logger.warning('Total mobike %s, match mobike %s' % (len(mobike_series), match_count))

    output_path = os.path.join(output_dir, spider_time.strftime('%H_%M_%S') + '.geojson')
    if not os.path.exists(output_path):
        road_segment_df.to_file(output_path, driver='GeoJSON', encoding='utf-8')


def mobikes_road_single_match_with_dump(road_segment, road_centroid, city, spider_time, output_dir, size):
    road_segment_df = road_segment.copy()
    mobike_df = get_epoch_mobike(city, spider_time, False)

    road_segment_df['mobike'] = road_centroid.apply(lambda row: match_bike_with_centroid(
        row['lng'], row['lat'], size, mobike_df), axis=1)
    road_segment_df['hot'] = 0
    logger.info('Total mobike %s, match junction %s' % (len(mobike_df), (road_segment_df['mobike'] > 0).sum()))

    output_path = os.path.join(output_dir, spider_time.strftime('%H_%M_%S') + '.geojson')
    if not os.path.exists(output_path):
        road_segment_df.to_file(output_path, driver='GeoJSON', encoding='utf-8')


def mobikes_road_match_with_density(road_segment, road_centroid, city, spider_time, output_dir,
                                    grid_search=False, bandwidth=0.04, metric='euclidean'):
    road_segment_df = road_segment.copy()
    mobike_df = get_epoch_mobike(city, spider_time, False)

    # use grid search cross-validation to optimize the bandwidth
    if grid_search:
        params = {'bandwidth': (0.001, 0.01, 0.1, 1, 10)}
        kde = GridSearchCV(
            KernelDensity(metric=metric, kernel='gaussian', algorithm='ball_tree'), params,
            n_jobs=2, verbose=2)
    else:
        kde = KernelDensity(bandwidth=bandwidth, metric=metric, kernel='gaussian', algorithm='ball_tree')

    kde_start_time = time.time()
    kde.fit(mobike_df[['lng', 'lat']].values)
    if grid_search:
        logger.info('Best parameters:{}'.format(kde.best_params_))
        kde = kde.best_estimator_

    road_segment_df['mobike'] = np.exp(kde.score_samples(road_centroid[['lng', 'lat']].values))

    logger.info('Max density score %s, mean density score %s, estimate time: %s' % (
        road_segment_df['mobike'].max(), road_segment_df['mobike'].mean(), time.time() - kde_start_time))

    output_path = os.path.join(output_dir, spider_time.strftime('%H_%M_%S') + '.geojson')
    if not os.path.exists(output_path):
        road_segment_df.to_file(output_path, driver='GeoJSON', encoding='utf-8')


def match_city_mobike_with_road(road_map_path, day_range, city_block, output_dir,
                                filter_type=True,
                                is_bound_road=True, is_weekday=True, parallel=True, match_mode='bound',
                                boundary_mode='dump',
                                size=0.0025, expand=0.0005,
                                kernel_grid_search=False, kernel_bandwidth=0.04):
    """
    match mobike with road segment during range data
    :return:
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pool = multiprocessing.Pool(processes=4)

    road_segment_df, road_segment_bound, road_segment_centroid = load_road_segment(road_map_path, city_block,
                                                                                   is_bound_road, filter_type)
    road_centroid_df = get_centroid_location(road_segment_centroid)

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

                if match_mode == 'bound':
                    if parallel:
                        pool.apply_async(
                            mobikes_road_single_match,
                            args=(
                                road_segment_df, road_segment_bound,
                                city, spider_time, day_output_dir,
                                expand, boundary_mode
                            )
                        )
                    else:
                        mobikes_road_single_match(
                            road_segment_df, road_segment_bound,
                            city, spider_time, day_output_dir, expand=expand, mode=boundary_mode
                        )
                elif match_mode == 'kernel':
                    if parallel:
                        pool.apply_async(
                            mobikes_road_match_with_density,
                            args=(road_segment_df, road_centroid_df, city, spider_time,
                                  day_output_dir, kernel_grid_search, kernel_bandwidth)
                        )
                    else:
                        mobikes_road_match_with_density(road_segment_df, road_centroid_df, city, spider_time,
                                                        day_output_dir, grid_search=kernel_grid_search,
                                                        bandwidth=kernel_bandwidth)
                else:
                    if parallel:
                        pool.apply_async(
                            mobikes_road_single_match_with_dump,
                            args=(road_segment_df, road_centroid_df,
                                  city, spider_time, day_output_dir, size)
                        )
                    else:
                        mobikes_road_single_match_with_dump(road_segment_df, road_centroid_df, city,
                                                            spider_time, day_output_dir, size)

        start_date += day_delta

    if parallel:
        pool.close()
        pool.join()


if __name__ == '__main__':
    run_city = City.SH
    half_road_length = 0.0025
    match_dis = int(half_road_length * 2 * 1000 * 100)
    logger.info('Run city: {}'.format(run_city.value))
    # get_road_segments(city_block_dict[run_city], output_path='./road_map/road_segment/%s_road_segment.geojson' %
    #                                                          run_city.value)

    # merge_road_segments(
    #     input_path='./road_map/road_segment/%s_road_segment.geojson' % run_city.value,
    #     output_path='./road_map/merge_segment_%s/%s.geojson' % (match_dis, run_city.value),
    #     distance_threshold=distance_threshold
    # )

    output_dir = '../data/road/match_res_bound_unique_new_type_bike'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    match_city_mobike_with_road(
        road_map_path='../data/road/road_map/merge_segment_%s/%s.geojson' % (match_dis, run_city.value),
        day_range=('2017-07-19', '2017-07-26'),
        city_block=city_block_dict[run_city],
        output_dir=output_dir + '/%s_%s' % (run_city.value, match_dis),
        filter_type=True,
        is_bound_road=True,
        is_weekday=False,
        parallel=True,
        match_mode='bound',
        kernel_grid_search=False,
        kernel_bandwidth=0.001,
        boundary_mode='unique',
        size=half_road_length,
        expand=0.0005
    )
