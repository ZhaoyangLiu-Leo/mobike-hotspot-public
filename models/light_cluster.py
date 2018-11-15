# -*- coding: utf-8 -*-

from util.db_helper import connect_to_db, close_db, query_city_light
import logging
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from util.const import city_block_dict, City


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

logger = logging.getLogger(__name__)


def load_city_light(city_block):
    """
    match light intensity with road segment
    :param city_block: city description object
    :return:
    """
    conn = connect_to_db()

    light_df = query_city_light(conn, city_block)
    logger.info('City light intensity point size %s' % light_df.shape[0])
    close_db(conn)

    return light_df


def light_cluster(city_block, top_threshold=30, n_cluster=30, output=True):
    light_df = load_city_light(city_block)

    u_threshold = np.percentile(light_df.light, 100 - top_threshold)
    light_df = light_df[light_df.light > u_threshold]
    cluster_model = KMeans(init='k-means++', n_clusters=n_cluster).fit(light_df[['lng', 'lat']].values)

    if output:
        cluster_df = pd.DataFrame(cluster_model.cluster_centers_, columns=['lng', 'lat'])
        cluster_df.to_csv('../data/meta_data/%s_light_cluster.csv' % city_block.city.value, index=False)
    return light_df, cluster_model


if __name__ == '__main__':
    light_cluster(city_block_dict[City.NB])
