# -*- coding: utf-8 -*-

import pymysql
import pandas as pd
from util.const import *


def connect_to_db(db_name='mobike'):
    return pymysql.connect(host='localhost', user='root', password='199457', db=db_name, charset='utf8')


def close_db(connection):
    connection.close()


def query_mobike_by_day_night(connection, city, day):
    table_name = 'mobike_' + city
    sql = "select distinct(bikeId), bikeType, lng, lat from %s where date(time)='%s'" % (table_name, day)
    df = pd.read_sql(sql, connection)
    return df.values


def query_mobike_day_range_spider_times(connection, city, start_day, end_day):
    table_name = 'mobike_' + city
    sql = "select distinct(spider_time) from %s where date(spider_time) between '%s' and '%s' order by spider_time" % (
        table_name, start_day, end_day)

    with connection.cursor() as cursor:
        cursor.execute(sql)
        records = cursor.fetchall()
        spider_times = [x[0] for x in records]

    return spider_times


def query_mobike_day_spider_times(connection, city, day, hour=None):
    table_name = 'mobike_' + city
    sql = "select distinct(spider_time) from %s where date(spider_time)='%s'" % (table_name, day)

    if hour:
        sql += ' and hour(spider_time)=%d' % (hour,)
    sql += ' order by spider_time'
    with connection.cursor() as cursor:
        cursor.execute(sql)
        records = cursor.fetchall()
        spider_times = [x[0] for x in records]

    return spider_times


def query_mobike_by_spider_time(connection, city, spider_time):
    table_name = 'mobike_' + city
    sql = "select distinct(bikeId), bikeType, lng, lat from %s where spider_time='%s'" % (table_name, spider_time)
    df = pd.read_sql(sql, connection)
    return df.values


def query_area_poi_vector(connection, left_lower_lat, left_lower_lng, right_upper_lat, right_upper_lng,
                          city, level=1):
    """
    查询方格内的POI向量
    """
    table_name = 'poi_' + city
    poi_class, poi_class_db_name = (POI_SECOND_CLASS, 'second_class') if level == 2 else (POI_FIRST_CLASS,
                                                                                          'first_class')
    poi_vec = [0] * len(poi_class)
    sql = "select %s, count(distinct(uid)) from %s where lat between %f and %f " \
          "and lng between %f and %f group by %s;" % (poi_class_db_name,
                                                      table_name,
                                                      left_lower_lat, right_upper_lat,
                                                      left_lower_lng, right_upper_lng,
                                                      poi_class_db_name)
    with connection.cursor() as cursor:
        cursor.execute(sql)
        pois = cursor.fetchall()
        for item in pois:
            poi_vec[poi_class.index(item[0])] = item[1]
    return poi_vec


def query_city_light(connection, city_block):
    """
    查询城市mobike
    """
    table_name = 'light_' + city_block.city.value
    left_lower_lat, left_lower_lng = city_block.left_lower.lat, city_block.left_lower.lng
    right_upper_lat, right_upper_lng = city_block.right_upper.lat, city_block.right_upper.lng

    sql = "select lng, lat, light from %s where lat between %f and %f " \
          "and lng between %f and %f;" % (table_name,
                                          left_lower_lat, right_upper_lat,
                                          left_lower_lng, right_upper_lng)

    light_df = pd.read_sql(sql, connection)
    return light_df


def query_area_light_value(connection, left_lower_lat, left_lower_lng, right_upper_lat, right_upper_lng, city):
    """
    查询方格内灯光的亮度值
    """
    table_name = 'light_' + city
    sql = "select light from %s where lat between %f and %f " \
          "and lng between %f and %f;" % (table_name,
                                          left_lower_lat, right_upper_lat,
                                          left_lower_lng, right_upper_lng)
    light_val = 0
    count = 0
    with connection.cursor() as cursor:
        cursor.execute(sql)
        pois = cursor.fetchall()

        for item in pois:
            light_val += item[0]
            count += 1

    return float(light_val) / count if count else 0


def batch_insert_mobike(connection, mobikes, city, spider_time):
    table_name = 'mobike_' + city
    sql_pattern = "insert into " + table_name + " values('%s', '%s', %s, '%s', %s, %s, %s, %s, %s, '" \
                  + spider_time + "')"
    flag = True

    try:
        cursor = connection.cursor()
        for record in mobikes:
            sql = sql_pattern % tuple(record)
            cursor.execute(sql)
        cursor.close()
        connection.commit()
    except pymysql.Error as e:
        print(e)
        cursor.close()
        connection.rollback()
        flag = False
    return flag


if __name__ == '__main__':
    conn = connect_to_db('mobike')
    print(query_area_poi_vector(conn, 30.985000, 121.290000, 31.250000, 121.785000))
    close_db(conn)
