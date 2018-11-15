# -*- coding: utf-8 -*-

import json
import urllib
import math
import pandas as pd
import numpy as np


x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 扁率
EARTH_RADIUS = 6378.137


def distance(lng1, lat1, lng2, lat2):
    """
    计算经纬度之间的直线距离
    :param lng1: 坐标1经度
    :param lat1: 坐标1维度
    :param lng2: 坐标2经度
    :param lat2: 坐标2维度
    :return: 经纬度之间的距离
    """
    rad_lat1 = math.radians(lat1)
    rad_lat2 = math.radians(lat2)
    diff = rad_lat1 - rad_lat2
    b = math.radians(lng1) - math.radians(lng2)
    s = 2 * math.asin(
        math.sqrt(pow(math.sin(diff / 2), 2) + math.cos(rad_lat1) * math.cos(rad_lat2) * pow(math.sin(b / 2), 2)))
    s *= EARTH_RADIUS
    s = round(s * 10000) / 10000
    return s


class Geocoding(object):
    def __init__(self, api_key):
        self.api_key = api_key

    def geocode(self, address):
        """
        利用高德geocoding服务解析地址获取位置坐标
        :param address:需要解析的地址
        :return:
        """
        geocoding = {'s': 'rsv3',
                     'key': self.api_key,
                     'city': '全国',
                     'address': address}
        geocoding = urllib.urlencode(geocoding)
        ret = urllib.urlopen("%s?%s" % ("http://restapi.amap.com/v3/geocode/geo", geocoding))

        if ret.getcode() == 200:
            res = ret.read()
            json_obj = json.loads(res)
            if json_obj['status'] == '1' and int(json_obj['count']) >= 1:
                geocodes = json_obj['geocodes'][0]
                lng = float(geocodes.get('location').split(',')[0])
                lat = float(geocodes.get('location').split(',')[1])
                return [lng, lat]
            else:
                return None
        else:
            return None


def gcj02_to_bd09(lng, lat):
    """
    火星坐标系(GCJ-02)转百度坐标系(BD-09)
    谷歌、高德——>百度
    :param lng:火星坐标经度
    :param lat:火星坐标纬度
    :return:
    """
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * x_pi)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * x_pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return [bd_lng, bd_lat]


def bd09_to_gcj02(bd_lon, bd_lat):
    """
    百度坐标系(BD-09)转火星坐标系(GCJ-02)
    百度——>谷歌、高德
    :param bd_lat:百度坐标纬度
    :param bd_lon:百度坐标经度
    :return:转换后的坐标列表形式
    """
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gg_lng = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return [gg_lng, gg_lat]


def wgs84_to_gcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    if out_of_china(lng, lat):  # 判断是否在国内
        return lng, lat
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return lng, lat
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def bd09_to_wgs84(bd_lon, bd_lat):
    lon, lat = bd09_to_gcj02(bd_lon, bd_lat)
    return gcj02_to_wgs84(lon, lat)


def wgs84_to_bd09(lon, lat):
    lon, lat = wgs84_to_gcj02(lon, lat)
    return gcj02_to_bd09(lon, lat)


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return lng > 73.66 or lng < 135.05 or lat > 3.86 or lat < 53.55


def transform_subway_coordinate(in_path, out_path):
    df = pd.read_csv(in_path, header=0)
    wgs_location = [bd09_to_wgs84(x[0], x[1]) for x in df.iloc[:, 1:3].values]
    df.iloc[:, 1:3] = np.array(wgs_location)
    df = df[['name', 'lat', 'lng']]
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    # lng = 128.543
    # lat = 37.065
    # result1 = gcj02_to_bd09(lng, lat)
    # result2 = bd09_to_gcj02(lng, lat)
    # result3 = wgs84_to_gcj02(lng, lat)
    # result4 = gcj02_to_wgs84(lng, lat)
    # result5 = bd09_to_wgs84(lng, lat)
    # result6 = wgs84_to_bd09(lng, lat)
    #
    # g = Geocoding('API_KEY')  # 这里填写你的高德api的key
    # result7 = g.geocode('北京市朝阳区朝阳公园')
    # print result1, result2, result3, result4, result5, result6, result7

    # sh_left_lower = (121.284373, 30.980175)
    # sh_right_upper = (121.777546, 31.005189)
    #
    # print gcj02_to_bd09(*sh_left_lower)
    # print gcj02_to_bd09(*sh_right_upper)
    # bj_bd = (116.409921,39.940118)
    # bj_nlgx = (116.4032471180, 39.9331372740)
    # print gcj02_to_bd09(*bj_nlgx)
    # print wgs84_to_bd09(*bj_nlgx)

    # 测试OpenStreetMap 道路连续两点间的距离
    # road1 = (120.1613447218582, 30.278617416695226)
    # road2 = (120.16198954166936, 30.27866167314397)
    # road3 = (120.16357028478888, 30.278680783883203)
    #
    # print(distance(road1[0], road1[1], *road2))
    # print(distance(road2[0], road2[1], *road3))
    transform_subway_coordinate('../data/meta_data/nb_subway_bd.csv', '../data/meta_data/nb_subway_wgs.csv')
