# -*- coding: utf-8 -*-

from enum import Enum, unique
import math

POI_FIRST_CLASS = [u'美食', u'酒店', u'购物', u'生活服务', u'丽人', u'旅游景点', u'休闲娱乐', u'运动健身', u'教育培训',
                   u'文化传媒', u'医疗', u'汽车服务', u'交通设施', u'金融', u'房地产', u'公司企业', u'政府机构']

POI_FIRST_CLASS_EN = ['food', 'hotel', 'shopping', 'life_service', 'beauty', 'tourist', 'entertainment', 'sports',
                      'education', 'culture_media', 'medical', 'car_service', 'transportation', 'finance', 'estate',
                      'company', 'government']

POI_SECOND_CLASS = [u'住宅区', u'写字楼', u'宿舍', u'公司', u'农林园艺', u'厂矿', u'园区', u'便利店', u'商铺', u'家居建材', u'家电数码', u'购物中心',
                    u'超市', u'集市', u'停车场', u'公交车站', u'加油加气站', u'地铁站', u'收费站', u'服务区', u'桥', u'港口', u'火车站', u'长途汽车站',
                    u'飞机场', u'中学', u'亲子教育', u'图书馆', u'培训机构', u'小学', u'幼儿园', u'成人教育', u'特殊教育学校', u'留学中介机构', u'科技馆',
                    u'科研机构', u'高等院校', u'ATM', u'信用社', u'典当行', u'投资理财', u'银行', u'公寓式酒店', u'快捷酒店', u'星级酒店', u'美体', u'美发',
                    u'美容', u'美甲', u'公园', u'动物园', u'博物馆', u'教堂', u'文物古迹', u'植物园', u'水族馆', u'海滨浴场', u'游乐园', u'风景区',
                    u'中餐厅', u'咖啡厅', u'外国餐厅', u'小吃快餐店', u'茶座', u'蛋糕甜品店', u'酒吧', u'汽车检测场', u'汽车租赁', u'汽车维修', u'汽车美容',
                    u'汽车配件', u'汽车销售', u'公共厕所', u'公用事业', u'售票处', u'图文快印店', u'宠物服务', u'家政服务', u'彩票销售点', u'房产中介机构', u'报刊亭',
                    u'殡葬服务', u'洗衣店', u'照相馆', u'物流公司', u'维修点', u'通讯营业厅', u'邮局', u'展览馆', u'广播电视', u'文化宫', u'新闻出版', u'美术馆',
                    u'艺术团体', u'KTV', u'休闲广场', u'农家院', u'剧院', u'度假村', u'歌舞厅', u'洗浴按摩', u'游戏场所', u'电影院', u'网吧', u'专科医院',
                    u'体检机构', u'急救中心', u'疗养院', u'疾控中心', u'综合医院', u'药店', u'诊所', u'体育场馆', u'健身中心', u'极限运动场所', u'中央机构',
                    u'党派团体', u'公检法机构', u'各级政府', u'政治教育机构', u'涉外机构', u'福利机构', u'行政单位']

GEO_PATH = '/Users/towardsun/Documents/Paper/DataSet/OpenStreetMap/%s/%s_china_roads.geojson'

STREET_TYPE = [u'highway-bridleway', u'highway-cycleway', u'highway-footway', u'highway-living_street',
               u'highway-motorway', u'highway-motorway_link', u'highway-path',
               u'highway-pedestrian', u'highway-primary', u'highway-primary_link', u'highway-raceway',
               u'highway-residential', u'highway-road', u'highway-secondary', u'highway-secondary_link',
               u'highway-service', u'highway-steps', u'highway-tertiary', u'highway-tertiary_link',
               u'highway-track', u'highway-trunk', u'highway-trunk_link', u'highway-unclassified',
               u'man_made-pier',
               u'railway-disused', u'railway-funicular', u'railway-light_rail',
               u'railway-monorail', u'railway-narrow_gauge',
               u'railway-preserved', u'railway-rail',
               u'railway-subway', u'railway-tram']

TIME_UNIT = Enum('Time', ('Week', 'Day', 'Night', 'Hour'))

FILTER_STREET_TYPE = ['cycleway', 'footway', 'living_street', 'pedestrian',
                      'residential', 'road', 'subway', 'service']

ROAD_SAVE_PATH = '../data/grid/road_neighbor/%s_%s.npy'

FEATURE_NAMES = ['cycleway', 'footway', 'living_street', 'motorway', 'motorway_link', 'path', 'pedestrian', 'pier',
                 'primary', 'primary_link', 'rail', 'residential', 'road', 'secondary', 'secondary_link', 'service',
                 'steps', 'subway', 'tertiary', 'tertiary_link', 'track', 'trunk', 'trunk_link', 'unclassified', 'food',
                 'hotel', 'shopping', 'life_service', 'beauty', 'tourist', 'entertainment', 'sports', 'education',
                 'culture_media', 'medical', 'car_service', 'transportation', 'finance', 'estate', 'company',
                 'government', 'num_pois', 'poi_entropy', 'light', 'light_dis', 'subway_dis', 'business_dis',
                 'business_level']

NEW_FEATURE_NAMES = ['cycleway', 'footway', 'living_street', 'pedestrian', 'residential', 'subway', 'service'] + \
                    POI_FIRST_CLASS_EN + [
    'num_pois', 'poi_entropy', 'light', 'light_dis', 'subway_dis', 'business_dis', 'business_level']


GENERAL_FEATURE_NAMES = [
    'food', 'hotel', 'shopping', 'life_service', 'beauty', 'tourist', 'entertainment', 'sports',
    'education', 'culture_media', 'medical', 'car_service', 'transportation', 'finance', 'estate', 'company',
    'government', 'num_pois', 'poi_entropy', 'light', 'light_dis', 'subway_dis', 'business_dis',
    'business_level'
]

WINDOW_SIZE_FEATURES = [
    'food', 'hotel', 'shopping', 'life_service', 'beauty', 'tourist', 'entertainment', 'sports',
    'education', 'culture_media', 'medical', 'car_service', 'transportation', 'finance', 'estate', 'company',
    'government', 'num_pois', 'light', 'light_dis', 'subway_dis', 'business_dis'
]


@unique
class City(Enum):
    SH = 'sh'
    BJ = 'bj'
    NB = 'nb'


class Location:
    def __init__(self, lat, lng):
        self._lat = lat
        self._lng = lng

    @property
    def lat(self):
        return self._lat

    @property
    def lng(self):
        return self._lng


class Block:
    def __init__(self, city, left_lower, right_upper, width=0.01, height=0.01):
        """
        define city block range
        :param city: city name
        :param left_lower: 城市左下角坐标
        :param right_upper: 城市右上角坐标
        :param width: 网格宽
        :param height: 网格高
        """
        self.city = city
        self.left_lower = left_lower
        self.right_upper = right_upper
        self.width = width
        self.height = height
        self.lat_steps, self.lng_steps = self.get_grid_steps()

    def get_grid_steps(self):
        """
        get the step number in the latitude and longitude directions
        """
        lat_steps = int(math.ceil((self.right_upper.lat - self.left_lower.lat) / self.height))
        lng_steps = int(math.ceil((self.right_upper.lng - self.left_lower.lng) / self.width))
        return lat_steps, lng_steps


SH_BLOCK = Block(
    City.SH,
    left_lower=Location(lat=30.705000, lng=120.915000),
    right_upper=Location(lat=31.495000, lng=121.965000)
)

BJ_BLOCK = Block(
    City.BJ,
    left_lower=Location(lat=39.720, lng=116.090),
    right_upper=Location(lat=40.280, lng=116.700)
)

NB_BLOCK = Block(
    City.NB,
    left_lower=Location(lat=29.80, lng=121.45),
    right_upper=Location(lat=29.99, lng=121.90)
)


def get_grid_steps(block):
    """
    get the step number in the latitude and longitude directions
    """
    left_lower = block.left_lower
    right_upper = block.right_upper

    lat_steps = int(math.ceil((right_upper.lat - left_lower.lat) / block.height))
    lng_steps = int(math.ceil((right_upper.lng - left_lower.lng) / block.width))
    return lat_steps, lng_steps


def get_sh_range(width=0.01, height=0.01):
    # Shanghai large range：31.495189, 120.914373, 30.700175, 121.967546
    sh_left_lower = Location(lat=30.705000, lng=120.915000)
    sh_right_upper = Location(lat=31.495000, lng=121.965000)

    sh_block = Block(City.SH, sh_left_lower, sh_right_upper, width=width, height=height)
    lat_steps, lng_steps = get_grid_steps(sh_block)
    return lat_steps, lng_steps, sh_block


def get_bj_range(width=0.01, height=0.01):
    # mobike spider 40.2912, 116.0796, 39.7126, 116.7086
    bj_lower_left = Location(lat=39.720, lng=116.090)
    bj_upper_right = Location(lat=40.280, lng=116.700)

    bj_block = Block(City.BJ, bj_lower_left, bj_upper_right, width=width, height=height)
    lat_steps, lng_steps = get_grid_steps(bj_block)
    return lat_steps, lng_steps, bj_block


def get_nb_range(width=0.01, height=0.01):
    # mobike range 29.99, 121.45, 29.80, 121.91
    nb_lower_left = Location(lat=29.80, lng=121.45)
    nb_upper_right = Location(lat=29.99, lng=121.90)

    nb_block = Block(City.BJ, nb_lower_left, nb_upper_right, width=width, height=height)
    lat_steps, lng_steps = get_grid_steps(nb_block)
    return lat_steps, lng_steps, nb_block


city_block_dict = {
    City.SH: SH_BLOCK,
    City.BJ: BJ_BLOCK,
    City.NB: NB_BLOCK
}
