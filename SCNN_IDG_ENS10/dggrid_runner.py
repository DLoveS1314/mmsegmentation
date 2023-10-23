# -*- coding: utf-8 -*-

from pathlib import Path
import uuid
import shutil
import os
import sys
import subprocess
import traceback
import tempfile
import numpy as np
import pandas as pd

import fiona
from fiona.crs import from_epsg

import geopandas as gpd

fiona_drivers = fiona.supported_drivers


def get_geo_out(legacy=True):
    if legacy is True:
        return {"driver": "GeoJSON", "ext": "geojson"}
        # return { "driver": "ESRI Shapefile", "ext": ".shp"}

    if "FlatGeobuf" in fiona_drivers.keys() and "w" in fiona_drivers["FlatGeobuf"]:
        return {"driver": "FlatGeobuf", "ext": "fgb"}

    return {"driver": "GPKG", "ext": "gpgk"}


# specify a ISEA3H
dggs_types = (
    'CUSTOM',  # parameters will be specified manually
    'SUPERFUND',  # Superfund_500m grid
    'PLANETRISK',
    'ISEA3H',  # ISEA projection with hexagon cells and an aperture of 3
    'ISEA4H',  # ISEA projection with hexagon cells and an aperture of 4
    'ISEA4T',  # ISEA projection with triangle cells and an aperture of 4
    'ISEA4D',  # ISEA projection with diamond cells and an aperture of 4
    'ISEA43H',
    # ISEA projection with hexagon cells and a mixed sequence of aperture 4 resolutions followed by aperture 3 resolutions
    'ISEA7H',  # ISEA projection with hexagon cells and an aperture of 7
    'FULLER3H',  # FULLER projection with hexagon cells and an aperture of 3
    'FULLER4H',  # FULLER projection with hexagon cells and an aperture of 4
    'FULLER4T',  # FULLER projection with triangle cells and an aperture of 4
    'FULLER4D',  # FULLER projection with diamond cells and an aperture of 4
    'FULLER43H',
    # FULLER projection with hexagon cells and a mixed sequence of aperture 4 resolutions followed by aperture 3 resolutions
    'FULLER7H',  # FULLER projection with hexagon cells and an aperture of 7
)

# control grid generation
clip_subset_types = (
    'SHAPEFILE',
    'WHOLE_EARTH',
    'GDAL',
    'AIGEN',
    'SEQNUMS'
)

# specify the output
cell_output_types = (
    'AIGEN',
    'GDAL',
    'GEOJSON',
    'SHAPEFILE',
    'NONE',
    'TEXT'
)

output_address_types = (
    'GEO',  # geodetic coordinates -123.36 43.22 20300 Roseburg
    'Q2DI',  # quad number and (i, j) coordinates on that quad
    'SEQNUM',
    # DGGS index - linear address (1 to size-of-DGG), not supported for parameter input_address_type if dggs_aperture_type is SEQUENCE
    'INTERLEAVE',
    # digit-interleaved form of Q2DI, only supported for parameter output_address_type; only available for hexagonal aperture 3 and 4 grids
    'PLANE',  # (x, y) coordinates on unfolded ISEA plane,  only supported for parameter output_address_type;
    'Q2DD',  # quad number and (x, y) coordinates on that quad
    'PROJTRI',  # PROJTRI - triangle number and (x, y) coordinates within that triangle on the ISEA plane
    'VERTEX2DD',  # vertex number, triangle number, and (x, y) coordinates on ISEA plane
    'AIGEN'  # Arc/Info Generate file format
)

input_address_types = (
    'GEO',  # geodetic coordinates -123.36 43.22 20300 Roseburg
    'Q2DI',  # quad number and (i, j) coordinates on that quad
    'SEQNUM',
    # DGGS index - linear address (1 to size-of-DGG), not supported for parameter input_address_type if dggs_aperture_type is SEQUENCE
    'Q2DD',  # quad number and (x, y) coordinates on that quad
    'PROJTRI',  # PROJTRI - triangle number and (x, y) coordinates within that triangle on the ISEA plane
    'VERTEX2DD',  # vertex number, triangle number, and (x, y) coordinates on ISEA plane
    'AIGEN'  # Arc/Info Generate file format
)

### CUSTOM args
dggs_projections = ("ISEA", "FULLER")

dggs_topologies = ('HEXAGON', 'TRIANGLE', 'DIAMOND')
dggs_aperture_types = ('PURE', 'MIXED43', 'SEQUENCE')

dggs_res_specify_types = ("SPECIFIED", "CELL_AREA", "INTERCELL_DISTANCE")
dggs_orient_specify_types = ('SPECIFIED', 'RANDOM', 'REGION_CENTER')

# specify the operation
dggrid_operations = (
    'GENERATE_GRID',
    'TRANSFORM_POINTS',
    'BIN_POINT_VALS',
    'BIN_POINT_PRESENCE',
    'OUTPUT_STATS'
)

"""
helper function to create a DGGS config quickly
 生成dggs 各种参数 最终会在dg_grid_meta 进行参数挑选1 而不是吧所有的参数都输入到meta中国
"""


def dgselect(dggs_type, **kwargs):
    dggs = None

    topo_dict = {
        'H': 'HEXAGON',
        'T': 'TRIANGLE',
        'D': 'DIAMOND'
    }

    if dggs_type in dggs_types:
        if dggs_type in ['SUPERFUND', 'PLANETRISK']:
            # keep it simple, only that spec PLANETRISK是一种六边形格网类型
            dggs = Dggs(dggs_type=dggs_type,
                        metric=True,
                        show_info=True)

            for res_opt in ['res',  # dggs_res_spec
                            'precision',  # default 7
                            'area',  # dggs_res_specify_area
                            'spacing',
                            'cls_val'  # dggs_res_specify_intercell_distance
                            ]:
                if res_opt in kwargs.keys():
                    dggs.set_par(res_opt, kwargs[res_opt])

        elif not dggs_type == 'CUSTOM':

            # if dggs_type == 'ISEA3H'
            #     projection, aperture, topology = 'ISEA', 3, 'HEXAGON'

            projection, aperture, topology = 'ISEA', 3, 'HEXAGON'

            if dggs_type.find('ISEA') > -1:
                projection == 'ISEA'
                sub1 = dggs_type.replace('ISEA', '')
                topology = topo_dict[sub1[-1]]
                aperture = int(sub1.replace(sub1[-1], ''))

            elif dggs_type.find('FULLER') > -1:
                projection == 'FULLER'
                sub1 = dggs_type.replace('FULLER', '')
                topology = topo_dict[sub1[-1]]
                aperture = int(sub1.replace(sub1[-1], ''))

            else:
                raise ValueError('projection not ISEA nor FULLER???')

            dggs = Dggs(dggs_type=dggs_type,
                        projection=projection,  # dggs_projection
                        aperture=aperture,  # dggs_aperture_type / dggs_aperture
                        topology=topology,  # dggs_topology
                        metric=True,
                        show_info=True)

            for res_opt in ['res',  # dggs_res_spec
                            'precision',  # default 7
                            'area',  # dggs_res_specify_area
                            'spacing',
                            'cls_val'  # dggs_res_specify_intercell_distance
                            ]:
                if res_opt in kwargs.keys():
                    dggs.set_par(res_opt, kwargs[res_opt])

            if aperture == 43:
                if 'mixed_aperture_level' in kwargs.keys():
                    dggs.set_par('mixed_aperture_level', kwargs['mixed_aperture_level'])


        elif dggs_type == 'CUSTOM':
            # load and insert grid definition from dggs obj

            # dggs_projections = ( "ISEA", "FULLER")
            # dggs_res_specify_types = ( "SPECIFIED", "CELL_AREA", "INTERCELL_DISTANCE" )

            # specify_resolution(proj_spec, dggs_res_spec_type)
            """
            proj_spec,
                        dggs_res_spec_type,
                        dggs_res_spec=9,
                        dggs_res_specify_area=120000,
                        dggs_res_specify_intercell_distance=4000,
                        dggs_res_specify_rnd_down=True
            """

            # dggs_topologies = ( 'HEXAGON', 'TRIANGLE', 'DIAMOND')
            # dggs_aperture_types = ( 'PURE', 'MIXED43', 'SEQUENCE')

            # specify_topo_aperture(topology_type, aperture_type, aperture_res)
            """
            specify_topo_aperture(topology_type, aperture_type, aperture_res, dggs_num_aperture_4_res=0, dggs_aperture_sequence="333333333333")
            """

            # dggs_orient_specify_types = ( 'SPECIFIED', 'RANDOM', 'REGION_CENTER' )

            if 'orient_type' in kwargs.keys() and kwargs['orient_type'] in dggs_orient_specify_types:
                orient_type = kwargs['orient_type']
                # specify_orient_type_args(orient_type)
                """
                                            dggs_vert0_lon=11.25,
                                            dggs_vert0_lat=58.28252559,
                                            dggs_vert0_azimuth=0.0,
                                            dggs_orient_rand_seed=42
                """

            raise ValueError('custom not yet implemented')

    # dggs.dgverify()

    return dggs


"""
helper function to generate the metafile from a DGGS config
只从dggs里提取res precision dggs_type操作 其它不提取
"""


def dg_grid_meta(dggs):
    dggridy_par_lookup = {
        'res': 'dggs_res_spec',
        'precision': 'precision',
        'area': 'dggs_res_specify_area',
        'cls_val': 'dggs_res_specify_intercell_distance',
        'mixed_aperture_level': 'dggs_num_aperture_4_res'
    }
    metafile = []

    if dggs.dggs_type in ['SUPERFUND', 'PLANETRISK']:
        metafile.append(f"dggs_type {dggs.dggs_type}")

    elif not dggs.dggs_type == 'CUSTOM':
        metafile.append(f"dggs_type {dggs.dggs_type}")

    elif dggs.dggs_type == 'CUSTOM':
        raise ValueError('custom not yet implemented')

    for res_opt in ['res',  # dggs_res_spec
                    'precision',  # default 7
                    'area',  # dggs_res_specify_area
                    'cls_val',  # dggs_res_specify_intercell_distance
                    'mixed_aperture_level'  # dggs_num_aperture_4_res 5
                    ]:
        if not dggs.get_par(res_opt, None) is None:
            opt_val = dggs.get_par(res_opt, None)
            if not opt_val is None:
                metafile.append(f"{dggridy_par_lookup[res_opt]} {opt_val}")

    return metafile


"""
class representing a DGGS grid system configuration, projection aperture etc
"""


class Dggs(object):
    """
        dggs_type: str     # = 'CUSTOM'
        projection: str     # = 'ISEA'
        aperture: int      #  = 3
        topology: str      #  = 'HEXAGON'
        res: int           #  = None
        precision: int     #  = 7
        area: float         #   = None
        spacing: float       #  = None
        cls_val: float        #     = None
        resround: str      #  = 'nearest'
        metric: bool        #  = True
        show_info: bool     #  = True
        azimuth_deg: float   #  = 0
        pole_lat_deg: float  #  = 58.28252559
        pole_lon_deg: float  # = 11.25

        mixed_aperture_level:  # e.g. 5 -> dggs_num_aperture_4_res 5  for ISEA_43_H etc
        metafile = []
    """

    def __init__(self, dggs_type, **kwargs):
        self.dggs_type = dggs_type

        for key, value in kwargs.items():
            self.set_par(key, value)

    # def dgverify(self):
    #     #See page 21 of documentation for further bounds
    #     if not self.projection in ['ISEA','FULLER']:
    #         raise ValueError('Unrecognised dggs projection')
    #
    #     if not self.topology in ['HEXAGON','DIAMOND','TRIANGLE']:
    #         raise ValueError('Unrecognised dggs topology')
    #     if not self.aperture in [ 3, 4 ]:
    #         raise ValueError('Unrecognised dggs aperture')
    #     if self.res < 0:
    #         raise ValueError('dggs resolution must be >=0')
    #     if self.res > 30:
    #         raise ValueError('dggs resolution must be <=30')
    #     if self.azimuth_deg < 0 or self.azimuth_deg > 360:
    #         raise ValueError('dggs azimuth_deg must be in the range [0,360]')
    #     if self.pole_lat_deg < -90  or self.pole_lat_deg > 90:
    #         raise ValueError('dggs pole_lat_deg must be in the range [-90,90]')
    #     if self.pole_lon_deg < -180 or self.pole_lon_deg > 180:
    #         raise ValueError('dggs pole_lon_deg must be in the range [-180,180]')

    def set_par(self, par_key, par_value):
        if par_key == 'dggs_type':
            self.dggs_type = par_value
        if par_key == 'projection':
            self.projection = par_value
        if par_key == 'aperture':
            self.aperture = par_value
        if par_key == 'topology':
            self.topology = par_value
        if par_key == 'res':
            self.res = par_value
        if par_key == 'precision':
            self.precision = par_value
        if par_key == 'area':
            self.area = par_value
        if par_key == 'spacing':
            self.spacing = par_value
        if par_key == 'cls_val':
            self.cls_val = par_value
        if par_key == 'resround':
            self.resround = par_value
        if par_key == 'metric':
            self.metric = par_value
        if par_key == 'show_info':
            self.show_info = par_value
        if par_key == 'azimuth_deg':
            self.azimuth_deg = par_value
        if par_key == 'pole_lat_deg':
            self.pole_lat_deg = par_value
        if par_key == 'pole_lon_deg':
            self.pole_lon_deg = par_value
        if par_key == 'mixed_aperture_level':
            self.mixed_aperture_level = par_value

        return self

    def get_par(self, par_key, alternative=None):
        if par_key == 'dggs_type':
            try:
                return self.dggs_type
            except AttributeError:
                return alternative
        if par_key == 'projection':
            try:
                return self.projection
            except AttributeError:
                return alternative
        if par_key == 'aperture':
            try:
                return self.aperture
            except AttributeError:
                return alternative
        if par_key == 'topology':
            try:
                return self.topology
            except AttributeError:
                return alternative
        if par_key == 'res':
            try:
                return self.res
            except AttributeError:
                return alternative
        if par_key == 'precision':
            try:
                return self.precision
            except AttributeError:
                return alternative
        if par_key == 'area':
            try:
                return self.area
            except AttributeError:
                return alternative
        if par_key == 'spacing':
            try:
                return self.spacing
            except AttributeError:
                return alternative
        if par_key == 'cls_val':
            try:
                return self.cls_val
            except AttributeError:
                return alternative
        if par_key == 'resround':
            try:
                return self.resround
            except AttributeError:
                return alternative
        if par_key == 'metric':
            try:
                return self.metric
            except AttributeError:
                return alternative
        if par_key == 'show_info':
            try:
                return self.show_info
            except AttributeError:
                return alternative
        if par_key == 'azimuth_deg':
            try:
                return self.azimuth_deg
            except AttributeError:
                return alternative
        if par_key == 'pole_lat_deg':
            try:
                return self.pole_lat_deg
            except AttributeError:
                return alternative
        if par_key == 'pole_lon_deg':
            try:
                return self.pole_lon_deg
            except AttributeError:
                return alternative
        if par_key == 'mixed_aperture_level':
            try:
                return self.mixed_aperture_level
            except AttributeError:
                return alternative
        else:
            return alternative

    def dg_closest_res_to_area(self, area, resround, metric, show_info=True):
        raise ValueError('not yet implemented')

    def dg_closest_res_to_spacing(self, spacing, resround, metric, show_info=True):
        raise ValueError('not yet implemented')

    def dg_closest_res_to_cls(self, cls_val, resround, metric, show_info=True):
        raise ValueError('not yet implemented')


"""
necessary instance object that needs to be instantiated once to tell where to use and execute the dggrid cmd tool
"""


class DGGRIDv7(object):

    def __init__(self, executable='dggrid', working_dir=None, capture_logs=True, silent=False, tmp_geo_out_legacy=True):
        self.executable = Path(executable).resolve()
        self.capture_logs = capture_logs
        self.silent = silent
        self.last_run_succesful = False
        self.last_run_logs = ''
        # 默认为GeoJson 在实验的时候 使用GDAL的GeoJson生成时错误的
        self.tmp_geo_out = get_geo_out(legacy=tmp_geo_out_legacy)

        if working_dir is None:
            self.working_dir = tempfile.mkdtemp(prefix='dggrid_')
        else:
            self.working_dir = working_dir

    def is_runnable(self):  ##没有使用
        is_runnable = 0

        takes = []
        take_1 = shutil.which(self.executable)
        if not take_1 is None:
            takes.append(take_1)

        take_2 = shutil.which(os.path.join(self.working_dir, self.executable))
        if not take_2 is None:
            takes.append(take_2)

        if len(takes) < 1:
            print(f"{self.executable} not in executable paths")
        else:
            for elem in takes:
                swx = Path(elem)
                if swx.exists() and swx.is_file():
                    if os.access(elem, os.X_OK):
                        # print(f"{elem} is executable")
                        self.executable = str(elem)
                        is_runnable = 1

        return is_runnable

    def run(self, dggs_meta_ops):

        curdir = os.getcwd()
        tmp_id = uuid.uuid4()

        # subprocess.call / Popen swat_exec, check if return val is 0 or not
        # yield logs?
        try:
            os.chdir(self.working_dir)

            with open('metafile_' + str(tmp_id), 'w', encoding='utf-8') as metafile:
                for line in dggs_meta_ops:
                    metafile.write(line + '\n')
            metafile.close()
            logs = []
            o = subprocess.Popen([os.path.join(self.working_dir, self.executable), 'metafile_' + str(tmp_id)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            while o.poll() is None:
                for b_line in o.stdout:
                    line = b_line.decode().strip()
                    # sys.stdout.write(line)
                    if not self.silent:
                        print(line)
                    if self.capture_logs:
                        logs.append(line.strip())

            if o.returncode == 0:
                self.last_run_succesful = True
                try:
                    pass
                    # os.remove( 'metafile_' + str(tmp_id) )
                except Exception:
                    pass
            else:
                self.last_run_succesful = False

            if self.capture_logs:
                self.last_run_logs = '\n'.join(logs)
            else:
                self.last_run_logs = ''

        except Exception as e:
            self.last_run_succesful = False
            print(repr(e))
            traceback.print_exc(file=sys.stdout)
            self.last_run_logs = repr(e)
        finally:
            os.chdir(curdir)

        return o.returncode

    """
    ##############################################################################################
    # lower level API
    ##############################################################################################
    """

    def dgapi_grid_gen(self, dggs, subset_conf, output_conf):
        """
        格网生成操作
        Grid Generation. Generate the cells of a DGG, either covering the complete surface of the earth or covering only a
        specific set of regions on the earth’s surface.
        返回值是 metafile 调用生成的metafile 和dggrid可执行文件来生成输出结果（利用run） 再读取到python中
        """
        dggrid_operation = 'GENERATE_GRID'
        metafile = []
        metafile.append("dggrid_operation " + dggrid_operation)

        dggs_config_meta = dg_grid_meta(dggs)

        for cmd in dggs_config_meta:
            metafile.append(cmd)

        # clip_subset_types
        if subset_conf['clip_subset_type'] == 'WHOLE_EARTH':
            metafile.append("clip_subset_type " + subset_conf['clip_subset_type'])
        else:
            raise ValueError('something is not correct in subset_conf')
        for elem in output_conf.keys():
            metafile.append(f"{elem} " + output_conf[elem])

        result = self.run(metafile)

        if not result == 0:
            if self.capture_logs == True:
                message = f"some error happened under the hood of dggrid (exit code {result}): " + self.last_run_logs
                raise ValueError(message)
            else:
                message = f"some error happened under the hood of dggrid (exit code {result}), try capture_logs=True for dggrid instance"
                raise ValueError(message)

        return {'metafile': metafile, 'output_conf': output_conf}

    # 只针对四边形能用 要重新写都能用的
    # 处理gdal collection s生成的csv字符串，
    def process_chi(self, gdf):
        '''由于生成的是字符串数组 需要拆开 处理生成的child和nei s
        '[ "2", "20480", "36928", "65" ]'
        '''
        children_name = ['0', '1', '2', '3']  # up left down right
        gd_child = gdf['children'].str.replace('[\[\]\"]', '').str.split(',', expand=True)
        gd_child.rename(columns={x: children_name[x] for x in gd_child.columns}, inplace=True)
        # print(gd_child)
        gdf = gdf.drop(columns={'children'}).rename(columns={'name': 'code'})
        # gdf.seqnum=gdf.seqnum.astype(int)
        gdf = pd.concat([gdf, gd_child], axis=1)
        return gdf.astype(int)

    def process_nei(self, gdf):
        '''由于生成的是字符串数组 需要拆开 处理生成的child和nei s
        '[ "2", "20480", "36928", "65" ]'
        '''
        neighbors_name = ['r', 'ru', 'u', 'lu', 'l', 'ld', 'd', 'rd']  # up left down right
        gd_nei = gdf['neighbors'].str.replace('[\[\]\"]', '').str.split(',',
                                                                        expand=True)  # 如果正则中没有\" 那么生成的数字是 ' "2048" '这种 就是 字符串中包含的字符串 所以需要吧" 也给消除
        gd_nei.rename(columns={x: neighbors_name[x] for x in gd_nei.columns}, inplace=True)
        # print(gd_child)
        gdf = gdf.drop(columns={'neighbors'}).rename(columns={'name': 'code'})
        # gdf.seqnum=gdf.seqnum.astype(int)
        gdf = pd.concat([gdf, gd_nei], axis=1)
        return gdf.astype(int)

    def gen_nei_chi(self, dggs_type, resolution, gen_type, mixed_aperture_level=None,
                    save_dir='../data/adjacency_table'):
        """_summary_
        根据 gen_type 生成相应的数据 'adj'/'chi/code'
        Args:
            dggs_type (_type_): 格网类型
            resolution (_type_): 格网等级
            gen_type (_type_): 生成临近还是子格网还是只生成编码 'adj'/'chi/code'
            mixed_aperture_level (_type_, optional): _description_. Defaults to None.
        Returns:
            _type_: 返回临近或者子格网
        """

        tmp_dir = self.working_dir
        os.makedirs(tmp_dir, exist_ok=True)
        dggs = dgselect(dggs_type=dggs_type, res=resolution, mixed_aperture_level=mixed_aperture_level)

        subset_conf = {'update_frequency': 100000, 'clip_subset_type': 'WHOLE_EARTH'}
        # self.tmp_geo_out.update({'ext':'geojson','driver':"GeoJSON"})
        self.tmp_geo_out.update({'ext': 'csv', 'driver': "CSV"})  # csv 不保存geometry信息 正好只保存child和nei信息

        collection_output_file_name = str(
            os.path.join(save_dir, f"{dggs_type}_{resolution}_{gen_type}.{self.tmp_geo_out['ext']}"))
        if not os.path.exists(collection_output_file_name):  ##如果存在了相关文件 就不在生成

            if gen_type == 'adj':
                output_conf = {
                    'collection_output_gdal_format': self.tmp_geo_out['driver'],
                    'neighbor_output_type': 'GDAL_COLLECTION',
                    # 'children_output_type' : 'GDAL_COLLECTION',
                    'point_output_type': 'GDAL_COLLECTION',
                    'collection_output_file_name': collection_output_file_name
                }
            elif gen_type == 'chi' or gen_type == 'dechi':  # 卷积和反卷积 生成结果是一样的 但是后续的处理中 反卷积不需要也不能排序
                output_conf = {
                    'collection_output_gdal_format': self.tmp_geo_out['driver'],
                    # 'neighbor_output_type' : 'GDAL_COLLECTION',
                    'children_output_type': 'GDAL_COLLECTION',
                    'point_output_type': 'GDAL_COLLECTION',
                    'collection_output_file_name': collection_output_file_name
                }
            else:  # code
                output_conf = {
                    'collection_output_gdal_format': self.tmp_geo_out['driver'],
                    # 'neighbor_output_type' : 'GDAL_COLLECTION',
                    # 'children_output_type' : 'GDAL_COLLECTION',
                    'point_output_type': 'GDAL_COLLECTION',
                    'collection_output_file_name': collection_output_file_name
                }
            # output_conf = {
            #     'point_output_type': 'GDAL',
            #     'point_output_gdal_format' : self.tmp_geo_out['driver'],
            #     'point_output_file_name': str( (Path(tmp_dir) / f"temp_{dggs_type}_{resolution}_out_{tmp_id}.{self.tmp_geo_out['ext']}").resolve())
            #     }
            dggs_ops = self.dgapi_grid_gen(dggs, subset_conf, output_conf)
        else:
            print(f'gen_nei_chi {collection_output_file_name} file have exits')
        df = pd.read_csv(collection_output_file_name)
        # 处理一下因为生成csv导致的数据读取问题 字符串转为数字
        if gen_type == 'adj':
            df = self.process_nei(df)
            df.sort_values(by='code', ascending=True)  ##确保按照从小到大排序
        elif gen_type == 'chi':
            df = self.process_chi(df)
            df.sort_values(by='code', ascending=True)  ##确保按照从小到大排序
        elif gen_type == 'dechi':  # 不需要排序 因为 反卷积本身要返回的就是排序过程中的索引值
            df = self.process_chi(df)
            # df.sort_values(by= 'code',ascending=True)##确保按照从小到大排序、/
        try:  # shp无法这样删除 直接删除文件夹
            shutil.rmtree(tmp_dir)  # 删除文件夹
            # 创建文件夹
            os.makedirs(tmp_dir)  # 创建空文件夹
            # os.remove( str( Path(tmp_dir) / f"temp_{dggs_type}_{resolution}_out_{tmp_id}.{self.tmp_geo_out['ext']}") )
            # os.remove( str( Path(tmp_dir) / f"temp_clip_{tmp_id}.txt") )
        except Exception:
            pass

        return df

    # 获取当前等级编码
    def getcode(self, dggs_type, resolution, save_dir='../data/chi_table'):
        tmp_dir = self.working_dir
        os.makedirs(tmp_dir, exist_ok=True)
        dggs = dgselect(dggs_type=dggs_type, res=resolution)
        subset_conf = {'update_frequency': 100000, 'clip_subset_type': 'WHOLE_EARTH'}
        # self.tmp_geo_out.update({'ext':'geojson','driver':"GeoJSON"})
        # self.tmp_geo_out.update({'ext':'csv','driver':"CSV"}) #csv 不保存geometry信息 正好只保存child和nei信息
        # 借助生成子格网的程序，求code
        output_file_name = str(os.path.join(save_dir, f"{dggs_type}_{resolution}_chi.csv"))
        if not os.path.exists(output_file_name):  ##如果存在了相关文件 就不在生成
            output_conf = {
                # 'neighbor_output_type' : 'TEXT',
                'children_output_type': 'TEXT',
                'children_output_file_name': output_file_name,
            }
            # output_conf = {
            #     'point_output_type': 'GDAL',
            #     'point_output_gdal_format' : self.tmp_geo_out['driver'],
            #     'point_output_file_name': str( (Path(tmp_dir) / f"temp_{dggs_type}_{resolution}_out_{tmp_id}.{self.tmp_geo_out['ext']}").resolve())
            #     }
            dggs_ops = self.dgapi_grid_gen(dggs, subset_conf, output_conf)
        else:
            print(f'getcode {output_file_name} file have exits')
            # 读取没有标题的 CSV 文件
        if "4H" in dggs_type:  # 六边形  四孔的子孩子是七个 算上code 有八个
            new_columns = ['code' if x == 0 else f'chi_{x - 1}' for x in range(8)]
            df = pd.read_csv(output_file_name, header=None, names=new_columns)
            df['chi_6'] = df['chi_6'].fillna(df['chi_5']).astype(int)
        elif "H" not in dggs_type:
            df = pd.read_csv(output_file_name, header=None)
            # 根据列的长度创建新的列名称列表
            new_columns = ['code' if x == 0 else f'chi_{x - 1}' for x in range(len(df.columns))]
            # 将新的列名称分配给 DataFrame.columns
            df.columns = new_columns
        else:
            raise f' {dggs_type}  尚未实现'
        df1 = df['code']
        df1.sort_values(ascending=True)  ##确保按照从小到大排序

        try:  # shp无法这样删除 直接删除文件夹
            shutil.rmtree(tmp_dir)  # 删除文件夹
            # 创建文件夹
            os.makedirs(tmp_dir)  # 创建空文件夹
            # os.remove( str( Path(tmp_dir) / f"temp_{dggs_type}_{resolution}_out_{tmp_id}.{self.tmp_geo_out['ext']}") )
            # os.remove( str( Path(tmp_dir) / f"temp_clip_{tmp_id}.txt") )
        except Exception:
            pass
        return df1

    # 获取当前等级邻近 用于卷积  六边形返回7个 三角形返回四个 菱形返回九个
    def getadjtable(self, dggs_type, resolution, save_dir):
        tmp_dir = self.working_dir
        os.makedirs(tmp_dir, exist_ok=True)
        dggs = dgselect(dggs_type=dggs_type, res=resolution)
        subset_conf = {'update_frequency': 100000, 'clip_subset_type': 'WHOLE_EARTH'}
        # self.tmp_geo_out.update({'ext':'geojson','driver':"GeoJSON"})
        # self.tmp_geo_out.update({'ext':'csv','driver':"CSV"}) #csv 不保存geometry信息 正好只保存child和nei信息
        # 借助生成子格网的程序，求code
        output_file_name = str(os.path.join(save_dir, f"{dggs_type}_{resolution}_nei.csv"))
        if not os.path.exists(output_file_name):  ##如果存在了相关文件 就不在生成
            output_conf = {
                'neighbor_output_type': 'TEXT',
                # 'children_output_type' : 'TEXT',
                # 'children_output_file_name':output_file_name,
                'neighbor_output_file_name': output_file_name
            }

            dggs_ops = self.dgapi_grid_gen(dggs, subset_conf, output_conf)
        else:
            print(f'getadj {output_file_name} file have exits')
            # 读取没有标题的 CSV 文件
        if "H" in dggs_type:  # 六边形 由于没有更改程序 会存在最后一列缺失值   用前一个值补充
            new_columns = ['code' if x == 0 else f'nei_{x - 1}' for x in range(7)]
            df = pd.read_csv(output_file_name, header=None, names=new_columns)
            df['nei_5'] = df['nei_5'].fillna(df['nei_4']).astype(int)
        else:
            df = pd.read_csv(output_file_name, header=None)
            # 根据列的长度创建新的列名称列表
            new_columns = ['code' if x == 0 else f'nei_{x - 1}' for x in range(len(df.columns))]
            # 将新的列名称分配给 DataFrame.columns
            df.columns = new_columns
            df.sort_values(by='code', ascending=True, inplace=True)  ## 对于求取子编码 不能对编码进行排序 因为生成的反卷积是按照子编码顺序来的
        try:  # shp无法这样删除 直接删除文件夹
            shutil.rmtree(tmp_dir)  # 删除文件夹
            # 创建文件夹
            os.makedirs(tmp_dir)  # 创建空文件夹
            # os.remove( str( Path(tmp_dir) / f"temp_{dggs_type}_{resolution}_out_{tmp_id}.{self.tmp_geo_out['ext']}") )
            # os.remove( str( Path(tmp_dir) / f"temp_clip_{tmp_id}.txt") )
        except Exception:
            pass
        return df
        # 获取上一层级的子格网编码 用于池化  。注意 res在maketable里需要减1去得到 这里求的是res的子格网编码

    def getpartable(self, dggs_type, resolution, save_dir):
        tmp_dir = self.working_dir
        os.makedirs(tmp_dir, exist_ok=True)
        dggs = dgselect(dggs_type=dggs_type, res=resolution)
        subset_conf = {'update_frequency': 100000, 'clip_subset_type': 'WHOLE_EARTH'}
        # self.tmp_geo_out.update({'ext':'geojson','driver':"GeoJSON"})
        # self.tmp_geo_out.update({'ext':'csv','driver':"CSV"}) #csv 不保存geometry信息 正好只保存child和nei信息
        # 借助生成子格网的程序，求code
        output_file_name = str(os.path.join(save_dir, f"{dggs_type}_{resolution}_chi.csv"))
        if not os.path.exists(output_file_name):  ##如果存在了相关文件 就不在生成
            output_conf = {
                # 'neighbor_output_type' : 'TEXT',
                'children_output_type': 'TEXT',
                'children_output_file_name': output_file_name,
                # 'neighbor_output_file_name':'output_file_name
            }
            # output_conf = {
            #     'point_output_type': 'GDAL',
            #     'point_output_gdal_format' : self.tmp_geo_out['driver'],
            #     'point_output_file_name': str( (Path(tmp_dir) / f"temp_{dggs_type}_{resolution}_out_{tmp_id}.{self.tmp_geo_out['ext']}").resolve())
            #     }
            dggs_ops = self.dgapi_grid_gen(dggs, subset_conf, output_conf)
        else:
            print(f'getpar {output_file_name} file have exits')
            # 读取没有标题的 CSV 文件
        if "4H" in dggs_type:  # 六边形  四孔的子孩子是七个 算上code 有八个
            new_columns = ['code' if x == 0 else f'chi_{x - 1}' for x in range(8)]
            df = pd.read_csv(output_file_name, header=None, names=new_columns)
            df['chi_6'] = df['chi_6'].fillna(df['chi_5']).astype(int)
        elif "H" not in dggs_type:
            df = pd.read_csv(output_file_name, header=None)
            # 根据列的长度创建新的列名称列表
            new_columns = ['code' if x == 0 else f'chi_{x - 1}' for x in range(len(df.columns))]
            # 将新的列名称分配给 DataFrame.columns
            df.columns = new_columns
        else:
            raise f' {dggs_type}  尚未实现'
        df.sort_values(by='code', ascending=True, inplace=True)  ## 对于求取子编码 不能对编码进行排序 因为生成的反卷积是按照子编码顺序来的
        try:  # shp无法这样删除 直接删除文件夹
            shutil.rmtree(tmp_dir)  # 删除文件夹
            # 创建文件夹
            os.makedirs(tmp_dir)  # 创建空文件夹
            # os.remove( str( Path(tmp_dir) / f"temp_{dggs_type}_{resolution}_out_{tmp_id}.{self.tmp_geo_out['ext']}") )
            # os.remove( str( Path(tmp_dir) / f"temp_clip_{tmp_id}.txt") )
        except Exception:
            pass
        return df

        # 获取当前等级的子格网 用于反卷积

    def getchitable(self, dggs_type, resolution, save_dir):
        tmp_dir = self.working_dir
        os.makedirs(tmp_dir, exist_ok=True)
        dggs = dgselect(dggs_type=dggs_type, res=resolution)
        subset_conf = {'update_frequency': 100000, 'clip_subset_type': 'WHOLE_EARTH'}
        # self.tmp_geo_out.update({'ext':'geojson','driver':"GeoJSON"})
        # self.tmp_geo_out.update({'ext':'csv','driver':"CSV"}) #csv 不保存geometry信息 正好只保存child和nei信息
        # 借助生成子格网的程序，求code
        output_file_name = str(os.path.join(save_dir, f"{dggs_type}_{resolution}_chi.csv"))
        if not os.path.exists(output_file_name):  ##如果存在了相关文件 就不在生成
            output_conf = {
                # 'neighbor_output_type' : 'TEXT',
                'children_output_type': 'TEXT',
                'children_output_file_name': output_file_name,
                # 'neighbor_output_file_name':'output_file_name
            }
            # output_conf = {
            #     'point_output_type': 'GDAL',
            #     'point_output_gdal_format' : self.tmp_geo_out['driver'],
            #     'point_output_file_name': str( (Path(tmp_dir) / f"temp_{dggs_type}_{resolution}_out_{tmp_id}.{self.tmp_geo_out['ext']}").resolve())
            #     }
            dggs_ops = self.dgapi_grid_gen(dggs, subset_conf, output_conf)
        else:
            print(f'getchi {output_file_name} file have exits')
            # 读取没有标题的 CSV 文件
        if "4H" in dggs_type:  # 六边形  四孔的子孩子是七个 算上code 有八个
            new_columns = ['code' if x == 0 else f'chi_{x - 1}' for x in range(8)]
            df = pd.read_csv(output_file_name, header=None, names=new_columns)
            df['chi_6'] = df['chi_6'].fillna(df['chi_5']).astype(int)
        elif "H" not in dggs_type:
            df = pd.read_csv(output_file_name, header=None)
            # 根据列的长度创建新的列名称列表
            new_columns = ['code' if x == 0 else f'chi_{x - 1}' for x in range(len(df.columns))]
            # 将新的列名称分配给 DataFrame.columns
            df.columns = new_columns
        else:
            raise f' {dggs_type}  尚未实现'

        # df1.sort_values(by='code',ascending=True)## 对于求取子编码 不能对编码进行排序 因为生成的反卷积是按照子编码顺序来的
        try:  # shp无法这样删除 直接删除文件夹
            shutil.rmtree(tmp_dir)  # 删除文件夹
            # 创建文件夹
            os.makedirs(tmp_dir)  # 创建空文件夹
            # os.remove( str( Path(tmp_dir) / f"temp_{dggs_type}_{resolution}_out_{tmp_id}.{self.tmp_geo_out['ext']}") )
            # os.remove( str( Path(tmp_dir) / f"temp_clip_{tmp_id}.txt") )
        except Exception:
            pass
        return df
        # 生成geopandas 包含cell 和 point    # 生成geopandas 包含cell 和 point dens代表每个边要生成的点个数

    def gen_cell_point(self, dggs_type, resolution, mixed_aperture_level=None, if_drop=True, dggs_vert0_lon="0.0",
                       dggs_vert0_lat="90.0", dens="0"):
        """
        方位固定为极点处
        生成点和cell geometry
        if_drop :是否删除dggrid生成的源文件
        """
        tmp_dir = self.working_dir

        if not os.path.isdir(tmp_dir):
            # 创建文件夹
            os.makedirs(tmp_dir)

        tmp_id = uuid.uuid4()  # 创建文件的唯一标识
        dggs = dgselect(dggs_type=dggs_type, res=resolution, mixed_aperture_level=mixed_aperture_level)

        subset_conf = {'update_frequency': 100000, 'clip_subset_type': 'WHOLE_EARTH'}
        self.tmp_geo_out.update({'ext': 'json', 'driver': "GeoJSON"})

        point_file_name = str(
            os.path.join(tmp_dir, f"temp_{dggs_type}_{resolution}_out_{tmp_id}_point.{self.tmp_geo_out['ext']}"))

        cell_file_name = str(
            os.path.join(tmp_dir, f"temp_{dggs_type}_{resolution}_out_{tmp_id}.{self.tmp_geo_out['ext']}"))

        output_conf = {
            'point_output_type': 'GDAL',
            'point_output_gdal_format': self.tmp_geo_out['driver'],
            'point_output_file_name': point_file_name,
            'cell_output_type': 'GDAL',
            'cell_output_gdal_format': self.tmp_geo_out['driver'],
            'cell_output_file_name': cell_file_name,
            'dggs_vert0_lon': dggs_vert0_lon,
            'dggs_vert0_lat': dggs_vert0_lat,
            'precision': '12',
            'densification': dens
        }
        dggs_ops = self.dgapi_grid_gen(dggs, subset_conf, output_conf)
        # 连接路径 路径可以用/运算符或joinpath()方法连接。
        gdf_point = gpd.read_file(point_file_name, driver=self.tmp_geo_out['driver']).rename_geometry('point').rename(
            columns={'name': 'seqnum'})
        gdf_cell = gpd.read_file(cell_file_name, driver=self.tmp_geo_out['driver']).rename_geometry('cell').rename(
            columns={'name': 'seqnum'})
        gdf = gdf_point.merge(gdf_cell, on='seqnum', how='inner')
        gdf.seqnum = gdf.seqnum.astype(int)

        if if_drop:
            try:  # shp无法这样删除 直接删除文件夹
                shutil.rmtree(tmp_dir)  # 删除文件夹
                # 创建文件夹
                os.makedirs(tmp_dir)  # 创建空文件夹
                # os.remove( str( Path(tmp_dir) / f"temp_{dggs_type}_{resolution}_out_{tmp_id}.{self.tmp_geo_out['ext']}") )
                # os.remove( str( Path(tmp_dir) / f"temp_clip_{tmp_id}.txt") )
            except Exception:
                pass

        return gdf


