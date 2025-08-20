
from json_handler import JSONHandler
from modular_generator import ModluarGenarator
from structure_FEM import Structural_model

from modular_scheme_gener import run_generate_data

import os
import ezdxf
import numpy as np
import json
from config import DXF_FILE, JSON_FILE, MATERIAL_FILE
import openseespy.opensees as ops
import opstool as opst
import opstool.vis.pyvista as opsvis


def run_FEM_analysis(building_data):
    dxf_data = run_generate_data(building_data)

    # 转化为json
    SaveJson = JSONHandler(dxf_data, JSON_FILE)
    SaveJson.run()

    '''读取json文件'''
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        building_data = json.load(f)
    with open(MATERIAL_FILE, 'r', encoding='utf-8') as f:
        material_data = json.load(f)

    '''生成3D数字模型'''
    ModularData = ModluarGenarator(building_data, 200, 200)
    structure_data = ModularData.run()

    '''基于Opensees的结构分析'''
    standard_story_num = len(building_data["GraphStandardBuildingStorey"])
    modudular_type_num = len(structure_data["modular_group"])
    modular_unit = {0: {'type': 0, 'top': 0, 'bottom': 0, 'column': 0},
                    1: {'type': 1, 'top': 3, 'bottom': 3, 'column': 8},
                    2: {'type': 2, 'top': 7, 'bottom': 7, 'column': 9},
                    3: {'type': 3, 'top': 10, 'bottom': 10, 'column': 10}}

    modular_variable = [3 for _ in range(standard_story_num * modudular_type_num)]

    StructureAnalysis = Structural_model(structure_data, modular_unit, modular_variable
                                         , building_data["GraphStandardBuildingStorey"], material_data)
    analysis_data = StructureAnalysis.run()
    total_weight = analysis_data['total_weight']
    max_force = analysis_data['max_force']
    inter_dis = analysis_data['inter_dis']
    story_dis = analysis_data['story_dis']

    inter_dis_max_x = max(inter_dis[key]['max_x'] for key in inter_dis)
    inter_dis_max_y = max(inter_dis[key]['max_y'] for key in inter_dis)

    story_dis_max_x = max(story_dis[key]['max_x'] for key in story_dis)
    story_dis_max_y = max(story_dis[key]['max_y'] for key in story_dis)

    dis_data = [inter_dis_max_x, inter_dis_max_y, story_dis_max_x, story_dis_max_y]
    max_dis = max(dis_data)

    if max_force >=1:
        result_force = -3
    else:
        result_force = 1-max_force


    if max_dis >=1:
        result_dis = -3
    else:
        result_dis = 1-max_force


    return total_weight, result_force,result_dis

if __name__ == "__main__":
    building_data=[3900, 3500, 3500, 3800, 3500, 3500, 3500, 3500, 3000, 3500, 3000, 3500, 3500, 3200, 3200, 3200, 3200, 3200, 3200, 3100, 3200, 3100, 3100, 3100, 4000, 3600, 3600, 3400, 3400, 3400, 3400, 3400, 3400, 3100, 3100]


    total_weight, max_force,dis_data=run_FEM_analysis(building_data)
    a= 1

