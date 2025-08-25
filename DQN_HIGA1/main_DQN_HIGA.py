
from json_handler import JSONHandler
from modular_generator import ModluarGenarator
from structure_FEM import Structural_model

from modular_scheme_gener import run_generate_data

import json
from config import DXF_FILE, JSON_FILE, MATERIAL_FILE



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
                    1: {'type': 1, 'top': 2, 'bottom': 2, 'column': 2},
                    2: {'type': 2, 'top': 8, 'bottom': 8, 'column': 8},
                    3: {'type': 3, 'top': 10, 'bottom': 10, 'column': 10}}

    modular_variable = [1 for _ in range(standard_story_num * modudular_type_num)]

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
        result_force = 10
    else:
        result_force = 1-max_force


    if max_dis >=1:
        result_dis = 10
    else:
        result_dis = 1-max_force


    return total_weight, result_force,result_dis


def process_data():
    with open('data_save_20_wlq.json', 'r', encoding='utf-8') as f:
        split_data = json.load(f)

    data_groups1 = [split_data[i:i + 3] for i in range(0, len(split_data), 3)]

    all_split_succeed = []
    for i in range(len(data_groups1)):
        all_split_succeed.append(data_groups1[i][0])
    targets = [25200, 29600, 25200, 37800]
    perfect_lists = find_perfect_groups(all_split_succeed, targets)



    groups = group_by_sum(perfect_lists[0], targets)
    return perfect_lists

def find_perfect_groups(number_lists: list, targets: list) -> list:
    """
    从多个数列中找出能够完美分组的数列

    参数：
    number_lists: 多个数字列表的列表
    targets: 目标和列表

    返回：
    list: 可以完美分组的数列列表
    """
    perfect_lists = []

    for i, numbers in enumerate(number_lists):
        if check_perfect_grouping(numbers, targets):
            perfect_lists.append(numbers)

    return perfect_lists

def check_perfect_grouping(numbers: list, targets: list) -> bool:
    """
    检查一个数列是否能够完美按照目标和进行分组

    参数：
    numbers: 需要检查的数字列表
    targets: 目标和列表

    返回：
    bool: 是否可以完美分组
    """
    if sum(numbers) != sum(targets):  # 首先检查总和是否相等
        return False

    current_sum = 0
    target_index = 0
    number_index = 0

    while number_index < len(numbers) and target_index < len(targets):
        current_sum += numbers[number_index]

        if abs(current_sum - targets[target_index]) < 0.1:  # 达到目标和
            current_sum = 0
            target_index += 1
        elif current_sum > targets[target_index]:  # 超过目标和
            return False

        number_index += 1

    return target_index == len(targets)  # 检查是否所有目标和都已匹配



def group_by_sum(numbers: list, targets: list) -> list:
    """
    根据目标和将数字列表分组

    参数：
    numbers: 需要分组的数字列表
    targets: 目标和列表

    返回：
    分组后的列表
    """
    result = []
    current_group = []
    current_sum = 0
    target_index = 0

    for num in numbers:
        # 如果已经处理完所有目标和，结束处理
        if target_index >= len(targets):
            break

        current_group.append(num)
        current_sum += num

        # 当前组的和达到目标值
        if abs(current_sum - targets[target_index]) < 0.1:  # 使用小误差范围处理浮点数
            result.append(current_group)
            current_group = []
            current_sum = 0
            target_index += 1

    # 检查是否还有剩余的数字
    if current_group:
        result.append(current_group)

    return result



if __name__ == "__main__":
    building_data=[3900, 3500, 3500, 3800, 3500, 3500, 3500, 3500, 3000, 3500, 3000, 3500, 3500, 3200, 3200, 3200, 3200, 3200, 3200, 3100, 3200, 3100, 3100, 3100, 4000, 3600, 3600, 3400, 3400, 3400, 3400, 3400, 3400, 3100, 3100]

    perfect_dara = process_data()


    total_weight, max_force,dis_data=run_FEM_analysis(perfect_dara[0])
    a= 1

