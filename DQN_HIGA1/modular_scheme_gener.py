
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import random
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
from typing import List, Dict
from collections import defaultdict


def plot_modules_by_type(modules_dict):
    """
    使用不同颜色绘制不同类型的模块

    参数：
    modules_dict: dict, 按类型分组的模块信息字典
    """
    # 设置图形大小
    plt.figure(figsize=(15, 10))

    # 定义颜色映射
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
              '#FFEEAD', '#D4A5A5', '#9B59B6', '#3498DB']

    # 跟踪坐标范围
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    # 为每种类型绘制模块
    for i, (type_key, modules) in enumerate(modules_dict.items()):
        color = colors[i % len(colors)]  # 循环使用颜色

        # 绘制该类型的所有模块
        for module in modules:
            segments = module['segments']

            # 收集所有点的坐标
            x_coords = []
            y_coords = []

            # 绘制每个段
            for segment in segments:
                start_x, start_y = segment['start']
                end_x, end_y = segment['end']

                x_coords.extend([start_x, end_x])
                y_coords.extend([start_y, end_y])

                # 更新坐标范围
                min_x = min(min_x, start_x, end_x)
                max_x = max(max_x, start_x, end_x)
                min_y = min(min_y, start_y, end_y)
                max_y = max(max_y, start_y, end_y)

            # 绘制填充多边形
            plt.fill(x_coords, y_coords, color=color, alpha=0.5)
            plt.plot(x_coords, y_coords, color=color, linewidth=1)

    # 添加图例
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i % len(colors)],
                                     alpha=0.5, label=type_key)
                       for i, type_key in enumerate(modules_dict.keys())]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    # 设置坐标轴比例相等
    plt.axis('equal')

    # 设置坐标轴范围，添加一些边距
    margin = (max_x - min_x) * 0.05  # 5%的边距
    plt.xlim(min_x - margin, max_x + margin)
    plt.ylim(min_y - margin, max_y + margin)

    # 添加标题和标签
    plt.title('Modules Layout by Type')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)

    # 调整布局以显示完整的图例
    plt.tight_layout()

    # 显示图形
    plt.show()


def generate_modular_scheme(modular_list, positions, direction, long_edge=12600):
    """
    生成模块排布方案
    :param modular_list: 每组模块的宽度列表
    :param positions: 每组模块的起始左下角坐标列表
    :param direction: 排布方向，'x'或'y'
    :param long_edge: 模块的长边长度，默认为12600
    :return: 所有模块的字典列表
    """
    modules = []
    for group_idx, start_pos in enumerate(positions):
        x0, y0 = start_pos
        for i, width in enumerate(modular_list):
            if direction == 'x':
                # 沿x正向排布，长边为x方向
                pos = (x0 + sum(modular_list[:i]), y0)
                module = {
                    'position': pos,
                    'width': width,
                    'height': long_edge,
                    'long_edge': long_edge,
                    'short_edge': width
                }
            elif direction == 'y':
                # 沿y正向排布，长边为y方向
                pos = (x0, y0 + sum(modular_list[:i]))
                module = {
                    'position': pos,
                    'width': long_edge,
                    'height': width,
                    'long_edge': long_edge,
                    'short_edge': width
                }
            else:
                raise ValueError("direction只能为'x'或'y'")
            modules.append(module)
    return modules


def plot_modules(modular_schemes):
    """
    绘制所有模块方案

    参数：
    modular_schemes: list of lists, 包含多个方案的模块数据
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 为不同方案设置不同的颜色
    colors = ['lightblue', 'lightgreen', 'lightpink']

    # 记录所有坐标以便自动调整显示范围
    all_x = []
    all_y = []

    # 遍历每个方案
    for scheme_idx, scheme in enumerate(modular_schemes):
        # 遍历方案中的每个模块
        for module in scheme:
            # 获取模块数据
            x, y = module['position']
            width = module['width']
            height = module['height']

            # 创建矩形
            rect = patches.Rectangle(
                (x, y),  # 左下角坐标
                width,  # 宽度
                height,  # 高度
                facecolor=colors[scheme_idx],  # 填充颜色
                edgecolor='black',  # 边框颜色
                alpha=0.5,  # 透明度
                linewidth=1  # 边框宽度
            )

            # 添加矩形到图中
            ax.add_patch(rect)

            # 在矩形中心添加文本标注
            center_x = x + width / 2
            center_y = y + height / 2
            ax.text(center_x, center_y, f'Scheme {scheme_idx + 1}\n{width}x{height}',
                    horizontalalignment='center',
                    verticalalignment='center')

            # 记录坐标范围
            all_x.extend([x, x + width])
            all_y.extend([y, y + height])

    # 设置坐标轴范围，留有一定边距
    margin = 1000
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # 设置等比例显示
    ax.set_aspect('equal')

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)

    # 添加标题和轴标签
    plt.title('Module Layout Visualization')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    # 添加图例
    legend_elements = [patches.Patch(facecolor=color, alpha=0.5,
                                     label=f'Scheme {i + 1}')
                       for i, color in enumerate(colors)]
    ax.legend(handles=legend_elements)

    # 显示图形
    plt.show()


def plot_modules2(modular_schemes):
    """
    绘制所有模块方案

    参数：
    modular_schemes: list of lists, 包含多个方案的模块数据
    """


    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 统计所有模块类型（long_edge, short_edge）并分配颜色
    module_types = {}
    color_map = {}
    color_list = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"
    ]
    color_idx = 0

    # 先遍历所有方案，收集所有模块类型
    for scheme in modular_schemes:
        for module in scheme:
            key = (module['long_edge'], module['short_edge'])
            if key not in module_types:
                module_types[key] = None

    # 为每种模块类型分配颜色
    for key in module_types:
        if color_idx < len(color_list):
            color_map[key] = color_list[color_idx]
        else:
            # 超过预设颜色则随机生成
            color_map[key] = "#%06x" % random.randint(0, 0xFFFFFF)
        color_idx += 1

    # 记录所有坐标以便自动调整显示范围
    all_x = []
    all_y = []

    # 遍历每个方案
    for scheme_idx, scheme in enumerate(modular_schemes):
        # 遍历方案中的每个模块
        for module in scheme:
            # 获取模块数据
            x, y = module['position']
            width = module['width']
            height = module['height']
            long_edge = module['long_edge']
            short_edge = module['short_edge']

            # 选择颜色
            color = color_map[(long_edge, short_edge)]

            # 创建矩形
            rect = patches.Rectangle(
                (x, y),  # 左下角坐标
                width,  # 宽度
                height,  # 高度
                facecolor=color,  # 填充颜色
                edgecolor='black',  # 边框颜色
                alpha=0.5,  # 透明度
                linewidth=1  # 边框宽度
            )

            # 添加矩形到图中
            ax.add_patch(rect)

            # 在矩形中心添加文本标注
            center_x = x + width / 2
            center_y = y + height / 2
            # ax.text(center_x, center_y, f'{long_edge}x{short_edge}',
            #         horizontalalignment='center',
            #         verticalalignment='center', fontsize=8)

            # 记录坐标
            all_x.extend([x, x + width])
            all_y.extend([y, y + height])

    # 自动调整显示范围
    if all_x and all_y:
        ax.set_xlim(min(all_x) - 500, max(all_x) + 500)
        ax.set_ylim(min(all_y) - 500, max(all_y) + 500)

    # 图例
    # handles = []
    # for key, color in color_map.items():
    #     handles.append(patches.Patch(color=color, label=f'{key[0]}x{key[1]}'))
    # ax.legend(handles=handles, title="模块类型(long_edge x short_edge)")

    ax.set_aspect('equal')
    ax.set_title("模块排布方案")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def merge_to_dict(modular_schemes: List[list]) -> Dict:
    """
    将所有模块合并为一个大字典，根据long_edge和short_edge分配type

    参数：
    modular_schemes: 包含多个方案的列表

    返回：
    以索引为key的模块字典
    """
    # 先收集所有不同的模块类型
    module_types = {}  # (long_edge, short_edge) -> type_index
    type_index = 0

    # 第一次遍历，识别所有不同的模块类型
    for scheme in modular_schemes:
        for module in scheme:
            # 创建模块类型的标识（排序确保一致性）
            type_key = tuple(sorted([module['long_edge'], module['short_edge']], reverse=True))
            if type_key not in module_types:
                module_types[type_key] = type_index
                type_index += 1

    # 创建最终的字典
    merged_dict = {}
    index = 0

    # 第二次遍历，创建最终字典
    for scheme in modular_schemes:
        for module in scheme:
            type_key = tuple(sorted([module['long_edge'], module['short_edge']], reverse=True))
            type_number = module_types[type_key]

            merged_dict[str(index)] = {
                'position': module['position'],
                'width': module['width'],
                'height': module['height'],
                'long_edge': module['long_edge'],
                'short_edge': module['short_edge'],
                'type': f'30000-{type_number}'  # 使用类型索引而不是模块索引
            }
            index += 1

    return merged_dict


def convert_to_polyline_format(modules_dict):
    """
    将模块信息转换为多段线格式

    参数：
    modules_dict: dict, 模块信息字典

    返回：
    转换后的字典，按类型分组
    """
    # 初始化结果字典
    result = {}

    # 遍历所有模块
    for module_id, module_info in modules_dict.items():
        # 获取模块类型
        module_type = module_info['type']
        type_key = f'{module_type}'  # 转换为指定格式的类型key

        # 如果该类型还没有在结果字典中，初始化一个空列表
        if type_key not in result:
            result[type_key] = []

        # 获取模块的坐标信息
        x, y = module_info['position']
        width = module_info['width']
        height = module_info['height']

        # 计算矩形四个角点的坐标
        points = [
            (x, y),  # 左下角
            (x + width, y),  # 右下角
            (x + width, y + height),  # 右上角
            (x, y + height),  # 左上角
            (x, y)  # 回到起点，形成闭合多边形
        ]

        # 创建segments列表
        segments = []
        for i in range(len(points) - 1):
            segment = {
                'start': points[i],
                'end': points[i + 1]
            }
            segments.append(segment)

        # 创建多段线对象
        polyline = {
            'type': 'LWPOLYLINE',
            'segments': segments,
            'closed': False
        }

        # 添加到结果字典中
        result[type_key].append(polyline)

    return result

def create_complete_building_dict(modules_dict, floor_texts):
    """
    创建完整的建筑信息字典集合

    参数：
    modules_dict: dict, 按类型分组的模块信息字典
    floor_texts: list, 每个建筑的楼层文本列表

    返回：
    dict, 包含多个建筑信息的大字典
    """

    # 创建建筑字典的基本结构
    def create_building_dict(floor_text):
        building_dict = {
            # 直接使用原始模块数据
            **modules_dict,
            # 文本信息
            'text': [{
                'type': 'MTEXT',
                'position': (0, 0),
                'text': floor_text
            }],
            # 中心点信息
            'center_point': (0, 0)
        }
        return building_dict

    # 创建多个建筑的大字典
    buildings_dict = {}
    for i, floor_text in enumerate(floor_texts):
        building_key = f'Building_{i + 1}'
        buildings_dict[building_key] = create_building_dict(floor_text)

    return buildings_dict


def calculate_bounding_box(building_data):
    """
    计算建筑所有模块的外包矩形

    参数：
    building_data: dict, 建筑数据

    返回：
    包含外包矩形信息的列表
    """
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    # 遍历所有模块类型和坐标
    for type_key, modules in building_data.items():
        if type_key not in ['text', 'center_point']:
            for module in modules:
                for segment in module['segments']:
                    # 更新最大最小坐标值
                    min_x = min(min_x, segment['start'][0], segment['end'][0])
                    max_x = max(max_x, segment['start'][0], segment['end'][0])
                    min_y = min(min_y, segment['start'][1], segment['end'][1])
                    max_y = max(max_y, segment['start'][1], segment['end'][1])

    # 创建外包矩形
    bounding_box = [{
        'type': 'LWPOLYLINE',
        'segments': [
            {'start': (min_x, min_y), 'end': (max_x, min_y)},
            {'start': (max_x, min_y), 'end': (max_x, max_y)},
            {'start': (max_x, max_y), 'end': (min_x, max_y)},
            {'start': (min_x, max_y), 'end': (min_x, min_y)}
        ],
        'closed': True
    }]

    return bounding_box


def shift_buildings_coordinates(buildings_dict):
    """
    平移建筑坐标并添加外包矩形
    """
    # 找出第一个建筑中的最大坐标值
    max_x, max_y = float('-inf'), float('-inf')
    first_building = list(buildings_dict.values())[0]

    # 计算最大坐标值
    for type_key, modules in first_building.items():
        if type_key not in ['text', 'center_point']:
            for module in modules:
                for segment in module['segments']:
                    max_x = max(max_x, segment['start'][0], segment['end'][0])
                    max_y = max(max_y, segment['start'][1], segment['end'][1])

    # 添加边距
    margin = 1000
    max_x += margin
    max_y += margin

    # 创建新的字典
    new_buildings = {}

    # 处理每个建筑
    for i, (_, building_data) in enumerate(buildings_dict.items()):
        shift_x = i * max_x
        shift_y = i * max_y

        new_building = {}

        # 处理每种类型的模块
        for type_key, modules in building_data.items():
            if type_key not in ['text', 'center_point']:
                new_modules = []
                for module in modules:
                    new_module = {
                        'type': module['type'],
                        'segments': [],
                        'closed': module['closed']
                    }

                    for segment in module['segments']:
                        if i == 0:
                            new_segment = segment.copy()
                        else:
                            new_segment = {
                                'start': (segment['start'][0] + shift_x, segment['start'][1] + shift_y),
                                'end': (segment['end'][0] + shift_x, segment['end'][1] + shift_y)
                            }
                        new_module['segments'].append(new_segment)

                    new_modules.append(new_module)
                new_building[type_key] = new_modules
            elif type_key == 'text':
                new_building[type_key] = [{
                    'type': 'MTEXT',
                    'position': (shift_x, shift_y) if i > 0 else (0, 0),
                    'text': modules[0]['text']
                }]
            else:  # center_point
                new_building[type_key] = (shift_x, shift_y)

        # 添加外包矩形
        new_building['20000-2'] = calculate_bounding_box(new_building)

        # 使用坐标作为键名
        new_key = f"({shift_x},{shift_y})"
        new_buildings[new_key] = new_building

    return new_buildings


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


def run_generate_data(building_data):
    # with open('data_save_20_wlq.json', 'r', encoding='utf-8') as f:
    #     split_data = json.load(f)
    # # split_data = building_data
    # data_groups1 = [split_data[i:i + 3] for i in range(0, len(split_data), 3)]
    #
    # all_split_succeed = []
    # for i in range(len(data_groups1)):
    #     all_split_succeed.append(data_groups1[i][0])
    targets = [25200, 29600, 25200, 37800]
    # perfect_lists = find_perfect_groups(all_split_succeed, targets)
    #
    #
    # groups = group_by_sum(perfect_lists[index_id], targets)


    groups =group_by_sum(building_data, targets)
    # print("\n分组结果：")
    # for i, group in enumerate(groups):
    #     print(f"\n第{i + 1}组 (目标和: {targets[i] if i < len(targets) else '未指定'}):")
    #     print(f"数据: {group}")
    #     print(f"实际和: {sum(group)}")

    # modular_list1 = [3400, 3000, 3100, 3100]
    # modular_list2 = [4000, 3800, 3800, 3800, 3800, 3800, 3300, 3300]
    # modular_list3 = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3900, 3900]

    modular_list1 = groups[0]
    modular_list2 = groups[1]
    modular_list3 = groups[2]
    modular_list4 = groups[3]



    list1_position=[[0,0]]
    list2_position=[[12600,0],[12600,12600],[12600,25200],[12600,37800],[12600,50400]]
    list3_position=[[54800,50400]]
    list4_position = [[42200, 12600], [54800, 12600], [67400, 12600]]

    list1_direction = 'y'
    list2_direction = 'x'
    list3_direction = 'x'
    list4_direction = 'y'


    # 示例：生成list1的模块方案
    modular_scheme1 = generate_modular_scheme(modular_list1, list1_position, list1_direction)
    # 示例：生成list2的模块方案
    modular_scheme2 = generate_modular_scheme(modular_list2, list2_position, list2_direction)
    # 示例：生成list3的模块方案
    modular_scheme3 = generate_modular_scheme(modular_list3, list3_position, list3_direction)
    modular_scheme4 = generate_modular_scheme(modular_list4, list4_position, list4_direction)

    # 合并和分类
    classified_modules = merge_to_dict([
        modular_scheme1,
        modular_scheme2,
        modular_scheme3,
        modular_scheme4
    ])

    type_2_modules = convert_to_polyline_format(classified_modules)
    # 绘制以上转化的结果
    # plot_modules_by_type(type_2_modules)

    # 指定每个建筑的楼层文本
    classified_modules = [
        '1-3F',
        '4-6F',
        '7-9F'
    ]
    buildings = create_complete_building_dict(type_2_modules, classified_modules)
    buildings = shift_buildings_coordinates(buildings)

    return buildings

if __name__ == "__main__":

    # with open('data_save_20_wlq.json', 'r', encoding='utf-8') as f:
    #     split_data = json.load(f)
    #
    # data_groups1 = [split_data[i:i + 3] for i in range(0, len(split_data), 3)]
    #
    # all_split_succeed = []
    # for i in range(len(data_groups1)):
    #     all_split_succeed.append(data_groups1[i][0])
    #
    # perfect_lists = find_perfect_groups(all_split_succeed, targets)
    #

    perfect_data = [
        4000,
        4000,
        4000,
        3200,
        3200,
        3200,
        3600,
        3200,
        3400,
        3400,
        4000,
        4000,
        3600,
        4000,
        4000,
        3600,
        3600,
        3600,
        3600,
        3600,
        3600,
        3600,
        3600,
        3500,
        3500,
        3500,
        3500,
        3500,
        3500,
        3200,
        3500,
        3500,
        3000
    ]
    targets = [25200, 29600, 25200, 37800]
    # groups = group_by_sum(perfect_lists[0], targets)
    groups = group_by_sum(perfect_data, targets)


    modular_list1 = groups[0]
    modular_list2 = groups[1]
    modular_list3 = groups[2]
    modular_list4 = groups[3]



    list1_position=[[0,0]]
    list2_position=[[12600,0],[12600,12600],[12600,25200],[12600,37800],[12600,50400]]
    list3_position=[[54800,50400]]
    list4_position = [[42200, 12600], [54800, 12600], [67400, 12600]]

    list1_direction = 'y'
    list2_direction = 'x'
    list3_direction = 'x'
    list4_direction = 'y'


    # 示例：生成list1的模块方案
    modular_scheme1 = generate_modular_scheme(modular_list1, list1_position, list1_direction)
    # 示例：生成list2的模块方案
    modular_scheme2 = generate_modular_scheme(modular_list2, list2_position, list2_direction)
    # 示例：生成list3的模块方案
    modular_scheme3 = generate_modular_scheme(modular_list3, list3_position, list3_direction)
    modular_scheme4 = generate_modular_scheme(modular_list4, list4_position, list4_direction)

    # 合并和分类
    classified_modules = merge_to_dict([
        modular_scheme1,
        modular_scheme2,
        modular_scheme3,
        modular_scheme4
    ])

    type_2_modules = convert_to_polyline_format(classified_modules)
    # 绘制以上转化的结果
    plot_modules_by_type(type_2_modules)

    # 指定每个建筑的楼层文本
    classified_modules = [
        '1-3F',
        '4-6F',
        '7-9F'
    ]
    buildings = create_complete_building_dict(type_2_modules, classified_modules)
    buildings = shift_buildings_coordinates(buildings)


    # 绘制三个模块方案
    # plot_modules2([modular_scheme1, modular_scheme2, modular_scheme3])
    a=1
