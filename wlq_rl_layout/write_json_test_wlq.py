import json
from typing import List, Tuple, Dict, Any


def cal_total_width(room_width: List[int], corridor_width: bool) -> List[int]:
    """计算总宽度，忽略corridor_width参数，因为代码中都设为0或False"""
    return room_width.copy()


def cal_total_length(length: List[List[int]]) -> List[int]:
    """计算每层房间长度总和"""
    return [sum(floor_lengths) for floor_lengths in length]


def cal_outer_space(length_total: List[int], width: List, location: List[List[int]], direction: str) -> List[
    List[Tuple[int, int]]]:
    """计算外部空间坐标"""
    room_width, corridor_width, corridor_para = width
    width_total = cal_total_width(room_width, corridor_width)

    outer_space = []
    for i, (x, y) in enumerate(location):
        if direction == "h":
            # 水平方向
            coords = [
                (x, y),
                (x + length_total[i], y),
                (x + length_total[i], y + width_total[i]),
                (x, y + width_total[i])
            ]
        else:  # direction == "v"
            # 垂直方向
            coords = [
                (x, y),
                (x, y + length_total[i]),
                (x + width_total[i], y + length_total[i]),
                (x + width_total[i], y)
            ]
        outer_space.append(coords)

    return outer_space


def cal_inner_space(length: List[List[int]], width: List, location: List[List[int]], direction: str) -> List[
    List[List[Tuple[int, int]]]]:
    """计算内部空间坐标"""
    room_width, _, _ = width
    inner_space = []

    for i, (floor_lengths, (x_start, y_start)) in enumerate(zip(length, location)):
        floor_rooms = []
        x_temp, y_temp = x_start, y_start

        for room_length in floor_lengths:
            if direction == "h":
                coords = [
                    (x_temp, y_temp),
                    (x_temp + room_length, y_temp),
                    (x_temp + room_length, y_temp + room_width[i]),
                    (x_temp, y_temp + room_width[i])
                ]
                x_temp += room_length
            else:  # direction == "v"
                coords = [
                    (x_temp, y_temp),
                    (x_temp + room_width[i], y_temp),
                    (x_temp + room_width[i], y_temp + room_length),
                    (x_temp, y_temp + room_length)
                ]
                y_temp += room_length

            floor_rooms.append(coords)
        inner_space.append(floor_rooms)

    return inner_space


def find_room_length(room_indices: List[List[int]], room_dict: Dict[str, int]) -> List[List[int]]:
    """根据房间索引查找房间长度"""
    return [[room_dict[str(index)] for index in floor_indices] for floor_indices in room_indices]


def create_zone_config(room_width: List[int], room_indices: List[List[int]],
                       room_dict: Dict[str, int], location: List[List[int]],
                       direction: str, corridor_width: bool = False) -> Tuple[
    List, List[List[int]], List[List[int]], str]:
    """创建区域配置"""
    width = [room_width, corridor_width, 0]
    length = find_room_length(room_indices, room_dict)
    return width, length, location, direction


def find_story_info_from_building(building_list: List[List[int]]) -> List[List[int]]:
    """从建筑信息中提取楼层信息"""
    max_len = max(len(building) for building in building_list)
    return [[building[i] for building in building_list if i < len(building)]
            for i in range(max_len)]


def horizontal_relationship(zone_pairs: List[List[int]], outer_space_per_building: Dict[str, List[int]]) -> Dict[
    str, List[int]]:
    """计算水平关系"""
    horizontal_rel = {}
    count = 0

    for zone1_idx, zone2_idx in zone_pairs:
        zone1 = outer_space_per_building[str(zone1_idx)]
        zone2 = outer_space_per_building[str(zone2_idx)]
        min_len = min(len(zone1), len(zone2))

        for i in range(min_len):
            horizontal_rel[str(count)] = [zone1[i], zone2[i]]
            count += 1

    return horizontal_rel


def main():
    # 房间字典定义
    room_dicts = [
        {"0": 29600, "1": 37800, "2": 12600, "3": 25200, "4": 13000, "5": 20000, "6": 6500, "7": 12500, "8": 38000,
         "9": 6300},

    ]

    # 区域配置数据
    zone_configs = [
        # # Zone 1
        ([10000] * 9, [[3,0,3,1]] * 9, room_dicts[0], [[0, 0]] * 9, "h"),
        # Zone 2
        # ([3000] * 9, [[1]] * 9, room_dicts[0], [[0, 10000]] * 9, "h"),
        # # Zone 3
        # ([10000] * 9, [[0, 0, 0, 0]] * 9, room_dicts[0], [[0, 13000]] * 9, "h"),
    ]



    # 处理所有区域
    zones = []
    zones_dir = []
    zones_index = []
    total_outer_space = []
    total_inner_space = []

    for room_width, room_indices, room_dict, location, direction in zone_configs:
        width, length, location, direction = create_zone_config(room_width, room_indices, room_dict, location,
                                                                direction)

        zones.append([width, length, location])
        zones_dir.append(direction)
        zones_index.append(room_indices)

        length_total = cal_total_length(length)
        outer_space = cal_outer_space(length_total, width, location, direction)
        inner_space = cal_inner_space(length, width, location, direction)

        total_outer_space.append(outer_space)
        total_inner_space.append(inner_space)

    # 构建最终的项目数据结构
    storey_num = max(len(term) for term in total_outer_space)

    # 构建outer_space相关数据
    outer_space_config = {}
    outer_space_per_building = {}
    outer_space_per_building_count_list = []

    space_count = 0
    for i, outer_spaces in enumerate(total_outer_space):
        building_count_list = []
        for j, space in enumerate(outer_spaces):
            outer_space_config[str(space_count)] = space
            building_count_list.append(space_count)
            space_count += 1
        outer_space_per_building[str(i)] = building_count_list
        outer_space_per_building_count_list.append(building_count_list)

    # 构建inner_space相关数据
    inner_space_config = {}
    rooms_per_outer_space = {}

    inner_count = 0
    outer_count = 0
    for i, inner_spaces in enumerate(total_inner_space):
        for j, floor_spaces in enumerate(inner_spaces):
            room_list = []
            for space in floor_spaces:
                inner_space_config[str(inner_count)] = space
                room_list.append(inner_count)
                inner_count += 1
            rooms_per_outer_space[str(outer_count)] = room_list
            outer_count += 1

    # 构建楼层信息
    outer_space_per_storey_count_list = find_story_info_from_building(outer_space_per_building_count_list)
    outer_space_per_storey = {str(i): spaces for i, spaces in enumerate(outer_space_per_storey_count_list)}

    # 构建关系数据
    vertical = {}
    for i, building_spaces in enumerate(outer_space_per_building.values()):
        for j in range(len(building_spaces) - 1):
            vertical[str(len(vertical))] = [building_spaces[j], building_spaces[j + 1]]

    # horizontal = horizontal_relationship([[1, 2], [4, 5]], outer_space_per_building)
    # horizontal = horizontal_relationship([[1, 2], [4, 5], [7, 8], [10, 11]], outer_space_per_building)
    # horizontal = horizontal_relationship([[0, 1], [1, 2]], outer_space_per_building)
    horizontal = horizontal_relationship([], outer_space_per_building)

    # 构建房间类型数据
    rooms_dict = {}
    zones_index_flat = [room for zone in zones_index for room in zone]
    room_list_flat = [room for rooms in rooms_per_outer_space.values() for room in rooms]

    for i, room_indices in enumerate(zones_index_flat):
        for j, room_type in enumerate(room_indices):
            room_type_str = str(room_type)
            if room_type_str not in rooms_dict:
                rooms_dict[room_type_str] = {"rooms": []}
            if i < len(room_list_flat):
                rooms_dict[room_type_str]["rooms"].append(room_list_flat[i])

    # 构建最终项目数据
    project = {
        "storey_num": storey_num,
        "zone_num": space_count,
        "outer_space_num": len(outer_space_config),
        "inner_space_num": len(inner_space_config),
        "outer_space_per_building": {
            str(i): {
                "index": str(space_id),
                "direction": zones_dir[i // max(len(spaces) for spaces in outer_space_per_building_count_list)],
                "story": str(j),
                "corridor": zones[i // max(len(spaces) for spaces in outer_space_per_building_count_list)][0][1],
                "zone": str(space_id)
            } for i, space_id in enumerate([space for spaces in outer_space_per_building.values() for space in spaces])
            for j in range(len([space for spaces in outer_space_per_building.values() for space in spaces]))
        },
        "outer_space_per_storey": outer_space_per_storey,
        "inner_space_config": inner_space_config,
        "outer_space_config": outer_space_config,
        "rooms_per_outer_space": rooms_per_outer_space,
        "room_type": rooms_dict,
        "x_axis": [0],
        "y_axis": [0],
        "outer_space_relationship": {"horizontal": horizontal, "vertical": vertical},
        "building_boundary": {i: [[0, 0], [40000, 0], [40000, 23000], [0, 23000]] for i in range(9)}
    }

    # 保存到文件
    with open('Buildingdata/building_data7.json', 'w') as f:
        json.dump(project, f, indent=4)

    return project


if __name__ == "__main__":
    main()