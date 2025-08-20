import copy
from typing import Dict
import collections
from copy import deepcopy
from collections import defaultdict
from collections import Counter

class Building2Case:
    def __init__(self, data: Dict):
        self.storey_num = data['storey_num']
        self.outer_space_num = data['outer_space_num']
        self.inner_space_num = data['inner_space_num']
        self.outer_space_per_storey = data['outer_space_per_storey']
        self.outer_space_per_building = data['outer_space_per_building']
        self.inner_space_per_outer_space = data['rooms_per_outer_space']
        self.outer_space_relationship = data["outer_space_relationship"]
        self.outer_space_config = data['outer_space_config']
        self.inner_space_config = data['inner_space_config']
        self.room_type = data['room_type']
        self.x_axis = sorted(data['x_axis'])
        self.y_axis = sorted(data['y_axis'])

        res3 = self.encode_for_outer_space(self.outer_space_relationship, self.outer_space_per_building)

        self.outer_space_idx_per_building = dict()
        tp_count = 0
        self.storey_groups = []
        for terms in res3:
            self.outer_space_idx_per_building[f"{tp_count}"] = terms
            tp_count += 1
            tp_list = []
            for term in terms:
                tp_list.append(f"{term}")
            self.storey_groups.append(tp_list)


        print(self.storey_groups)



        self.storey_dict = dict()
        for s in self.storey_groups:
            key = tuple([int(i) for i in s])
            self.storey_dict[key] = list()

        print("self.storey_dict")
        print(self.storey_dict)

        self.inner = list()
        self.outer = list()
        self.outer_bound = list()
        self.inner_bound = list()
        self.inner_counter = list()
        self.outer_counter = list()
        self.direction_by_building = list()
        self.width_by_building = list()
        self.outer_space_idx_per_building = res3
        self._run()
        self.offsets = self._get_offsets()

    def _run(self):
        for storey_group in self.storey_dict.keys():
            sy_in = list()
            sx_in = list()
            sy_out = list()
            sx_out = list()
            s_width = list()

            cnt_in = collections.Counter()
            cnt_out = collections.Counter()

            outer = list()
            inner = list()

            for j, o in enumerate(storey_group):
                # o is the key for outer space

                sy_out.append(
                    sorted(list(set([coord[0] for coord in
                                     self.outer_space_config[str(o)]])))
                )
                sx_out.append(
                    sorted(list(set([coord[1] for coord in
                                     self.outer_space_config[str(o)]])))
                )

                if j == 0:
                    print(self.outer_space_per_building[str(o)])
                    print(self.outer_space_per_building[str(o)]["direction"])
                    if self.outer_space_per_building[str(o)]["direction"] == 'h':
                        self.direction_by_building.append("x")
                        self.width_by_building.append(
                            max(sx_out[0]) - min(sx_out[0]))
                        cnt_out += collections.Counter(
                            list(set([coord[0] for coord in
                                      self.outer_space_config[str(o)]])))
                    else:
                        self.direction_by_building.append('y')
                        self.width_by_building.append(
                            max(sy_out[0]) - min(sy_out[0]))
                        cnt_out += collections.Counter(
                            list(set([coord[1] for coord in
                                      self.outer_space_config[str(o)]])))
                if self.direction_by_building[-1] == 'x':
                    outer.append(
                        sorted(list(set([coord[0] for coord in
                                         self.outer_space_config[
                                             str(o)]])))
                    )
                else:
                    outer.append(
                        sorted(list(set([coord[1] for coord in
                                         self.outer_space_config[
                                             str(o)]])))
                    )

                ib = list()
                for i in self.inner_space_per_outer_space[str(o)]:
                    if self.direction_by_building[-1] == 'x':
                        ib.extend(
                            sorted(list(set([coord[0] for coord in
                                             self.inner_space_config[
                                                 str(i)]])))
                        )
                        # sy_in.append(
                        #     sorted(list(set([coord[0] for coord in
                        #                      self.inner_space_config[
                        #                          str(i)]])))
                        # )

                    else:
                        ib.extend(
                            sorted(list(set([coord[1] for coord in
                                             self.inner_space_config[
                                                 str(i)]])))
                        )
                        # sx_in.append(
                        #     sorted(list(set([coord[1] for coord in
                        #                      self.inner_space_config[
                        #                          str(i)]])))
                        # )

                inner.append(sorted(list(set(ib))))
                if self.direction_by_building[-1] == 'x':
                    sy_in.append(sorted(list(set(ib))))
                else:
                    sx_in.append(sorted(list(set(ib))))
                cnt_in += collections.Counter(list(set(ib)))

            self.inner_bound.append(inner)
            # sy_out = list(set(sy_out))
            # sx_out = list(set(sx_out))
            # sy_in = list(set(sy_in))
            # sx_in = list(set(sx_in))
            # sy_out.sort()
            # sx_out.sort()
            # sy_in.sort()
            # sx_in.sort()

            self.inner_counter.append(cnt_in)
            self.outer_counter.append(cnt_out)

            if self.direction_by_building[-1] == 'x':
                self.outer.append(sy_out)
                self.inner.append(sy_in)
            else:
                self.outer.append(sx_out)
                self.inner.append(sx_in)

            self.outer_bound.append(outer)

    def _get_offsets(self):
        offsets = []
        for storey in range(len(self.outer)):
            left = min([self.outer[storey][i][0] for i in
                        range(len(self.outer[storey]))])
            offsets.append(left)
        return offsets

    def get_algorithm_data(self):
        outer_segs = deepcopy(self.outer)
        inner_segs = deepcopy(self.inner)
        axis_segs = list()
        outer_counter = deepcopy(self.outer_counter)
        inner_counter = deepcopy(self.inner_counter)
        for building in range(len(self.outer)):
            left = self.offsets[building]
            if self.direction_by_building[building] == 'x':
                axis_segs.append([i - left for i in self.x_axis])
            else:
                axis_segs.append([i - left for i in self.y_axis])
            for i in range(len(self.outer[building])):
                outer_segs[building][i] = sorted(
                    [self.outer[building][i][j] - left for j in
                     range(len(self.outer[building][i]))])

                inner_segs[building][i] = sorted(
                    [self.inner[building][i][j] - left for j in
                     range(len(self.inner[building][i]))])
            outer_counter_ = collections.Counter()
            for key in outer_counter[building].keys():
                outer_counter_[key - left] = self.outer_counter[building][key]
            outer_counter[building] = outer_counter_
            inner_counter_ = collections.Counter()
            for key in self.inner_counter[building].keys():
                inner_counter_[key - left] = self.inner_counter[building][key]
            inner_counter[building] = inner_counter_

        rooms_per_outer_space = deepcopy(self.inner_space_per_outer_space)
        room_index_list = []
        building_index = []


        for value in self.outer_space_idx_per_building:
            temp_list_2 = []
            for index in value:
                label = rooms_per_outer_space[f"{index}"]
                temp_list_2.append(label)
            room_index_list.append(temp_list_2)
        room_index_list_ = deepcopy(room_index_list)

        room_type_new = {}
        room_type = deepcopy(self.room_type)
        for key, values in room_type.items():
            room_type_new[f"{key}"] = values["rooms"]

        for ii in range(len(room_index_list_)):
            for jj in range(len(room_index_list_[ii])):
                for kk in range(len(room_index_list_[ii][jj])):
                    for key, value in room_type_new.items():
                        if room_index_list_[ii][jj][kk] in value:
                            room_index_list_[ii][jj][kk] = int(key)

        return outer_segs, inner_segs, axis_segs, outer_counter, inner_counter, room_index_list_, self.width_by_building, self.storey_groups

    def convert_to_case(self, results):
        """
        :param results: [[[modules] x storey] x building]
        :return: case =
        {
            'zone1': {
                'story': 1,
                'direction': 'x',
                'modular': [3200, 3200],
                'width': 12600,
                'location': [left, bottom]
            },
        }
        """
        case_ = dict()
        zone_i = 1
        for i, building in enumerate(results):
            for (j, modules) in enumerate(building):
                outer_key = str(self.storey_groups[i][j])
                case_['zone{}'.format(zone_i)] = {
                    'story': self.outer_to_storey[outer_key] + 1,
                    'direction': self.direction_by_building[i],
                    'modular': modules,
                    'width': self.width_by_building[i],
                    'location': [
                        min([self.outer_space_config[outer_key][k][0] for k in
                             range(4)]),
                        min([self.outer_space_config[outer_key][k][1] for k in
                             range(4)]),
                    ]
                }
                zone_i += 1

        return [case_]



    # 构建图
    def build_graph(self, vertical):
        graph = defaultdict(list)
        for key, values in vertical.items():
            for value in values:
                graph[value].append(key)
        return graph

    # 深度优先搜索
    def dfs(self, node, visited, component, graph, vertical):
        visited.add(node)
        component.append(node)
        for neighbor in vertical[node]:
            for adjacent_node in graph[neighbor]:
                if adjacent_node not in visited:
                    self.dfs(adjacent_node, visited, component, graph, vertical)

    # 找到所有连通分量
    def find_connected_components(self, vertical):
        graph = self.build_graph(vertical)
        visited = set()
        components = []

        for node in vertical:
            if node not in visited:
                component = []
                self.dfs(node, visited, component, graph, vertical)
                components.append(component)

        return components

    def unique_list(self, list_):
        result_list = []
        for terms in list_:
            for term in terms:
                if term not in result_list:
                    result_list.append(term)
        return result_list

    def find_connected_value(self, _list, _dict, outer_space_per_building=None, _type=None):

        if _type == "h":
            result_ = []
            for terms in _list:
                temp_lst = []
                for term in terms:
                    temp_lst2 = _dict[term]
                    if outer_space_per_building[f"{temp_lst2[0]}"]["direction"] == \
                            outer_space_per_building[f"{temp_lst2[1]}"]["direction"]:
                        temp_lst.append(_dict[term])
                result_.append(self.unique_list(temp_lst))
            return result_
        else:
            result_ = []
            for terms in _list:
                temp_lst = []
                for term in terms:
                    temp_lst.append(_dict[term])
                result_.append(self.unique_list(temp_lst))
            return result_

    def encode_for_outer_space(self, outer_space_relationship, outer_space_per_building):

        print(outer_space_relationship)
        vertical = outer_space_relationship["vertical"]
        horizontal = outer_space_relationship["horizontal"]

        components = self.find_connected_components(vertical)
        res1 = self.find_connected_value(components, vertical)

        components = self.find_connected_components(horizontal)
        res2 = self.find_connected_value(components, horizontal, outer_space_per_building, "h")

        temp_list = []
        for term in res1:
            if len(term) != 0:
                temp_list.append(term)
        for term in res2:
            if len(term) != 0:
                temp_list.append(term)

        temp_dict = dict()
        for i in range(len(temp_list)):
            temp_dict[f"{i}"] = temp_list[i]

        components = self.find_connected_components(temp_dict)
        res3 = self.find_connected_value(components, temp_dict)

        return res3

    def plot_td_error(self, iter, td_error):

        import matplotlib.pyplot as plt
        # # 示例数据
        # iter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 迭代次数
        # td_error = [0.9, 0.7, 0.5, 0.3, 0.1, 0.15, 0.1, 0.005, 0.004, 0.003, 0.001, 0.0002]  # 对应的td_error

        # 绘制完整图形
        plt.figure(figsize=(10, 6))
        plt.plot(iter, td_error, marker='o', linestyle='-', color='b', label='TD Error')

        # 数据筛选
        filtered_iter = []
        filtered_td_error = []

        for ii, error in zip(iter, td_error):
            if 0.0001 <= error <= 0.01:
                filtered_iter.append(ii)
                filtered_td_error.append(error)

        # 在同一图上绘制放大图形
        plt.plot(filtered_iter, filtered_td_error, marker='o', linestyle='-', color='r',
                 label='Filtered TD Error (0.01 to 0.0001)')

        # 添加标题和标签
        plt.title('TD Error vs Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('TD Error')

        # 显示图例
        plt.legend()

        # 设置Y轴的对数尺度以更好地可视化
        plt.yscale('log')

        # 显示网格
        plt.grid(True)

        # 显示图形
        plt.show()

    def plot_reward(self, iter, reward):

        import matplotlib.pyplot as plt
        # # 示例数据
        # iter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 迭代次数
        # td_error = [0.9, 0.7, 0.5, 0.3, 0.1, 0.15, 0.1, 0.005, 0.004, 0.003, 0.001, 0.0002]  # 对应的td_error

        # 绘制完整图形
        plt.figure(figsize=(10, 6))
        plt.plot(iter, reward, marker='o', linestyle='-', color='grey', label='Reward')

        # 添加标题和标签
        plt.title('Reward vs Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')

        # 显示图例
        plt.legend()

        # 显示网格
        plt.grid(True)

        # 显示图形
        plt.show()

    def merge_buildings(self, outer, inner, inner_boundary, room, width):
        """
        合并多个建筑并返回更新后的信息
        :return: 合并后的数据和建筑位置信息
        """

        outer = copy.deepcopy(outer)
        inner = copy.deepcopy(inner)
        inner_boundary = copy.deepcopy(inner_boundary)
        room = copy.deepcopy(room)
        width = copy.deepcopy(width)

        def flatten_list(nested_list):
            """
            展平嵌套列表并去重
            :param nested_list: 嵌套列表
            :return: 展平后的列表
            """
            flattened = []

            def flatten_helper(lst):
                for item in lst:
                    if isinstance(item, list):
                        flatten_helper(item)
                    else:
                        flattened.append(item)

            flatten_helper(nested_list)
            return sorted(list(set(flattened)))  # 去重并排序

        num_buildings = len(outer)
        if num_buildings <= 1:
            # total_length = outer[0][0][1] if outer else 0
            total_length = max(pair[1] for pair in outer[0])
            return {
                'outer_range': [0, total_length],  # 合并后的单个范围
                'outer': outer,  # 合并后的单个范围
                'inner': inner,
                'inner_element': flatten_list(inner),
                'inner_boundary': inner_boundary[0] if inner_boundary else Counter(),
                'room': room,
                'width': width,
                'building_positions': [0],
                'total_length': total_length,
                'building_axis': []
            }
            # result = {
            #     'outer_range': [0, total_length],  # 合并后的单个范围
            #     'outer': outer,  # 合并后的单个范围
            #     'inner': new_inner,
            #     'inner_element': flatten_list(new_inner),
            #     'inner_boundary': merged_boundary,  # 合并后的boundary
            #     'room': [item for sublist in room for item in sublist],
            #     'width': width,
            #     'building_positions': building_positions,
            #     'building_lengths': building_lengths,
            #     'total_length': total_length,
            #     'building_ranges': building_ranges,
            #     'building_axis': building_axis
            # }
        # 获取每栋建筑的起始位置和长度信息
        building_positions = [0]  # 记录每栋建筑的起始位置
        building_lengths = []  # 记录每栋建筑的长度

        # 计算第一栋建筑的长度
        first_length = max(outer[0][j][1] for j in range(len(outer[0])))
        building_lengths.append(first_length)

        # 计算后续建筑的起始位置和长度
        current_position = first_length
        for i in range(1, num_buildings):
            building_positions.append(current_position)
            current_length = max(outer[i][j][1] - outer[i][j][0] for j in range(len(outer[i])))
            building_lengths.append(current_length)
            current_position += current_length

        # 调整每栋建筑的坐标（除第一栋外）
        for building_idx in range(1, num_buildings):
            start_pos = building_positions[building_idx]

            # 调整outer
            for i in range(len(outer[building_idx])):
                length = outer[building_idx][i][1] - outer[building_idx][i][0]
                outer[building_idx][i] = [start_pos, start_pos + length]

            # 调整inner
            for i in range(len(inner[building_idx])):
                old_inner = inner[building_idx][i]
                new_inner = [x + start_pos for x in old_inner]
                inner[building_idx][i] = new_inner

            new_inner = [item for sublist in inner for item in sublist]

            # 调整inner_boundary
            new_boundary = Counter()
            for key, value in inner_boundary[building_idx].items():
                new_boundary[key + start_pos] = value

            inner_boundary[building_idx] = new_boundary

        # 合并inner_boundary
        merged_boundary = Counter()
        for boundary in inner_boundary:
            for key, value in boundary.items():
                merged_boundary[key] += value

        # 计算总长度
        total_length = sum(building_lengths)
        building_ranges = list(zip(building_positions,
                                   [pos + length for pos, length in zip(building_positions, building_lengths)]))

        building_axis = sorted(set(number for tuple_item in building_ranges for number in tuple_item))

        # 返回合并后的数据和位置信息
        result = {
            'outer_range': [0, total_length],  # 合并后的单个范围
            'outer': outer,  # 合并后的单个范围
            'inner': new_inner,
            'inner_element': flatten_list(new_inner),
            'inner_boundary': merged_boundary,  # 合并后的boundary
            'room': [item for sublist in room for item in sublist],
            'width': width,
            'building_positions': building_positions,
            'building_lengths': building_lengths,
            'total_length': total_length,
            'building_ranges': building_ranges,
            'building_axis': building_axis
        }

        return result


    def replace_with_room_sequence(self, buildings, room_widths, grouped_idx, goal_idx):
        """
        将建筑范围替换为房间宽度序列
        :param buildings: 建筑范围列表
        :param room_widths: 房间宽度列表
        :return: 替换后的列表
        """
        result = [[] for _ in buildings]

        # 遍历每个子列表
        for i, building_group in enumerate(buildings):
            # 遍历每个建筑范围
            for building_range in building_group:
                start, end = building_range
                room_sequence = []
                current_pos = 0
                width_sum = 0

                # 从位置0开始累加房间宽度
                for width in room_widths:
                    if start <= current_pos < end:  # 如果当前位置在范围内
                        room_sequence.append(width)
                    current_pos += width

                    if current_pos >= end:  # 如果超出范围就停止
                        break

                result[i].append(room_sequence)

        # 创建一个字典，将 grouped_idx 与 result 关联
        group_to_result = {tuple(group): res for group, res in zip(grouped_idx, result)}

        # 初始化一个空列表来存放重新排序的结果
        ordered_result = []

        # 遍历 goal_idx，根据分组找到对应的值，并添加到结果列表中
        for idx in goal_idx:
            for group in grouped_idx:
                if idx in group:
                    group_index = group.index(idx)
                    ordered_result.append(group_to_result[tuple(group)][group_index])
                    break
        return ordered_result