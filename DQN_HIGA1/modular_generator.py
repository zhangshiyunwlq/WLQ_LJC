import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pyvista as pv


class ModluarGenarator:
    def __init__(self, json_data,hor_connection, ver_connection):
        self.json_data = json_data
        self.hor_connection=hor_connection
        self.ver_connection = ver_connection
    def run(self):
        result = self.json_data
        # 生成数据并进行编号
        modular_data = self.create_modular_dict(result)
        # 缩小模块尺寸
        modular_data =self.scale_modular_coordinates(modular_data, self.hor_connection)
        # 可视化缩小后的模块
        # self.visualize_modulars_simple(modular_data)
        # 增加每个模块的顶部面和柱数据
        modular_data =self.add_top_data(modular_data,self.ver_connection)
        # 可视化增加顶面的数据
        # self.visualize_modulars_simple(modular_data)
        # 增加垂直连接
        modular_data,ver_connection =self.add_vertical_connections(modular_data)
        # self.visualize_modulars_simple(modular_data)
        # 按照水平面划分数据
        hor_data = self.classify_by_story_position(modular_data)
        # 生成水平连接
        hor_connection = self.find_all_floor_connections(hor_data,self.hor_connection)
        # self.visualize_modulars_simple(modular_data, ver_connection=ver_connection, hor_connection=hor_connection)
        # 按照模块单元类型划分模块
        modular_divided=self.group_modulars_by_type(result['GraphModular'])

        modular_data = self.add_planes_to_modular(modular_data)

        return {'modular_data':modular_data,'hor_connection':hor_connection,'ver_connection':ver_connection,'modular_group':modular_divided}

    def group_modulars_by_type(self,graph_modular):
        """
        将modulars按照type分组

        参数:
        - graph_modular: 输入的模块数据字典

        返回:
        - 按type分组的字典
        """
        type_groups = {}

        # 遍历所有modular
        for modular_id, modular_info in graph_modular.items():
            modular_type = modular_info['type']

            # 如果这个type还没有在字典中，创建一个新的列表
            if modular_type not in type_groups:
                type_groups[modular_type] = []

            # 将modular_id添加到对应type的列表中
            type_groups[modular_type].append(int(modular_id))

        # 对每个类型的列表进行排序
        for type_key in type_groups:
            type_groups[type_key].sort()

        return type_groups

    def generate_id(self,modular_id, type_code, number):
        """
        生成编码
        modular_id: 模块编号
        type_code: 0表示node, 1表示frame
        number: 节点或构件编号
        """
        return int(f"{modular_id}{type_code}{number:03d}")

    def get_plane_nodes_label(self, frames_dict, node_type):
        """获取指定类型平面的节点"""
        """获取指定类型平面的节点和构件"""
        nodes = set()
        frames = []

        # 收集指定类型frame中的所有节点和构件id
        for frame_id, frame in frames_dict.items():
            if frame['type'] == node_type:
                nodes.update(frame['nodes'])
                frames.append(int(frame_id))

        return sorted(list(nodes)), sorted(frames)

    def add_planes_to_modular(self,modular_dict):
        """
        为模块单元字典添加平面信息

        参数:
        modular_dict: dict, 模块单元字典

        返回:
        更新后的模块单元字典
        """

        for modular_id, modular_data in modular_dict.items():
            # 初始化planes字典
            modular_data['planes'] = {}

            # 获取底部和顶部的节点和构件
            bottom_nodes, bottom_frames = self.get_plane_nodes_label(modular_data['frames'], 'bottom')
            top_nodes, top_frames = self.get_plane_nodes_label(modular_data['frames'], 'top')

            # 生成底部平面ID
            bottom_plane_id = self.generate_id(int(modular_id), 2, 1)
            # 生成顶部平面ID
            top_plane_id = self.generate_id(int(modular_id), 2, 2)

            # 添加底部平面
            modular_data['planes'][str(bottom_plane_id)] = {
                'id': int(bottom_plane_id),
                'nodes': bottom_nodes,
                'frames': bottom_frames,  # 添加底部构件列表
                'position': 'bottom',
                # 'rect_id': int(modular_id)
            }

            # 添加顶部平面
            modular_data['planes'][str(top_plane_id)] = {
                'id': int(top_plane_id),
                'nodes': top_nodes,
                'frames': top_frames,  # 添加顶部构件列表
                'position': 'top',
                # 'rect_id': int(modular_id)
            }

        return modular_dict


    def create_modular_dict(self,data):
        result = {}

        # 首先获取所有楼层的高度信息
        story_heights = {}
        for story_id, story_info in data["GraphBuildingStorey"].items():
            story_heights[story_id] = story_info["story_height"]

        # 遍历GraphModular中的每个模块
        for modular_id, modular_data in data["GraphModular"].items():
            modular_id = int(modular_id)

            # 获取该模块所在的space信息
            space_idx = str(modular_data["space_idx"])
            story_idx = data["GraphSpace"][space_idx]["story_idx"]

            # 为每个模块创建新的字典结构
            modular_info = {
                "story": story_idx,
                "story_height": story_heights.get(str(story_idx), 0),  # 使用字符串作为键获取高度
                "nodes": {},
                "frames": {},
                'type':modular_data['type']
            }

            # 处理节点信息
            coordinates = modular_data["coordinates"]
            for i, coord in enumerate(coordinates, 1):
                node_id = self.generate_id(modular_id, 0, i)  # 生成node_id
                modular_info["nodes"][node_id] = coord

            # 处理frame信息
            for i in range(len(coordinates)):
                frame_id = self.generate_id(modular_id, 1, i + 1)  # 生成frame_id
                start_node = self.generate_id(modular_id, 0, i + 1)
                end_node = self.generate_id(modular_id, 0, (i + 1) % len(coordinates) + 1)

                modular_info["frames"][frame_id] = {
                    "nodes": [start_node, end_node]
                }

            result[modular_id] = modular_info

        return result

    def scale_modular_coordinates(self,modular_dict, reduce):
        """
        缩放模块尺寸
        modular_dict: 原始模块字典
        x_reduce: x方向缩短距离
        y_reduce: y方向缩短距离
        """
        x_reduce = reduce
        y_reduce = reduce
        def get_rectangle_center(coords):
            """计算矩形中心点"""
            x_sum = sum(coord[0] for coord in coords)
            y_sum = sum(coord[1] for coord in coords)
            return [x_sum / 4, y_sum / 4]

        def scale_point(point, center, x_scale, y_scale):
            """根据中心点缩放单个点的坐标"""
            dx = point[0] - center[0]
            dy = point[1] - center[1]

            # 计算缩放后的偏移量
            if dx > 0:
                dx -= x_scale / 2
            elif dx < 0:
                dx += x_scale / 2

            if dy > 0:
                dy -= y_scale / 2
            elif dy < 0:
                dy += y_scale / 2

            return [
                center[0] + dx,
                center[1] + dy,
                point[2]  # z坐标保持不变
            ]

        result = {}

        # 遍历每个模块
        for modular_id, modular_info in modular_dict.items():
            result[modular_id] = {
                "story": modular_info["story"],
                "story_height": modular_info["story_height"],
                "nodes": {},
                "frames": {},
                "type": modular_info["type"],
            }

            # 获取原始坐标列表
            original_coords = [coord for coord in modular_info["nodes"].values()]
            center = get_rectangle_center(original_coords)

            # 缩放每个节点
            for node_id, coord in modular_info["nodes"].items():
                new_coord = scale_point(coord, center, x_reduce, y_reduce)
                result[modular_id]["nodes"][node_id] = new_coord

            # 保持frame的连接关系不变
            result[modular_id]["frames"] = modular_info["frames"]

        return result

    def visualize_modulars_simple(self, modular_dict, ver_connection=None, hor_connection=None, highlight_frames=None):
        """
        使用PyVista进行3D可视化，支持高亮显示特定构件

        参数：
        modular_dict: 模块单元字典
        ver_connection: 垂直连接信息
        hor_connection: 水平连接信息
        highlight_frames: 需要高亮显示的构件ID字典
        """
        # 创建PyVista plotter
        plotter = pv.Plotter()
        plotter.set_background('white')

        # 设置样式参数
        node_color = 'red'
        frame_color = 'blue'
        highlight_color = 'red'  # 高亮构件的颜色
        hor_conn_color = 'green'
        ver_conn_color = 'green'

        node_size = 3
        frame_width = 3.5
        highlight_width = 4.0  # 高亮构件的线宽
        hor_conn_width = 2.5
        ver_conn_width = 2.5

        # 创建所有节点的字典
        all_nodes = {node_id: coords for mod_data in modular_dict.values()
                     for node_id, coords in mod_data["nodes"].items()}

        # 1. 绘制节点
        for modular_info in modular_dict.values():
            if "nodes" in modular_info:
                for node_id, coord in modular_info["nodes"].items():
                    if len(coord) == 3:
                        sphere = pv.Sphere(radius=node_size, center=coord)
                        plotter.add_mesh(sphere, color=node_color)

        # 2. 绘制框架
        for modular_info in modular_dict.values():
            if "frames" in modular_info:
                for frame_id, frame_info in modular_info["frames"].items():
                    node1_id = frame_info["nodes"][0]
                    node2_id = frame_info["nodes"][1]

                    if node1_id in modular_info["nodes"] and node2_id in modular_info["nodes"]:
                        coord1 = modular_info["nodes"][node1_id]['coord']
                        coord2 = modular_info["nodes"][node2_id]['coord']
                        line = pv.Line(coord1, coord2)

                        # 判断是否为需要高亮的构件
                        if highlight_frames and int(frame_id) in highlight_frames:
                            plotter.add_mesh(line, color=highlight_color, line_width=highlight_width)
                        else:
                            plotter.add_mesh(line, color=frame_color, line_width=frame_width)

        # 3. 绘制水平连接
        if hor_connection:
            for connectionid, connection in hor_connection.items():
                node1_id, node2_id = connection
                if node1_id in all_nodes and node2_id in all_nodes:
                    node1_coords = all_nodes[node1_id]['coord']
                    node2_coords = all_nodes[node2_id]['coord']
                    line = pv.Line(node1_coords, node2_coords)
                    plotter.add_mesh(line, color=hor_conn_color, line_width=hor_conn_width)

        # 4. 绘制垂直连接
        if ver_connection:
            for connectionid, connection in ver_connection.items():
                node1_id, node2_id = connection
                if node1_id in all_nodes and node2_id in all_nodes:
                    node1_coords = all_nodes[node1_id]['coord']
                    node2_coords = all_nodes[node2_id]['coord']
                    line = pv.Line(node1_coords, node2_coords)
                    plotter.add_mesh(line, color=ver_conn_color, line_width=ver_conn_width)

        # 添加坐标轴和设置相机
        plotter.add_axes()
        plotter.camera_position = 'iso'

        # 显示场景
        plotter.show()
    def add_top_data(self,modular_dict, a3):
        """
        缩放模块坐标并添加顶面
        参数:
        - modular_dict: 模块字典
        - a1, a2, a3: 缩放参数
        """
        result = {}

        for modular_id, modular_info in modular_dict.items():
            new_modular = {
                "story": modular_info["story"],
                "story_height": modular_info["story_height"],
                "nodes": {},
                "frames": {},
                "type": modular_info["type"],
            }

            # 获取该模块的节点数量（原始底面节点数）
            num_original_nodes = len(modular_info["nodes"])

            # 处理底面节点（前4个节点）
            for node_id, coord in modular_info["nodes"].items():
                new_coord = [
                    coord[0],
                    coord[1],
                    coord[2]
                ]
                new_modular["nodes"][node_id] = {'coord':new_coord,'type':'bottom'}

                # 计算并添加对应的顶面节点（后4个节点）
                # 新节点ID为原节点ID + 4
                top_node_id = self.generate_id(modular_id, 0, int(str(node_id)[-4:]) + num_original_nodes)
                height_diff = modular_info["story_height"] - a3
                top_coord = [
                    new_coord[0],
                    new_coord[1],
                    new_coord[2] + height_diff
                ]
                # new_modular["nodes"][top_node_id] = top_coord
                new_modular["nodes"][top_node_id] = {'coord': top_coord, 'type': 'top'}

            # 处理底面frames（前4个frame）
            for frame_id, frame_info in modular_info["frames"].items():
                new_modular["frames"][frame_id] = frame_info.copy()
                new_modular["frames"][frame_id]["type"] = 'bottom'
                # 添加顶面对应的frame（后4个frame）
                new_frame_id = self.generate_id(modular_id, 1, int(str(frame_id)[-4:]) + num_original_nodes)
                # 计算顶面frame的节点
                top_nodes = [
                    self.generate_id(modular_id, 0, int(str(node_id)[-4:]) + num_original_nodes)
                    for node_id in frame_info["nodes"]
                ]
                new_modular["frames"][new_frame_id] = {"nodes": top_nodes, "type": 'top'}

                # 添加连接底面和顶面的竖直frame
                vertical_frame_id = self.generate_id(modular_id, 1, int(str(frame_id)[-4:]) + 2 * num_original_nodes)
                bottom_node = frame_info["nodes"][0]  # 使用frame的第一个节点
                top_node = self.generate_id(modular_id, 0, int(str(bottom_node)[-4:]) + num_original_nodes)
                new_modular["frames"][vertical_frame_id] = {"nodes": [bottom_node, top_node],"type": 'column'}

            all_coords =[]
            for key, node_data in new_modular["nodes"].items():
                all_coords.append(node_data["coord"])

            # 计算中心点坐标（8个顶点的平均值）
            center_x = sum(coord[0] for coord in all_coords) / len(all_coords)
            center_y = sum(coord[1] for coord in all_coords) / len(all_coords)
            center_z = sum(coord[2] for coord in all_coords) / len(all_coords)
            new_modular["center"] = [center_x, center_y, center_z]
            result[modular_id] = new_modular

        return result

    def add_vertical_connections(self,modular_dict):
        """
        为相同水平位置的模块单元添加垂直连接
        参数:
        - modular_dict: 包含中心点信息的模块字典
        返回:
        - 添加了垂直连接的模块字典
        """
        # 根据中心点的x,y坐标对模块进行分类
        vertical_connection = {}
        vertical_groups = {}
        all_id = 1
        for modular_id, modular_info in modular_dict.items():
            # 使用中心点的x,y坐标作为分类键（保留两位小数以处理浮点数误差）
            center = modular_info["center"]
            key = (round(center[0], 2), round(center[1], 2))
            if key not in vertical_groups:
                vertical_groups[key] = []
            vertical_groups[key].append((modular_id, modular_info))

        # 复制原始字典并为每个模块添加vertical_connections字典
        result = {}
        for k, v in modular_dict.items():
            result[k] = v.copy()
            # result[k]["vertical_connections"] = {}  # 添加新的垂直连接字典

        # 在每个垂直组内添加连接
        for group in vertical_groups.values():
            # 按story排序，确保从底层到顶层
            sorted_group = sorted(group, key=lambda x: int(x[1]["story"]))

            # 对除最顶层外的每个模块添加与上层模块的连接
            for i in range(len(sorted_group) - 1):
                current_modular = sorted_group[i]
                next_modular = sorted_group[i + 1]

                current_id = current_modular[0]
                current_info = current_modular[1]
                next_info = next_modular[1]

                # 获取当前模块的顶部节点和下一个模块的底部节点
                current_top_nodes = {k: v for k, v in current_info["nodes"].items()
                                     if int(str(k)[-4:]) > 4}  # 顶部节点
                next_bottom_nodes = {k: v for k, v in next_info["nodes"].items()
                                     if int(str(k)[-4:]) <= 4}  # 底部节点

                # 匹配水平位置相同的节点并添加连接
                connection_count = 1
                for top_id, top_coord in current_top_nodes.items():
                    for bottom_id, bottom_coord in next_bottom_nodes.items():
                        if (round(top_coord['coord'][0], 2) == round(bottom_coord['coord'][0], 2) and
                                round(top_coord['coord'][1], 2) == round(bottom_coord['coord'][1], 2)):
                            # 生成新的connection_id
                            connection_id = self.generate_id(current_id, 2, connection_count)
                            # 添加垂直连接
                            # result[current_id]["vertical_connections"][connection_id] = {
                            #     "nodes": [top_id, bottom_id]
                            # }
                            ver_id= all_id+130000000
                            vertical_connection[ver_id]=[top_id, bottom_id]
                            # vertical_connection.append([top_id, bottom_id])
                            all_id+=1
                            connection_count += 1

        return result,vertical_connection

    def classify_by_story_position(self,result):
        """
        按照楼层的顶部和底部对模块的节点和框架进行分类
        参数:
        - result: 包含模块信息的字典
        返回:
        - 分类后的字典
        """
        classified_dict = {}

        for modular_id, modular_info in result.items():
            story = modular_info["story"]

            # 创建底部和顶部的键
            bottom_key = f"{story}bottom"
            top_key = f"{story}top"

            # 初始化字典
            if bottom_key not in classified_dict:
                classified_dict[bottom_key] = {}
            if top_key not in classified_dict:
                classified_dict[top_key] = {}

            # 初始化模块ID对应的字典
            if modular_id not in classified_dict[bottom_key]:
                classified_dict[bottom_key][modular_id] = {"nodes": {}, "frames": {}}
            if modular_id not in classified_dict[top_key]:
                classified_dict[top_key][modular_id] = {"nodes": {}, "frames": {}}

            # 分类节点
            for node_id, coord in modular_info["nodes"].items():
                # 判断节点是否为底部节点（通过节点编号的最后四位）
                node_position = int(str(node_id)[-4:])
                if node_position <= 4:  # 底部节点
                    classified_dict[bottom_key][modular_id]["nodes"][node_id] = coord
                else:  # 顶部节点
                    classified_dict[top_key][modular_id]["nodes"][node_id] = coord

            # 分类框架
            for frame_id, frame_info in modular_info["frames"].items():
                node1_id = frame_info["nodes"][0]
                node2_id = frame_info["nodes"][1]

                # 判断框架是否属于底部或顶部
                node1_position = int(str(node1_id)[-4:])
                node2_position = int(str(node2_id)[-4:])

                # 如果两个节点都是底部节点
                if node1_position <= 4 and node2_position <= 4:
                    classified_dict[bottom_key][modular_id]["frames"][frame_id] = frame_info
                # 如果两个节点都是顶部节点
                elif node1_position > 4 and node2_position > 4:
                    classified_dict[top_key][modular_id]["frames"][frame_id] = frame_info

        return classified_dict

    def find_all_floor_connections(self,classified_dict, connection_distance):
        """
        找出所有楼层的水平连接

        参数:
        - classified_dict: 按照楼层分类好的数据
        - connection_distance: 连接距离参数

        返回:
        - floor_connections: 包含所有楼层连接的字典
        """
        floor_connections = {}
        hor_connection = {}
        all_id = 0
        # 遍历每个楼层
        for floor_name, floor_data in classified_dict.items():
            # 获取当前楼层的连接
            connections = self.find_connections_between_modulars(floor_data, connection_distance)
            floor_connections[floor_name] = connections

        connection_dict = {'connections': []}

        # 遍历所有楼层的连接并添加到列表中
        for floor_connections in floor_connections.values():
            for connection in floor_connections:
                # connection已经是[node1, node2]格式
                connection_dict['connections'].append(list(connection))

        for i in range(len(connection_dict['connections'])):
            hor_id = all_id + 140000000
            hor_connection[hor_id] = connection_dict['connections'][i]
            all_id += 1
        return hor_connection

    def find_connections_between_modulars(self,floor_data, connection_distance):
        """
        寻找模块间的连接
        """
        connections = []

        for mod1_id, mod1_data in floor_data.items():
            mod1_nodes = mod1_data["nodes"]

            for node1_id, node1_coords in mod1_nodes.items():
                node1_pos = np.array(node1_coords['coord'][:2])

                for mod2_id, mod2_data in floor_data.items():
                    if mod1_id == mod2_id:
                        continue

                    mod2_nodes = mod2_data["nodes"]

                    for node2_id, node2_coords in mod2_nodes.items():
                        node2_pos = np.array(node2_coords['coord'][:2])

                        distance = np.linalg.norm(node2_pos - node1_pos)

                        if (abs(node1_pos[0] - node2_pos[0]) < 0.1 or
                                abs(node1_pos[1] - node2_pos[1]) < 0.1):
                            if abs(distance - connection_distance) < 0.1:
                                connection = tuple(sorted([node1_id, node2_id]))
                                if connection not in connections:
                                    connections.append(connection)

        return connections
