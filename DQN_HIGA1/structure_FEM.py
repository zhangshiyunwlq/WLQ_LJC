import copy
import os
import openseespy.opensees as ops
import opstool_1 as opst
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from collections import defaultdict
import math
from scipy.spatial import ConvexHull



class Structural_model:
    def __init__(self, all_data_info,unit_data,modular_variable,standardstory_data,material_data):
        self.all_data_info = all_data_info
        self.unit_data = unit_data
        self.modular_variable=modular_variable
        self.standardstory_data=standardstory_data
        self.material_data=material_data
    def run(self):
        self.heights = []
        for standard_floor in self.standardstory_data.values():
            # 获取该标准层的高度和包含的楼层
            height = standard_floor['story_height']
            floors = standard_floor['story_id']

            # 为该标准层包含的每个楼层添加相应的高度
            for _ in floors:
                self.heights.append(height)
        modular_group=self.classify_modulars_by_story_and_type(self.all_data_info["modular_data"], self.all_data_info["modular_group"],self.standardstory_data)
        modular_all_data = self.assign_section_to_modulars(self.all_data_info["modular_data"], modular_group, self.modular_variable)

        self.story_weight = self.calculate_level_weight(self.all_data_info)

        Seismic_Force = Seismic_force(self.story_weight, self.heights)
        # 计算地震折算到节点的力
        self.base_shear = Seismic_Force.calculate_base_shear(0.04)  # 基底剪力
        self.story_force = Seismic_Force.distribute_lateral_forces(self.base_shear, mode_shape=None)  # 各层剪力

        node_data = self.extract_nodes_info(self.all_data_info['modular_data'])
        frames_data = self.extract_frame_info(self.all_data_info['modular_data'])

        self.nodes_force = Seismic_Force.distribute_forces_to_nodes(self.story_force,node_data)  # 各节点力

        # 计算风荷载折算到节点的力
        Wind_Force = Wind_force(self.heights)
        self.floor_dimensions = self.get_floor_dimensions(node_data)
        uz = Wind_Force.calculate_wind_load(self.heights, 'C')
        us = Wind_Force.calculate_vibration_coefficient()
        self.story_wind_force = Wind_Force.distribute_wind_load_to_stories(self.floor_dimensions, self.heights,
                                                                           0.35, uz, us, 0.8, -0.5)
        self.node_wind_force = Wind_Force.distribute_wind_to_nodes(self.story_wind_force,node_data,is_moment_frame=True)

        # 计算重力荷载折算到柱节点的力
        node_column = self.column_node_division(node_data,frames_data)
        GravityForce = Gravity_force(self.story_weight)
        node_gravity_force = GravityForce.calculate_gravity_loads(node_column)

        # 计算水平荷载
        self.result_dict = {}

        # 遍历第一个字典的键
        for key in self.node_wind_force:
            # 确保两个字典都有这个键
            if key in self.nodes_force:
                # 将对应元组的值相加
                self.result_dict[key] = tuple(x + y for x, y in zip(self.node_wind_force[key], self.nodes_force[key]))

        # 重力荷载和水平荷载相加
        self.all_force_result = copy.deepcopy(self.result_dict)
        for node_id, forces_A in node_gravity_force.items():
            if node_id in self.all_force_result:
                # 节点在两个字典中都存在，相加力分量
                forces_B = self.all_force_result[node_id]
                combined_forces = tuple(a + b for a, b in zip(forces_A, forces_B))
                self.all_force_result[node_id] = combined_forces
            else:
                # 节点只在node_forces中存在
                self.all_force_result[node_id] = forces_A

        self.structure_model(modular_all_data,node_data,frames_data)

        self.displacement_result, self.force_result, self.node_tag, self.ele_tag = self.out_dis_data()

        self.story_displacements = self.calculate_dis_index(self.all_data_info, self.displacement_result,self.node_tag)
        self.inter_displacements = self.calculate_interdis_index(self.all_data_info, self.displacement_result,
                                                                 self.heights)
        self.force_verify, self.max_force,total_weight,out_value = self.calculate_frame_index(self.all_data_info, self.force_result,self.ele_tag)

        analysis_data = {
            'story_dis':self.story_displacements,
            'inter_dis': self.inter_displacements,
            'max_force':self.max_force,
            'total_weight':total_weight,
            'out_value': out_value
        }
        return analysis_data
    def structure_model(self,modular_all_data,node_data,frames_data):
        modular_data = modular_all_data
        hor_connection = self.all_data_info['hor_connection']
        ver_connection = self.all_data_info['ver_connection']
        modular_group = self.all_data_info['modular_group']


        # 初始化模型
        ops.wipe()  # 清除之前的模型数据
        ops.model('basic', '-ndm', 3, '-ndf', 6)  # 3D模型，6个自由度（x, y, z 平移 + x, y, z 旋转）
        section_data = self.material_data['section_types']
        ops.geomTransf('Linear', 1, *[0, 1, 0])

        # 对于沿 Y 方向的构件，局部 y 轴选择 [0, 0, 1] (全局 Z 方向)
        ops.geomTransf('Linear', 2, *[0, 0, 1])

        # 对于沿 Z 方向的构件，局部 y 轴选择 [1, 0, 0] (全局 X 方向)
        ops.geomTransf('Linear', 3, *[1, 0, 0])

        # 定义材料和截面
        ops.uniaxialMaterial('Elastic', 1, 206000)
        # ops.section('Elastic', self.initial_id+1, 3000.0, 0.1, 0.1)

        G = 80000.0  # 剪切模量 (N/mm²)
        for i in range(len(section_data)):
            A = section_data[f'{i}']['Area']
            E = 206000
            Ix = section_data[f'{i}']['I22']
            Iy = section_data[f'{i}']['I33']
            J = section_data[f'{i}']['Torsion']
            # 定义截面
            ops.section('Elastic', i, E, A, Ix, Iy, G, J)
            # 定义梁柱单元的积分方法 (Lobatto积分，5个积分点)
            ops.beamIntegration('Lobatto', i, i, 5)


        node_all_data = {}

        # 建立节点
        for i in range(len(modular_data)):
            # 创建节点 (编号, x坐标, y坐标, z坐标)
            for node_key,node_crood in modular_data[i+1]['nodes'].items():
                ops.node(node_key, node_crood['coord'][0], node_crood['coord'][1],node_crood['coord'][2])
                node_all_data[node_key]=node_crood
        # 建立构件
        for modular_id,modular_id_data in modular_data.items():
            for frame_id, frame_data in modular_id_data['frames'].items():
                index1, index2 = frame_data['nodes']
                direct = self.loacl_direction(node_all_data[index1]['coord'], node_all_data[index2]['coord'])
                frame_location = frame_data['type']
                frame_section = self.unit_data[modular_id_data['modular_type']][frame_location]
                A = section_data[f'{frame_section}']['Area']
                E = 206000
                Ix = section_data[f'{frame_section}']['I22']
                Iy = section_data[f'{frame_section}']['I33']
                J = section_data[f'{frame_section}']['Torsion']
                ops.element('elasticBeamColumn', frame_id, index1,
                            index2, A,E, G, J, Ix, Iy, direct, '-integration', 'Lobatto',5)


        # 建立水平连接
        for hor_id, hor_id_data in hor_connection.items():
            frame_section = 11
            A = section_data[f'{frame_section}']['Area']
            E = 206000
            Ix = section_data[f'{frame_section}']['I22']
            Iy = section_data[f'{frame_section}']['I33']
            J = section_data[f'{frame_section}']['Torsion']
            index1, index2 = hor_id_data
            direct = self.loacl_direction(node_all_data[index1]['coord'], node_all_data[index2]['coord'])
            # 创建梁柱构件 (梁柱连接的节点号) 1是定义刚度矩阵的方向或变形模式等几何变换编号
            ops.element('elasticBeamColumn', hor_id, index1, index2, A, E, G, J, Ix, Iy, direct,'-integration', 'Lobatto', 5)


        # 建立垂直连接
        for ver_id, ver_id_data in ver_connection.items():
            frame_section = 11
            A = section_data[f'{frame_section}']['Area']
            E = 206000
            Ix = section_data[f'{frame_section}']['I22']
            Iy = section_data[f'{frame_section}']['I33']
            J = section_data[f'{frame_section}']['Torsion']
            index1, index2 = ver_id_data
            direct = self.loacl_direction(node_all_data[index1]['coord'], node_all_data[index2]['coord'])
            # 创建梁柱构件 (梁柱连接的节点号) 1是定义刚度矩阵的方向或变形模式等几何变换编号
            ops.element('elasticBeamColumn', ver_id, index1, index2, A, E, G, J, Ix, Iy, direct,'-integration', 'Lobatto', 5)

        self.boundary_modelling(modular_data)
        self.load_modelling(modular_data,node_data,frames_data)

        # 设置求解器和分析选项
        ops.system('BandGeneral')  # 线性方程求解器
        ops.numberer('RCM')  # 自由度编号方法
        # ops.numberer('Plain')
        ops.constraints('Plain')  # 约束处理方法
        # ops.constraints('Transformation')
        ops.test('NormDispIncr', 1.0e-6, 10)  # 收敛判据
        ops.integrator('LoadControl', 1.0)  # 负载控制积分器
        # 'Newton', 'ModifiedNewton', 'KrylovNewton','NewtonLineSearch'
        ops.algorithm('Newton')  # 牛顿迭代算法
        ops.analysis('Static')  # 静力分析

        ops.analyze(1)

        # print(f"线程:{self.work_id}")

        # 输出结果
        ODB = opst.post.CreateODB(odb_tag=1)
        ODB.fetch_response_step()
        ODB.save_response()

        # nodal_resp = opst.post.get_nodal_responses(
        #     odb_tag=1,
        #     resp_type="disp",  # 指定获取位移数据
        #     node_tags=None
        # )
        # 获取最后一步的位移数据

    def loacl_direction(self, node1, node2):
        if node1[0] != node2[0]:
            direction = 1
        elif node1[1] != node2[1]:
            direction = 2
        elif node1[2] != node2[2]:
            direction = 3
        return direction

    def classify_modulars_by_story_and_type(self,modular_units, type_groups,standardstory_data):
        """
        将模块单元按照楼层和类型进行分类

        参数:
        - modular_units: 包含模块单元信息的字典
        - type_groups: 按type分组的模块编号字典

        返回:
        - 按楼层和类型分类的嵌套字典
        """
        # 创建结果字典
        classified_dict = {}

        # 遍历每个类型组
        for type_id, modular_ids in type_groups.items():
            # 遍历该类型组中的每个模块
            for modular_id in modular_ids:
                if modular_id in modular_units:
                    # 获取楼层信息
                    story = modular_units[modular_id]['story']

                    # 如果楼层不存在，创建新的楼层字典
                    if story not in classified_dict:
                        classified_dict[story] = {}

                    # 如果该楼层中不存在该类型，创建新的类型列表
                    if type_id not in classified_dict[story]:
                        classified_dict[story][type_id] = []

                    # 将模块ID添加到对应的分类中
                    classified_dict[story][type_id].append(modular_id)

        # 对每个列表进行排序
        for story in classified_dict:
            for type_id in classified_dict[story]:
                classified_dict[story][type_id].sort()

        merged_dict = {}

        # 创建楼层映射关系
        story_to_standard = {}
        for std_floor, info in standardstory_data.items():
            for story in info['story_id']:
                story_to_standard[str(story)] = std_floor

        # 遍历原始分类字典
        for story in classified_dict:
            # 获取对应的标准层
            standard_floor = story_to_standard.get(story)
            if standard_floor is None:
                continue

            # 如果标准层还没有在新字典中，创建它
            if standard_floor not in merged_dict:
                merged_dict[standard_floor] = {}

            # 遍历该楼层的所有类型
            for type_id in classified_dict[story]:
                # 如果该类型还没有在标准层中，创建它
                if type_id not in merged_dict[standard_floor]:
                    merged_dict[standard_floor][type_id] = []

                # 添加模块ID
                merged_dict[standard_floor][type_id].extend(classified_dict[story][type_id])

        # 对每个分组内的模块ID进行排序
        for standard_floor in merged_dict:
            for type_id in merged_dict[standard_floor]:
                merged_dict[standard_floor][type_id] = sorted(set(merged_dict[standard_floor][type_id]))

        return merged_dict

    def get_floor_dimensions(self, nodes_dict):
        # 初始化结果字典
        floor_dimensions = {}  # {level: {'length': length, 'width': width}}

        # 初始化每层的最大最小坐标值
        level_coords = {}  # {level: {'max_x': val, 'min_x': val, 'max_y': val, 'min_y': val}}

        # 遍历所有节点
        for node_info in nodes_dict.values():
            level = node_info['level']
            x, y = node_info['coord'][0], node_info['coord'][1]

            # 如果该层还未记录，初始化该层的数据
            if level not in level_coords:
                level_coords[level] = {
                    'max_x': x,
                    'min_x': x,
                    'max_y': y,
                    'min_y': y
                }
            else:
                # 更新最大最小值
                level_coords[level]['max_x'] = max(level_coords[level]['max_x'], x)
                level_coords[level]['min_x'] = min(level_coords[level]['min_x'], x)
                level_coords[level]['max_y'] = max(level_coords[level]['max_y'], y)
                level_coords[level]['min_y'] = min(level_coords[level]['min_y'], y)

        # 计算每层的实际长度和宽度
        for level, coords in level_coords.items():
            length = coords['max_x'] - coords['min_x']
            width = coords['max_y'] - coords['min_y']
            floor_dimensions[level] = {
                'length': length,
                'width': width
            }

        return floor_dimensions

    def column_node_division(self, node_data,frames_data):

        """
        根据构件信息提取每层柱的顶部和底部节点

        参数:
        node_dict - 节点字典，格式为 {节点ID: 节点信息字典}
        member_dict - 构件字典，格式为 {构件ID: 构件信息字典}

        返回:
        按楼层分类的柱节点字典，格式为 {楼层: {'top': [节点ID列表], 'bottom': [节点ID列表]}}
        """

        # node_dict = all_data_info.full_structure['Nodes']
        # member_dict = all_data_info.full_structure['Frames']

        node_dict = node_data
        member_dict = frames_data


        # 初始化结果字典
        result = {}

        # 1. 找出所有柱构件，并按level分组
        columns = {}
        for member_id, member in member_dict.items():
            # 只处理类型为'column'的构件
            if member['type'] == 'column':
                level = member['level']
                if level not in columns:
                    columns[level] = []
                columns[level].append(member)

        # 2. 对每层的柱构件，提取其顶部和底部节点
        for level, level_columns in columns.items():
            if level not in result:
                result[level] = {'top': [], 'bottom': []}

            for column in level_columns:
                # 获取柱的两个节点ID
                node1_id, node2_id = column['nodes']

                # 查找这两个节点的位置信息
                node1_pos = node_dict[node1_id]['position']
                node2_pos = node_dict[node2_id]['position']

                # 将节点分配到正确的类别
                if node1_pos == 'top':
                    result[level]['top'].append(node1_id)
                elif node1_pos == 'bottom':
                    result[level]['bottom'].append(node1_id)

                if node2_pos == 'top':
                    result[level]['top'].append(node2_id)
                elif node2_pos == 'bottom':
                    result[level]['bottom'].append(node2_id)

        # 3. 移除可能的重复节点
        for level in result:
            result[level]['top'] = list(set(result[level]['top']))
            result[level]['bottom'] = list(set(result[level]['bottom']))

        return result



    def assign_section_to_modulars(self,modular_units, classified_dict, section_list):
        """
        将截面编号分配给各个模块单元

        参数:
        - modular_units: 原始模块单元信息字典
        - classified_dict: 按楼层和类型分类的字典
        - section_list: 截面编号列表

        返回:
        - 更新后的modular_units字典
        """
        # 创建深拷贝以避免修改原始数据
        updated_modular_units = modular_units.copy()

        # 用于追踪section_list的索引
        section_index = 0

        # 按楼层顺序遍历
        for story in sorted(classified_dict.keys()):
            # 按type顺序遍历
            for type_id in sorted(classified_dict[story].keys()):
                # 获取当前截面编号
                current_section = section_list[section_index]

                # 将该截面编号分配给该楼层该type的所有模块
                for modular_id in classified_dict[story][type_id]:
                    updated_modular_units[modular_id]['modular_type'] = current_section

                # 更新索引
                section_index += 1

        return updated_modular_units

    def boundary_modelling(self, modular_data):
        for modular_id,modular_id_data in modular_data.items():
            if modular_id_data['story'] == '1':
                for nodes_id, node_id_data in modular_id_data['nodes'].items():
                    if node_id_data['type']=='bottom':
                        ops.fix(nodes_id, 1, 1, 1, 1, 1, 1)


    def load_modelling(self,modular_data,node_data,frames_data):

        ops.timeSeries('Linear', 1, '-factor', 1.2)
        ops.pattern('Plain', 1, 1)  # Pattern ID 1 for Dead Load

        # for modular_id,modular_id_data in self.all_data_info['modular_data'].items():
        #         for nodes_id, node_id_data in modular_id_data['nodes'].items():
        #             if node_id_data['type']=='bottom':
        #                 ops.load(nodes_id, -1000, -1000, -2000, 0, 0, 0)
        Nodes = node_data
        Frames =frames_data

        for modular_id_data in modular_data.values():
            plane_data = modular_id_data['planes']
            for plane in plane_data.values():
                node_coords = []
                frame_plane = plane['frames']
                for node_id in plane['nodes']:
                    if node_id in Nodes:
                        coord = Nodes[node_id]['coord']
                        node_coords.append(tuple(coord))  # 转为元组更直观
                if plane['position'] == 'bottom':
                    load = self.calculate_edge_loads(node_coords, 0.001)
                    max_load, max_edge,min_load,min_edge = self.get_max_edge_load(load)
                    for f_id in frame_plane:
                        index1, index2 = Frames[f_id]['nodes']
                        y_load, z_load = self.local_gravity_loading(Nodes[index1]['coord'], Nodes[index2]['coord'],
                                                                    min_load)
                        ops.eleLoad('-ele', f_id, '-type', '-beamUniform', -y_load, -z_load)  # 线荷载
                elif plane['position'] == 'top':
                    load = self.calculate_edge_loads(node_coords, 0.001)
                    max_load, max_edge,min_load,min_edge = self.get_max_edge_load(load)
                    for f_id in frame_plane:
                        index1, index2 = Frames[f_id]['nodes']
                        y_load, z_load = self.local_gravity_loading(Nodes[index1]['coord'], Nodes[index2]['coord'],
                                                                    min_load)
                        ops.eleLoad('-ele', f_id, '-type', '-beamUniform', -y_load, -z_load)  # 线荷载



        for force_id,force_data in self.all_force_result.items():
            load_node = force_data
            ops.load(force_id, load_node[0], load_node[1], load_node[2], load_node[3], load_node[4], load_node[5])

        # for i in range(len(self.all_force_result)):
        #     load_node = self.all_force_result[i]
        #     ops.load(i, load_node[0], load_node[1], load_node[2], load_node[3], load_node[4], load_node[5])


    def out_dis_data(self):
        # 打开 NetCDF 文件

        APIPath = os.path.join(os.getcwd(), '_OPSTOOL_ODB')
        SpecifyPath = True
        if not os.path.exists(APIPath):
            try:
                os.makedirs(APIPath)
            except OSError:
                pass

        path1 = os.path.join(APIPath, f'RespStepData-{1}.nc')

        nc_data = Dataset(path1, mode="r")

        # 进入 /NodalResponses 分组
        if "NodalResponses" in nc_data.groups:
            # 数据格式  分析布*节点数量*六个自由度的位移
            nodal_responses = nc_data.groups["NodalResponses"]
            # 提取位移数据
            if "disp" in nodal_responses.variables:
                disp = nodal_responses.variables["disp"][:]  # 位移数据
            if 'nodeTags' in nodal_responses.variables:
                nodetag = nodal_responses.variables['nodeTags'][:]
        # 提取force信息
        if "FrameResponses" in nc_data.groups:
            Frame_responses = nc_data.groups["FrameResponses"]

            # 提取位移数据
            if "sectionForces" in Frame_responses.variables:
                # 数据格式  分析步数量*构件数量*16个截面*六种内力

                # 'N' 'MZ' 'VY' 'MY' 'VZ' 'T'
                sectionForces = Frame_responses.variables['sectionForces'][:]
                # localForce = Frame_responses.variables['localForces'][:]
                ele_tags = Frame_responses.variables['eleTags'][:]  # 节点标签
                # secdofs = Frame_responses.variables['secDofs'][:]  # 自由度标签
                # time = Frame_responses.variables['time'][:]

        nc_data.close()
        return disp[1], sectionForces[1],nodetag,ele_tags

    def calculate_edge_loads(self, points, area_load):
        """
        计算均布面荷载传递到矩形四边的线荷载

        参数:
        points - 矩形各点坐标，格式为 [(x1,y1), (x2,y2), ...]
        area_load - 均布面荷载，单位: kN/m²

        返回:
        各边线荷载列表
        """
        # 提取二维坐标用于凸包计算（只使用x和y坐标）
        points_2d = [(p[0], p[1]) for p in points]

        # 确保点坐标按顺序排列并形成凸包
        hull = ConvexHull(points_2d)
        vertices = hull.vertices
        hull_points_2d = [points_2d[i] for i in vertices]
        hull_points_3d = [points[i] for i in vertices]  # 保留原始三维点

        # 计算矩形尺寸
        # 这里假设输入是严格的矩形，所以可以找出最大的x和y距离
        x_vals = [p[0] for p in hull_points_2d]
        y_vals = [p[1] for p in hull_points_2d]

        a = max(x_vals) - min(x_vals)  # x方向尺寸
        b = max(y_vals) - min(y_vals)  # y方向尺寸

        # 确定长边和短边
        if a >= b:
            long_side = a
            short_side = b
        else:
            long_side = b
            short_side = a

        # 计算矩形面积
        area = a * b

        # 计算每条边接收的线荷载 (kN/m)
        # 标准的双向板荷载分配
        edge_load_long = area_load * short_side / 2  # 长边上的线荷载
        edge_load_short = area_load * long_side / 2  # 短边上的线荷载

        # 识别每条边的长度
        edges = []
        for i in range(len(hull_points_3d)):
            p1 = hull_points_3d[i]
            p2 = hull_points_3d[(i + 1) % len(hull_points_3d)]
            # 计算二维平面上的边长（忽略z坐标差异）
            edge_length = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            edges.append((p1, p2, edge_length))

        # 分配线荷载
        edge_loads = []
        for p1, p2, length in edges:
            if abs(length - long_side) < 1e-6:  # 是长边
                edge_loads.append((p1, p2, edge_load_long))
            else:  # 是短边
                edge_loads.append((p1, p2, edge_load_short))

        return edge_loads



    def get_max_edge_load(self, edge_loads):
        """
        从边界线荷载计算结果中找出最大荷载值

        参数:
        edge_loads - calculate_edge_loads函数返回的结果
                     格式为 [(p1, p2, load), ...]

        返回:
        最大线荷载值(float)和对应的边((p1, p2))
        """
        max_load = 0
        max_edge = None

        min_load = 100000
        min_edge = None

        for p1, p2, load in edge_loads:
            if load > max_load:
                max_load = load
                max_edge = (p1, p2)

            if load<min_load:
                min_load=load
                min_edge =(p1, p2)
        return max_load, max_edge,min_load,min_edge

    def local_gravity_loading(self, node1, node2, load_value):
        if node1[0] != node2[0]:
            if node1[0] > node2[0]:
                y_local = load_value
                z_local = 0
            else:
                y_local = -load_value
                z_local = 0
        elif node1[1] != node2[1]:
            if node1[1] > node2[1]:
                y_local = 0
                z_local = load_value
            else:
                y_local = 0
                z_local = load_value
        return y_local, z_local


    def calculate_dis_index(self, all_data_info, out_dis,node_tag):

        nodes_data = self.extract_nodes_info(all_data_info['modular_data'])
        displacements = np.array(out_dis)
        node_tag = np.array(node_tag)
        value_to_id = {value: i for i, value in enumerate(node_tag)}
        # 初始化用于存储每个楼层的最大位移
        max_displacements = defaultdict(lambda: {"max_x": 0, "max_y": 0})
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(nodes_data.keys())}

        for node_id, info in nodes_data.items():
            if info["position"] == 'top':
                height = info["coord"][2]  # 节点高度
                story = info["level"]  # 节点楼层

                # 找到矩阵中对应的行索引
                idx = node_id_to_index[node_id]

                # 节点的 X 和 Y 方向位移
                x_displacement = displacements[idx, 0]
                y_displacement = displacements[idx, 1]

                # 计算 X 和 Y 方向的楼层位移
                x_floor_displacement = abs(x_displacement / height * 600)
                y_floor_displacement = abs(y_displacement / height * 600)

                # 更新当前楼层的最大值
                max_displacements[story]["max_x"] = max(max_displacements[story]["max_x"], x_floor_displacement)
                max_displacements[story]["max_y"] = max(max_displacements[story]["max_y"], y_floor_displacement)

        # 打印每个楼层的最大 X 和 Y 位移
        # for story, values in sorted(max_displacements.items()):
        #     print(f"楼层 {story}: 最大 X 方向位移 = {values['max_x']:.6f}, 最大 Y 方向位移 = {values['max_y']:.6f}")
        return max_displacements

    def extract_nodes_info(self,modular_units):
        """
        从模块单元中提取节点信息并转换为指定格式

        参数:
        - modular_units: 模块单元字典
        - standard_stories: 标准层信息字典

        返回:
        - 转换后的节点信息字典
        """
        nodes_dict = {}

        # 遍历每个模块
        for modular_id, modular_info in modular_units.items():
            # 获取该模块的截面编号和楼层
            story = modular_info['story']

            # 获取节点信息
            nodes = modular_info['nodes']
            # 遍历模块中的所有节点
            for node_id, node_id_data in nodes.items():
                # 确定节点位置（这里需要根据具体规则来判断）
                # 假设根据z坐标判断：如果在模块高度的上半部分，则为'top'，否则为'bottom'
                position = node_id_data['type']
                coord = node_id_data['coord']
                nodes_dict[node_id]={
                    'id': node_id,
                    'coord': coord,
                    'level': story,
                    'position': position
                }

        # 给每个点添加一个label
        coord_to_label = {}
        label_counter = 0  # **label 从 1 开始编号**

        # **遍历所有节点**
        for node_id, node in nodes_dict.items():
            coord_key = (round(node["coord"][0], 2), round(node["coord"][1], 2))  # **(x, y) 作为唯一标识**

            # **如果该 (x, y) 还没有分配 label，则创建新 label**
            if coord_key not in coord_to_label:
                coord_to_label[coord_key] = label_counter
                label_counter += 1  # **更新 label 计数**

            # **给当前点赋值 label**
            nodes_dict[node_id]["label"] = coord_to_label[coord_key]

        return nodes_dict


    def calculate_interdis_index(self, all_data_info, out_dis, all_height):

        node_data = self.extract_nodes_info(all_data_info['modular_data'])

        displacements = copy.deepcopy(out_dis)

        displacements = np.array(displacements)

        # 创建按楼层分组的节点字典
        # 例如 {1: [节点0, 节点1], 2: [节点2, 节点3]}
        nodes_by_story = defaultdict(list)
        for node_id, info in node_data.items():
            nodes_by_story[int(info["level"])-1].append(node_id)

        # 初始化用于存储每层最大层间位移比的字典
        max_drift_ratios_by_story = defaultdict(lambda: {"max_x": 0, "max_y": 0})

        node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_data.keys())}

        # 遍历每层，计算层间位移比
        for story, nodes in sorted(nodes_by_story.items()):
            for node_id in nodes:
                info = node_data[node_id]

                # 只处理 location_type = "top" 的点
                if info["position"] != "top":
                    continue

                height = all_height[story]  # 当前楼层高度
                position = info["label"]  # 当前节点的位置标识（用于匹配上一层）

                # 找到矩阵中对应的行索引
                idx = node_id_to_index[node_id]
                # 当前节点的 X 和 Y 位移
                x_displacement = displacements[idx, 0]
                y_displacement = displacements[idx, 1]

                if story == 0:
                    # 第一层计算方法
                    x_drift_ratio = x_displacement / height * 250
                    y_drift_ratio = y_displacement / height * 250
                else:
                    # 查找上一层中水平位置相同的点
                    previous_story_nodes = nodes_by_story[story - 1]
                    previous_node_id = None
                    for prev_node_id in previous_story_nodes:
                        if node_data[prev_node_id]["label"] == position:
                            previous_node_id = prev_node_id
                            break

                    if previous_node_id is None:
                        raise ValueError(f"未找到楼层 {story - 1} 中位置 {position} 的对应节点！")

                    # 上一层节点的 X 和 Y 位移
                    idx = node_id_to_index[previous_node_id]
                    prev_x_displacement = displacements[idx, 0]
                    prev_y_displacement = displacements[idx, 1]

                    # 计算层间位移比
                    x_drift_ratio = abs((x_displacement - prev_x_displacement) / height * 250)
                    y_drift_ratio = abs((y_displacement - prev_y_displacement) / height * 250)

                # 更新当前楼层的最大层间位移比
                max_drift_ratios_by_story[story]["max_x"] = max(max_drift_ratios_by_story[story]["max_x"],
                                                                x_drift_ratio)
                max_drift_ratios_by_story[story]["max_y"] = max(max_drift_ratios_by_story[story]["max_y"],
                                                                y_drift_ratio)
        return max_drift_ratios_by_story


    def calculate_frame_index(self, all_data_info, force_result,ele_tag):

        frames = self.extract_frame_info(all_data_info['modular_data'])
        section_data = self.material_data['section_types']
        ele_tag = np.array(ele_tag).tolist()
        frame_id_to_index = {idx: frame_id for idx, frame_id in enumerate(frames.keys())}
        # print(frame_id_to_index[4535])
        # print(ele_tag.index(37811013))
        # 构件数量、截面数量和截面内力数量
        num_members = len(frames)
        num_sections = len(force_result[0])
        num_forces = len(force_result[0][0])

        # 随机生成构件内力数据 (20 x 5 x 6)
        # 实际使用时替换为你的输入数据
        all_forces = copy.deepcopy(force_result)

        # 存储每个构件的最大内力
        max_forces_by_member = []
        total_weight = 0
        # 遍历每个构件
        for idx in range(num_members):
            member_id = frame_id_to_index[idx]
            frames_section = frames[member_id]['section']
            frames_area = section_data[f'{frames_section}']["Area"]
            frames_length = frames[member_id]['length']
            frames_property = section_data[f'{frames_section}']

            # 获取该构件的内力数据 (15 x 6)
            # member_forces = all_forces[idx]
            member_forces = all_forces[ele_tag.index(member_id)]
            # 对该构件的所有截面计算 fun1，并找到最大值
            max_force = max(
                self.calculate_g(frames_property, section_force, frames_length) for section_force in member_forces)

            # frames[member_id]['weight']=0.00000000785*frames_area*frames_length
            frames[member_id]['max_force'] = max(max_force)
            total_weight+=frames[member_id]['weight']
            # 存储最大内力值
            max_forces_by_member.append((member_id, max_force))  # 构件编号从 1 开始

        # 初始化变量
        global_max_force = float("-inf")  # 全局最大内力
        global_max_member_id = None  # 对应构件编号
        global_max_component = None  # 对应内力分量值
        out_value = {}

        # 遍历每个构件，找到每个构件的最大内力值
        for member_id, forces in max_forces_by_member:
            # 找到当前构件的最大内力值
            local_max_force = max(forces)
            if local_max_force>1:
                out_value[member_id]=forces
            # 检查是否为全局最大值
            if local_max_force > global_max_force:
                global_max_force = local_max_force
                global_max_member_id = member_id
                global_max_component = forces

        return max_forces_by_member, global_max_force,total_weight,out_value


    def extract_frame_info(self,modular_units):
        """
        从模块单元中提取节点信息并转换为指定格式

        参数:
        - modular_units: 模块单元字典
        - standard_stories: 标准层信息字典

        返回:
        - 转换后的节点信息字典
        """
        frame_dict = {}

        # 遍历每个模块
        for modular_id, modular_info in modular_units.items():
            # 获取该模块的截面编号和楼层
            story = modular_info['story']

            # 获取节点信息
            frames = modular_info['frames']
            # 遍历模块中的所有节点
            for frame_id, frame_id_data in frames.items():
                # 确定节点位置（这里需要根据具体规则来判断）
                # 假设根据z坐标判断：如果在模块高度的上半部分，则为'top'，否则为'bottom'
                types = frame_id_data['type']
                nodes = frame_id_data['nodes']
                index1,index2 = nodes
                x1,y1,z1 = modular_info['nodes'][index1]['coord']
                x2,y2,z2 = modular_info['nodes'][index2]['coord']
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                frame_section = self.unit_data[modular_info['modular_type']][types]
                section_area = self.material_data['section_types'][f'{frame_section}']['Area']
                weight = 0.00000000785 * section_area * distance
                frame_dict[frame_id]={
                    'id': frame_id,
                    'nodes': nodes,
                    'level': story,
                    'type': types,
                    'section':frame_section,
                    'length':distance,
                    'weight':weight
                }

        return frame_dict


    def calculate_g(self, section_properties, frame_reactions, frame_length):

        N_axis = frame_reactions[0]
        My = frame_reactions[3]
        Mz = frame_reactions[1]
        # 柱强度验算
        rx = 1
        ry = 1
        f = 355
        wnx = 906908.8
        wny = 101172.04
        faix = 0.8
        faiy = 0.8
        bmx = 0.9
        btx = 0.9
        bmy = 0.9
        bty = 0.9
        Nex = 5
        Ney = 5
        n_canshu = 1
        faiby = 0.8
        faibx = 0.8

        G11 = (abs(N_axis) / f / section_properties['Area']) + (
                abs(My) / f / rx / section_properties['S22']
        ) + (abs(Mz) / f / ry / section_properties['S33'])

        G21 = (abs(N_axis) / f / section_properties['Area'] / faix) + (
                bmx * abs(My) / f / rx / section_properties['S22']
                / (1 - 0.8 * abs(N_axis) / abs(section_properties['I22']) / 1846434.18 *
                   frame_length *
                   frame_length)) + n_canshu * (
                      bty * abs(Mz) / f / section_properties['S33'] / faiby)

        G31 = (abs(N_axis) / f / section_properties['Area'] / faiy) + n_canshu * (
                btx * abs(My) / f / section_properties['S22']
                / faibx) + (bmy * abs(My) / f / ry / section_properties['I22'] / (
                1 - 0.8 * abs(N_axis) / section_properties[
            'I33'] / 1846434.18 * frame_length * frame_length))

        return [G11, G21, G31]

    def calculate_level_weight(self,all_data_info):

        frames = self.extract_frame_info(all_data_info['modular_data'])
        level_weights = {}

        # 遍历大字典中的每个子字典
        for item in frames.values():
            level = int(item['level'])   # 如果需要从0开始计数，将level转为整数并-1
            weight = item['weight']

            # 累加每个level的weight
            if level in level_weights:
                level_weights[level] += weight
            else:
                level_weights[level] = weight

        # 对字典按键排序
        sorted_weights = dict(sorted(level_weights.items()))

        return sorted_weights

    def calculate_frame_weight(self, all_data_info):



        frames = all_data_info.full_structure['Frames']


        section_data = self.material_data['section_types']

        # 构件数量、截面数量和截面内力数量
        num_members = len(frames)

        # 遍历每个构件
        for member_id in range(num_members):
            frames_section = frames[member_id]['section']
            frames_area = section_data[f'{frames_section}']["Area"]
            frames_length = frames[member_id]['length']

            # 对该构件的所有截面计算 fun1，并找到最大值

            frames[member_id]['weight'] = 0.00000000785 * frames_area * frames_length


'''地震荷载'''
class Seismic_force:
    def __init__(self, weight, height):
        self.weight = weight
        self.height = height


    '''计算底部剪力'''
    def calculate_base_shear(self,alpha):
        """
        计算底部剪力

        参数:
        alpha - 地震影响系数
        weight - 结构总重(kN)
        damping_ratio - 阻尼比(默认0.05)

        返回:
        底部剪力值(kN)
        """
        # 对于一般框架结构
        # 根据规范确定地震影响系数α
        # 计算底部剪力
        total_weight = sum(weight for weight in self.weight.values())
        base_shear = alpha * total_weight*10
        return base_shear


    '''各层水平力分配'''
    def distribute_lateral_forces(self, base_shear, mode_shape=None):
        """
        分配各层水平力

        参数:
        base_shear - 底部剪力(kN)
        weights - 各层重量列表(kN) [W1, W2, ..., Wn]
        heights - 各层高度列表(m) [h1, h2, ..., hn]
        mode_shape - 振型向量(可选)，默认按规范公式分配

        返回:
        各层水平力列表(kN)
        """

        weights_list = [weight for weight in [self.weight[i] for i in sorted(self.weight.keys())]]
        heights_list = self.height
        n_floors = len(weights_list)

        # 若未提供振型，按规范计算等效侧向力分配
        if mode_shape is None:
            # 计算各层Wh^γ乘积
            gamma = 1.0  # 一般取1.0(适用于高度<60m的框架结构)
            wh_products = [weights_list[i] * (heights_list[i] ** gamma) for i in range(n_floors)]
            sum_wh = sum(wh_products)

            # 计算各层水平力
            forces = [base_shear * wh_products[i] / sum_wh for i in range(n_floors)]
        else:
            # 使用振型进行分配
            modal_products = [weights_list[i] * mode_shape[i] for i in range(n_floors)]
            sum_modal = sum(modal_products)

            # 计算各层水平力
            forces = [base_shear * modal_products[i] / sum_modal for i in range(n_floors)]

        return forces

    '''节点水平力分配'''
    def distribute_forces_to_nodes(self, story_forces,Nodes_data):
        """
        分配层间水平力至各节点

        参数:
        story_forces - 各层水平力列表(kN)
        floor_node_coords - 各层节点坐标字典 {floor_id: [(x1,y1,z1), (x2,y2,z2), ...]}
        floor_node_ids - 各层节点ID字典 {floor_id: [node_id1, node_id2, ...]}

        返回:
        节点力字典 {node_id: (fx, fy, fz, mx, my, mz)}


        """

        # 初始化两个结果字典
        floor_node_coords = {}  # {floor_id: [(x1,y1,z1), (x2,y2,z2), ...]}
        floor_node_ids = {}  # {floor_id: [node_id1, node_id2, ...]}

        # 遍历所有节点
        for node_id, node_info in Nodes_data.items():
            # 获取楼层信息和节点坐标
            floor_id = node_info['level']
            coord = tuple(node_info['coord'])  # 转换为元组格式

            # 添加坐标信息
            if floor_id not in floor_node_coords:
                floor_node_coords[floor_id] = []
            floor_node_coords[floor_id].append(coord)

            # 添加节点ID信息
            if floor_id not in floor_node_ids:
                floor_node_ids[floor_id] = []
            floor_node_ids[floor_id].append(node_id)




        node_forces = {}

        for floor_id in range(len(story_forces)):
            floor_force = story_forces[floor_id]
            # nodes = floor_node_coords[floor_id]
            # node_ids = floor_node_ids[floor_id]
            nodes = floor_node_coords[f'{floor_id+1}']
            node_ids = floor_node_ids[f'{floor_id+1}']

            # 确定层中心坐标
            # center_x = sum(node[0] for node in nodes) / len(nodes)
            # center_y = sum(node[1] for node in nodes) / len(nodes)

            # 计算各节点与中心距离和相应惯性力矩(刚性楼板假定)
            # total_inertia_x = sum((node[1] - center_y) ** 2 for node in nodes)
            # total_inertia_y = sum((node[0] - center_x) ** 2 for node in nodes)

            # 风/地震力方向
            direction_x = 1.0  # 假设力在x方向
            direction_y = 1.0  # 可以修改为所需方向

            # 分配节点力
            for i, node_id in enumerate(node_ids):
                node = nodes[i]

                # 转动效应(如有偏心)可以在此考虑
                # 简化为仅考虑平动分量
                fx = floor_force * direction_x / len(nodes)
                fy = floor_force * direction_y / len(nodes)

                # 存储节点力(fx, fy, fz, mx, my, mz)
                node_forces[node_id] = (fx, fy, 0, 0, 0, 0)
        new_dict = {k: tuple(x * 1000 for x in v) for k, v in node_forces.items()}
        return new_dict

    '''考虑偏心效应的扭转分量'''
    def calculate_torsional_forces(self,story_forces, floor_node_coords, floor_node_ids,
                                   centers_of_mass, centers_of_rigidity, eccentricity=0.05):
        """
        计算考虑偏心效应的扭转力分量

        参数:
        story_forces - 各层水平力列表(kN)
        floor_node_coords - 各层节点坐标
        floor_node_ids - 各层节点ID
        centers_of_mass - 各层质心坐标 [(x1,y1), (x2,y2), ...]
        centers_of_rigidity - 各层刚心坐标 [(x1,y1), (x2,y2), ...]
        eccentricity - 附加偏心率(默认5%)

        返回:
        包含扭转分量的节点力字典
        """
        node_forces = {}

        for floor_id in range(len(story_forces)):
            floor_force = story_forces[floor_id]
            nodes = floor_node_coords[floor_id]
            node_ids = floor_node_ids[floor_id]

            # 质心和刚心
            cm_x, cm_y = centers_of_mass[floor_id]
            cr_x, cr_y = centers_of_rigidity[floor_id]

            # 计算偏心距离(考虑附加偏心)
            # 规范要求考虑附加偏心
            building_size_x = max(node[0] for node in nodes) - min(node[0] for node in nodes)
            building_size_y = max(node[1] for node in nodes) - min(node[1] for node in nodes)

            ex = cm_x - cr_x + eccentricity * building_size_x  # x方向偏心
            ey = cm_y - cr_y + eccentricity * building_size_y  # y方向偏心

            # 扭矩
            torsion_moment_x = floor_force * ey  # 由x向力产生的扭矩
            torsion_moment_y = floor_force * ex  # 由y向力产生的扭矩

            # 计算各节点与刚心的距离和极惯性矩
            polar_moment = sum((node[0] - cr_x) ** 2 + (node[1] - cr_y) ** 2 for node in nodes)

            # 分配节点力(包括平动和转动效应)
            for i, node_id in enumerate(node_ids):
                node = nodes[i]

                # 平动分量
                fx_trans = floor_force / len(nodes)

                # 转动分量
                dist_x = node[0] - cr_x
                dist_y = node[1] - cr_y
                fx_rot = torsion_moment_y * dist_y / polar_moment
                fy_rot = torsion_moment_x * dist_x / polar_moment

                # 合力
                fx = fx_trans + fx_rot
                fy = fy_rot

                # 存储节点力
                node_forces[node_id] = (fx, fy, 0, 0, 0, 0)

        return node_forces

    '''钢结构框架节点力计算案例'''

    def base_shear_analysis_for_steel_frame(self):
        """完整的钢结构框架底部剪力法分析示例"""

        # 结构参数
        n_floors = 3
        floor_heights = [3.6, 7.2, 10.8]  # 各层高度(m)
        floor_weights = [1200, 1200, 900]  # 各层重量(kN)

        # 各层节点坐标
        # 假设每层有4个节点，格式为(x,y,z)
        floor_nodes = {
            0: [(0, 0, 3.6), (6, 0, 3.6), (6, 4, 3.6), (0, 4, 3.6)],
            1: [(0, 0, 7.2), (6, 0, 7.2), (6, 4, 7.2), (0, 4, 7.2)],
            2: [(0, 0, 10.8), (6, 0, 10.8), (6, 4, 10.8), (0, 4, 10.8)]
        }

        # 节点ID
        floor_node_ids = {
            0: [1, 2, 3, 4],
            1: [5, 6, 7, 8],
            2: [9, 10, 11, 12]
        }

        # 计算基底剪力
        # 假设地震影响系数α=0.08，总重3300kN
        base_shear = self.calculate_base_shear(alpha=0.08, weight=sum(floor_weights))
        print(f"底部剪力: {base_shear:.2f} kN")

        # 分配各层水平力
        story_forces = self.distribute_lateral_forces(base_shear, floor_weights, floor_heights)
        print("\n各层水平力(kN):")
        for i, force in enumerate(story_forces):
            print(f"第{i + 1}层: {force:.2f}")

        # 各层质心和刚心(简化假设为几何中心)
        centers_of_mass = [(3, 2) for _ in range(n_floors)]

        # 假设存在偏心(刚心相对于质心偏移)
        centers_of_rigidity = [(3.3, 2.2) for _ in range(n_floors)]

        # 计算考虑扭转效应的节点力
        node_forces = self.calculate_torsional_forces(
            story_forces, floor_nodes, floor_node_ids,
            centers_of_mass, centers_of_rigidity
        )

        # 显示结果
        print("\n各节点水平力(kN):")
        for node_id, forces in node_forces.items():
            print(f"节点{node_id}: Fx={forces[0]:.2f}, Fy={forces[1]:.2f}")

        # 可视化结果
        self.visualize_forces(floor_nodes, node_forces)

        return node_forces

    '''可视化节点力分布'''

    def visualize_forces(self,floor_nodes, node_forces, floor_node_ids):
        """
        可视化节点力分布

        参数:
        floor_nodes - 各层节点坐标字典
        node_forces - 节点力字典
        floor_node_ids - 各层节点ID字典
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 节点坐标
        all_nodes = []
        for floor_id, nodes in floor_nodes.items():
            all_nodes.extend(nodes)

        x = [node[0] for node in all_nodes]
        y = [node[1] for node in all_nodes]
        z = [node[2] for node in all_nodes]

        # 绘制节点
        ax.scatter(x, y, z, c='blue', marker='o', s=50)

        # 绘制水平力矢量
        for node_id, forces in node_forces.items():
            # 找到节点坐标
            node = None
            for floor_id, nodes in floor_nodes.items():
                node_ids = floor_node_ids[floor_id]
                if node_id in node_ids:
                    index = node_ids.index(node_id)
                    node = nodes[index]
                    break

            if node:
                fx, fy = forces[0], forces[1]
                # 缩放因子使矢量长度适中
                # 根据力的大小自动调整缩放因子
                max_force = max(abs(fx), abs(fy))
                scale = 0.5 / max_force if max_force > 0 else 0.5

                ax.quiver(node[0], node[1], node[2],
                          fx * scale, fy * scale, 0,
                          color='red', arrow_length_ratio=0.3)

                # 标注力值
                ax.text(node[0], node[1], node[2],
                        f"{node_id}:{fx:.1f},{fy:.1f}",
                        color='black', fontsize=8)

        # 绘制框架连接线
        for floor_id, nodes in floor_nodes.items():
            # 连接同一层的节点形成平面
            n = len(nodes)
            for i in range(n):
                next_i = (i + 1) % n
                ax.plot([nodes[i][0], nodes[next_i][0]],
                        [nodes[i][1], nodes[next_i][1]],
                        [nodes[i][2], nodes[next_i][2]], 'k-', alpha=0.5)

            # 连接上下层的节点形成柱
            if floor_id > 0:
                prev_floor_nodes = floor_nodes[floor_id - 1]
                for i in range(n):
                    ax.plot([nodes[i][0], prev_floor_nodes[i][0]],
                            [nodes[i][1], prev_floor_nodes[i][1]],
                            [nodes[i][2], prev_floor_nodes[i][2]], 'k-', alpha=0.5)

        ax.set_xlabel('X轴(m)')
        ax.set_ylabel('Y轴(m)')
        ax.set_zlabel('Z轴(m)')
        ax.set_title('钢结构框架节点水平力分布')

        # 设置坐标轴比例一致
        max_range = max([max(x) - min(x), max(y) - min(y), max(z) - min(z)])
        mid_x = (max(x) + min(x)) / 2
        mid_y = (max(y) + min(y)) / 2
        mid_z = (max(z) + min(z)) / 2
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        plt.tight_layout()
        plt.show()

'''风荷载'''
class Wind_force:
    def __init__(self,  height):
        self.height = height

    '''根据GB50009-2012《建筑结构荷载规范》，计算风压高度变化系数'''
    def calculate_wind_load(self, height_all, terrain_category):
        """
        计算风荷载标准值

        参数:
        basic_wind_pressure - 基本风压(kN/m²)，依据当地气象资料确定
        height - 计算点离地高度[h1,h2,...](m)
        terrain_category - 地形类别(A,B,C,D类)
        shape_coef - 风荷载体型系数
        importance_factor - 结构重要性系数，默认为1.0

        返回:
        风荷载标准值(kN/m²)
        """
        # 根据GB50009-2012风压高度变化系数表查表获取
        # 这里以常用数据建立简化模型
        terrain_params = {
            'A': {5: 1.09, 10: 1.28,15: 1.42, 20: 1.52,30: 1.67, 40: 1.79,50: 1.89, 60: 1.97,70: 2.05, 80: 2.12},
            'B': {5: 1.00, 10: 1.00,15: 1.13, 20: 1.23,30: 1.39, 40: 1.52,50: 1.62, 60: 1.71,70: 1.79, 80: 1.87},
            'C': {5: 0.65, 10: 0.65,15: 0.65, 20: 0.74,30: 0.88, 40: 1.00,50: 1.10, 60: 1.20,70: 1.28, 80: 1.36},
            'D': {5: 0.51, 10: 0.51,15: 0.51, 20: 0.51,30: 0.51, 40: 0.60,50: 0.69, 60: 0.77,70: 0.84, 80: 0.91}
        }

        # 确保地形类别有效
        if terrain_category not in terrain_params:
            raise ValueError(f"无效的地形类别: {terrain_category}，请使用 A, B, C 或 D")

        height_stories = []
        uz = []
        # 风压高度变化系数计算，根据GB50009-2012公式4.8.2
        params = terrain_params[terrain_category]
        height_coef = 0.0
        cumulative_heights = (np.cumsum(height_all) * 0.001).tolist()

        height_ranges = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80]
        for height in cumulative_heights:
            story = next(h for h in height_ranges if height <= h)
            height_stories.append(story)
        for u in height_stories:
            uz.append(terrain_params[terrain_category][u])


        return uz

    '''根据GB50009-2012《建筑结构荷载规范》，计算风振系数'''
    def calculate_vibration_coefficient(self):
        return 1

    '''风荷载需要转换为各楼层的集中力'''

    def distribute_wind_load_to_stories(self, floor_dimensions, story_heights,
                                        basic_wind_pressure,uz,us,windward_coef, leeward_coef):

        """
        将风荷载分配到各层，同时计算X和Y方向

        参数:
        floor_dimensions - 各层尺寸字典 {level: {'length': length, 'width': width}}
        story_heights - 各层高度列表(m)
        basic_wind_pressure - 基本风压(kN/m²)
        terrain_category - 地形类别(A,B,C,D)
        windward_coef - 迎风面风荷载体型系数，默认0.8
        leeward_coef - 背风面风荷载体型系数，默认-0.5
        importance_factor - 重要性系数，默认1.0

        返回:
        各层风荷载字典 {'x': x方向风荷载列表, 'y': y方向风荷载列表}
        """
        n_stories = len(story_heights)
        story_forces = {'x': [], 'y': []}

        # 累计高度
        cumulative_heights = np.cumsum(story_heights)

        # 计算每层受风面积（X和Y方向）
        story_areas_x = []  # X方向受风面积
        story_areas_y = []  # Y方向受风面积

        for i in range(n_stories):

            # 获取当前层的尺寸
            # current_dimensions = floor_dimensions[i]
            current_dimensions = floor_dimensions[f'{i+1}']
            # X方向受风面积（使用宽度width）
            area_x = story_heights[i] * current_dimensions['width']
            story_areas_x.append(area_x)

            # Y方向受风面积（使用长度length）
            area_y = story_heights[i] * current_dimensions['length']
            story_areas_y.append(area_y)

        # 计算各层风荷载
        for i in range(n_stories):
            net_pressure = basic_wind_pressure*uz[i]*us*(windward_coef-leeward_coef)*0.001
            # X方向风荷载
            story_force_x = net_pressure * story_areas_x[i]
            story_forces['x'].append(story_force_x)

            # Y方向风荷载
            story_force_y = net_pressure * story_areas_y[i]
            story_forces['y'].append(story_force_y)

        return story_forces


        # n_stories = len(story_heights)
        # story_forces = []
        #
        # # 累计高度
        # cumulative_heights = np.cumsum(story_heights)
        #
        # # 计算每层受风面积
        # story_areas = []
        # for i in range(n_stories):
        #     # 层高
        #     if i == 0:
        #         story_height = story_heights[0]
        #     else:
        #         story_height = story_heights[i] - story_heights[i - 1]
        #
        #     # 获取当前层的尺寸
        #     current_dimensions = floor_dimensions[i]
        #
        #     # 受风面积
        #     if wind_direction == 'X':
        #         # X方向风荷载使用宽度(width)作为受风面宽度
        #         area = story_height * story_heights['width']
        #     else:  # 'Y'方向
        #         # Y方向风荷载使用长度(length)作为受风面宽度
        #         area = story_height * story_heights['length']
        #
        #     story_areas.append(area)
        #
        # # 计算各层风荷载
        # for i in range(n_stories):
        #     height = cumulative_heights[i]
        #
        #     net_pressure = basic_wind_pressure*uz[i]*us*(windward_coef-leeward_coef)
        #
        #
        #     # 各层风荷载
        #     story_force = net_pressure * story_areas[i]
        #     story_forces.append(story_force)

        # return story_forces




    '''刚性楼板假定，将各层风荷载分配到各节点'''

    def distribute_wind_to_nodes(self, story_forces, Nodes_data, is_moment_frame=True):
        """
        将风荷载分配到各节点

        参数:
        story_forces - 各层风荷载字典 {'x': x方向风荷载列表, 'y': y方向风荷载列表}
        floor_nodes - 各层节点坐标字典 {floor_id: [(x,y,z), ...]}
        floor_node_ids - 各层节点ID字典 {floor_id: [node_id1, ...]}
        is_moment_frame - 是否为刚接框架，影响负担分配方式

        返回:
        节点力字典 {node_id: (fx, fy, fz, mx, my, mz)}
        """

        floor_node_coords = {}  # {floor_id: [(x1,y1,z1), (x2,y2,z2), ...]}
        floor_node_ids = {}  # {floor_id: [node_id1, node_id2, ...]}

        # 遍历所有节点
        for node_id, node_info in Nodes_data.items():
            # 获取楼层信息和节点坐标
            floor_id = node_info['level']
            coord = tuple(node_info['coord'])  # 转换为元组格式

            # 添加坐标信息
            if floor_id not in floor_node_coords:
                floor_node_coords[floor_id] = []
            floor_node_coords[floor_id].append(coord)

            # 添加节点ID信息
            if floor_id not in floor_node_ids:
                floor_node_ids[floor_id] = []
            floor_node_ids[floor_id].append(node_id)



        node_forces_x = {}  # X方向风荷载产生的节点力
        node_forces_y = {}  # Y方向风荷载产生的节点力
        floor_nodes= floor_node_coords
        # 处理X方向风荷载
        for floor_id in range(len(story_forces['x'])):
            floor_force = story_forces['x'][floor_id]
            # nodes = floor_nodes[floor_id]
            # node_ids = floor_node_ids[floor_id]
            nodes = floor_nodes[f'{floor_id+1}']
            node_ids = floor_node_ids[f'{floor_id+1}']



            if is_moment_frame:
                # 刚接框架，所有柱均承担水平力
                n_effective_nodes = len(nodes)
                force_per_node = floor_force / n_effective_nodes

                # 分配风荷载
                for node_id in node_ids:
                    node_forces_x[node_id] = (force_per_node, 0.0, 0.0, 0.0, 0.0, 0.0)
            else:
                # 铰接框架处理
                y_coords = sorted(set(node[1] for node in nodes))
                n_frames = len(y_coords)
                force_per_frame = floor_force / n_frames

                for i, node_id in enumerate(node_ids):
                    node = nodes[i]
                    for y in y_coords:
                        if abs(node[1] - y) < 0.01:
                            frame_nodes = [n for n in nodes if abs(n[1] - y) < 0.01]
                            n_frame_nodes = len(frame_nodes)
                            fx = force_per_frame / n_frame_nodes
                            node_forces_x[node_id] = (fx, 0.0, 0.0, 0.0, 0.0, 0.0)
                            break

        # 处理Y方向风荷载
        for floor_id in range(len(story_forces['y'])):
            floor_force = story_forces['y'][floor_id]
            # nodes = floor_nodes[floor_id]
            # node_ids = floor_node_ids[floor_id]
            nodes = floor_nodes[f'{floor_id + 1}']
            node_ids = floor_node_ids[f'{floor_id + 1}']

            if is_moment_frame:
                # 刚接框架，所有柱均承担水平力
                n_effective_nodes = len(nodes)
                force_per_node = floor_force / n_effective_nodes

                # 分配风荷载
                for node_id in node_ids:
                    node_forces_y[node_id] = (0.0, force_per_node, 0.0, 0.0, 0.0, 0.0)
            else:
                # 铰接框架处理
                x_coords = sorted(set(node[0] for node in nodes))
                n_frames = len(x_coords)
                force_per_frame = floor_force / n_frames

                for i, node_id in enumerate(node_ids):
                    node = nodes[i]
                    for x in x_coords:
                        if abs(node[0] - x) < 0.01:
                            frame_nodes = [n for n in nodes if abs(n[0] - x) < 0.01]
                            n_frame_nodes = len(frame_nodes)
                            fy = force_per_frame / n_frame_nodes
                            node_forces_y[node_id] = (0.0, fy, 0.0, 0.0, 0.0, 0.0)
                            break

        # 合并X和Y方向的节点力
        node_forces = {}
        all_node_ids = set(node_forces_x.keys()) | set(node_forces_y.keys())

        for node_id in all_node_ids:
            fx = node_forces_x.get(node_id, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))[0]
            fy = node_forces_y.get(node_id, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))[1]
            node_forces[node_id] = (fx, fy, 0.0, 0.0, 0.0, 0.0)

        return node_forces

    '''完整的多层钢框架风荷载计算示例'''
    def wind_load_analysis_for_steel_frame(self):
        """
        根据中国规范计算多层钢框架在风荷载作用下的节点力
        """
        # 结构几何参数
        n_floors = 4  # 楼层数
        building_width = 18.0  # 建筑宽度(m)
        building_length = 30.0  # 建筑长度(m)
        story_heights = [4.2, 8.4, 12.6, 16.8]  # 各层累计高度(m)
        bay_width_x = 6.0  # X方向跨度(m)
        bay_width_y = 6.0  # Y方向跨度(m)

        # 风荷载参数(根据GB50009-2012)
        basic_wind_pressure = 0.35  # 基本风压(kN/m²)，取普通城市
        terrain_category = 'B'  # 地形类别B类
        importance_factor = 1.0  # 重要性系数(一般建筑)

        # 生成框架节点坐标
        floor_nodes = {}
        floor_node_ids = {}

        node_id = 1
        for floor in range(n_floors):
            floor_nodes[floor] = []
            floor_node_ids[floor] = []

            for j in range(4):  # Y方向4个分格
                for i in range(4):  # X方向4个分格
                    x = i * bay_width_x
                    y = j * bay_width_y
                    z = story_heights[floor]

                    floor_nodes[floor].append((x, y, z))
                    floor_node_ids[floor].append(node_id)
                    node_id += 1

        # 计算各层风荷载
        # X方向风荷载
        x_story_forces = self.distribute_wind_load_to_stories(
            building_width, building_length, story_heights,
            basic_wind_pressure, terrain_category,
            windward_coef=0.8, leeward_coef=-0.5,
            importance_factor=importance_factor,
            wind_direction='X'
        )

        # 将风荷载分配到节点
        node_forces_x = self.distribute_wind_to_nodes(
            x_story_forces, floor_nodes, floor_node_ids,
            is_moment_frame=True, wind_direction='X'
        )

        # 打印结果
        print("根据中国规范GB50009计算的钢结构框架风荷载")
        print(f"基本风压: {basic_wind_pressure} kN/m²")
        print(f"地形类别: {terrain_category}")
        print(f"建筑尺寸: {building_width}m × {building_length}m × {story_heights[-1]}m")

        print("\n各层X方向风荷载:")
        for i, force in enumerate(x_story_forces):
            print(f"第{i + 1}层: {force:.2f} kN")

        print("\n节点水平力(X方向):")
        print("节点ID\tFx(kN)\tFy(kN)")
        for node_id, forces in sorted(node_forces_x.items()):
            print(f"{node_id}\t{forces[0]:.2f}\t{forces[1]:.2f}")

        # 可视化结果
        self.visualize_wind_forces(floor_nodes, node_forces_x, floor_node_ids, wind_direction='X')

        return node_forces_x

    def visualize_wind_forces(self,floor_nodes, node_forces, floor_node_ids, wind_direction='X'):
        """
        可视化风荷载节点力分布
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制框架
        # 绘制柱
        for floor_id in range(len(floor_nodes) - 1):
            upper_nodes = floor_nodes[floor_id + 1]
            lower_nodes = floor_nodes[floor_id]

            for i in range(len(upper_nodes)):
                # 连接上下层节点形成柱
                ax.plot([upper_nodes[i][0], lower_nodes[i][0]],
                        [upper_nodes[i][1], lower_nodes[i][1]],
                        [upper_nodes[i][2], lower_nodes[i][2]],
                        'k-', linewidth=1.5, alpha=0.7)

        # 绘制梁
        for floor_id in range(len(floor_nodes)):
            nodes = floor_nodes[floor_id]
            x_coords = sorted(set(node[0] for node in nodes))
            y_coords = sorted(set(node[1] for node in nodes))

            # 沿X方向的梁
            for y in y_coords:
                x_nodes = [node for node in nodes if abs(node[1] - y) < 0.01]
                x_nodes.sort(key=lambda n: n[0])

                for i in range(len(x_nodes) - 1):
                    ax.plot([x_nodes[i][0], x_nodes[i + 1][0]],
                            [x_nodes[i][1], x_nodes[i + 1][1]],
                            [x_nodes[i][2], x_nodes[i + 1][2]],
                            'b-', linewidth=1, alpha=0.5)

            # 沿Y方向的梁
            for x in x_coords:
                y_nodes = [node for node in nodes if abs(node[0] - x) < 0.01]
                y_nodes.sort(key=lambda n: n[1])

                for i in range(len(y_nodes) - 1):
                    ax.plot([y_nodes[i][0], y_nodes[i + 1][0]],
                            [y_nodes[i][1], y_nodes[i + 1][1]],
                            [y_nodes[i][2], y_nodes[i + 1][2]],
                            'b-', linewidth=1, alpha=0.5)

        # 绘制节点和力
        for floor_id in range(len(floor_nodes)):
            nodes = floor_nodes[floor_id]
            node_ids = floor_node_ids[floor_id]

            for i, node_id in enumerate(node_ids):
                node = nodes[i]

                # 绘制节点
                ax.scatter(node[0], node[1], node[2], c='red', s=30)

                # 绘制风荷载矢量
                if node_id in node_forces:
                    forces = node_forces[node_id]
                    fx, fy = forces[0], forces[1]

                    # 确定箭头比例因子
                    force_mag = np.sqrt(fx ** 2 + fy ** 2)
                    if force_mag > 0:
                        scale = 0.5 / force_mag

                        # 绘制力矢量
                        ax.quiver(node[0], node[1], node[2],
                                  fx * scale, fy * scale, 0,
                                  color='green', arrow_length_ratio=0.3)

                        # 标注力值
                        if wind_direction == 'X':
                            ax.text(node[0], node[1], node[2],
                                    f"{fx:.1f}", color='blue', fontsize=8)
                        else:
                            ax.text(node[0], node[1], node[2],
                                    f"{fy:.1f}", color='blue', fontsize=8)

        # 添加风向指示箭头
        # 确定箭头位置(框架下方)
        min_x = min(node[0] for floor in floor_nodes.values() for node in floor)
        max_x = max(node[0] for floor in floor_nodes.values() for node in floor)
        min_y = min(node[1] for floor in floor_nodes.values() for node in floor)
        max_y = max(node[1] for floor in floor_nodes.values() for node in floor)
        min_z = min(node[2] for floor in floor_nodes.values() for node in floor)

        # 绘制风向箭头
        if wind_direction == 'X':
            arrow_x = min_x - (max_x - min_x) * 0.2
            arrow_y = (min_y + max_y) / 2
            arrow_z = min_z / 2

            ax.quiver(arrow_x, arrow_y, arrow_z,
                      1, 0, 0,
                      color='red', arrow_length_ratio=0.3, linewidth=3)
            ax.text(arrow_x, arrow_y, arrow_z,
                    "风向", color='red', fontsize=12)
        else:
            arrow_x = (min_x + max_x) / 2
            arrow_y = min_y - (max_y - min_y) * 0.2
            arrow_z = min_z / 2

            ax.quiver(arrow_x, arrow_y, arrow_z,
                      0, 1, 0,
                      color='red', arrow_length_ratio=0.3, linewidth=3)
            ax.text(arrow_x, arrow_y, arrow_z,
                    "风向", color='red', fontsize=12)

        # 设置图形属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'钢框架结构{wind_direction}方向风荷载分布 (GB50009)')

        # 保持坐标轴比例一致
        max_range = max([max_x - min_x, max_y - min_y,
                         max(node[2] for floor in floor_nodes.values() for node in floor)])
        mid_x = (max_x + min_x) / 2
        mid_y = (max_y + min_y) / 2
        mid_z = max(node[2] for floor in floor_nodes.values() for node in floor) / 2

        ax.set_xlim(mid_x - max_range / 1.5, mid_x + max_range / 1.5)
        ax.set_ylim(mid_y - max_range / 1.5, mid_y + max_range / 1.5)
        ax.set_zlim(0, max_range)

        plt.tight_layout()
        plt.show()

'''每层重力荷载'''
class Gravity_force:
    def __init__(self,  weight):
        self.weight = weight

    def calculate_gravity_loads(self, column_nodes):
        """
        计算各层柱节点重力荷载分布

        参数:
        level_weights - 各层重量字典，格式为 {level: weight_in_tons}
        column_nodes - 各层柱节点字典，格式为 {level: {'top': [node_ids], 'bottom': [node_ids]}}

        返回:
        节点力字典，格式为 {node_id: (fx, fy, fz, mx, my, mz)}
        """

        level_weights=self.weight
        # 将吨转换为牛顿 (1吨 = 9.81 kN)
        conversion_factor = 9.81 * 1000  # 牛顿

        # 初始化结果字典
        node_forces = {}

        # 获取所有楼层，确保按从小到大排序
        levels = sorted(column_nodes.keys())
        levels_int = copy.deepcopy(levels)
        for i in range(len(levels_int)):
            levels_int[i] = int(levels_int[i])
        # 计算每层以上的累计重量
        cumulative_weights = {}

        for level in levels_int:
            # 计算当前层以上的总重量
            upper_weight = sum(level_weights.get(l, 0) for l in level_weights if l > level)
            # 计算当前层及以上的总重量
            current_and_upper_weight = upper_weight + level_weights.get(level, 0)

            cumulative_weights[level] = {
                'upper': upper_weight,
                'current_and_upper': current_and_upper_weight
            }

        # 计算每个节点的力
        for level in levels:
            # 处理顶部节点
            top_nodes = column_nodes[level]['top']
            if top_nodes:
                # 每个top节点均分承担上部重量
                upper_weight = cumulative_weights[int(level)]['upper']
                force_per_top_node = (upper_weight * conversion_factor) / len(top_nodes)

                for node_id in top_nodes:
                    # 力的方向为竖直向下(Z方向为负)
                    node_forces[node_id] = (0, 0, -force_per_top_node, 0, 0, 0)

            # 处理底部节点
            bottom_nodes = column_nodes[level]['bottom']
            if bottom_nodes:
                # 每个bottom节点均分承担当前层及上部重量
                current_and_upper_weight = cumulative_weights[int(level)]['current_and_upper']
                force_per_bottom_node = (current_and_upper_weight * conversion_factor) / len(bottom_nodes)

                for node_id in bottom_nodes:
                    # 力的方向为竖直向下(Z方向为负)
                    node_forces[node_id] = (0, 0, -force_per_bottom_node, 0, 0, 0)

        return node_forces

