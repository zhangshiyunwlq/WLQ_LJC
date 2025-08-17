import os
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import ezdxf
from shapely.geometry import Point, Polygon, LineString
import json
import pyvista as pv

class JSONHandler:
    def __init__(self, input_data, file_name):
        self.input_data = input_data
        self.file_name = file_name

    def run(self):
        building_info = self.convert_to_building_info(self.input_data)
        result = self.convert_to_simple_format(building_info)
        # 增加楼层信息
        result = self.transform_building_data(result)
        # 按楼层生成模块单元和space
        result = self.transform_space_and_modular_data(result)

        # # 或者绘制所有楼层的矩形
        # self.plot_building_rectangles(result)
        #
        # # 或者以多子图方式绘制多个楼层
        # self.plot_building_by_floors(result)


        # 不同楼层平移
        result = self.align_floors_horizontally(result)

        result = self.convert_2d_to_3d_coordinates(result)

        # plotter =self.visualize_building_3d(result, show_spaces=False, show_modulars=True, color_by_floor=True)
        # plotter.show()
        new_standard_story = {str(i + 1): v for i, v in enumerate(result['GraphStandardBuildingStorey'].values())}
        result['GraphStandardBuildingStorey']=new_standard_story
        self.save_to_json(result)

    def visualize_building_3d(self,data_3d, show_spaces=True, show_modulars=True, color_by_floor=True):
        """
        使用 PyVista 可视化三维建筑数据

        Args:
            data_3d: 包含三维坐标的建筑数据字典
            show_spaces: 是否显示空间
            show_modulars: 是否显示模块
            color_by_floor: 是否按楼层着色

        Returns:
            plotter: PyVista 绘图器对象，可用于交互或保存
        """
        # 创建 PyVista 绘图器
        plotter = pv.Plotter()

        # 设置背景色为白色
        plotter.background_color = 'white'

        # 启用抗锯齿
        plotter.enable_anti_aliasing()

        # 获取楼层数据以确定颜色映射
        if "GraphBuildingStorey" in data_3d:
            storeys = data_3d["GraphBuildingStorey"]
            n_floors = len(storeys)
            # 为每个楼层生成一个颜色
            floor_colors = {}

            # 预定义不同楼层的颜色，避免相邻楼层颜色太相似
            base_colors = [
                (0.7, 0.7, 0.9),  # 淡蓝色
                (0.9, 0.7, 0.7),  # 淡红色
                (0.7, 0.9, 0.7),  # 淡绿色
                (0.9, 0.9, 0.7),  # 淡黄色
                (0.7, 0.9, 0.9),  # 淡青色
                (0.9, 0.7, 0.9),  # 淡紫色
                (0.8, 0.8, 0.8),  # 淡灰色
                (0.9, 0.8, 0.7),  # 淡橙色
            ]

            for i, (storey_id, _) in enumerate(sorted(storeys.items(), key=lambda x: int(x[0]))):
                floor_colors[storey_id] = base_colors[i % len(base_colors)]

        # 添加坐标轴
        plotter.add_axes()

        # 可视化 GraphSpace
        if show_spaces and "GraphSpace" in data_3d:
            spaces = data_3d["GraphSpace"]
            for space_id, space_data in spaces.items():
                if "coordinates" in space_data and len(space_data["coordinates"]) >= 3:
                    # 获取坐标
                    coords = np.array(space_data["coordinates"])

                    # 确保坐标是三维的
                    if coords.shape[1] == 3:
                        # 获取楼层高度
                        height = space_data.get("z_height", 100)  # 默认高度为100

                        # 创建底面多边形
                        bottom_face = coords

                        # 创建顶面多边形
                        top_face = bottom_face.copy()
                        top_face[:, 2] += height  # 增加Z坐标以形成高度

                        # 组合所有点
                        points = np.vstack((bottom_face, top_face))

                        # 创建底面和顶面的面索引
                        n_points = len(bottom_face)
                        bottom_indices = list(range(n_points))
                        top_indices = [i + n_points for i in range(n_points)]

                        # 创建一个单元格列表，表示底面和顶面
                        faces = []
                        # 底面
                        faces.append(n_points)
                        faces.extend(bottom_indices)
                        # 顶面
                        faces.append(n_points)
                        faces.extend(top_indices[::-1])  # 反向，确保法线朝外

                        # 添加侧面
                        for i in range(n_points):
                            faces.append(4)  # 四边形
                            faces.append(bottom_indices[i])
                            faces.append(bottom_indices[(i + 1) % n_points])
                            faces.append(top_indices[(i + 1) % n_points])
                            faces.append(top_indices[i])

                        # 创建 PyVista 网格对象
                        poly = pv.PolyData(points, faces=np.array(faces))

                        # 设置颜色
                        if color_by_floor:
                            story_idx = space_data.get("story_idx")
                            if story_idx in floor_colors:
                                color = floor_colors[story_idx]
                            else:
                                color = (0.8, 0.8, 0.8)  # 默认灰色
                        else:
                            color = (0.8, 0.8, 0.9)  # 蓝灰色

                        # 添加到场景
                        plotter.add_mesh(poly, color=color, opacity=0.7,
                                         show_edges=True, line_width=1,
                                         edge_color='black', label=f"Space {space_id}")

                        # 添加空间ID标签（如果需要）
                        center = poly.center
                        plotter.add_point_labels(
                            [center], [f"S{space_id}"],
                            font_size=10, point_color='red',
                            text_color='black', always_visible=True
                        )

        # 可视化 GraphModular
        if show_modulars and "GraphModular" in data_3d:
            modulars = data_3d["GraphModular"]
            for modular_id, modular_data in modulars.items():
                if "coordinates" in modular_data and len(modular_data["coordinates"]) >= 3:
                    # 获取坐标
                    coords = np.array(modular_data["coordinates"])

                    # 确保坐标是三维的
                    if coords.shape[1] == 3:
                        # 获取楼层高度
                        height = modular_data.get("z_height", 100)  # 默认高度为100

                        # 创建底面多边形
                        bottom_face = coords

                        # 创建顶面多边形
                        top_face = bottom_face.copy()
                        top_face[:, 2] += height  # 增加Z坐标以形成高度

                        # 组合所有点
                        points = np.vstack((bottom_face, top_face))

                        # 创建底面和顶面的面索引
                        n_points = len(bottom_face)
                        bottom_indices = list(range(n_points))
                        top_indices = [i + n_points for i in range(n_points)]

                        # 创建一个单元格列表，表示底面和顶面
                        faces = []
                        # 底面
                        faces.append(n_points)
                        faces.extend(bottom_indices)
                        # 顶面
                        faces.append(n_points)
                        faces.extend(top_indices[::-1])  # 反向，确保法线朝外

                        # 添加侧面
                        for i in range(n_points):
                            faces.append(4)  # 四边形
                            faces.append(bottom_indices[i])
                            faces.append(bottom_indices[(i + 1) % n_points])
                            faces.append(top_indices[(i + 1) % n_points])
                            faces.append(top_indices[i])

                        # 创建 PyVista 网格对象
                        poly = pv.PolyData(points, faces=np.array(faces))

                        # 设置颜色 - 模块使用更鲜艳的颜色
                        story_idx = None
                        for space_id, space_data in data_3d.get("GraphSpace", {}).items():
                            if int(space_id) == modular_data.get("space_idx"):
                                story_idx = space_data.get("story_idx")
                                break

                        if color_by_floor and story_idx in floor_colors:
                            # 使用比楼层更鲜艳的颜色
                            r, g, b = floor_colors[story_idx]
                            color = (min(r + 0.2, 1.0), min(g + 0.2, 1.0), min(b + 0.2, 1.0))
                        else:
                            color = (1.0, 0.6, 0.6)  # 浅红色

                        # 添加到场景 - 模块透明度较低，更好地区分
                        plotter.add_mesh(poly, color=color, opacity=0.8,
                                         show_edges=True, line_width=2,
                                         edge_color='red', label=f"Modular {modular_id}")

                        # 添加模块ID标签
                        center = poly.center
                        plotter.add_point_labels(
                            [center], [f"M{modular_id}"],
                            font_size=10, point_color='blue',
                            text_color='black', always_visible=True
                        )

        # 添加图例
        if show_spaces and show_modulars:
            plotter.add_legend([("Spaces", 'gray'), ("Modulars", 'red')])

        # 优化视角
        plotter.view_isometric()

        # 启用鼠标交互
        plotter.enable_trackball_style()

        return plotter


    def convert_2d_to_3d_coordinates(self,aligned_data):
        """
        将已对齐的二维坐标转换为三维坐标，并替换原有坐标
        同一层的modular和space具有相同的Z坐标高度

        流程：
        1. 创建三维坐标并存为coordinates_3d
        2. 删除原有的coordinates
        3. 将coordinates_3d重命名为coordinates

        Args:
            aligned_data: 已水平对齐的建筑数据字典

        Returns:
            包含三维坐标的建筑数据
        """
        # 复制输入数据，避免修改原始数据
        data_3d = aligned_data.copy()

        # 确保有楼层数据
        if "GraphBuildingStorey" not in aligned_data:
            print("错误: 未找到GraphBuildingStorey数据")
            return data_3d

        # 获取楼层数据
        storeys = aligned_data["GraphBuildingStorey"]

        # 计算每层的累积高度
        cumulative_heights = {}
        current_height = 0

        # 按楼层编号排序，从低到高处理
        sorted_storeys = sorted(storeys.items(), key=lambda x: int(x[0]))

        for storey_id, storey_data in sorted_storeys:
            # 获取当前楼层高度，默认为0如果未指定
            storey_height = storey_data.get("story_height", 0)

            # 存储当前层的起始高度（楼层底部高度）
            cumulative_heights[storey_id] = current_height

            # 为下一层准备累积高度
            current_height += storey_height

            # 打印调试信息
            # print(f"楼层 {storey_id}: 起始高度 = {cumulative_heights[storey_id]}, 层高 = {storey_height}")

        # 调整GraphSpace坐标
        if "GraphSpace" in aligned_data:
            spaces = data_3d["GraphSpace"]
            for space_id, space_data in spaces.items():
                story_idx = space_data.get("story_idx")

                # 如果该空间属于有高度数据的楼层，则添加Z坐标
                if story_idx in cumulative_heights and "coordinates" in space_data:
                    z_height = cumulative_heights[story_idx]

                    # 创建三维坐标
                    new_coordinates_3d = []
                    for coord in space_data["coordinates"]:
                        # 添加Z坐标
                        new_coord_3d = [coord[0], coord[1], z_height]
                        new_coordinates_3d.append(new_coord_3d)

                    # 先创建coordinates_3d
                    spaces[space_id]["coordinates"] = new_coordinates_3d

                    # 添加楼层高度信息
                    # spaces[space_id]["z_level"] = z_height
                    # spaces[space_id]["z_height"] = storeys[story_idx].get("story_height", 0)

        # 调整GraphModular坐标
        if "GraphModular" in aligned_data:
            modulars = data_3d["GraphModular"]
            spaces = aligned_data.get("GraphSpace", {})

            for modular_id, modular_data in modulars.items():
                # 获取该模块对应的空间ID
                space_idx = modular_data.get("space_idx")

                # 查找该空间所属的楼层
                story_idx = None
                for space_id, space_data in spaces.items():
                    if int(space_id) == space_idx:
                        story_idx = space_data.get("story_idx")
                        break

                # 如果找到了楼层且有高度数据，添加Z坐标
                if story_idx in cumulative_heights and "coordinates" in modular_data:
                    z_height = cumulative_heights[story_idx]

                    # 创建三维坐标
                    new_coordinates_3d = []
                    for coord in modular_data["coordinates"]:
                        # 添加Z坐标
                        new_coord_3d = [coord[0], coord[1], z_height]
                        new_coordinates_3d.append(new_coord_3d)

                    # 先创建coordinates_3d
                    modulars[modular_id]["coordinates"] = new_coordinates_3d
                    # 获取所有x坐标值(第一列)
                    x_values = [coord[0] for coord in new_coordinates_3d]
                    # 获取所有y坐标值(第二列)
                    y_values = [coord[1] for coord in new_coordinates_3d]

                    # 找出最大最小值
                    x_min = min(x_values)  # 2367143.84046855
                    x_max = max(x_values)  # 2370343.84046855
                    y_min = min(y_values)  # -961984.173038333
                    y_max = max(y_values)  # -949384.173038333

                    modular_length = x_max-x_min
                    modular_width = y_max - y_min
                    modular_height = storeys[story_idx].get("story_height", 0)-z_height

                    length_final = max(modular_length,modular_width)
                    width_final = min(modular_length, modular_width)
                    modulars[modular_id]["modular_size"] = [length_final,width_final,modular_height]
                    # 添加楼层高度信息
                    # modulars[modular_id]["z_level"] = z_height
                    # modulars[modular_id]["z_height"] = storeys[story_idx].get("story_height", 0)

        # 添加楼层高度信息到GraphBuildingStorey
        for storey_id, storey_data in data_3d["GraphBuildingStorey"].items():
            if storey_id in cumulative_heights:
                # storey_data["z_level"] = cumulative_heights[storey_id]
                # 添加楼层顶部高度
                storey_data["z_top"] = cumulative_heights[storey_id] + storey_data.get("story_height", 0)

        # # 第二步：删除原有的coordinates，将coordinates_3d重命名为coordinates
        # if "GraphSpace" in data_3d:
        #     for space_id, space_data in data_3d["GraphSpace"].items():
        #         if "coordinates_3d" in space_data:
        #             # 先删除原有的coordinates
        #             if "coordinates" in space_data:
        #                 del space_data["coordinates"]
        #
        #             # 将coordinates_3d重命名为coordinates
        #             space_data["coordinates"] = space_data["coordinates_3d"]
        #             del space_data["coordinates_3d"]
        #
        # if "GraphModular" in data_3d:
        #     for modular_id, modular_data in data_3d["GraphModular"].items():
        #         if "coordinates_3d" in modular_data:
        #             # 先删除原有的coordinates
        #             if "coordinates" in modular_data:
        #                 del modular_data["coordinates"]
        #
        #             # 将coordinates_3d重命名为coordinates
        #             modular_data["coordinates"] = modular_data["coordinates_3d"]
        #             del modular_data["coordinates_3d"]

        return data_3d

    def align_floors_horizontally(self,building_data):
        """
        对建筑数据中的各楼层进行水平对齐
        以第一层的label_point为基准，调整其他楼层的space和modular坐标

        Args:
            building_data: 包含GraphBuildingStorey、GraphSpace和GraphModular的建筑数据字典

        Returns:
            水平对齐后的建筑数据
        """
        # 复制输入数据，避免修改原始数据
        aligned_data = building_data.copy()

        # 确保有楼层数据
        if "GraphBuildingStorey" not in building_data:
            print("错误: 未找到GraphBuildingStorey数据")
            return aligned_data

        # 获取楼层数据
        storeys = building_data["GraphBuildingStorey"]

        # 找到第一层的label_point作为基准
        reference_point = None
        for storey_id, storey_data in storeys.items():
            if storey_id == "1" and "label_point" in storey_data:
                reference_point = storey_data["label_point"]
                break

        # 如果没有找到第一层的label_point，则终止
        if reference_point is None:
            print("错误: 未找到第一层的label_point")
            return aligned_data

        # 为每个楼层计算偏移量并调整坐标
        offset_by_floor = {}
        for storey_id, storey_data in storeys.items():
            if "label_point" in storey_data:
                # 计算与基准点的水平偏移
                floor_point = storey_data["label_point"]
                offset_x = floor_point[0] - reference_point[0]
                offset_y = floor_point[1] - reference_point[1]
                offset_by_floor[storey_id] = (offset_x, offset_y)

        # 调整GraphSpace坐标
        if "GraphSpace" in building_data:
            spaces = aligned_data["GraphSpace"]
            for space_id, space_data in spaces.items():
                story_idx = space_data.get("story_idx")

                # 如果该空间属于有偏移量的楼层，则调整坐标
                if story_idx in offset_by_floor and "coordinates" in space_data:
                    offset_x, offset_y = offset_by_floor[story_idx]

                    # 应用偏移量到坐标
                    new_coordinates = []
                    for coord in space_data["coordinates"]:
                        new_x = coord[0] - offset_x
                        new_y = coord[1] - offset_y
                        new_coordinates.append([new_x, new_y])

                    # 更新坐标
                    spaces[space_id]["coordinates"] = new_coordinates

        # 调整GraphModular坐标
        if "GraphModular" in building_data:
            modulars = aligned_data["GraphModular"]
            spaces = building_data.get("GraphSpace", {})

            for modular_id, modular_data in modulars.items():
                # 获取该模块对应的空间ID
                space_idx = modular_data.get("space_idx")

                # 查找该空间所属的楼层
                story_idx = None
                for space_id, space_data in spaces.items():
                    if int(space_id) == space_idx:
                        story_idx = space_data.get("story_idx")
                        break

                # 如果找到了楼层且有偏移量，调整坐标
                if story_idx in offset_by_floor and "coordinates" in modular_data:
                    offset_x, offset_y = offset_by_floor[story_idx]

                    # 应用偏移量到坐标
                    new_coordinates = []
                    for coord in modular_data["coordinates"]:
                        new_x = coord[0] - offset_x
                        new_y = coord[1] - offset_y
                        new_coordinates.append([new_x, new_y])

                    # 更新坐标
                    modulars[modular_id]["coordinates"] = new_coordinates

        # 更新GraphStandardBuildingStorey的label_point
        if "GraphStandardBuildingStorey" in building_data:
            std_storeys = aligned_data["GraphStandardBuildingStorey"]
            for storey_id, storey_data in std_storeys.items():
                if "label_point" in storey_data:
                    # 处理单层和多层的情况
                    if "-" in storey_id:
                        # 对于标准层包含多个楼层的情况，使用第一个楼层的偏移量
                        start_floor, _ = storey_id.split("-")
                        if start_floor in offset_by_floor:
                            offset_x, offset_y = offset_by_floor[start_floor]
                            storey_data["label_point"] = [
                                storey_data["label_point"][0] - offset_x,
                                storey_data["label_point"][1] - offset_y
                            ]
                    else:
                        # 单层的情况直接使用该层的偏移量
                        if storey_id in offset_by_floor:
                            offset_x, offset_y = offset_by_floor[storey_id]
                            storey_data["label_point"] = [
                                storey_data["label_point"][0] - offset_x,
                                storey_data["label_point"][1] - offset_y
                            ]

        # 更新GraphBuildingStorey的label_point
        for storey_id, storey_data in storeys.items():
            if "label_point" in storey_data and storey_id in offset_by_floor:
                offset_x, offset_y = offset_by_floor[storey_id]
                storey_data["label_point"] = [
                    storey_data["label_point"][0] - offset_x,
                    storey_data["label_point"][1] - offset_y
                ]

        return aligned_data

    def transform_building_data(self,input_json):
        """
        将标准层信息转换为包含单独楼层信息的建筑数据

        Args:
            input_json: 输入的建筑标准层数据

        Returns:
            包含GraphBuildingStorey字典的更新后的建筑数据
        """
        # 复制输入数据，避免修改原始数据
        output_json = input_json.copy()

        # 初始化GraphBuildingStorey字典
        output_json["GraphBuildingStorey"] = {}

        # 从GraphStandardBuildingStorey提取楼层信息
        if "GraphStandardBuildingStorey" in input_json:
            standard_storeys = input_json["GraphStandardBuildingStorey"]

            # 处理每个标准层
            for storey_group, storey_data in standard_storeys.items():
                # 检查是否包含连字符，表示多个楼层
                if "-" in storey_group:
                    # 分割楼层范围
                    start_floor, end_floor = storey_group.split("-")
                    floor_range = range(int(start_floor), int(end_floor) + 1)
                    output_json["GraphStandardBuildingStorey"][storey_group]['story_id']=list(floor_range)
                else:
                    # 单个楼层
                    floor_range = [int(storey_group)]
                    output_json["GraphStandardBuildingStorey"][storey_group]['story_id'] = floor_range
                # 为每个楼层创建条目
                for floor in floor_range:
                    floor_str = str(floor)
                    output_json["GraphBuildingStorey"][floor_str] = {
                        "story_id": int(floor_str),
                        "story_height": storey_data.get("story_height"),
                        "standard_storey": storey_group
                    }

                    # 如果有label_point，也添加它
                    if "label_point" in storey_data:
                        output_json["GraphBuildingStorey"][floor_str]["label_point"] = storey_data["label_point"]


        return output_json

    def transform_space_data(self,input_json):
        """
        将GraphSpace中的标准层信息转换为单独楼层的信息

        Args:
            input_json: 输入的建筑数据，包含GraphSpace字典

        Returns:
            更新后的建筑数据，GraphSpace按楼层重新组织
        """
        # 复制输入数据，避免修改原始数据
        output_json = input_json.copy()

        # 处理GraphSpace数据，按楼层重新组织
        if "GraphSpace" in input_json:
            # 临时存储原始GraphSpace数据
            original_space = input_json["GraphSpace"]
            new_space = {}

            # 创建一个计数器来为新空间分配唯一ID
            space_counter = 1

            # 按照story_idx分类空间
            space_by_story = {}
            for space_id, space_data in original_space.items():
                story_idx = space_data.get("story_idx")
                if story_idx not in space_by_story:
                    space_by_story[story_idx] = []

                # 将空间数据复制并添加原始ID以供参考
                space_copy = space_data.copy()
                space_copy["original_id"] = space_id
                space_by_story[story_idx].append(space_copy)

            # 为每个楼层生成空间数据
            for story_idx, spaces in space_by_story.items():
                # 检查是否包含连字符，表示多个楼层
                if "-" in story_idx:
                    # 分割楼层范围
                    start_floor, end_floor = story_idx.split("-")
                    floor_range = [str(i) for i in range(int(start_floor), int(end_floor) + 1)]
                else:
                    # 单个楼层
                    floor_range = [story_idx]

                # 为每个楼层复制空间数据
                for floor in floor_range:
                    for space in spaces:
                        # 创建新的空间数据
                        new_space_data = space.copy()
                        # 更新story_idx为当前楼层
                        new_space_data["story_idx"] = floor
                        # 删除original_id，除非您想保留它
                        if "original_id" in new_space_data:
                            del new_space_data["original_id"]

                        # 添加到新的GraphSpace字典中
                        new_space[str(space_counter)] = new_space_data
                        space_counter += 1

            # 更新输出JSON中的GraphSpace
            output_json["GraphSpace"] = new_space

        return output_json

    def transform_space_and_modular_data(self,input_json):
        """
        将GraphSpace和GraphModular中的标准层信息转换为单独楼层的信息

        Args:
            input_json: 输入的建筑数据，包含GraphSpace和GraphModular字典

        Returns:
            更新后的建筑数据，GraphSpace和GraphModular按楼层重新组织
        """
        # 复制输入数据，避免修改原始数据
        output_json = input_json.copy()

        # 临时存储原始数据
        original_space = input_json.get("GraphSpace", {})
        original_modular = input_json.get("GraphModular", {})

        new_space = {}
        new_modular = {}

        # 创建计数器来为新空间和模块分配唯一ID
        space_counter = 1
        modular_counter = 1

        # 创建映射表，记录原始modular_id到新modular_id的映射
        modular_id_mapping = {}

        # 按照story_idx分类空间
        space_by_story = {}
        for space_id, space_data in original_space.items():
            story_idx = space_data.get("story_idx")
            if story_idx not in space_by_story:
                space_by_story[story_idx] = []

            # 将空间数据复制并添加原始ID以供参考
            space_copy = space_data.copy()
            space_copy["original_id"] = space_id
            space_by_story[story_idx].append(space_copy)

        # 为每个楼层生成空间和模块数据
        for story_idx, spaces in space_by_story.items():
            # 检查是否包含连字符，表示多个楼层
            if "-" in story_idx:
                # 分割楼层范围
                start_floor, end_floor = story_idx.split("-")
                floor_range = [str(i) for i in range(int(start_floor), int(end_floor) + 1)]
            else:
                # 单个楼层
                floor_range = [story_idx]

            # 为每个楼层复制空间数据
            for floor in floor_range:
                for space in spaces:
                    # 创建新的空间数据
                    new_space_data = space.copy()
                    # 更新story_idx为当前楼层
                    new_space_data["story_idx"] = floor

                    # 获取原始modular_id列表
                    original_modular_ids = new_space_data.get("modular_id", [])
                    new_modular_ids = []

                    # 为每个模块创建新条目
                    for original_modular_id in original_modular_ids:
                        original_modular_id_str = str(original_modular_id)

                        # 检查是否已经为当前楼层创建了这个模块的复制
                        mapping_key = f"{original_modular_id_str}_{floor}"
                        if mapping_key not in modular_id_mapping:
                            # 如果原始模块存在，创建它的复制
                            if original_modular_id_str in original_modular:
                                # 复制模块数据
                                modular_data = original_modular[original_modular_id_str].copy()
                                # 更新space_idx引用
                                modular_data["space_idx"] = space_counter

                                # 添加到新的GraphModular字典
                                modular_id_mapping[mapping_key] = modular_counter
                                new_modular[str(modular_counter)] = modular_data
                                new_modular_ids.append(modular_counter)
                                modular_counter += 1
                        else:
                            # 如果已经创建了，使用现有的映射
                            new_modular_ids.append(modular_id_mapping[mapping_key])

                    # 更新modular_id列表
                    new_space_data["modular_id"] = new_modular_ids

                    # 删除original_id，除非您想保留它
                    if "original_id" in new_space_data:
                        del new_space_data["original_id"]

                    # 添加到新的GraphSpace字典
                    new_space[str(space_counter)] = new_space_data
                    space_counter += 1

        # 更新输出JSON中的GraphSpace和GraphModular
        output_json["GraphSpace"] = new_space
        output_json["GraphModular"] = new_modular

        for space_id, space_data in output_json["GraphSpace"].items():
            output_json["GraphSpace"][space_id]['space_id']=int(space_id)
            output_json["GraphSpace"][space_id]['type_parameters'] = [0,0,0,0,0,0]

        for modular_id, modular_data in output_json["GraphModular"].items():
            output_json["GraphModular"][modular_id]['modular_id']=int(modular_id)

        return output_json

    def plot_building_rectangles(self,building_data, floor=None, figsize=(12, 10)):
        """
        使用matplotlib绘制建筑中GraphSpace和GraphModular的矩形

        Args:
            building_data: 包含GraphSpace和GraphModular的建筑数据字典
            floor: 指定要绘制哪一层的矩形，如果为None则绘制所有层
            figsize: 图表大小
        """
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=figsize)

        # 提取空间和模块数据
        spaces = building_data.get("GraphSpace", {})
        modulars = building_data.get("GraphModular", {})

        # 用于存储所有坐标以确定图的范围
        all_x = []
        all_y = []

        # 绘制GraphSpace矩形
        for space_id, space_data in spaces.items():
            # 如果指定了楼层，只绘制该楼层的空间
            if floor is not None and space_data.get("story_idx") != str(floor):
                continue

            if "coordinates" in space_data:
                coords = space_data["coordinates"]
                # 将坐标提取为x和y列表
                x = [p[0] for p in coords]
                y = [p[1] for p in coords]

                # 存储用于确定图范围
                all_x.extend(x)
                all_y.extend(y)

                # 创建多边形
                polygon = patches.Polygon(coords, closed=True, fill=True,
                                          edgecolor='blue', facecolor='lightblue',
                                          alpha=0.5, label=f'Space {space_id}')
                ax.add_patch(polygon)

                # 添加空间ID文本
                center_x = np.mean(x)
                center_y = np.mean(y)
                ax.text(center_x, center_y, f'S{space_id}',
                        ha='center', va='center', fontsize=9)

        # 绘制GraphModular矩形
        for modular_id, modular_data in modulars.items():
            # 找到这个modular对应的space
            space_idx = modular_data.get("space_idx")

            # 如果指定了楼层，检查相关联的空间是否在该楼层
            if floor is not None:
                space_floor = None
                for space_id, space_data in spaces.items():
                    if int(space_id) == space_idx:
                        space_floor = space_data.get("story_idx")
                        break

                if space_floor != str(floor):
                    continue

            if "coordinates" in modular_data:
                coords = modular_data["coordinates"]
                # 将坐标提取为x和y列表
                x = [p[0] for p in coords]
                y = [p[1] for p in coords]

                # 存储用于确定图范围
                all_x.extend(x)
                all_y.extend(y)

                # 创建多边形
                polygon = patches.Polygon(coords, closed=True, fill=True,
                                          edgecolor='red', facecolor='none',
                                          linewidth=1.5, label=f'Modular {modular_id}')
                ax.add_patch(polygon)

                # 添加模块ID文本
                center_x = np.mean(x)
                center_y = np.mean(y)
                ax.text(center_x, center_y, f'M{modular_id}',
                        ha='center', va='center', fontsize=8, color='red')

        # 设置图的范围，稍微扩大以便更好地查看
        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            # 扩大范围的缓冲
            buffer_x = (max_x - min_x) * 0.05
            buffer_y = (max_y - min_y) * 0.05

            ax.set_xlim(min_x - buffer_x, max_x + buffer_x)
            ax.set_ylim(min_y - buffer_y, max_y + buffer_y)

        # 设置标题和标签
        if floor:
            ax.set_title(f'Building Floor {floor} - Spaces and Modules')
        else:
            ax.set_title('Building Spaces and Modules (All Floors)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        # 设置相等的纵横比，使矩形看起来正确
        ax.set_aspect('equal')

        # 添加图例
        space_patch = patches.Patch(color='lightblue', alpha=0.5, edgecolor='blue', label='Spaces')
        modular_patch = patches.Patch(edgecolor='red', facecolor='none', label='Modulars')
        ax.legend(handles=[space_patch, modular_patch])

        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

        return fig, ax

    def plot_building_by_floors(self,building_data, floors=None, figsize=(15, 12)):
        """
        绘制建筑的多个楼层，每个楼层一个子图

        Args:
            building_data: 包含GraphSpace和GraphModular的建筑数据字典
            floors: 要绘制的楼层列表，如果为None则从数据中推断
            figsize: 整个图表的大小
        """
        # 如果没有指定楼层，从数据中推断
        if floors is None:
            floors = set()
            for _, space_data in building_data.get("GraphSpace", {}).items():
                if "story_idx" in space_data:
                    floors.add(space_data["story_idx"])
            floors = sorted(floors, key=lambda x: int(x))

        # 计算子图的行和列数
        n_floors = len(floors)
        n_cols = min(3, n_floors)  # 最多3列
        n_rows = (n_floors + n_cols - 1) // n_cols  # 向上取整

        # 创建图形和轴
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # 确保axes是二维数组，即使只有一行或一列
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # 提取空间和模块数据
        spaces = building_data.get("GraphSpace", {})
        modulars = building_data.get("GraphModular", {})

        # 为每个楼层创建子图
        for idx, floor in enumerate(floors):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # 用于存储所有坐标以确定图的范围
            all_x = []
            all_y = []

            # 绘制GraphSpace矩形
            for space_id, space_data in spaces.items():
                if space_data.get("story_idx") == str(floor) and "coordinates" in space_data:
                    coords = space_data["coordinates"]
                    # 将坐标提取为x和y列表
                    x = [p[0] for p in coords]
                    y = [p[1] for p in coords]

                    # 存储用于确定图范围
                    all_x.extend(x)
                    all_y.extend(y)

                    # 创建多边形
                    polygon = patches.Polygon(coords, closed=True, fill=True,
                                              edgecolor='blue', facecolor='lightblue',
                                              alpha=0.5)
                    ax.add_patch(polygon)

                    # 添加空间ID文本
                    center_x = np.mean(x)
                    center_y = np.mean(y)
                    ax.text(center_x, center_y, f'S{space_id}',
                            ha='center', va='center', fontsize=8)

            # 绘制GraphModular矩形
            for modular_id, modular_data in modulars.items():
                # 找到这个modular对应的space
                space_idx = modular_data.get("space_idx")

                # 检查相关联的空间是否在当前楼层
                space_floor = None
                for space_id, space_data in spaces.items():
                    if int(space_id) == space_idx:
                        space_floor = space_data.get("story_idx")
                        break

                if space_floor == str(floor) and "coordinates" in modular_data:
                    coords = modular_data["coordinates"]
                    # 将坐标提取为x和y列表
                    x = [p[0] for p in coords]
                    y = [p[1] for p in coords]

                    # 存储用于确定图范围
                    all_x.extend(x)
                    all_y.extend(y)

                    # 创建多边形
                    polygon = patches.Polygon(coords, closed=True, fill=False,
                                              edgecolor='red', linewidth=1.5)
                    ax.add_patch(polygon)

                    # 添加模块ID文本
                    center_x = np.mean(x)
                    center_y = np.mean(y)
                    ax.text(center_x, center_y, f'M{modular_id}',
                            ha='center', va='center', fontsize=7, color='red')

            # 设置图的范围，稍微扩大以便更好地查看
            if all_x and all_y:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)

                # 扩大范围的缓冲
                buffer_x = (max_x - min_x) * 0.05
                buffer_y = (max_y - min_y) * 0.05

                ax.set_xlim(min_x - buffer_x, max_x + buffer_x)
                ax.set_ylim(min_y - buffer_y, max_y + buffer_y)

            # 设置标题和标签
            ax.set_title(f'Floor {floor}')

            # 只为最左侧的子图添加Y轴标签
            if col == 0:
                ax.set_ylabel('Y Coordinate')

            # 只为最底部的子图添加X轴标签
            if row == n_rows - 1:
                ax.set_xlabel('X Coordinate')

            # 设置相等的纵横比，使矩形看起来正确
            ax.set_aspect('equal')

            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.7)

        # 禁用未使用的子图
        for idx in range(len(floors), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        # 添加图例到整个图形
        space_patch = patches.Patch(color='lightblue', alpha=0.5, edgecolor='blue', label='Spaces')
        modular_patch = patches.Patch(edgecolor='red', facecolor='none', label='Modulars')
        fig.legend(handles=[space_patch, modular_patch], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2)

        # 调整子图之间的间距和边距
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为图例留出空间
        plt.suptitle('Building Spaces and Modules by Floor', fontsize=16)

        plt.show()

        return fig, axes



    def convert_to_simple_format(self, building_info):
        """
        将building_info转换为简化的格式，处理spaces和modules，
        确保每个space和modular都有完整的四点矩形coordinates

        Args:
            building_info: 按楼层组织的建筑信息字典

        Returns:
            转换后的简化格式字典
        """
        result = {
            "GraphProject": "建筑项目",
            "GraphSite": "建筑场地",
            "GraphBuilding": "主建筑",
            "GraphStandardBuildingStorey": {},
            "GraphSpace": {}
        }

        # 空间计数器
        space_counter = 1

        # 添加所有楼层信息
        for floor_name, floor_data in building_info.items():
            floor_number = floor_name.replace('F', '')
            result["GraphStandardBuildingStorey"][f"{floor_number}"] = {
                "story_height": 3500  # 使用固定层高
            }
        for floor_name, floor_data in building_info.items():
            floor_number = floor_name.replace('F', '')
            result["GraphStandardBuildingStorey"][f"{floor_number}"]['label_point'] = floor_data['center_point']

        # 空间数据存储，用于后续查找包含模块的空间
        all_spaces = []

        # 处理每个楼层的房间
        for floor_name, floor_data in building_info.items():
            floor_number = floor_name.replace('F', '')

            # 处理每种房间类型
            for room_type, rooms in floor_data.get("rooms", {}).items():
                # 处理该房间类型中的每个具体房间
                for room in rooms:
                    # 去除重复的坐标点
                    coords = room["coordinates"]
                    unique_coords = self.remove_duplicate_coords(coords)

                    # 计算矩形边界框
                    min_x = min(coord[0] for coord in unique_coords)
                    min_y = min(coord[1] for coord in unique_coords)
                    max_x = max(coord[0] for coord in unique_coords)
                    max_y = max(coord[1] for coord in unique_coords)

                    # 创建矩形的四个角点（顺时针排列）
                    rect_corners = [
                        (min_x, min_y),  # 左下
                        (max_x, min_y),  # 右下
                        (max_x, max_y),  # 右上
                        (min_x, max_y)  # 左上
                    ]

                    # 每个坐标列表对应一个space
                    space_key = f"{space_counter}"
                    result["GraphSpace"][space_key] = {
                        "story_idx": floor_number,
                        "room_type": int(room_type.split('-')[1]),
                        "coordinates": rect_corners  # 添加空间的矩形坐标
                    }

                    # 存储空间信息，用于后续查找
                    all_spaces.append({
                        "id": space_counter,
                        "coordinates": rect_corners,
                        "story_idx": floor_number
                    })

                    space_counter += 1

        # 处理所有模块
        if "GraphModular" not in result.keys():
            result["GraphModular"] = {}

        modular_counter = 1

        space_modular = {}

        # 处理每个楼层的模块
        for floor_name, floor_data in building_info.items():
            floor_number = floor_name.replace('F', '')

            # 处理每种模块类型
            for module_type, modules in floor_data.get("modules", {}).items():
                # 处理该类型中的每个具体模块
                for module in modules:
                    # 去除重复的坐标点
                    coords = module["coordinates"]
                    unique_coords = self.remove_duplicate_coords(coords)

                    # 计算矩形边界框
                    min_x = min(coord[0] for coord in unique_coords)
                    min_y = min(coord[1] for coord in unique_coords)
                    max_x = max(coord[0] for coord in unique_coords)
                    max_y = max(coord[1] for coord in unique_coords)

                    # 创建矩形的四个角点（顺时针排列）
                    rect_corners = [
                        (min_x, min_y),  # 左下
                        (max_x, min_y),  # 右下
                        (max_x, max_y),  # 右上
                        (min_x, max_y)  # 左上
                    ]

                    # 计算模块中心点
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    center_point = (center_x, center_y)

                    # 查找包含该中心点的空间
                    space_idx = self.find_containing_space(center_point, all_spaces, floor_number)

                    # 如果没有找到包含该点的空间，分配给默认空间
                    if space_idx is None:
                        space_idx = 1  # 默认分配给第一个空间

                    # 添加模块
                    modular_key = f"{modular_counter}"
                    result["GraphModular"][modular_key] = {
                        "space_idx": space_idx,
                        "type_parameters": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        "type": int(module_type.split('-')[1]),
                        "coordinates": rect_corners  # 添加模块的矩形坐标
                    }
                    if space_idx not in space_modular:
                        space_modular[space_idx] = [modular_counter]
                    else:
                        space_modular[space_idx].append(modular_counter)
                    modular_counter += 1

        for space_id, modular_data in space_modular.items():
            result['GraphSpace'][f'{space_id}']['modular_id'] = modular_data
        return result

    def convert_to_building_info(self, input_data):
        """
        将输入的DXF实体信息转换为按楼层组织的建筑信息字典

        Args:
            input_data: 从DXF文件中提取的原始实体信息字典

        Returns:
            一个按楼层组织的建筑信息字典
        """
        building_info = {}

        # 遍历每个坐标键（代表不同的框架/楼层）
        for frame_key, frame_data in input_data.items():
            floor_text = None

            # 寻找楼层号文本
            if 'text' in frame_data:
                for text_entity in frame_data['text']:
                    if text_entity['type'] == 'MTEXT' and text_entity['text'].endswith('F'):
                        floor_text = text_entity['text']
                        break

            # 如果没有找到楼层文本，跳过
            if floor_text is None:
                continue

            # 为该楼层创建字典结构
            floor_dict = {
                "outline": {"coordinates": []},
                "rooms": {},
                "modules": {}
            }

            # 处理outline（区域边界）
            if 'outline' in frame_data:
                outline_entity = frame_data['outline'][0]  # 假设只有一个outline
                coords = []
                for segment in outline_entity['segments']:
                    if len(coords) == 0:  # 第一个点
                        coords.append(segment['start'])
                    coords.append(segment['end'])
                floor_dict["outline"]["coordinates"] = coords

            # 处理rooms（房间）
            for layer_name, entities in frame_data.items():
                # 跳过非房间层
                if layer_name in ['text', 'print_frame', 'outline'] or '×' in layer_name:
                    continue

                # 初始化该类型房间的列表
                # if layer_name not in floor_dict["rooms"]:
                if layer_name.split('-')[0] == '20000':
                    floor_dict["rooms"][layer_name] = []

                    # 添加每个房间的坐标信息
                    for entity in entities:
                        if entity["type"] in ["POLYLINE", "LWPOLYLINE"]:
                            coords = []
                            for segment in entity['segments']:
                                if len(coords) == 0:  # 第一个点
                                    coords.append(segment['start'])
                                coords.append(segment['end'])
                            if layer_name.split('-')[0] == '20000':
                                # 添加调试信息
                                # print("Floor dict rooms type:", type(floor_dict["rooms"]))
                                # print("Layer name:", layer_name)
                                # print("Current rooms content:", floor_dict["rooms"])
                                #
                                floor_dict["rooms"][layer_name].append({"coordinates": coords})

            # 处理modules（模块单元）
            for layer_name, entities in frame_data.items():
                # 只处理带尺寸的图层（模块单元）
                if layer_name.split('-')[0] == '30000':
                    # 初始化该类型模块的列表
                    if layer_name not in floor_dict["modules"]:
                        floor_dict["modules"][layer_name] = []

                    # 添加每个模块的坐标信息
                    for entity in entities:
                        if entity["type"] in ["POLYLINE", "LWPOLYLINE"]:
                            coords = []
                            for segment in entity['segments']:
                                if len(coords) == 0:  # 第一个点
                                    coords.append(segment['start'])
                                coords.append(segment['end'])

                            # 只有当实体是闭合的，或者我们能构成闭合路径时才添加
                            if entity.get('closed', False) or (len(coords) >= 3 and coords[0] == coords[-1]):
                                floor_dict["modules"][layer_name].append({"coordinates": coords})
                            else:
                                # 如果轮廓不闭合，尝试通过首尾相连来闭合
                                coords.append(coords[0])
                                floor_dict["modules"][layer_name].append({"coordinates": coords})

            # 将处理好的楼层信息添加到结果字典
            building_info[floor_text] = floor_dict

        # 存入每层楼中心点
        # 获取源字典和目标字典的键列表
        source_keys = list(input_data.keys())
        target_keys = list(building_info.keys())

        key_to_copy = 'center_point'

        # 遍历键，将input_data中的中心点信息复制到building_info中
        for i, (source_key, target_key) in enumerate(zip(source_keys, target_keys)):
            # 确保两个子字典都存在
            if isinstance(input_data[source_key], dict) and isinstance(building_info[target_key], dict):
                # 检查源字典中是否存在要复制的键
                if key_to_copy in input_data[source_key]:
                    # 将键值对复制到目标字典
                    building_info[target_key][key_to_copy] = input_data[source_key][key_to_copy]

        return building_info

    def remove_duplicate_coords(self, coords):
        """去除重复的坐标点，包括闭合多边形中的重复点"""
        unique_coords = []
        for coord in coords:
            if not any(self.coords_equal(coord, existing) for existing in unique_coords):
                unique_coords.append(coord)

        # 如果这是一个闭合多边形且第一个点和最后一个点相同，也去除
        if len(unique_coords) > 3 and self.coords_equal(unique_coords[0], unique_coords[-1]):
            unique_coords.pop()

        return unique_coords

    def coords_equal(self, coord1, coord2, tolerance=1e-6):
        """检查两个坐标是否相同（考虑浮点误差）"""
        return (abs(coord1[0] - coord2[0]) < tolerance and
                abs(coord1[1] - coord2[1]) < tolerance)

    def find_containing_space(self, point, spaces, floor_number):
        """
        查找包含指定点的空间

        Args:
            point: 要检查的点 (x, y)
            spaces: 所有空间的列表
            floor_number: 当前楼层编号

        Returns:
            包含该点的空间ID，如果没找到则返回None
        """
        # 首先检查同一楼层的空间
        for space in spaces:
            if space["story_idx"] == floor_number and self.is_point_in_polygon(point, space["coordinates"]):
                return space["id"]

        # 如果同一楼层没找到，尝试其他楼层
        for space in spaces:
            if self.is_point_in_polygon(point, space["coordinates"]):
                return space["id"]

        return None

    def is_point_in_polygon(self, point, polygon):
        """
        判断点是否在多边形内部
        这里我们处理的是矩形，所以可以简化判断

        Args:
            point: 要检查的点 (x, y)
            polygon: 矩形的四个角点

        Returns:
            布尔值，表示点是否在矩形内
        """
        x, y = point

        # 假设polygon是四个角点，顺序为：左下、右下、右上、左上
        min_x = min(p[0] for p in polygon)
        max_x = max(p[0] for p in polygon)
        min_y = min(p[1] for p in polygon)
        max_y = max(p[1] for p in polygon)

        # 判断点是否在矩形内
        return min_x <= x <= max_x and min_y <= y <= max_y

    def save_to_json(self, data):
        """
        将数据字典保存为JSON文件

        Args:
            data: 要保存的数据字典
            output_file: 输出的JSON文件路径
        """
        output_file = self.file_name
        try:
            # 使用自定义的编码器处理numpy数组和其他非标准JSON类型
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    import numpy as np
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    return super(NumpyEncoder, self).default(obj)

            # 确保坐标是列表格式，而不是元组（JSON不支持元组）
            def convert_tuples_to_lists(obj):
                if isinstance(obj, tuple):
                    return list(obj)
                elif isinstance(obj, list):
                    return [convert_tuples_to_lists(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_tuples_to_lists(value) for key, value in obj.items()}
                else:
                    return obj

            # 转换数据中的元组为列表
            processed_data = convert_tuples_to_lists(data)

            # 将数据写入JSON文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

            # print(f"数据已成功保存到 {output_file}")

        except Exception as e:
            print(f"保存JSON文件时出错: {str(e)}")
