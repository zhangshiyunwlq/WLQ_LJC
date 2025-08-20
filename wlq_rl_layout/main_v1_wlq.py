import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from rl_split_v3 import train_dqn
import matplotlib.pyplot as plt
from rl_data_proc import *
import numpy as np
import os
import json
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

time1 = time.time()

def plot_results(rewards_history, loss_history):
    # 设置全局样式和字体
    plt.style.use('seaborn')  # 使用简洁风格
    plt.rcParams.update({
        'font.size': 12,  # 全局字体大小
        'axes.titlesize': 14,  # 图表标题字体大小
        'axes.labelsize': 12,  # 轴标签字体大小
        'xtick.labelsize': 10,  # x轴刻度字体大小
        'ytick.labelsize': 10,  # y轴刻度字体大小
        'lines.linewidth': 2,  # 线条宽度
    })

    # 绘制训练过程
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), gridspec_kw={'wspace': 0.3})

    # 子图1：奖励曲线
    ax[0].plot(rewards_history, color='royalblue', alpha=0.8, label='Total Rewards')
    ax[0].set_title('Training Rewards over Episodes', fontsize=14)
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Total Reward')
    ax[0].grid(alpha=0.3)  # 设置网格线
    ax[0].legend()

    # 子图2：损失曲线
    ax[1].plot(loss_history, color='seagreen', alpha=0.8, label='Loss')
    ax[1].set_title('Training Loss over Episodes', fontsize=14)
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Loss')
    ax[1].grid(alpha=0.3)  # 设置网格线
    ax[1].legend()

    # 子图3：测试奖励
    ax[2].plot(test_reward, color='seagreen', alpha=0.8, label='Loss')
    ax[2].set_title('Calculation Rewards', fontsize=14)
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('Reward')
    ax[2].grid(alpha=0.3)  # 设置网格线
    ax[2].legend()

    # 自动调整布局
    plt.tight_layout()
    plt.show()

    # 打印最终解决方案
    print("\n### 最终解决方案 ###")
    print("使用的砖块尺寸序列:")
    print(f"{solution}")
    print(f"总长度: {sum(solution)} 米")


def append_to_json(new_list, filename):
    """
    向JSON文件追加新的列表数据
    如果文件不存在则创建，如果存在则追加

    参数:
    new_list: 要追加的新列表
    filename: JSON文件名
    """
    try:
        # 检查文件是否存在
        if os.path.exists(filename):
            # 读取现有数据
            with open(filename, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    # 如果文件为空或格式错误，初始化为空列表
                    existing_data = []
        else:
            # 文件不存在，初始化为空列表
            existing_data = []

        # 将新列表添加到现有数据中
        if isinstance(new_list, list):
            existing_data.append(new_list)
        else:
            existing_data.append([new_list])

        # 保存更新后的数据
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

        print(f"成功将新数据添加到 {filename}")

    except Exception as e:
        print(f"发生错误: {str(e)}")


def evaluate_the_sol(building_data, result_data):
    out_space_num = len(building_data['outer_space_config'])
    out_space_info = building_data["outer_space_per_building"]
    out_space_cfg = building_data["outer_space_config"]
    inner_space_info = building_data["rooms_per_outer_space"]
    inner_space_cfg = building_data["inner_space_config"]
    # out_space_relationship = building_data["outer_space_relationship"]


    modular_plan_x = {}
    temp_count = 0
    for key, value in result_data.items():
        modular_plan_x[temp_count] = value["modular"]
        temp_count += 1

    print("modular_plan_x")
    print(modular_plan_x)
    # print(out_space_info)
    # print(out_space_cfg)
    # print(result_data)

    def evaluate_modulars(modular_dic):
        """
        :param modular_dic: 模块排列方案（dict）
        :return: 每一种模块的总数量（dict）
        """

        modular_list = []
        f1_result = {}
        for value in modular_dic.values():
            modular_list.append(value)

        def unique_list(list_):
            result_list = []
            for term in list_:
                temp = np.unique(term)
                for value in temp:
                    result_list.append(value)
            return np.unique(result_list).tolist()

        modular_type_used = unique_list(modular_list)

        # 统计字典初始化
        for key in modular_type_used:
            f1_result[key] = 0

        for value in modular_dic.values():
            for term in value:
                f1_result[term] += 1

        modular_num = 0
        for value in modular_dic.values():
            modular_num += len(value)

        return f1_result, modular_num

    def evaluate_outspace(result_data, out_space_cfg, modular_dic) -> list:
        """
        :param out_space_info_:
        :param out_space_cfg_:
        :param modular_dic:
        :param modular_type:
        :return:
        """
        out_space_cfg_ = out_space_cfg

        is_covered_list = []
        cover_rate_list = []
        out_index_list = []

        for index, value in enumerate(result_data.values()):
            total_width = 0
            out_dir = value["direction"]
            out_width_nodes = out_space_cfg_[f"{index}"]
            out_node_max = np.max(out_width_nodes, axis=0)
            out_node_min = np.min(out_width_nodes, axis=0)
            if out_dir == "x":
                out_width = out_node_max[0] - out_node_min[0]
            elif out_dir == "y":
                out_width = out_node_max[1] - out_node_min[1]

            for temp in modular_dic[index]:
                total_width += temp

            cover_rate = total_width / out_width

            if abs(cover_rate - 1) <= 1e-5:
                is_covered = True
            else:
                is_covered = False
            is_covered_list.append(is_covered)
            cover_rate_list.append(abs(cover_rate - 1))

        return is_covered_list, cover_rate_list

    def evaluate_innerspace(result_data, inner_space_info_, inner_space_cfg_, modular_dic) -> dict:
        """

        :param out_space_info_:
        :param inner_space_info_:
        :param inner_space_cfg_:
        :param modular_dic:
        :param modular_type:
        :return: 左右调整量绝对差值和  f3 = {1:10%, 2:5%}
        """

        is_covered_list = []
        cover_rate_list = []
        out_index_list = []
        f3_return = {}
        f3_room_info = {}

        for index, value in enumerate(result_data.values()):
            total_width = 0
            # 计算房间边界坐标
            out_dir = value["direction"]
            if out_dir == "x":
                total_inner_width = value["location"][0]
                inner_width_list = []
                inner_width_list.append(total_inner_width)
            elif out_dir == "y":
                total_inner_width = value["location"][1]
                inner_width_list = []
                inner_width_list.append(total_inner_width)

            for index2 in inner_space_info_[f"{index}"]:
                inner_width_nodes = inner_space_cfg_[f"{index2}"]
                inner_node_max = np.max(inner_width_nodes, axis=0)
                inner_node_min = np.min(inner_width_nodes, axis=0)
                if out_dir == "x":
                    inner_width = inner_node_max[0] - inner_node_min[0]
                elif out_dir == "y":
                    inner_width = inner_node_max[1] - inner_node_min[1]
                total_inner_width += inner_width
                inner_width_list.append(total_inner_width)
            # 计算模块与房间的调整值
            modular_boundary = []
            loss_list = []
            room_covered_list = []

            if out_dir == "x":
                temp_width = value["location"][0]
                modular_boundary.append(temp_width)
            elif out_dir == "y":
                temp_width = value["location"][1]
                modular_boundary.append(temp_width)

            for term in value["modular"]:
                temp_width += term
                modular_boundary.append(temp_width)

            for temp_i in range(len(inner_width_list)):
                dis_function_boundary = inner_width_list[temp_i] - np.array(modular_boundary)
                dis_ab_func_bd = np.absolute(dis_function_boundary)
                room_adjustment_value_ab = np.min(dis_ab_func_bd)
                index_temp = np.where(dis_ab_func_bd == room_adjustment_value_ab)[0]
                # index_temp_list.append(index_temp)

                if len(index_temp) == 1:
                    temp_a = dis_ab_func_bd[index_temp[0]]
                    loss_list.append(temp_a)
                elif len(index_temp) > 1:
                    temp_a = 0
                    loss_list.append(temp_a)
            # 将结果转化为输出形式
            for temp_i in range(int(len(loss_list) - 1)):
                loss = loss_list[temp_i + 1] + loss_list[temp_i]
                rorm_width_temp = inner_width_list[temp_i + 1] - inner_width_list[temp_i]
                room_covered_list.append(round(loss / rorm_width_temp, 3))
                f3_return[inner_space_info_[f"{index}"][temp_i]] = round(loss / rorm_width_temp, 3)
                f3_room_info[inner_space_info_[f"{index}"][temp_i]] = [inner_width_list[temp_i],
                                                                       inner_width_list[temp_i + 1]]

                # new_fuction.append(temp_a[0])

        values = list(f3_return.values())
        max_val = max(values)  # 最大值
        min_val = min(values)  # 最小值
        avg_val = sum(values) / len(values)  # 平均值
        threshold = 0.1
        abnormal = {k: v for k, v in f3_return.items() if v > threshold}
        adjust_num = len([v for v in f3_return.values() if v != 0])

        return f3_return, max_val, min_val, avg_val, abnormal, len(abnormal), adjust_num, f3_room_info

    # 计算模块用量
    data1 = evaluate_modulars(modular_plan_x)
    print("data1")
    print(data1)

    # 评估outer space 对齐性
    data2 = evaluate_outspace(result_data, out_space_cfg, modular_plan_x)
    print("data2")
    print(data2)

    # 评估inner space对齐性
    data3 = evaluate_innerspace(result_data, inner_space_info, inner_space_cfg, modular_plan_x)
    # f3_return, max_val, min_val, avg_val, abnormal, len_abnormal, adjust_num = data3
    f3_return, max_val, min_val, avg_val, abnormal, len_abnormal, adjust_num, f3_room_info = data3
    print("data3")
    print("max_val:", max_val, "min_val:", min_val, "avg_val:", avg_val, "abnormal:", abnormal, "len_abnormal:", adjust_num, f3_room_info)

    # room_list, data4 = ut.evaluate_roomtype(building_data, result_data)

    return data1, data3[:4]


filename_ = "building_data7"
Js_file_dir = os.path.join(os.getcwd(), f"Buildingdata/{filename_}.json")

# Js_file_dir = os.path.join(os.getcwd(), "test_3.json")
with open(Js_file_dir, 'r') as f:
    building_data = json.load(f)

print("building_data is ok")

data_ = Building2Case(building_data)
_outer_segs, _inner_segs, _axis_segs, _outer_counters, _inner_counters, _inner_indices, width_, groupd_idx = \
data_.get_algorithm_data()
building_info = data_.merge_buildings(_outer_segs, _inner_segs, _inner_counters, _inner_indices, width_)

print(building_info)
# building_info = {'outer_range': [0, 49200],
#  'outer': [[[0, 49200], [0, 49200], [0, 49200], [0, 49200], [0, 49200], [0, 49200], [0, 49200], [0, 49200], [0, 49200], [0, 49200], [0, 49200], [0, 49200], [0, 49200], [0, 49200]],
#           ],
# 'inner': [[0, 9900, 19800, 25800, 32700, 39600, 49200], [0, 9900, 19800, 25800, 32700, 39600, 49200],
#           [0, 9900, 19800, 25800, 32700, 39600, 49200], [0, 9900, 19800, 25800, 32700, 39600, 49200],
#           [0, 13000, 16200, 29200, 39600, 49200], [0, 13000, 16200, 29200, 39600, 49200], [0, 13000, 16200, 29200, 39600, 49200],
#           [0, 3600, 13200, 22800, 32400, 42000, 45600, 49200], [0, 3600, 13200, 22800, 32400, 42000, 45600, 49200],
#           [0, 3600, 13200, 22800, 32400, 42000, 45600, 49200], [0, 3600, 13200, 22800, 32400, 42000, 45600, 49200],
#           [0, 3600, 16400, 29200, 42000, 45600, 49200], [0, 3600, 16400, 29200, 42000, 45600, 49200],
#           [0, 3600, 16400, 29200, 42000, 45600, 49200]],
#                  'inner_element': [0, 3600, 9900, 13000, 13200, 16200, 16400, 19800, 22800, 25800, 29200, 32400, 32700, 39600, 42000, 45600, 49200],
#                  'inner_boundary': Counter({49200: 21, 0: 14, 58200: 14, 39600: 7, 45600: 7, 3600: 7, 42000: 7, 70200: 7, 29200: 6, 25800: 4, 9900: 4, 19800: 4, 32700: 4, 22800: 4, 13200: 4, 32400: 4, 64200: 4, 16200: 3, 13000: 3, 16400: 3, 56200: 3, 63200: 3}),
#                  'room': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [6, 7, 6, 8, 1], [6, 7, 6, 8, 1], [6, 7, 6, 8, 1], [0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0], [0, 9, 9, 9, 0, 0], [0, 9, 9, 9, 0, 0], [0, 9, 9, 9, 0, 0], [4], [4], [4], [4], [10, 11], [10, 11], [10, 11], [1, 1], [1, 1], [1, 1], [1, 1], [12, 10], [12, 10], [12, 10]],
#                  'width': [15000, 12000, 12000],
#                  'building_positions': [0, 49200, 58200],
#                  'building_lengths': [49200, 9000, 12000],
#                  'total_length': 70200,
#                  'building_ranges': [(0, 49200), (49200, 58200), (58200, 70200)],
#                  'building_axis': [0, 49200, 56200, 58200, 63200, 64200, 70200]}

outer_segs_ = building_info['outer_range']
outer_list_ = building_info['outer']
inner_segs_ = building_info['inner_element']
room_segs_ = building_info['inner'][0] if len(building_info['inner']) == 1 else building_info['inner']
room_idces_ = building_info['room']
inner_counter_ = building_info['inner_boundary']
axis_segs_ = building_info['building_axis']
inner_boundary = building_info['inner_boundary']
axis_segs_.append(25200)
axis_segs_.append(54800)
axis_segs_.append(80000)
# axis_segs_.append(69000)
# axis_segs_.append(72000)
# axis_segs_.append(108000)
# axis_segs_.append(111000)
# modules = [3000, 3100, 3200, 3300, 3400, 3500, 3600]
# modules = [3000, 3500, 4000]
# modules = [3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000]
# modules = [3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900]
# modules = [3000, 3100, 3200, 3300, 3600]
modules = [3000, 3200, 3400, 3500, 3600, 3800, 4000]
# modules = [3000, 3200, 3600, 4000]
# modules = [3000, 3400, 3600]
# modules = [3100, 3200,3300,3400]


print("outer_segs_", outer_segs_)
# 测试模型
wall_length = outer_segs_[1]
mo_opt_type = modules
alignment_points = [x for x in inner_segs_ if x not in axis_segs_ and x not in outer_segs_]
alignment_points2 = [x for x in axis_segs_ if x not in outer_segs_]
inner_boundary_ = copy.deepcopy(inner_boundary)
a = inner_boundary.keys()
for x in inner_boundary.keys():
    if x in axis_segs_ or x in outer_segs_:
        inner_boundary_.pop(x)
print(inner_boundary_)

building_info = {
"wall_length": wall_length,
"modules": mo_opt_type,
"alignment_points": alignment_points,
"alignment_points2": alignment_points2,
"inner_boundary": inner_boundary_,
"room_segs": room_segs_
}

print(f", 建筑尺寸: {wall_length}米,测试模型: 砖块尺寸序列: {mo_opt_type}, 砖块对齐点: {alignment_points}, 砖块对齐点2: {alignment_points2}, inner_boundary_{inner_boundary_}")

parameter_ = {
    "episode": 1500,
    "lr_parameter": {"cos_T": 1800, "lr_min": 1e-5},
    # "reward": [8, 0.18, 0.02, 1, 1],
    "reward": [5, 0.18, 0.02, 0, 0],
    # "reward": [1, 0.18, 0.02, 0, 0],
    # "reward": [2, 0, 0, 0, 0],
    # "reward": [8, 0.045, 0.005, 0, 0],
    # "reward": [2, 0.05, 3, 4],
    # "reward": [1, 0.2, 4, 6],
    "adjust_limited": 2

#     "reward": [对齐, 添加模块, 种类数, 数量],
}


# 训练模型
agent, rewards_history, loss_history, solution, test_reward = train_dqn(building_info, parameter_)
# solution = [3500, 4000, 3500, 4000, 3500, 4000, 3500, 4000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3500, 4000, 3500, 4000, 3500, 4000, 3500, 4000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000]
time2 = time.time()
print("算法用时:", time2 - time1)

building_info = data_.merge_buildings(_outer_segs, _inner_segs, _inner_counters, _inner_indices, width_)

outer_space_per_building = building_data["outer_space_per_building"]
outer_space_config = building_data["outer_space_config"]
tp_list = []
tp_width_list = []
tp_story_list = []
tp_dir_list = []
tp_loc_list = []
for value in outer_space_per_building.values():
    tp_list.append(value["index"])
    tp_story_list.append(int(value["story"]))
    tp_dir_list.append(value["direction"])
for i, term in enumerate(tp_list):
    tp_loc_list.append(outer_space_config[term][0])
    if tp_dir_list[i] == 'h':
        tp_width_list.append(outer_space_config[term][2][1]-outer_space_config[term][0][1])
    elif tp_dir_list[i] == 'v':
        tp_width_list.append(outer_space_config[term][2][0]-outer_space_config[term][0][0])
outer_list_ = building_info['outer']
modular_list = data_.replace_with_room_sequence(outer_list_, solution, groupd_idx, tp_list)

print(tp_loc_list)

print("modular_list")
print(modular_list)
project = {}
temp_num = 1
modular_idx = 0
print(modular_list)
print(tp_story_list)

for term in modular_list:

    tp_length = tp_width_list[temp_num - 1]
    tp_location_x, tp_location_y = tp_loc_list[temp_num - 1]

    # for modular in term[i]:
    zone = {}
    zone['story'] = tp_story_list[temp_num - 1]
    if tp_dir_list[temp_num - 1] == 'h':
        zone['direction'] = 'x'
    elif tp_dir_list[temp_num - 1] == 'v':
        zone['direction'] = 'y'
    zone['modular'] = np.array(term).tolist()
    zone['width'] = term
    zone['length'] = tp_length
    zone['location'] = (tp_location_x, tp_location_y)

    project[f'{modular_idx}'] = zone
    modular_idx += 1
    temp_num += 1

# for term in modular_list:
#
#     print("term")
#     print(term)
#
#
#     for i in range(len(term)):
#         tp_length = tp_width_list[temp_num - 1]
#         tp_location_x, tp_location_y = tp_loc_list[temp_num - 1]
#
#         # for modular in term[i]:
#         zone = {}
#         zone['story'] = tp_story_list[temp_num - 1]
#         if tp_dir_list[temp_num - 1] == 'h':
#             zone['direction'] = 'x'
#         elif tp_dir_list[temp_num - 1] == 'v':
#             zone['direction'] = 'y'
#         zone['modular'] = np.array(term[i]).tolist()
#         zone['width'] = term[i]
#         zone['length'] = tp_length
#         zone['location'] = (tp_location_x, tp_location_y)
#
#
#         project[f'{modular_idx}'] = zone
#         modular_idx += 1
#         temp_num += 1

print(project)

with open('Resultdata/cal_result.json', 'w') as f:
# with open('cal_result.json', 'w') as f:
    json.dump(project, f, indent=4)

# T2 = time.time()
# print('程序运行时间:%s秒' % (round((T2 - T1), 2)))
# plot_results(rewards_history, loss_history)


data1, data3 = evaluate_the_sol(building_data, project)



filename = f'Resultdata/{filename_}_wlq.json'

append_to_json(solution, filename)
append_to_json(data1, filename)
# append_to_json(data3[:4], filename)
append_to_json(max(rewards_history), filename)

# 读取并显示最终结果
with open(filename, 'r', encoding='utf-8') as f:
    final_data = json.load(f)