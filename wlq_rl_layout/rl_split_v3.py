import copy
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple, Counter
import matplotlib.pyplot as plt
from segment_tree import MinSegmentTree, SumSegmentTree

from DQN_HIGA1.main_DQN_HIGA import run_FEM_analysis

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 经验回放缓冲区
class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.modified_reward = reward  # 新增一个可修改的奖励属性

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=30000):
        # 保持原有初始化参数
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.episode_counter = 0

        # 原有buffer相关
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        # 精英经验相关
        self.elite_buffer = []
        self.elite_size = 700
        self.min_reward = float('-inf')
        self.episode_max_reward = float('-inf')

        # 最优轨迹追踪
        self.best_trajectory = []
        self.best_trajectory_reward = float('-inf')
        self.current_episode = []
        self.reward_propagation_factor = 0.95

        # 轨迹权重追踪
        self.trajectory_weights = {}

        # 原有树结构
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0

    def push(self, experience, episode_total_reward=None, is_episode_end=False):
        """增强版存储函数，支持轨迹追踪和奖励传播"""
        # 存储当前episode的经验
        self.current_episode.append(experience)

        if is_episode_end and episode_total_reward is not None:
            # 处理完整轨迹
            self._process_complete_trajectory(episode_total_reward)
            # 更新精英缓冲区
            self._update_elite_buffer(self.current_episode, episode_total_reward)
            self.current_episode = []  # 重置当前轨迹

        # 原有的经验存储逻辑
        max_priority = max(self.max_priority, 1e-5) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        # 更新优先级，考虑轨迹权重
        base_priority = max_priority ** self.alpha
        trajectory_bonus = self._calculate_trajectory_bonus(experience)
        final_priority = max(base_priority * trajectory_bonus, 1e-5)

        self.sum_tree[self.position] = final_priority
        self.min_tree[self.position] = final_priority

        self.position = (self.position + 1) % self.capacity

        # 状态打印逻辑
        self.episode_counter += 1
        if self.episode_counter % 1000 == 0:
            self.print_elite_samples()

    def _update_elite_buffer(self, experiences, episode_reward):
        """更新精英经验缓冲区"""
        if episode_reward > self.min_reward or len(self.elite_buffer) < self.elite_size:
            # 更新最大/最小奖励记录
            self.episode_max_reward = max(self.episode_max_reward, episode_reward)

            # 如果精英缓冲区未满，直接添加
            if len(self.elite_buffer) < self.elite_size:
                self.elite_buffer.extend(experiences)
                self.min_reward = min(self.min_reward, episode_reward)
            else:
                # 如果当前episode比最差的精英经验更好，替换最差的
                worst_reward = float('inf')
                worst_idx_start = 0
                worst_idx_end = 0

                # 找到最差的轨迹
                current_idx = 0
                while current_idx < len(self.elite_buffer):
                    # 找到当前轨迹的结束位置
                    end_idx = current_idx
                    current_reward = 0
                    while end_idx < len(self.elite_buffer) and not self.elite_buffer[end_idx].done:
                        current_reward += self.elite_buffer[end_idx].reward
                        end_idx += 1
                    if end_idx < len(self.elite_buffer):
                        current_reward += self.elite_buffer[end_idx].reward
                        end_idx += 1

                    if current_reward < worst_reward:
                        worst_reward = current_reward
                        worst_idx_start = current_idx
                        worst_idx_end = end_idx

                    current_idx = end_idx

                # 替换最差的轨迹
                if episode_reward > worst_reward:
                    self.elite_buffer[worst_idx_start:worst_idx_end] = experiences
                    # 更新最小奖励
                    self._update_min_reward()

    def _update_min_reward(self):
        """更新精英缓冲区中的最小奖励"""

        if not self.elite_buffer:
            self.min_reward = float('-inf')
            return

        current_idx = 0
        min_trajectory_reward = float('inf')

        while current_idx < len(self.elite_buffer):
            # 计算当前轨迹的总奖励
            trajectory_reward = 0
            end_idx = current_idx

            while end_idx < len(self.elite_buffer) and not self.elite_buffer[end_idx].done:
                trajectory_reward += self.elite_buffer[end_idx].reward
                end_idx += 1

            if end_idx < len(self.elite_buffer):
                trajectory_reward += self.elite_buffer[end_idx].reward
                end_idx += 1

            min_trajectory_reward = min(min_trajectory_reward, trajectory_reward)
            current_idx = end_idx

        self.min_reward = min_trajectory_reward

    def print_elite_samples(self):
        """打印精英样本的统计信息"""
        if self.elite_buffer:
            elite_rewards = [exp.reward for exp in self.elite_buffer]
            avg_reward = sum(elite_rewards) / len(elite_rewards)
            max_reward = max(elite_rewards)
            min_reward = min(elite_rewards)

            print(f"\nElite Buffer Statistics:")
            print(f"Size: {len(self.elite_buffer)}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Max Reward: {max_reward:.2f}")
            print(f"Min Reward: {min_reward:.2f}")
            print(f"Best Trajectory Reward: {self.best_trajectory_reward:.2f}")
            print(f"Current Episode Counter: {self.episode_counter}")
        else:
            print("\nElite Buffer is empty")

    def _process_complete_trajectory(self, episode_total_reward):
        """处理完整轨迹"""
        if episode_total_reward is None:
            episode_total_reward = sum(exp.reward for exp in self.current_episode)

        # 更新最优轨迹
        if episode_total_reward > self.best_trajectory_reward:
            self.best_trajectory = self.current_episode.copy()
            self.best_trajectory_reward = episode_total_reward
            self._propagate_rewards(self.current_episode, episode_total_reward)

        # 更新轨迹权重
        trajectory_quality = self._calculate_trajectory_quality(episode_total_reward)
        for exp in self.current_episode:
            exp_id = id(exp)
            self.trajectory_weights[exp_id] = trajectory_quality

    def _propagate_rewards(self, trajectory, total_reward):
        """向后传播奖励"""
        n_steps = len(trajectory)
        for i, exp in enumerate(trajectory):
            # 计算传播奖励
            propagation_strength = self.reward_propagation_factor ** (n_steps - i - 1)
            reward_bonus = total_reward * propagation_strength
            # 更新修改后的奖励
            exp.modified_reward = exp.reward + reward_bonus * 0.1

    def _calculate_trajectory_quality(self, episode_reward):
        """计算轨迹质量分数"""
        if self.episode_max_reward == float('-inf'):
            return 1.0
        normalized_reward = (episode_reward - self.min_reward) / (self.episode_max_reward - self.min_reward + 1e-5)
        return np.clip(normalized_reward, 0.1, 1.0)

    def _calculate_trajectory_bonus(self, experience):
        """计算轨迹额外奖励"""
        exp_id = id(experience)
        if exp_id in self.trajectory_weights:
            return 1.0 + self.trajectory_weights[exp_id]
        return 1.0

    def sample(self, batch_size):
        """修改后的采样方法，增加了额外的边界检查"""
        if len(self.buffer) < batch_size:
            return None

        # 调整采样比例
        best_trajectory_sample_size = min(batch_size // 8, len(self.best_trajectory))
        elite_sample_size = min(batch_size // 4, len(self.elite_buffer))
        normal_sample_size = batch_size - best_trajectory_sample_size - elite_sample_size

        experiences = []
        weights = []
        indices = []

        # 采样最优轨迹
        if best_trajectory_sample_size > 0 and self.best_trajectory:
            best_samples = random.sample(self.best_trajectory, best_trajectory_sample_size)
            experiences.extend(best_samples)
            weights.extend([2.0] * best_trajectory_sample_size)
            indices.extend([-2] * best_trajectory_sample_size)

        # 普通样本采样
        if normal_sample_size > 0:
            try:
                p_total = self.sum_tree.sum(0, len(self.buffer) - 1)

                # 确保p_total是有效值
                if p_total <= 0:
                    p_total = 1e-8

                segment = p_total / normal_sample_size
                beta = self.beta_by_frame(self.frame)

                for i in range(normal_sample_size):
                    a = segment * i
                    b = segment * (i + 1)

                    # 确保upperbound在有效范围内
                    upperbound = random.uniform(a, b)
                    upperbound = min(upperbound, p_total)
                    upperbound = max(upperbound, 0)

                    try:
                        idx = self.sum_tree.find_prefixsum_idx(upperbound)

                        # 验证idx的有效性
                        if 0 <= idx < len(self.buffer):
                            p_sample = self.sum_tree[idx] / p_total
                            weight = (p_sample * len(self.buffer)) ** (-beta)

                            experiences.append(self.buffer[idx])
                            weights.append(weight)
                            indices.append(idx)
                        else:
                            # 如果idx无效，使用随机采样作为后备方案
                            random_idx = random.randint(0, len(self.buffer) - 1)
                            experiences.append(self.buffer[random_idx])
                            weights.append(1.0)
                            indices.append(random_idx)

                    except Exception as e:
                        # 采样失败时的后备方案
                        random_idx = random.randint(0, len(self.buffer) - 1)
                        experiences.append(self.buffer[random_idx])
                        weights.append(1.0)
                        indices.append(random_idx)

            except Exception as e:
                # 如果优先级采样完全失败，退回到统一采样
                selected_indices = np.random.choice(len(self.buffer), normal_sample_size)
                for idx in selected_indices:
                    experiences.append(self.buffer[idx])
                    weights.append(1.0)
                    indices.append(idx)

        # 采样精英经验
        if elite_sample_size > 0 and self.elite_buffer:
            elite_samples = random.sample(self.elite_buffer, elite_sample_size)
            experiences.extend(elite_samples)
            weights.extend([1.5] * elite_sample_size)
            indices.extend([-1] * elite_sample_size)

        # 确保我们有足够的样本
        if len(experiences) < batch_size:
            additional_needed = batch_size - len(experiences)
            random_indices = np.random.choice(len(self.buffer), additional_needed)
            for idx in random_indices:
                experiences.append(self.buffer[idx])
                weights.append(1.0)
                indices.append(idx)

        # 归一化权重
        weights = np.array(weights)
        max_weight = np.max(weights)
        if max_weight > 0:
            weights = weights / max_weight
        else:
            weights = np.ones_like(weights)

        # 构建返回数据
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([getattr(exp, 'modified_reward', exp.reward) for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])

        return (states, actions, rewards, next_states, dones, indices, weights)

    def beta_by_frame(self, frame_idx):
        """计算当前beta值"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            priority = max(priority, 1e-6)  # 避免优先级为0
            self.max_priority = max(self.max_priority, priority)

            # 更新线段树
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

    def __len__(self):
        """返回当前缓冲区中的经验数量"""
        return len(self.buffer)

class BrickEnv:
    def __init__(self, wall_length, mo_opt_type, alignment_points, axis_segs, inner_boundary, room_segs, reward_para, adjust_limited):
        self.wall_length = wall_length
        self.mo_opt_type = mo_opt_type
        self.alignment_points = alignment_points
        self.axis_segs = axis_segs
        self.inner_count = inner_boundary
        self.module_layout = list()
        self.room_segs = room_segs
        self.reward_para = reward_para
        self.adjust_limited = adjust_limited
        self.reset()

    def reset(self):
        self.current_length = 0.0
        # self.brick_positions = list()
        self.brick_positions = [0.]
        self.module_layout = list()
        return self._get_state()

    def adjust_room_boundaries(self, boundaries, module_positions):
            new_boundaries = []
            boundary_crossed = False

            if len(module_positions) == 0:
                return [None for _ in range(len(boundaries))]

            for boundary in boundaries:
                if boundary_crossed:
                    new_boundaries.append(None)
                    continue

                left = max((pos for pos in module_positions if pos <= boundary), default=None)
                right = min((pos for pos in module_positions if pos > boundary), default=None)

                if left is not None and right is not None:
                    if abs(boundary - left) <= abs(boundary - right):
                        new_boundaries.append(left)
                    else:
                        new_boundaries.append(right)
                elif boundaries[-1] - module_positions[-1] < 1e-6:
                    new_boundaries.append(module_positions[-1])
                    boundary_crossed = True
                else:
                    new_boundaries.append(None)
                    boundary_crossed = True

            return new_boundaries

    def _get_state(self):

        module_num = len(self.module_layout)
        module_type_num = len(set(self.module_layout))
        # 统计 mo_use 中每种 opt_type 的数量
        count = Counter(self.module_layout)

        # 只输出 opt_type 中的统计结果
        opt_type_usage = [count[key] / module_num for key in self.mo_opt_type] if module_num > 0 else [0] * len(self.mo_opt_type)

        max_module_num = int(self.current_length/min(self.mo_opt_type))+1

        module_info = [
            module_num/max_module_num,  # 模块数量
            module_type_num/len(self.mo_opt_type)  # 模块种类数
        ] + opt_type_usage

        layout_info = [
            self.current_length / self.wall_length,  # 当前长度
            (self.wall_length - self.current_length) / self.wall_length,  # 剩余长度
            len(self.brick_positions) / (int(self.wall_length / min(self.mo_opt_type))+1)  # 已使用的砖块数
        ]

        room_counts = Counter(tuple(sublist) for sublist in self.room_segs)
        tp_room_list = list()
        tp_room_count = list()
        tp_room_num = 0

        for sublist, count in room_counts.items():
            tp_room_list.append(sublist)
            tp_room_count.append(count)
            tp_room_num += count

        room_info = []

        for i, tp_room in enumerate(tp_room_list):
            tp_room_new = self.adjust_room_boundaries(tp_room, self.brick_positions)
            for j in range(len(tp_room)-1):



                room_info.append(
                    [
                        tp_room[j] / self.wall_length,
                        tp_room[j+1] / self.wall_length,
                        tp_room_new[j] / self.wall_length if tp_room_new[j] != None else 0,
                        tp_room_new[j+1] / self.wall_length if tp_room_new[j+1] != None else 0,
                        # tp_room_count[i] / tp_room_num
                    ])
                tp_a = tp_room[j + 1] - tp_room[j]

                room_info[-1].append(0 if tp_room_new[j + 1] is None else
                -1 if (abs(tp_room_new[j + 1] - tp_room[j + 1]) + abs(
                    tp_room_new[j] - tp_room[j])) / tp_a > self.adjust_limited
                else 1)


        # 增加状态信息
        state = np.array(
            module_info +
            layout_info +
            [item for room in room_info for item in room],
            dtype=np.float32
            )

        # state = np.array(
        #     module_info +
        #     layout_info)


        # # 添加对齐点状态
        # for point in self.alignment_points:
        #     state.append(1.0 if any(abs(pos - point) < 1e-6 for pos in self.brick_positions) else 0.0)
        # 添加对齐点状态
        for point in self.alignment_points:
            if point != 0:
                state = np.append(state, (abs(self.current_length - point)/self.wall_length))

        for point in self.axis_segs:
            if point != 0:
                if len(self.brick_positions) >= 2:
                    if self.brick_positions[-2] < point <self.brick_positions[-1]:
                        state = np.append(state, -1.)
                    else:
                        state = np.append(state, (1.0 if abs(self.current_length - point) < 1e-6 else 0.0))
                else:
                    state = np.append(state, (1.0 if abs(self.current_length - point) < 1e-6 else 0.0))

        return np.array(state)

    def step(self, action):
        brick_size = self.mo_opt_type[action]
        self.module_layout.append(brick_size)
        new_length = self.current_length + brick_size

        # 优化奖励函数
        reward = 0
        done = False
        success = False

        self.current_length = new_length
        self.brick_positions.append(new_length)

        # 检查是否超出墙长
        if new_length > self.wall_length + 1e-6:
            done = True
            return self._get_state(), -2, done, success
        # 检查房间边界调整
        room_counts = Counter(tuple(sublist) for sublist in self.room_segs)
        tp_room_list = list()
        tp_room_count = list()
        tp_room_num = 0

        for sublist, count in room_counts.items():
            tp_room_list.append(sublist)
            tp_room_count.append(count)
            tp_room_num += count
        room_info = []
        for i, tp_room in enumerate(tp_room_list):
            tp_room_new = self.adjust_room_boundaries(tp_room, self.brick_positions)
            for j in range(len(tp_room)-1):
                if tp_room_new[j+1] != None:
                    a = (abs(tp_room_new[j] - tp_room[j]) + abs(tp_room_new[j+1] - tp_room[j+1]))
                    b = (tp_room[j+1] - tp_room[j])
                    c = a/b
                    # if (abs(tp_room_new[j] - tp_room[j]) + abs(tp_room_new[j+1] - tp_room[j+1]))/(tp_room[j+1] - tp_room[j]) > 0.12:
                    if c > self.adjust_limited:
                        reward -= 1 + len(self.module_layout)/20
                        done = True
                        return self._get_state(), -2, done, success
                else:
                    break

        # 对齐点奖励
        for point in self.axis_segs:
            if abs(new_length - point) < 1e-6:
                reward += 2
                break
            if len(self.brick_positions) >= 2:
                if self.brick_positions[-2] < point < self.brick_positions[-1]:
                    done = True
                    return self._get_state(), -2, done, success

        total_num = 0
        for value in self.inner_count.values():
            total_num += value

        for point in self.alignment_points:
            point_num = self.inner_count[point]
            distance = abs(new_length - point)
            reward += (1.0 / (1.0 + distance/100))*point_num/total_num * self.reward_para[0]  # 距离越近，奖励越高

        # 使用砖块数量惩罚
        reward -= self.reward_para[1]  # 每使用一块砖扣除小量分数，鼓励使用更少的砖
        count = Counter(self.module_layout)
        module_type = len(count)
        reward -= module_type * self.reward_para[2]
        # 检查是否超出墙长
        #

        def calculate_normalized_entropy(opt_type_usage: list, c2: float = 1.0) -> float:
            """
            根据给定的公式计算归一化熵。

            该函数实现了公式: Result = c2 * (Σ [p_i * log2(p_i)]) / log2(k)
            其中 p_i = n_i / N 是列表中给出的比例。

            Args:
                opt_type_usage (list): 一个列表，其中包含了每一类模块的使用比例 (p_i)。
                                       例如 [0.5, 0.5, 0.0]。
                c2 (float, optional): 公式中的常数系数 c₂。默认为 1.0。

            Returns:
                float: 计算出的归一化熵的值。
            """

            import math
            # k 是模块类型的总数，即列表的长度
            k = len(opt_type_usage)

            # --- 处理边界情况 ---
            # 如果只有一种类型 (k=1)，或者列表为空 (k=0)，熵为0。
            # log2(1) = 0 会导致分母为零，但此时分子也为0 (1*log2(1)=0)，因此结果应为0。
            if k <= 1:
                return 0.0

            # --- 计算分子中的求和项 Σ [p_i * log2(p_i)] ---
            # 这是信息熵的核心部分 (缺少一个负号)
            entropy_sum = 0.0
            for p_i in opt_type_usage:
                # 关键：当 p_i 为 0 时，p_i * log2(p_i) 的值为 0，对求和没有贡献
                # 我们只处理大于零的项，以避免 math.log2(0) 的错误
                if p_i > 0:
                    entropy_sum += p_i * math.log2(p_i)

            # --- 计算分母 log2(k) ---
            denominator = math.log2(k)

            # 避免潜在的除零错误（尽管上面的 if k<=1 已经处理了）
            if denominator == 0:
                return 0.0

            # --- 组合成最终公式 ---
            normalized_entropy = c2 * (entropy_sum / denominator)

            return normalized_entropy


        # 完成奖励
        if abs(new_length - self.wall_length) < 1e-6:
            reward = 2
            done = True
            # for i in range(9):
            #     self.module_layout.append(3000)
            # 统计 mo_use 中每种 opt_type 的数量
            usage_eva = 0
            count = Counter(self.module_layout)
            module_num = len(self.module_layout)
            module_type = len(count)
            opt_type_usage = [count[key] / module_num for key in self.mo_opt_type] if module_num > 0 else [0] * len(self.mo_opt_type)
            total_weight, max_force, dis_data = run_FEM_analysis(self.module_layout)

            for term in opt_type_usage:
                if term != 0:
                    usage_eva += 1/term

            success = True

            if max_force < 1.:
                reward += 1
            else:
                success = False
                reward -= 5

            if dis_data < 1.:
                reward += 1
            else:
                success = False
                reward -= 5
            if total_weight<=8200 :
                weight_reward = ((8200-total_weight)/(8200-2500))*1.5
            elif total_weight>8200:
                weight_reward =-1.5
            reward +=weight_reward
        # if done:
        #     reward -= len(self.module_layout) * 0.1

        return self._get_state(), reward, done, success

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # 使用LayerNorm替代BatchNorm
        self.fc1 = nn.Linear(state_size, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        self.fc4 = nn.Linear(64, action_size)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)

# 更新DQNAgent类
class DQNAgent:
    def __init__(self, state_size, action_size, lr_parameter):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.memory = PrioritizedReplayBuffer(50000)

        self.batch_size = 128
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=self.epsilon_decay)

        # # 余弦退火学习率调度器
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=lr_parameter["cos_T"],  # 一个完整的余弦周期的步数
        #     eta_min=lr_parameter["lr_min"]  # 最小学习率
        # )


    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # 获取带有优先级的样本
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return

        states, actions, rewards, next_states, dones, indices, weights = batch

        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # 计算当前Q值和目标Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # 计算TD误差
        td_errors = torch.abs(current_q_values.squeeze() - expected_q_values).detach().cpu().numpy()

        # 使用Huber损失并应用重要性采样权重
        loss = (weights * nn.SmoothL1Loss(reduction='none')(current_q_values.squeeze(), expected_q_values)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # 更新优先级
        self.memory.update_priorities(indices, td_errors)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def select_action(self, state, is_test=False):
        # 测试时仅使用贪婪策略
        if not is_test and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def test_agent(self, env, num_test_episodes=5):
        test_rewards = []

        for _ in range(num_test_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                # 使用贪婪策略
                action = self.select_action(state, is_test=True)
                next_state, reward, done, success = env.step(action)
                total_reward += reward
                state = next_state
            test_rewards.append(total_reward)
            # if success:
                # print("success", total_reward, reward)
        return np.mean(test_rewards), success


    def update_target_network(self):
        # 软更新
        tau = 0.001
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


def train_dqn(building_info, parameter):

    wall_length = building_info["wall_length"]
    mo_opt_type = building_info["modules"]
    alignment_points = building_info["alignment_points"]
    axis_segs = building_info["alignment_points2"]
    inner_boundary = building_info["inner_boundary"]
    room_segs = building_info["room_segs"]

    episodes = parameter["episode"]
    lr_parameter = parameter["lr_parameter"]
    reward_para = parameter["reward"]
    adjust_limited = parameter["adjust_limited"]

    # env = BrickEnv(wall_length, mo_opt_type, alignment_points, axis_segs)
    env = BrickEnv(wall_length, mo_opt_type, alignment_points, axis_segs, inner_boundary, room_segs, reward_para, adjust_limited)

    def calculate_room_info_elements(room_segs):
        """
        计算 room_info 中的总元素个数
        :param room_segs: 原始房间分段点列表
        :return: room_info 的总元素个数
        """
        # 统计唯一子列表及其出现次数
        room_counts = Counter(tuple(sublist) for sublist in room_segs)

        # 计算房间段总数
        room_segment_count = 0
        for tp_room, count in room_counts.items():
            room_segment_count += (len(tp_room) - 1)  # 每个子列表的段数

        # 每个房间段有 5 个元素，计算总元素个数
        total_elements = room_segment_count * 5

        return total_elements

    room_info_num = calculate_room_info_elements(room_segs)

    state_size = 3 + len(alignment_points) + len(axis_segs) + 2 + len(mo_opt_type) + room_info_num  # 更新状态空间大小
    # state_size = 3 + len(alignment_points) + len(axis_segs) + 2 + len(mo_opt_type)  # 更新状态空间大小


    # state_size = 3 + len(alignment_points) - 1
    action_size = len(mo_opt_type)
    agent = DQNAgent(state_size, action_size, lr_parameter)

    episodes = episodes  # 增加训练轮数
    test_rewards_history = []  # 记录测试奖励
    rewards_history = []
    loss_history = []
    episode_list = []
    r_max = -1000
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []
        action_output = []
        episode_experiences = []
        episode_states = []
        episode_actions = []

        # 记录轨迹信息
        for step in range(50):
            action = agent.select_action(state)
            next_state, reward, done, success = env.step(action)

            experience = Experience(state, action, reward, next_state, done)
            episode_experiences.append(experience)
            episode_states.append(state)
            episode_actions.append(action)

            state = next_state
            total_reward += reward
            action_output.append(env.mo_opt_type[action])

            if done:
                break
            if success:
                print("success", total_reward, reward)

        # 在episode结束后处理经验
        for i, exp in enumerate(episode_experiences):
            is_last_exp = (i == len(episode_experiences) - 1)
            agent.memory.push(exp, total_reward, is_episode_end=is_last_exp)

            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)

            # 更新目标网络
            agent.update_target_network()

        # 记录历史数据
        rewards_history.append(total_reward)
        if episode_losses:
            loss_history.append(np.mean(episode_losses))

        # 周期性评估和输出
        if episode % 10 == 0:
            test_reward, success = agent.test_agent(env)
            test_rewards_history.append(test_reward)
            episode_list.append(episode)
            print(
                f"Episode {episode}, total,reward: {total_reward},r_max: {r_max}, Average Reward: {np.mean(rewards_history[-100:]):.2f}, {success}")

        # 更新最佳轨迹
        if total_reward > r_max:
            r_max = total_reward
            output = action_output

            # 保存最佳轨迹的详细信息
            best_trajectory_info = {
                'states': episode_states,
                'actions': episode_actions,
                'total_reward': total_reward,
                'action_sequence': action_output
            }

            # 如果agent有相应方法，更新其最佳轨迹信息
            if hasattr(agent, 'update_best_trajectory'):
                agent.update_best_trajectory(best_trajectory_info)

        if episode % 100 == 0:
            print(total_reward)
            print(action_output)
            print(output)

    current_dir = os.getcwd()
    # 保存文件
    # save_path_r = os.path.join(current_dir, f'r_{r_max}.npy')
    # save_path_e = os.path.join(current_dir, f'e_{r_max}.npy')
    # 使用辅助函数获取唯一的、最终要使用的文件路径
    # final_save_path_r = get_unique_filename(save_path_r)
    # final_save_path_e = get_unique_filename(save_path_e)

    # np.save(final_save_path_r, test_rewards_history)
    # np.save(final_save_path_e, episode_list)

    return agent, rewards_history, loss_history, output, test_rewards_history


def get_unique_filename(path):
    """
    检查文件路径是否存在。
    如果不存在，则直接返回原路径。
    如果存在，则在文件名后添加一个数字后缀（如 _1, _2），直到找到一个不冲突的文件名。

    参数:
    path (str): 原始文件路径，例如 'data/result.npy'

    返回:
    str: 一个唯一的、可用的文件路径
    """
    # 如果原始路径的文件不存在，直接返回原始路径
    if not os.path.exists(path):
        return path

    # 如果文件已存在，开始寻找新的文件名
    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)

    counter = 1
    while True:
        # 构建新的文件名，例如 'r_100_1.npy'
        new_filename = f"{name}_{counter}{ext}"
        # 构建新的完整文件路径
        new_path = os.path.join(directory, new_filename)

        # 如果新路径的文件不存在，则返回这个新路径
        if not os.path.exists(new_path):
            return new_path

        # 如果仍然存在，则增加计数器，继续下一次循环
        counter += 1
# building_info = {'outer_range': [0, 67000],
#                  'outer': [
#                      [[0, 40000], [0, 40000], [0, 40000], [0, 40000], [0, 40000], [0, 40000], [0, 40000], [0, 40000]],
#                      [[40000, 67000], [40000, 61000], [40000, 61000], [40000, 61000]]],
#                  'inner': [[0, 20000, 40000], [0, 12000, 28000, 40000], [0, 12000, 28000, 40000],
#                            [0, 12000, 28000, 40000],
#                            [0, 12000, 28000, 40000], [0, 8000, 16000, 24000, 40000], [0, 8000, 16000, 24000, 40000],
#                            [0, 8000, 16000, 24000, 40000], [40000, 49000, 52000, 61000, 67000],
#                            [40000, 49000, 52000, 61000],
#                            [40000, 49000, 52000, 61000], [40000, 49000, 52000, 61000]],
#                  'inner_element': [0, 8000, 12000, 16000, 20000, 24000, 28000, 40000, 49000, 52000, 61000, 67000],
#                  'inner_boundary': Counter(
#                      {40000: 12, 0: 8, 28000: 4, 12000: 4, 52000: 4, 61000: 4, 49000: 4, 8000: 3, 16000: 3, 24000: 3,
#                       20000: 1,
#                       67000: 1}),
#                  'room': [[0, 0], [1, 2, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1], [3, 3, 3, 2], [3, 3, 3, 2], [3, 3, 3, 2],
#                           [4, 5, 4, 7], [4, 5, 4], [4, 5, 4], [4, 5, 4]], 'width': [12000, 12000],
#                  'building_positions': [0, 40000], 'building_lengths': [40000, 27000], 'total_length': 67000,
#                  'building_ranges': [(0, 40000), (40000, 67000)], 'building_axis': [40000, 67000]}
#
# building_info = {'outer_range': [0, 26000],
#                  'outer': [
#                      [[0, 40000], [0, 40000], [0, 40000], [0, 40000], [0, 40000], [0, 40000], [0, 40000], [0, 40000]],
#                      [[40000, 67000], [40000, 61000], [40000, 61000], [40000, 61000]]],
#                  'inner': [[0, 20000, 40000], [0, 12000, 28000, 40000], [0, 12000, 28000, 40000],
#                            [0, 12000, 28000, 40000],
#                            [0, 12000, 28000, 40000], [0, 8000, 16000, 24000, 40000], [0, 8000, 16000, 24000, 40000],
#                            [0, 8000, 16000, 24000, 40000], [40000, 49000, 52000, 61000, 67000],
#                            [40000, 49000, 52000, 61000],
#                            [40000, 49000, 52000, 61000], [40000, 49000, 52000, 61000]],
#                  'inner_element': [0, 9000, 12000, 21000, 27000],
#                  'inner_boundary': Counter(
#                      {40000: 12, 0: 8, 28000: 4, 12000: 4, 52000: 4, 61000: 4, 49000: 4, 8000: 3, 16000: 3, 24000: 3,
#                       20000: 1,
#                       67000: 1}),
#                  'room': [[0, 0], [1, 2, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1], [3, 3, 3, 2], [3, 3, 3, 2], [3, 3, 3, 2],
#                           [4, 5, 4, 7], [4, 5, 4], [4, 5, 4], [4, 5, 4]], 'width': [12000, 12000],
#                  'building_positions': [0, 40000], 'building_lengths': [40000, 27000], 'total_length': 67000,
#                  'building_ranges': [(0, 40000), (40000, 67000)], 'building_axis': [10500]}
#
# outer_segs_ = building_info['outer_range']
# outer_list_ = building_info['outer']
# inner_segs_ = building_info['inner_element']
# room_segs_ = building_info['inner']
# room_idces_ = building_info['room']
# inner_counter_ = building_info['inner_boundary']
# axis_segs_ = building_info['building_axis']
# inner_boundary = building_info['inner_boundary']
# # modules = [3000, 3200, 3600, 4000]
# # modules = [3000, 3500, 4000]
# modules = [3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000]
#
# # 测试模型
# wall_length = 2.0
# mo_opt_type = [0.15, 0.20, 0.25]
# alignment_points = [0.8, 1.1]
#
# # 测试模型
# wall_length = outer_segs_[1]
# mo_opt_type = modules
# alignment_points = [x for x in inner_segs_ if x not in axis_segs_]
# alignment_points2 = [x for x in axis_segs_ if x not in outer_segs_]
# inner_boundary_ = copy.deepcopy(inner_boundary)
# a = inner_boundary.keys()
# for x in inner_boundary.keys():
#     if x in axis_segs_ or x in outer_segs_:
#         inner_boundary_.pop(x)
# print(inner_boundary_)
#
# building_info = {
# "wall_length": wall_length,
# "modules": mo_opt_type,
# "alignment_points": alignment_points,
# "alignment_points2": alignment_points2,
# "inner_boundary": inner_boundary_
# }
#
# print(f", 建筑尺寸: {wall_length}米,测试模型: 砖块尺寸序列: {mo_opt_type}, 砖块对齐点: {alignment_points}, 砖块对齐点2: {alignment_points2}")
#
# parameter_ = {
#     "episode": 3800,
#     "lr_parameter": {"cos_T": 3000, "lr_min": 1e-5}
#
# }
#
# # 训练模型
# agent, rewards_history, loss_history, solution, test_reward = train_dqn(building_info, parameter_)
#
# # 设置全局样式和字体
# plt.style.use('seaborn')  # 使用简洁风格
# plt.rcParams.update({
#     'font.size': 12,        # 全局字体大小
#     'axes.titlesize': 14,   # 图表标题字体大小
#     'axes.labelsize': 12,   # 轴标签字体大小
#     'xtick.labelsize': 10,  # x轴刻度字体大小
#     'ytick.labelsize': 10,  # y轴刻度字体大小
#     'lines.linewidth': 2,   # 线条宽度
# })
#
#
#
#
#
# # 绘制训练过程
# fig, ax = plt.subplots(1, 3, figsize=(14, 6), gridspec_kw={'wspace': 0.3})
#
# # 子图1：奖励曲线
# ax[0].plot(rewards_history, color='royalblue', alpha=0.8, label='Total Rewards')
# ax[0].set_title('Training Rewards over Episodes', fontsize=14)
# ax[0].set_xlabel('Episode')
# ax[0].set_ylabel('Total Reward')
# ax[0].grid(alpha=0.3)  # 设置网格线
# ax[0].legend()
#
# # 子图2：损失曲线
# ax[1].plot(loss_history, color='seagreen', alpha=0.8, label='Loss')
# ax[1].set_title('Training Loss over Episodes', fontsize=14)
# ax[1].set_xlabel('Episode')
# ax[1].set_ylabel('Loss')
# ax[1].grid(alpha=0.3)  # 设置网格线
# ax[1].legend()
#
# # 子图3：测试奖励
# ax[2].plot(test_reward, color='seagreen', alpha=0.8, label='Loss')
# ax[2].set_title('Calculation Rewards', fontsize=14)
# ax[2].set_xlabel('Episode')
# ax[2].set_ylabel('Reward')
# ax[2].grid(alpha=0.3)  # 设置网格线
# ax[2].legend()
#
#
# # 自动调整布局
# plt.tight_layout()
# plt.show()
#
# # 打印最终解决方案
# print("\n### 最终解决方案 ###")
# print("使用的砖块尺寸序列:")
# print(f"{solution}")
# print(f"总长度: {sum(solution)} 米")

def calculate_proportional_tanh_reward(module_counts, proportion_threshold, steepness=1, reward_scale=1.0):
    """
    根据模块占总数的“比例”来计算奖励。

    当一个模块的比例超过阈值时，它才开始产生正向奖励。

    参数:
    module_counts (dict): 一个字典，键为模块类型，值为该模块的数量。
                          例如: {'3000': 100, '4000': 20}
    proportion_threshold (float): 比例阈值 (P_threshold)。例如，0.15 代表 15%。
    steepness (float): 曲线陡峭度 (k)。因为比例的数值范围很小（0到1），
                       这个值通常需要设置得比较大（如 30-100）才能形成明显的过渡区。
    reward_scale (float): 奖励缩放因子。

    返回:
    float: 计算出的总奖励值。
    """
    module_counts = Counter(module_counts)
    # 1. 计算模块总数
    total_count = sum(module_counts.values())

    # 如果没有模块，奖励为0，避免除以零的错误
    if total_count == 0:
        return 0.0

    total_reward = 0

    # 2. 遍历每一种模块
    for module_type, count in module_counts.items():
        # 3. 计算该模块的占比
        proportion = count / total_count

        # 4. 计算核心“差距”并应用 tanh 函数
        # 核心公式: reward_scale * tanh(k * (proportion - proportion_threshold))
        x = steepness * (proportion - proportion_threshold)
        single_module_reward = reward_scale * np.tanh(x)

        total_reward += single_module_reward

        # (可选) 打印调试信息
        # print(f"模块 '{module_type}': 数量={count}, 占比={proportion:.2%}, 贡献分={single_module_reward:.3f}")

    return total_reward