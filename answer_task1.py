import numpy as np
import copy

import scipy.signal as signal
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import math

# ------------- lab1里的代码 -------------#
import smooth_utils


def load_meta_data(bvh_path):
    with open(bvh_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                end_sites.append(len(joints))
                joints.append(parent_stack[-1] + '_end')
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()

            elif 'OFFSET' in line:
                joint_offsets[-1] = np.array(
                    [float(x) for x in line.split()[-3:]]).reshape(1, 3)

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = ''.join(trans_order) + ''.join(rot_order)

            elif 'Frame Time:' in line:
                break

    joint_parents = [-1] + [joints.index(i) for i in joint_parents[1:]]
    channels = [len(i) for i in channels]
    return joints, joint_parents, channels, joint_offsets


def load_motion_data(bvh_path):
    with open(bvh_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


# ------------- 实现一个简易的BVH对象，进行数据处理 -------------#
'''
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
'''


class BVHMotion():
    def __init__(self, bvh_file_name=None) -> None:

        # 一些 meta data
        self.joint_name = []
        self.joint_channel = []
        self.joint_parent = []

        # 一些local数据, 对应bvh里的channel, XYZposition和 XYZrotation
        # ! 这里我们把没有XYZ position的joint的position设置为offset, 从而进行统一
        self.joint_position = None  # (N,M,3) 的ndarray, 局部平移
        self.joint_rotation = None  # (N,M,4)的ndarray, 用四元数表示的局部旋转

        if bvh_file_name is not None:
            self.load_motion(bvh_file_name)
        pass

    # ------------------- 一些辅助函数 ------------------- #
    def load_motion(self, bvh_file_path):
        '''
            读取bvh文件，初始化元数据和局部数据
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset = \
            load_meta_data(bvh_file_path)

        motion_data = load_motion_data(bvh_file_path)

        # 把motion_data里的数据分配到joint_position和joint_rotation里
        self.joint_position = np.zeros(
            (motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros(
            (motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:, :, 3] = 1.0  # 四元数的w分量默认为1

        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                self.joint_position[:, i, :] = joint_offset[i].reshape(1, 3)
                continue
            elif self.joint_channel[i] == 3:
                self.joint_position[:, i, :] = joint_offset[i].reshape(1, 3)
                rotation = motion_data[:, cur_channel:cur_channel + 3]
            elif self.joint_channel[i] == 6:
                self.joint_position[:, i, :] = motion_data[:, cur_channel:
                                                              cur_channel + 3]
                rotation = motion_data[:, cur_channel + 3:cur_channel + 6]
            self.joint_rotation[:,
            i, :] = R.from_euler('XYZ',
                                 rotation,
                                 degrees=True).as_quat()
            cur_channel += self.joint_channel[i]

        return

    def batch_forward_kinematics(self,
                                 joint_position=None,
                                 joint_rotation=None):
        '''
        利用自身的metadata进行批量前向运动学
        joint_position: (N,M,3)的ndarray, 局部平移
        joint_rotation: (N,M,4)的ndarray, 用四元数表示的局部旋转
        '''
        if joint_position is None:
            joint_position = self.joint_position
        if joint_rotation is None:
            joint_rotation = self.joint_rotation

        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:, :, 3] = 1.0  # 四元数的w分量默认为1

        # 一个小hack是root joint的parent是-1, 对应最后一个关节
        # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向

        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:, pi, :])
            joint_translation[:, i, :] = joint_translation[:, pi, :] + \
                                         parent_orientation.apply(joint_position[:, i, :])
            joint_orientation[:, i, :] = (
                    parent_orientation *
                    R.from_quat(joint_rotation[:, i, :])).as_quat()
        return joint_translation, joint_orientation

    def adjust_joint_name(self, target_joint_name):
        '''
        调整关节顺序为target_joint_name
        '''
        idx = [
            self.joint_name.index(joint_name)
            for joint_name in target_joint_name
        ]
        idx_inv = [
            target_joint_name.index(joint_name)
            for joint_name in self.joint_name
        ]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [idx_inv[self.joint_parent[i]] for i in idx]
        self.joint_parent[0] = -1
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:, idx, :]
        self.joint_rotation = self.joint_rotation[:, idx, :]
        pass

    def raw_copy(self):
        '''
        返回一个拷贝
        '''
        return copy.deepcopy(self)

    @property
    def motion_length(self):
        return self.joint_position.shape[0]

    def sub_sequence(self, start, end):
        '''
        返回一个子序列
        start: 开始帧
        end: 结束帧
        '''
        res = self.raw_copy()
        res.joint_position = res.joint_position[start:end, :, :]
        res.joint_rotation = res.joint_rotation[start:end, :, :]
        return res

    def append(self, other):
        '''
        在末尾添加另一个动作
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate(
            (self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate(
            (self.joint_rotation, other.joint_rotation), axis=0)
        pass

    def batch_fk_with_velocities(self):
        local_v, local_angular_v = self.compute_velocities()
        global_pos = [self.joint_position[:, 0, :]]
        global_rot = [self.joint_rotation[:, 0, :]]
        global_v = [local_v[:, 0, :]]
        global_rot_v = [local_angular_v[:, 0, :]]
        for i in range(1, len(self.joint_parent)):
            parent = self.joint_parent[i]
            global_pos.append(
                R.from_quat(global_rot[parent]).apply(self.joint_position[..., i, :]) + global_pos[parent])
            global_rot.append(
                (R.from_quat(global_rot[parent]) * R.from_quat(self.joint_rotation[..., i, :])).as_quat())
            global_v.append(R.from_quat(global_rot[parent]).apply(local_v[..., i, :]) +
                            np.cross(global_rot_v[parent],
                                     R.from_quat(global_rot[parent]).apply(self.joint_position[..., i, :])) +
                            global_v[parent])
            global_rot_v.append(
                R.from_quat(global_rot[parent]).apply(local_angular_v[..., i, :]) + global_rot_v[parent])
        for t in range(len(global_pos)):
            global_pos[t] = global_pos[t][:, None, :]
            global_rot[t] = global_rot[t][:, None, :]
            global_v[t] = global_v[t][:, None, :]
            global_rot_v[t] = global_rot_v[t][:, None, :]
        return (
            np.concatenate(global_pos, axis=1),
            np.concatenate(global_rot, axis=1),
            np.concatenate(global_v, axis=1),
            np.concatenate(global_rot_v, axis=1))

    def compute_velocities(self):
        """
        计算关节速度与角速度
        """
        velocities = np.empty_like(self.joint_position)
        velocities[1:-1] = (
                0.5 * (self.joint_position[2:] - self.joint_position[1:-1]) * 60.0 +
                0.5 * (self.joint_position[1:-1] - self.joint_position[:-2]) * 60.0
        )
        velocities[0] = velocities[1] - (velocities[3] - velocities[2])
        velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

        angular_velocities = smooth_utils.quat_to_avel(self.joint_rotation, 1.0 / 60.0)
        angular_velocities = np.append(angular_velocities, angular_velocities[-1:, :, :] + (
                angular_velocities[-1:, :, :] - angular_velocities[-2:-1, :, :]), axis=0)
        return velocities, angular_velocities

    def add_simulation_bone(self):
        translations, orientations = self.batch_forward_kinematics()
        sim_translation = np.array([1.0, 0.0, 1.0]) * translations[:, self.joint_name.index('lowerback_torso')]
        sim_translation = signal.savgol_filter(sim_translation, 31, 3, axis=0, mode='interp')
        sim_direction = np.array([1.0, 0.0, 1.0]) * (R.from_quat(
            orientations[:, self.joint_name.index('RootJoint')]).apply(np.array([0.0, 0.0, 1.0])))
        sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode='interp')
        sim_direction = sim_direction / np.sqrt(np.sum(sim_direction ** 2, axis=-1))[..., np.newaxis]
        sim_orientation = np.repeat(np.array([[0.0, 1.0, 0.0]]), self.joint_position.shape[0], 0)
        d = np.cross(sim_direction, sim_orientation)
        rot_rad = np.arccos(sim_direction[:, 2])
        for i in range(self.joint_position.shape[0]):
            sim_orientation[i, :] = sim_orientation[i, :] * rot_rad[i]
            if d[i, 2] < 0.0:
                sim_orientation[i, :] = -sim_orientation[i, :]
        sim_orientation = R.from_rotvec(sim_orientation)
        self.joint_position[:, 0] = sim_orientation.inv().apply(self.joint_position[:, 0] - sim_translation)
        self.joint_rotation[:, 0] = (sim_orientation.inv() * R.from_quat(self.joint_rotation[:, 0])).as_quat()
        self.joint_position = np.concatenate([sim_translation[:, None, :], self.joint_position], axis=1)
        self.joint_rotation = np.concatenate([sim_orientation.as_quat()[:, None, :], self.joint_rotation], axis=1)
        self.joint_name.insert(0, 'Simulation')
        self.joint_channel[0] = 3
        self.joint_channel.insert(0, 6)
        for i in range(len(self.joint_parent)):
            self.joint_parent[i] += 1
        self.joint_parent.insert(0, -1)

        # --------------------- 你的任务 -------------------- #

    def decompose_rotation_with_yaxis(self, rotation):
        """
        输入: rotation 形状为(4,)的ndarray, 四元数旋转
        输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
        """
        rot_y = np.zeros_like(rotation)
        rot_xz = np.zeros_like(rotation)
        # TODO: 你的代码

        y_axis = np.array([0, 1, 0])
        y_axis_local = R.from_quat(rotation).as_matrix() @ y_axis
        y_axis_local = y_axis_local / np.sqrt(np.sum(y_axis_local ** 2))

        rot_axis = np.cross(y_axis_local, y_axis)
        rot_axis = rot_axis / np.sqrt(np.sum(rot_axis ** 2))

        theta = np.arccos(np.dot(y_axis, y_axis_local))

        # R'
        r = R.from_rotvec(theta * rot_axis)

        rot_y = (r * R.from_quat(rotation)).as_quat()
        rot_xz = (R.from_quat(rot_y).inv() * R.from_quat(rotation)).as_quat()
        return rot_y, rot_xz

    # part 1
    def translation_and_rotation(self, frame_num, target_translation_xz,
                                 target_facing_direction_xz):
        """
        计算出新的joint_position和joint_rotation
        使第frame_num帧的根节点平移为target_translation_xz, 水平面朝向为target_facing_direction_xz
        frame_num: int
        target_translation_xz: (2,)的ndarray
        target_faceing_direction_xz: (2,)的ndarray，表示水平朝向。你可以理解为原本的z轴被旋转到这个方向。
        Tips:
            主要是调整root节点的joint_position和joint_rotation
            frame_num可能是负数，遵循python的索引规则
            你需要完成并使用decompose_rotation_with_yaxis
            输入的target_facing_direction_xz的norm不一定是1
        """

        res = self.raw_copy()  # 拷贝一份，不要修改原始数据

        # 比如说，你可以这样调整第frame_num帧的根节点平移
        target_pos = np.array([
            target_translation_xz[0], res.joint_position[frame_num, 0, 1],
            target_translation_xz[1]
        ])
        Ry, Rxz = self.decompose_rotation_with_yaxis(
            res.joint_rotation[frame_num, 0])
        target_facing_direction_xz = target_facing_direction_xz / \
                                     np.sqrt(np.sum(target_facing_direction_xz ** 2))
        Ry_new = R.from_rotvec(
            np.array([0, 1, 0]) * np.arccos(target_facing_direction_xz[1]))

        res.joint_rotation[:, 0] = (
                Ry_new * R.from_quat(Ry).inv() *
                R.from_quat(res.joint_rotation[:, 0])).as_quat()

        N = res.joint_position.shape[0]
        for i in range(N):
            offset = self.joint_position[i, 0] - \
                     self.joint_position[frame_num, 0]
            new_position = (Ry_new * R.from_quat(Ry).inv()
                            ).as_matrix() @ offset + target_pos
            res.joint_position[i, 0] = new_position

        return res


# part2


def blend_two_motions(bvh_motion1, bvh_motion2, alpha):
    '''
    blend两个bvh动作
    假设两个动作的帧数分别为n1, n2
    alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作应该有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    '''

    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros(
        (len(alpha), res.joint_position.shape[1], res.joint_position.shape[2]))
    res.joint_rotation = np.zeros(
        (len(alpha), res.joint_rotation.shape[1], res.joint_rotation.shape[2]))
    res.joint_rotation[..., 3] = 1.0

    def my_lerp(v1, v2, w):
        return (1 - w) * v1 + w * v2

    N3 = len(alpha)
    N1 = bvh_motion1.joint_position.shape[0]
    N2 = bvh_motion2.joint_position.shape[0]
    for i in range(N3):
        alpha_cur = max(min(alpha[i], 1.0), 0.0)
        ratio = i / (N3 - 1)
        j = ratio * (N1 - 1)
        k = ratio * (N2 - 1)
        res.joint_position[i] = my_lerp(
            my_lerp(bvh_motion1.joint_position[math.floor(j)],
                    bvh_motion1.joint_position[min(math.floor(j) + 1, N1 - 1)],
                    j - math.floor(j)),
            my_lerp(bvh_motion2.joint_position[math.floor(k)],
                    bvh_motion2.joint_position[min(math.floor(k) + 1, N2 - 1)],
                    k - math.floor(k)), alpha_cur)

        def my_slerp(v1, v2, w):
            s = Slerp([0, 1], R.from_quat([v1, v2]))
            return s(w).as_quat()

        for joint in range(bvh_motion1.joint_rotation.shape[1]):
            r1 = my_slerp(
                bvh_motion1.joint_rotation[math.floor(j)][joint],
                bvh_motion1.joint_rotation[min(math.floor(j) + 1,
                                               N1 - 1)][joint],
                j - math.floor(j))
            r2 = my_slerp(
                bvh_motion2.joint_rotation[math.floor(k)][joint],
                bvh_motion2.joint_rotation[min(math.floor(k) + 1,
                                               N2 - 1)][joint],
                k - math.floor(k))
            res.joint_rotation[i][joint] = my_slerp(r1, r2, alpha_cur)
    return res


# part3


def build_loop_motion(bvh_motion):
    '''
    将bvh动作变为循环动作
    由于比较复杂,作为福利,不用自己实现
    (当然你也可以自己实现试一下)
    推荐阅读 https://theorangeduck.com/
    Creating Looping Animations from Motion Capture
    '''
    res = bvh_motion.raw_copy()

    from smooth_utils import build_loop_motion
    return build_loop_motion(res)


# part4


def concatenate_two_motions(bvh_motion1, bvh_motion2, mix_frame1, mix_time):
    '''
    将两个bvh动作平滑地连接起来，mix_time表示用于混合的帧数
    混合开始时间是第一个动作的第mix_frame1帧
    虽然某些混合方法可能不需要mix_time，但是为了保证接口一致，我们还是保留这个参数
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    res = bvh_motion1.raw_copy()
    res2 = bvh_motion2.raw_copy()

    dt = 1 / 60

    # 对齐
    pos = bvh_motion1.joint_position[mix_frame1, 0, [0, 2]]
    rot = bvh_motion1.joint_rotation[mix_frame1, 0]
    res2 = res2.translation_and_rotation(
        0, pos,
        R.from_quat(rot).apply(np.array([0, 0, 1]))[[0, 2]])

    # 更新 position
    offset_pos = res.joint_position[mix_frame1] - res2.joint_position[0]
    v1 = (res.joint_position[1:] - res.joint_position[:-1]) / dt
    v2 = (res2.joint_position[1:] - res2.joint_position[:-1]) / dt
    offset_v = v1[mix_frame1] - v2[0]
    for i in range(mix_time):
        offset_pos, offset_v = smooth_utils.decay_spring_implicit_damping_pos(offset_pos, offset_v, 0.2, dt)
        res2.joint_position[i] += offset_pos

    # 更新 rotation
    offset_rot = (R.from_quat(res.joint_rotation[mix_frame1]) * R.from_quat(res2.joint_rotation[0]).inv()).as_rotvec()
    rot_v1 = smooth_utils.quat_to_avel(res.joint_rotation, dt)
    rot_v2 = smooth_utils.quat_to_avel(res2.joint_rotation, dt)
    offset_rot_v = rot_v1[mix_frame1] - rot_v2[0]
    for i in range(mix_time):
        offset_rot, offset_rot_v = smooth_utils.decay_spring_implicit_damping_rot(offset_rot, offset_rot_v, 0.2, dt)
        res2.joint_rotation[i] = (R.from_rotvec(offset_rot) * R.from_quat(res2.joint_rotation[i])).as_quat()

    res.joint_position = np.concatenate(
        [res.joint_position[:mix_frame1], res2.joint_position], axis=0)
    res.joint_rotation = np.concatenate(
        [res.joint_rotation[:mix_frame1], res2.joint_rotation], axis=0)

    return res
