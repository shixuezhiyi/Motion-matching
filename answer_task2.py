# 以下部分均为可更改部分
import numpy as np

from answer_task1 import *


def database_trajectory_index_clamp(db, frame, offset):
    for i in range(len(db.range_start)):
        if db.range_start[i] <= frame < db.range_stop[i]:
            return max(min(frame + offset, db.range_stop[i] - 1), db.range_start[i])
    return -1


def length(vec):
    return np.sqrt(np.sum(vec ** 2))


def normalize(vec):
    return vec / length(vec)


def clamp(a, low, high):
    return min(max(a, low), high)


def ik_look_at(bone_rotation, global_parent_rotation, global_rotation,
               global_position, child_position, target_position, eps=1e-5):
    curr_dir = normalize(child_position - global_position)
    targ_dir = normalize(target_position - global_position)

    if abs(1.0 - np.dot(curr_dir, targ_dir)) > eps:
        rot_vec_between = R.from_rotvec(normalize(np.cross(curr_dir, targ_dir)) * np.arccos(np.dot(curr_dir, targ_dir)))
        bone_rotation = (
                R.from_quat(global_parent_rotation).inv() * rot_vec_between * R.from_quat(global_rotation)).as_quat()

    return bone_rotation


def partial_forward_kinematics(joint_position, joint_rotation, joint_parent, begin_joint, end_joint, joint_translation,
                               joint_orientation):
    # 一个小hack是root joint的parent是-1, 对应最后一个关节
    # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向

    if end_joint != begin_joint:
        parent_translation, parent_oritation, joint_translation, joint_orientation = partial_forward_kinematics(
            joint_position,
            joint_rotation,
            joint_parent,
            begin_joint,
            joint_parent[end_joint],
            joint_translation,
            joint_orientation)
        translation = R.from_quat(parent_oritation).apply(joint_position[end_joint]) + parent_translation
        oritation = (R.from_quat(parent_oritation) * R.from_quat(joint_rotation[end_joint])).as_quat()
    else:
        translation = joint_translation[end_joint]
        oritation = joint_orientation[end_joint]

    joint_translation[end_joint] = translation
    joint_orientation[end_joint] = oritation
    return translation, oritation, joint_translation, joint_orientation


def ik_two_bone(bone_root, bone_mid, bone_end, target, fwd, bone_root_gr, bone_mid_gr, bone_par_gr, max_length_buffer):
    max_extension = np.sqrt(np.sum((bone_root - bone_mid) ** 2)) + np.sqrt(
        np.sum((bone_mid - bone_end) ** 2)) - max_length_buffer
    target_clamp = target
    if np.sqrt(np.sum((target - bone_root) ** 2)) > max_extension:
        target_clamp = bone_root + max_extension * normalize(target - bone_root)

    axis_dwn = normalize(bone_end - bone_root)
    axis_rot = normalize(np.cross(axis_dwn, fwd))

    a = bone_root
    b = bone_mid
    c = bone_end
    t = target_clamp

    lab = length(b - a)
    lcb = length(b - c)
    lat = length(t - a)

    ac_ab_0 = np.arccos(clamp(np.dot(normalize(c - a), normalize(b - a)), -1.0, 1.0))
    ba_bc_0 = np.arccos(clamp(np.dot(normalize(a - b), normalize(c - b)), -1.0, 1.0))

    ac_ab_1 = np.arccos(clamp((lab * lab + lat * lat - lcb * lcb) / (2.0 * lab * lat), -1.0, 1.0))
    ba_bc_1 = np.arccos(clamp((lab * lab + lcb * lcb - lat * lat) / (2.0 * lab * lcb), -1.0, 1.0))

    r0 = R.from_rotvec((ac_ab_1 - ac_ab_0) * axis_rot)
    r1 = R.from_rotvec((ba_bc_1 - ba_bc_0) * axis_rot)

    c_a = normalize(bone_end - bone_root)
    t_a = normalize(target_clamp - bone_root)

    angle = np.arccos(clamp(np.dot(c_a, t_a), -1.0, 1.0))
    r2 = R.from_rotvec(np.arccos(clamp(np.dot(c_a, t_a), -1.0, 1.0)) * normalize(np.cross(c_a, t_a)))

    if angle > 0.001:
        bone_root_lr = (R.from_quat(bone_par_gr).inv() * r2 * r0 * R.from_quat(bone_root_gr)).as_quat()
    else:
        bone_root_lr = (R.from_quat(bone_par_gr).inv() * r0 * R.from_quat(bone_root_gr)).as_quat()
    bone_mid_lr = (R.from_quat(bone_root_gr).inv() * r1 * R.from_quat(bone_mid_gr)).as_quat()

    return bone_root_lr, bone_mid_lr


def forward_kinematics(joint_position, joint_rotation, joint_parent):
    '''
    joint_position: (M,3)的ndarray, 局部平移
    joint_rotation: (M,4)的ndarray, 用四元数表示的局部旋转
    '''

    joint_translation = np.zeros_like(joint_position)
    joint_orientation = np.zeros_like(joint_rotation)
    joint_orientation[:, 3] = 1.0  # 四元数的w分量默认为1

    # 一个小hack是root joint的parent是-1, 对应最后一个关节
    # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向

    for i in range(len(joint_parent)):
        pi = joint_parent[i]
        parent_orientation = R.from_quat(joint_orientation[pi, :])
        joint_translation[i, :] = joint_translation[pi, :] + \
                                  parent_orientation.apply(joint_position[i, :])
        joint_orientation[i, :] = (parent_orientation * R.from_quat(joint_rotation[i, :])).as_quat()
    return joint_translation, joint_orientation


def inertialize_transition_p(off_x, off_v, src_x, src_v, dst_x, dst_v):
    off_x = (src_x + off_x) - dst_x
    off_v = (src_v + off_v) - dst_v
    return off_x, off_v


def inertialize_transition_q(off_x, off_v, src_x, src_v, dst_x, dst_v):
    off_x = (R.from_quat(off_x) * R.from_quat(src_x) * R.from_quat(dst_x).inv()).as_quat()
    off_v = (off_v + src_v) - dst_v
    return off_x, off_v


def inertialize_update_p(off_x, off_v, in_x, in_v, halflife, dt):
    off_x, off_v = smooth_utils.decay_spring_implicit_damping_pos(off_x, off_v, halflife, dt)
    out_x = in_x + off_x
    out_v = in_v + off_v
    return out_x, out_v, off_x, off_v


def inertialize_update_q(off_x, off_v, in_x, in_v, halflife, dt):
    off_x = R.from_quat(off_x).as_rotvec()
    off_x, off_v = smooth_utils.decay_spring_implicit_damping_rot(off_x, off_v, halflife, dt)
    out_x = (R.from_rotvec(off_x) * R.from_quat(in_x)).as_quat()
    off_x = R.from_rotvec(off_x).as_quat()
    out_v = off_v + in_v
    return out_x, out_v, off_x, off_v


def batch_forward_kinematics_with_velocity(position, rotation, velocity, angular_velocity, joint_parent):
    gp, gr, gv, ga = [position[..., 0, :]], [rotation[..., 0, :]], [velocity[..., 0, :]], [angular_velocity[..., 0, :]]
    for i in range(1, len(joint_parent)):
        gp.append(R.from_quat(gr[joint_parent[i]]).apply(position[..., i, :]) + gp[joint_parent[i]])
        gr.append((R.from_quat(gr[joint_parent[i]]) * R.from_quat(rotation[..., i, :])).as_quat())
        gv.append(R.from_quat(gr[joint_parent[i]]).apply(velocity[..., i, :]) +
                  np.cross(ga[joint_parent[i]], R.from_quat(gr[joint_parent[i]]).apply(position[..., i, :])) +
                  gv[joint_parent[i]])
        ga.append(R.from_quat(gr[joint_parent[i]]).apply(angular_velocity[..., i, :]) + ga[joint_parent[i]])

    for t in range(len(gp)):
        gp[t] = gp[t][:, None, :]
        gr[t] = gr[t][:, None, :]
        gv[t] = gv[t][:, None, :]
        ga[t] = ga[t][:, None, :]
    return (
        np.concatenate(gp, axis=1),
        np.concatenate(gr, axis=1),
        np.concatenate(gv, axis=1),
        np.concatenate(ga, axis=1))


def batch_fk_single_joint(positions, rotations, parents, joint_idx):
    if parents[joint_idx] != -1:
        parent_t, parent_o = batch_fk_single_joint(positions, rotations, parents, parents[joint_idx])
        translation = R.from_quat(parent_o).apply(positions[:, joint_idx]) + parent_t
        orientation = (R.from_quat(parent_o) * R.from_quat(rotations[:, joint_idx])).as_quat()
    else:
        translation = positions[:, joint_idx]
        orientation = rotations[:, joint_idx]

    return translation, orientation


def normalize_feature(features, features_offset, features_scale, offset, size, weight=1.0):
    features_offset[offset:offset + size] = 0.0
    nframe = features.shape[0]
    features_offset[offset:offset + size] = np.sum(features[:, offset:offset + size] / nframe, axis=0)

    std = np.std(features[:, offset:offset + size])
    features_scale[offset:offset + size] = std / weight

    features[:, offset:offset + size] = (features[:, offset:offset + size] - features_offset[
                                                                             offset:offset + size]) / features_scale[
                                                                                                      offset:offset + size]


def batch_fk_with_velocity_single_joint(positions, rotations, velocities, angular_velocities, parents,
                                        joint_idx):
    if parents[joint_idx] != -1:
        parent_translation, parent_orientation, parent_velocity, parent_angular_velocity = batch_fk_with_velocity_single_joint(
            positions,
            rotations,
            velocities,
            angular_velocities,
            parents,
            parents[joint_idx])
        translation = R.from_quat(parent_orientation).apply(positions[:, joint_idx]) + parent_translation
        orientation = (R.from_quat(parent_orientation) * R.from_quat(rotations[:, joint_idx])).as_quat()
        v = R.from_quat(parent_orientation).apply(velocities[:, joint_idx, :]) + \
            np.cross(parent_angular_velocity,
                     R.from_quat(parent_orientation).apply(positions[:, joint_idx, :])) + \
            parent_velocity
        rot_v = R.from_quat(parent_orientation).apply(
            angular_velocities[:, joint_idx, :]) + parent_angular_velocity
    else:
        translation = positions[:, joint_idx]
        orientation = rotations[:, joint_idx]
        v = velocities[:, joint_idx]
        rot_v = angular_velocities[:, joint_idx]
    return translation, orientation, v, rot_v


class Database:
    def __init__(self, motions):
        bone_positions = []
        bone_velocities = []
        bone_rotations = []
        bone_angular_velocities = []
        contact_states = []
        range_starts = []
        range_stops = []
        for motion in motions:
            motion.add_simulation_bone()
            local_v, local_rot_v = motion.compute_velocities()
            global_pos, global_rot, global_v, global_rot_v = motion.batch_fk_with_velocities()

            # 算脚的速度
            contact_velocity_threshold = 0.3
            tv = global_v[:, np.array([
                motion.joint_name.index("lToeJoint"),
                motion.joint_name.index("rToeJoint")])]
            contact_velocity = np.sqrt(np.sum(tv ** 2, axis=-1))
            contacts = contact_velocity < contact_velocity_threshold

            import scipy.ndimage as ndimage
            for ci in range(contacts.shape[1]):
                contacts[:, ci] = ndimage.median_filter(
                    contacts[:, ci],
                    size=6,
                    mode='nearest')
            bone_positions.append(motion.joint_position)
            bone_rotations.append(motion.joint_rotation)
            bone_velocities.append(local_v)
            bone_angular_velocities.append(local_rot_v)
            contact_states.append(contacts)
            offset = 0 if len(range_starts) == 0 else range_stops[-1]
            range_starts.append(offset)
            range_stops.append(offset + len(motion.joint_position))
        bone_parents = motions[0].joint_parent
        bone_names = motions[0].joint_name
        bone_positions = np.concatenate(bone_positions, axis=0)
        bone_velocities = np.concatenate(bone_velocities, axis=0)
        bone_rotations = np.concatenate(bone_rotations, axis=0)
        bone_angular_velocities = np.concatenate(bone_angular_velocities, axis=0)
        contact_states = np.concatenate(contact_states, axis=0).astype(np.uint8)

        nfeatures = 3 + 3 + 3 + 3 + 3 + 6 + 6
        nframes = bone_positions.shape[0]
        features = np.zeros([nframes, nfeatures])
        features_offset = np.zeros(nfeatures)
        features_scale = np.zeros(nfeatures)
        # self.db = {'position': bone_positions, 'velocity': bone_velocities, 'rotation': bone_rotations,
        #            'parent': bone_parents, 'bone_name': bone_names, 'angular_velocity': bone_angular_velocities,
        #            'feature': features, 'feature_offset': features_offset, 'feature_scale': features_scale,
        #            'range_start': range_starts, 'range_stop': range_stops, 'contact_state': contact_states}
        self.position = bone_positions
        self.velocity = bone_velocities
        self.rotation = bone_rotations
        self.parent = bone_parents
        self.bone_name = bone_names
        self.angular_velocity = bone_angular_velocities
        self.feature = features
        self.feature_offset = features_offset
        self.feature_scale = features_scale
        self.range_start = range_starts
        self.range_stop = range_stops
        self.contact_state = contact_states

        # feature 权重
        feature_weight_foot_position = 0.75
        feature_weight_foot_velocity = 0.75
        feature_weight_hip_velocity = 1.0
        feature_weight_trajectory_positions = 1.0
        feature_weight_trajectory_directions = 1.5
        offset = 0
        offset = self.compute_bone_position_feature(offset, 'lAnkle', feature_weight_foot_position)
        offset = self.compute_bone_position_feature(offset, 'rAnkle', feature_weight_foot_position)
        offset = self.compute_bone_velocity_feature(offset, 'lAnkle', feature_weight_foot_velocity)
        offset = self.compute_bone_velocity_feature(offset, 'rAnkle', feature_weight_foot_velocity)
        offset = self.compute_bone_velocity_feature(offset, 'RootJoint', feature_weight_hip_velocity)
        offset = self.compute_trajectory_position_feature(offset, feature_weight_trajectory_positions)
        offset = self.compute_trajectory_direction_feature(offset, feature_weight_trajectory_directions)
        assert (nfeatures == offset)
        print('Done! Feature size is: ')
        print(self.feature.shape)

    def compute_trajectory_position_feature(self, offset, weight=1.0):
        for i in range(len(self.position)):
            t0 = self.database_trajectory_index_clamp(i, 20)
            t1 = self.database_trajectory_index_clamp(i, 40)
            t2 = self.database_trajectory_index_clamp(i, 60)
            trajectory_pos0 = R.from_quat(self.rotation[i, 0]).inv().apply(
                self.position[t0, 0] - self.position[i, 0])
            trajectory_pos1 = R.from_quat(self.rotation[i, 0]).inv().apply(
                self.position[t1, 0] - self.position[i, 0])
            trajectory_pos2 = R.from_quat(self.rotation[i, 0]).inv().apply(
                self.position[t2, 0] - self.position[i, 0])
            self.feature[i, offset + 0] = trajectory_pos0[0]
            self.feature[i, offset + 1] = trajectory_pos0[2]
            self.feature[i, offset + 2] = trajectory_pos1[0]
            self.feature[i, offset + 3] = trajectory_pos1[2]
            self.feature[i, offset + 4] = trajectory_pos2[0]
            self.feature[i, offset + 5] = trajectory_pos2[2]
            normalize_feature(self.feature, self.feature_offset, self.feature_scale, offset, 6,
                              weight)
            return offset + 6

    def compute_trajectory_direction_feature(self, offset, weight=1.0):
        for i in range(len(self.position)):
            t0 = self.database_trajectory_index_clamp(i, 20)
            t1 = self.database_trajectory_index_clamp(i, 40)
            t2 = self.database_trajectory_index_clamp(i, 60)

            trajectory_dir0 = R.from_quat(self.rotation[i, 0]).inv().apply(
                R.from_quat(self.rotation[t0, 0]).apply(np.array([0.0, 0.0, 1.0])))
            trajectory_dir1 = R.from_quat(self.rotation[i, 0]).inv().apply(
                R.from_quat(self.rotation[t1, 0]).apply(np.array([0.0, 0.0, 1.0])))
            trajectory_dir2 = R.from_quat(self.rotation[i, 0]).inv().apply(
                R.from_quat(self.rotation[t2, 0]).apply(np.array([0.0, 0.0, 1.0])))

            self.feature[i, offset + 0] = trajectory_dir0[0]
            self.feature[i, offset + 1] = trajectory_dir0[2]
            self.feature[i, offset + 2] = trajectory_dir1[0]
            self.feature[i, offset + 3] = trajectory_dir1[2]
            self.feature[i, offset + 4] = trajectory_dir2[0]
            self.feature[i, offset + 5] = trajectory_dir2[2]
        normalize_feature(self.feature, self.feature_offset, self.feature_scale, offset, 6, weight)

        return offset + 6

    def database_trajectory_index_clamp(self, frame, offset):
        for i in range(len(self.range_start)):
            if self.range_start[i] <= frame < self.range_start[i]:
                return max(min(frame + offset, self.range_start[i] - 1), self.range_start[i])
        return -1

    def compute_bone_velocity_feature(self, offset, joint_name, weight=1.0):
        joint_idx = self.bone_name.index(joint_name)
        _, _, bone_velocity, _ = batch_fk_with_velocity_single_joint(
            self.position, self.rotation,
            self.velocity, self.angular_velocity,
            self.parent, joint_idx)
        bone_velocity = R.from_quat(self.rotation[:, 0]).inv().apply(bone_velocity)
        self.feature[:, offset:offset + 3] = bone_velocity
        normalize_feature(self.feature, self.feature_offset, self.feature_scale, offset, 3, weight)
        return offset + 3

    def compute_bone_position_feature(self, offset, joint_name, weight=1.0):
        joint_idx = self.bone_name.index(joint_name)
        bone_transition, _ = batch_fk_single_joint(self.position, self.rotation,
                                                   self.parent, joint_idx)
        position = R.from_quat(self.rotation[:, 0]).inv().apply(bone_transition - self.position[:, 0])
        self.feature[:, offset:offset + 3] = position
        normalize_feature(self.feature, self.feature_offset, self.feature_scale, offset, 3, weight)
        return offset + 3


class CharacterController:
    def __init__(self, controller):
        self.motions = []
        self.motions.append(BVHMotion('motion_material/physics_motion/long_walk.bvh'))
        self.motions.append(BVHMotion('motion_material/physics_motion/long_run.bvh'))
        self.motions.append(BVHMotion('motion_material/physics_motion/long_run_mirror.bvh'))
        self.motions.append(BVHMotion('motion_material/physics_motion/long_walk_mirror.bvh'))
        self.controller = controller
        self.cur_root_pos = None
        self.cur_root_rot = None
        self.cur_frame = 0
        self.db = Database(self.motions)
        del self.motions
        start_frame = 0

        # 避免频繁切换动作
        self.search_time = 0.2
        self.search_timer = self.search_time
        self.force_search_timer = self.search_time
        self.desired_velocity = np.zeros(3)
        self.desired_velocity_change_curr = np.zeros(3)
        self.desired_velocity_change_prev = np.zeros(3)
        self.desired_velocity_change_threshold = 50.0
        self.desired_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        self.desired_rotation_change_curr = np.array([0.0, 0.0, 0.0])
        self.desired_rotation_change_prev = np.array([0.0, 0.0, 0.0])
        self.desired_rotation_change_threshold = 50.0

        # Inertialization
        njoint = self.db.position.shape[1]
        self.offset_positions = np.zeros([njoint, 3])
        self.offset_velocities = np.zeros([njoint, 3])
        self.offset_rotations = np.zeros([njoint, 4])
        self.offset_rotations[:, 3] = 1
        self.offset_angular_velocities = np.zeros([njoint, 3])

        self.cur_bone_positions = self.db.position[start_frame]
        self.cur_bone_rotations = self.db.rotation[start_frame]
        self.cur_bone_velocities = self.db.velocity[start_frame]
        self.cur_bone_angular_velocities = self.db.angular_velocity[start_frame]

        self.trns_bone_positions = self.db.position[start_frame]
        self.trns_bone_rotations = self.db.rotation[start_frame]
        self.trns_bone_velocities = self.db.velocity[start_frame]
        self.trns_bone_angular_velocities = self.db.angular_velocity[start_frame]

        self.bone_positions = self.db.position[start_frame]
        self.bone_rotations = self.db.rotation[start_frame]
        self.bone_velocities = self.db.velocity[start_frame]
        self.bone_angular_velocities = self.db.angular_velocity[start_frame]

        self.transition_src_position = self.bone_positions[0]
        self.transition_src_rotation = self.bone_rotations[0]
        self.transition_dst_position = self.bone_positions[0]
        self.transition_dst_rotation = self.bone_rotations[0]
        # simulation position
        self.simulation_position = None
        self.simulation_rotation = None
        self.simulation_velocity = None
        self.simulation_angular_velocity = None
        # 控制
        self.adjustment_position_halflife = 0.1
        self.adjustment_rotation_halflife = 0.2
        self.adjustment_position_max_ratio = 0.5
        self.adjustment_rotation_max_ratio = 0.5
        controller.set_pos(self.bone_positions[0])
        controller.set_rot(self.bone_rotations[0])
        # 锁足
        self.contact_bones = [self.db.bone_name.index('lToeJoint'), self.db.bone_name.index('rToeJoint')]
        self.contact_states = [False, False]
        self.contact_locks = [False, False]
        self.contact_positions = np.zeros([len(self.contact_bones), 3])
        self.contact_velocities = np.zeros([len(self.contact_bones), 3])
        self.contact_points = np.zeros([len(self.contact_bones), 3])
        self.contact_targets = np.zeros([len(self.contact_bones), 3])
        self.contact_offset_positions = np.zeros([len(self.contact_bones), 3])
        self.contact_offset_velocities = np.zeros([len(self.contact_bones), 3])

        self.ik_unlock_radius = 0.2
        self.ik_blending_halflife = 0.1
        self.ik_foot_height = 0.02
        self.ik_toe_length = 0.15

        t, o, v, a = batch_forward_kinematics_with_velocity(self.bone_positions[None, :, :],
                                                            self.bone_rotations[None, :, :],
                                                            self.bone_velocities[None, :, :],
                                                            self.bone_angular_velocities[None, :, :],
                                                            self.db.parent)
        self.contact_positions = t[0, self.contact_bones]
        self.contact_points = t[0, self.contact_bones]
        self.contact_targets = t[0, self.contact_bones]
        self.contact_velocities = v[0, self.contact_bones]
        self.cur_bone_contacts = self.db.contact_state[start_frame]
        controller.move_speed *= 0.75
        self.controller = controller
        self.cur_root_pos = np.zeros(3)
        self.cur_root_rot = np.array([0.0, 0.0, 0.0, 1.0])
        self.cur_frame = 0
        self.max_frame = self.db.position.shape[0]

    def update_state(self,
                     desired_pos_list,
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        self.simulation_position = desired_pos_list[0]
        self.simulation_velocity = desired_vel_list[0]
        self.simulation_rotation = desired_rot_list[0]
        self.simulation_angular_velocity = desired_avel_list[0]
        force_search = False
        dt = 1.0 / 60.0

        self.desired_velocity_change_prev = self.desired_velocity_change_curr
        self.desired_velocity_change_curr = (desired_vel_list[0] - self.desired_velocity) / dt
        self.desired_velocity = desired_vel_list[0]
        self.desired_rotation_change_prev = self.desired_rotation_change_curr
        self.desired_rotation_change_curr = (R.from_quat(desired_rot_list[0]) * R.from_quat(
            self.desired_rotation).inv()).as_rotvec() / dt
        self.desired_rotation = desired_rot_list[0]
        tmp = np.sqrt(
            np.sum(self.desired_rotation_change_prev ** 2)) >= self.desired_rotation_change_threshold > np.sqrt(
            np.sum(self.desired_rotation_change_curr ** 2))
        tmp2 = np.sqrt(
            np.sum(self.desired_velocity_change_prev ** 2)) >= self.desired_velocity_change_threshold > np.snp.sqrt(
            np.sum(self.desired_velocity_change_curr ** 2))
        if self.force_search_timer <= 0.0 and (tmp or tmp2):
            force_search = True
            self.force_search_timer = self.search_time
        elif self.force_search_timer > 0:
            self.force_search_timer -= dt
        # 当前片段是否播放完毕
        end_of_anim = database_trajectory_index_clamp(self.db, self.cur_frame, 1) == self.cur_frame

        # 当很久没search or 片段播放完毕 or 速度或旋转变化很大时才需要搜索db
        if (force_search and self.search_timer <= 0.0) or end_of_anim:
            query_feature = np.zeros_like(self.db.feature[self.cur_frame])
            query_feature[:15] = self.db.feature[self.cur_frame, :15] * self.db.feature_scale[
                                                                        :15] + self.db.feature_offset[:15]
            offset = 15
            # 更新位置
            traj0 = R.from_quat(desired_rot_list[0]).inv().apply(desired_pos_list[1] - desired_pos_list[0])
            traj1 = R.from_quat(desired_rot_list[0]).inv().apply(desired_pos_list[2] - desired_pos_list[0])
            traj2 = R.from_quat(desired_rot_list[0]).inv().apply(desired_pos_list[3] - desired_pos_list[0])

            query_feature[offset + 0] = traj0[0]
            query_feature[offset + 1] = traj0[2]
            query_feature[offset + 2] = traj1[0]
            query_feature[offset + 3] = traj1[2]
            query_feature[offset + 4] = traj2[0]
            query_feature[offset + 5] = traj2[2]
            offset += 6
            traj0 = R.from_quat(desired_rot_list[0]).inv().apply(
                R.from_quat(desired_rot_list[1]).apply(np.array([0.0, 0.0, 1.0])))
            traj1 = R.from_quat(desired_rot_list[0]).inv().apply(
                R.from_quat(desired_rot_list[2]).apply(np.array([0.0, 0.0, 1.0])))
            traj2 = R.from_quat(desired_rot_list[0]).inv().apply(
                R.from_quat(desired_rot_list[3]).apply(np.array([0.0, 0.0, 1.0])))
            query_feature = (query_feature - self.db.feature_offset) / self.db.feature_scale
            # 暴力搜索 cost 最小的 frame
            cost = np.sum((query_feature - self.db.feature) ** 2, axis=1)
            best_index = np.argmin(cost)
            if best_index != self.cur_frame:
                # inertialize pose transition
                self.trns_bone_positions = self.db.position[best_index]
                self.trns_bone_rotations = self.db.rotation[best_index]
                self.trns_bone_velocities = self.db.velocity[best_index]
                self.trns_bone_angular_velocities = self.db.angular_velocity[best_index]
                self.transition_dst_position = self.cur_root_pos
                self.transition_dst_rotation = self.cur_root_rot
                self.transition_src_position = self.db.position[best_index, 0]
                self.transition_src_rotation = self.db.rotation[best_index, 0]
                world_space_dst_velocity = R.from_quat(self.transition_dst_rotation).apply(
                    R.from_quat(self.transition_src_rotation).inv().apply(self.trns_bone_velocities[0])
                )
                world_space_dst_angular_velocity = R.from_quat(self.transition_dst_rotation).apply(
                    R.from_quat(self.transition_src_rotation).inv().apply(self.trns_bone_angular_velocities[0])
                )
                self.offset_positions[0], self.offset_velocities[0] = inertialize_transition_p(self.offset_positions[0],
                                                                                               self.offset_velocities[
                                                                                                   0],
                                                                                               self.bone_positions[0],
                                                                                               self.bone_velocities[0],
                                                                                               self.bone_positions[0],
                                                                                               world_space_dst_velocity)
                self.offset_rotations[0], self.offset_angular_velocities[0] = inertialize_transition_q(
                    self.offset_rotations[0], self.offset_angular_velocities[0], self.bone_rotations[0],
                    self.bone_angular_velocities[0], self.bone_rotations[0], world_space_dst_angular_velocity)
                self.offset_positions[1:], self.offset_velocities[1:] = inertialize_transition_p(
                    self.offset_positions[1:],
                    self.offset_velocities[1:],
                    self.cur_bone_positions[1:],
                    self.cur_bone_velocities[1:],
                    self.trns_bone_positions[1:],
                    self.trns_bone_velocities[1:]
                )
                self.offset_rotations[1:], self.offset_angular_velocities[1:] = inertialize_transition_q(
                    self.offset_rotations[1:],
                    self.offset_angular_velocities[1:],
                    self.cur_bone_rotations[1:],
                    self.cur_bone_angular_velocities[1:],
                    self.trns_bone_rotations[1:],
                    self.trns_bone_angular_velocities[1:]
                )
                self.cur_frame = best_index
                self.search_timer = self.search_time
            self.search_timer -= dt
        self.search_timer -= dt
        self.cur_frame += 1
        self.cur_frame = min(self.cur_frame, self.max_frame - 1)

        # next pose
        self.cur_bone_positions = self.db.position[self.cur_frame]
        self.cur_bone_velocities = self.db.velocity[self.cur_frame]
        self.cur_bone_rotations = self.db.rotation[self.cur_frame]
        self.cur_bone_angular_velocities = self.db.angular_velocity[self.cur_frame]
        self.cur_bone_contacts = self.db.contact_state[self.cur_frame]

        world_space_position = R.from_quat(self.transition_dst_rotation).apply(
            R.from_quat(self.transition_src_rotation).inv().apply(
                self.cur_bone_positions[0] - self.transition_src_position)) + self.transition_dst_position
        world_space_velocity = R.from_quat(self.transition_dst_rotation).apply(
            R.from_quat(self.transition_src_rotation).inv().apply(
                self.cur_bone_velocities[0]))

        def quat_normalize(q, eps=1e-8):
            return q / (length(q) + eps)

        world_space_rotation = quat_normalize(
            (R.from_quat(self.transition_dst_rotation) * R.from_quat(self.transition_src_rotation).inv() * R.from_quat(
                self.cur_bone_rotations[0])).as_quat())
        world_space_angular_velocity = R.from_quat(self.transition_dst_rotation).apply(
            R.from_quat(self.transition_src_rotation).inv().apply(
                self.cur_bone_angular_velocities[0]))

        self.bone_positions[0], self.bone_velocities[0], self.offset_positions[0], self.offset_velocities[0] = \
            inertialize_update_p(self.offset_positions[0],
                                 self.offset_velocities[0],
                                 world_space_position,
                                 world_space_velocity,
                                 0.2,
                                 dt)
        self.bone_rotations[0], self.bone_angular_velocities[0], self.offset_rotations[0, :], \
        self.offset_angular_velocities[0, :] = inertialize_update_q(self.offset_rotations[0, :],
                                                                    self.offset_angular_velocities[0, :],
                                                                    world_space_rotation,
                                                                    world_space_angular_velocity,
                                                                    0.1,
                                                                    dt)

        self.bone_positions[1:], self.bone_velocities[1:], self.offset_positions[1:], self.offset_velocities[1:] = \
            inertialize_update_p(self.offset_positions[1:],
                                 self.offset_velocities[1:],
                                 self.cur_bone_positions[1:],
                                 self.cur_bone_velocities[1:],
                                 0.2,
                                 dt)

        self.bone_rotations[1:], self.bone_angular_velocities[1:], self.offset_rotations[1:,
                                                                   :], self.offset_angular_velocities[1:, :] = \
            inertialize_update_q(self.offset_rotations[1:],
                                 self.offset_angular_velocities[1:],
                                 self.cur_bone_rotations[1:],
                                 self.cur_bone_angular_velocities[1:],
                                 0.1,
                                 dt)

        joint_translation, joint_orientation = forward_kinematics(self.bone_positions, self.bone_rotations,
                                                                  self.db.parent)

        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]

        return self.db.bone_name, joint_translation, joint_orientation

    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''

        # 一个简单的例子，将手柄的位置与角色对齐
        controller.set_pos(self.cur_root_pos)
        controller.set_rot(self.cur_root_rot)

        dt = 1.0 / 60.0
        joint_translation, joint_orientation = forward_kinematics(self.bone_positions, self.bone_rotations,
                                                                  self.db.parent)

        # 锁足
        eps = 1e-8
        adjusted_position = self.bone_positions.copy()
        adjusted_rotation = self.bone_rotations.copy()
        for i in range(len(self.contact_bones)):
            toe_bone = self.contact_bones[i]
            heel_bone = self.db.parent[toe_bone]
            knee_bone = self.db.parent[heel_bone]
            hip_bone = self.db.parent[knee_bone]
            root_bone = self.db.parent[hip_bone]
            input_contact_velocity = (self.contact_positions[i] - self.contact_targets[i]) / (dt + eps)
            self.contact_targets[i] = self.contact_positions[i]
            # 锁足,速度置 0
            if self.contact_locks[i]:
                contact_point = self.contact_points[i]
                contact_velocity = np.zeros(3)
            else:
                contact_point = joint_translation[toe_bone]
                contact_velocity = input_contact_velocity

            self.contact_positions[i], self.contact_velocities[i], self.contact_offset_positions[i], \
            self.contact_offset_velocities[i] = inertialize_update_p(self.contact_offset_positions[i],
                                                                     self.contact_offset_velocities[i], contact_point,
                                                                     contact_velocity, self.ik_blending_halflife, dt)
            tlength = length(self.contact_points[i] - joint_translation[toe_bone])
            unlock_contact = self.contact_locks[i] and tlength > self.ik_unlock_radius

            # 需要锁足
            if not self.contact_states[i] and self.cur_bone_contacts[i]:
                self.contact_locks[i] = True
                self.contact_points[i] = self.contact_positions[i]
                self.contact_points[i][1] = self.ik_foot_height
                self.contact_offset_positions[i], self.contact_offset_velocities[i] = inertialize_transition_p(
                    self.contact_offset_positions[i],
                    self.contact_offset_velocities[i],
                    joint_translation[toe_bone],
                    input_contact_velocity,
                    self.contact_points[i],
                    np.zeros(3)
                )
            elif (self.contact_locks[i] and self.contact_states[i] and not self.cur_bone_contacts[
                i]) or unlock_contact:
                self.contact_locks[i] = False
                self.contact_offset_positions[i], self.contact_offset_velocities[i] = inertialize_transition_p(
                    self.contact_offset_positions[i],
                    self.contact_offset_velocities[i],
                    self.contact_points[i],
                    np.zeros(3),
                    joint_translation[toe_bone],
                    input_contact_velocity
                )
            self.contact_states[i] = self.cur_bone_contacts[i]
            # 避免穿模
            contact_position_clamp = self.contact_positions[i].copy()
            contact_position_clamp[1] = max(contact_position_clamp[1], self.ik_foot_height)

            adjusted_rotation[hip_bone], adjusted_rotation[knee_bone] = ik_two_bone(
                joint_translation[hip_bone],
                joint_translation[knee_bone],
                joint_translation[heel_bone],
                contact_position_clamp + (joint_translation[heel_bone] - joint_translation[toe_bone]),
                R.from_quat(joint_orientation[knee_bone]).apply(np.array([0.0, 1.0, 0.0])),
                joint_orientation[hip_bone],
                joint_orientation[knee_bone],
                joint_orientation[root_bone],
                0.015)
            _, _, joint_translation, joint_orientation = partial_forward_kinematics(adjusted_position,
                                                                                    adjusted_rotation,
                                                                                    self.db.parent,
                                                                                    hip_bone, toe_bone,
                                                                                    joint_translation,
                                                                                    joint_orientation)
            toe_end_curr = R.from_quat(joint_orientation[toe_bone]).apply(np.array([self.ik_toe_length, 0.0, 0.0])) + \
                           joint_translation[toe_bone]

            toe_end_targ = toe_end_curr
            toe_end_targ[1] = max(toe_end_targ[1], self.ik_foot_height)

            adjusted_rotation[toe_bone] = ik_look_at(adjusted_rotation[toe_bone],
                                                     joint_orientation[heel_bone],
                                                     joint_orientation[toe_bone],
                                                     joint_translation[toe_bone],
                                                     toe_end_curr,
                                                     toe_end_targ)

        joint_translation, joint_orientation = forward_kinematics(adjusted_position, adjusted_rotation,
                                                                  self.db.parent)

        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]

        character_state = (self.db.bone_name, joint_translation, joint_orientation)
        return character_state
    # 你的其他代码,state matchine, motion matching, learning, etc.
