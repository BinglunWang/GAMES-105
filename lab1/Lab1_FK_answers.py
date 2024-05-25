import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    
    # resotre the following items as the same order as the bvh file
    joint_name = []
    joint_parent = []
    joint_offset = []

    # build a stack to follow dfs order
    stack = [-1]
    # initial index as -1, so the root father is -1
    index = -1
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line_i = lines[i].strip()
            # go to the child
            if line_i.startswith('ROOT') or line_i.startswith('JOINT') or line_i.startswith('End Site'):
                if line_i.startswith('End Site'):
                    joint_name.append(f'{joint_name[-1]}_end')                
                else:
                    joint_name.append(line_i.split()[1])
                joint_parent.append(stack[-1])
                index += 1
                stack.append(index)
            # go back to the parent
            if line_i.startswith('}'):
                stack.pop()
            # get the data of the joint
            if line_i.startswith('OFFSET'):
                joint_offset.append(np.array([float(x) for x in line_i.split()[1:]]))

    joint_offset = np.array(joint_offset)

    return joint_name, joint_parent, joint_offset

    

def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """

    joint_positions = np.zeros((len(joint_name), 3))
    joint_orientations = np.zeros((len(joint_name), 4))
    non_end_idx = 0

    for i in range(0, len(joint_name)):
        # end site position = parent position
        if joint_name[i].endswith('_end'):
            joint_positions[i, :] = joint_positions[joint_parent[i]]
        # first is root
        elif joint_name[i].startswith('Root'):
            joint_positions[i, :] = motion_data[frame_id, 0:3]
            joint_orientations[i, :] = R.from_euler('XYZ', motion_data[frame_id, 3:6], degrees = True).as_quat()
        else:
            local_rotation = R.from_euler('XYZ', motion_data[frame_id, 6 + 3*non_end_idx : 9 + 3*non_end_idx], degrees = True).as_matrix()
            parent_rotation = R.from_quat(joint_orientations[joint_parent[i]]).as_matrix()
            # 4 values of quaternion
            joint_orientations[i, :] = R.from_matrix(parent_rotation @ local_rotation).as_quat()
            joint_positions[i, :] = joint_positions[joint_parent[i]] + parent_rotation @  joint_offset[i]
            non_end_idx += 1


    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    A_data=load_motion_data(A_pose_bvh_path) # (14269, 63)
    T_joint_names, _, _=part1_calculate_T_pose(T_pose_bvh_path) # 25
    A_joint_names, _, _=part1_calculate_T_pose(A_pose_bvh_path) # 25

    index_T_in_A = np.zeros(len(T_joint_names), dtype=int)
    pre_end_A = np.zeros(len(T_joint_names), dtype=int)
    pre_end_cnt = 0

    # how many end sites before each joint
    for i in range(len(A_joint_names)):
        if A_joint_names[i].endswith('_end'):
            pre_end_cnt += 1
            continue
        pre_end_A[i] = pre_end_cnt
    # get the index of T in A
    for i in range(len(T_joint_names)):
        index_T_in_A[i] = A_joint_names.index(T_joint_names[i])

    rotate_z_n45 = R.from_euler('XYZ', [0, 0, -45], degrees=True).as_matrix()
    rotate_z_45 = R.from_euler('XYZ', [0, 0, 45], degrees=True).as_matrix()

    motion_data = np.zeros(A_data.shape)
    for frame_i in range(A_data.shape[0]):
        non_end_idx = 0
        for joint_i in range(len(T_joint_names)):
            # motion data dont require end sites
            if T_joint_names[joint_i].endswith('_end'):
                continue
        
            if T_joint_names[joint_i].startswith('Root'):
                motion_data[frame_i, 0 : 6] = A_data[frame_i, 0 : 6]
            else:
                start_A_data_index = (index_T_in_A[joint_i] - pre_end_A[index_T_in_A[joint_i]]) * 3 + 3 
                A_s = start_A_data_index
                A_e = start_A_data_index + 3
                T_s = non_end_idx * 3 + 3
                T_e = non_end_idx * 3 + 6
                A_rotation_euler = A_data[frame_i, A_s : A_e]

                # lShoulder and rShoulder need to rotate -45 and 45 to T pose
                if T_joint_names[joint_i] == 'lShoulder':
                    A_roatation_matrix = R.from_euler('XYZ', A_rotation_euler, degrees=True).as_matrix()
                    T_rotation_matrix = A_roatation_matrix @ rotate_z_n45
                    T_rotation_euler = R.from_matrix(T_rotation_matrix).as_euler('XYZ', degrees=True)
                elif T_joint_names[joint_i] == 'rShoulder':
                    A_roatation_matrix = R.from_euler('XYZ', A_rotation_euler, degrees=True).as_matrix()
                    T_rotation_matrix = A_roatation_matrix @ rotate_z_45
                    T_rotation_euler = R.from_matrix(T_rotation_matrix).as_euler('XYZ', degrees=True)
                else:
                    T_rotation_euler = A_rotation_euler
                
                motion_data[frame_i, T_s : T_e] = T_rotation_euler
            non_end_idx += 1


    return motion_data
