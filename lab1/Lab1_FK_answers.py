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
    

    joint_name = []
    joint_parent = []
    joint_offset = []

    stack = [-1]
    index = -1
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line_i = lines[i].strip()
            if line_i.startswith('ROOT') or line_i.startswith('JOINT') or line_i.startswith('End Site'):
                if line_i.startswith('End Site'):
                    joint_name.append(f'{joint_name[-1]}_end')                
                else:
                    joint_name.append(line_i.split()[1])
                joint_parent.append(stack[-1])
                index += 1
                stack.append(index)

            if line_i.startswith('}'):
                stack.pop()

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
        if joint_name[i].endswith('_end'):
            joint_positions[i, :] = joint_positions[joint_parent[i]] 
        elif joint_name[i].startswith('Root'):
            joint_positions[i, :] = motion_data[frame_id, 0:3]
            joint_orientations[i, :] = R.from_euler('XYZ', motion_data[frame_id, 3:6], degrees = True).as_quat()
        else:
            local_rotation = R.from_euler('XYZ', motion_data[frame_id, 6 + 3*non_end_idx : 9 + 3*non_end_idx], degrees = True).as_matrix()
            parent_rotation = R.from_quat(joint_orientations[joint_parent[i]]).as_matrix()
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
    motion_data = None
    return motion_data
