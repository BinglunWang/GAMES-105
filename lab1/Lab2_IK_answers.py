import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    # return joint_positions, joint_orientations
    # ccd IK
    ITERATIONS = 50
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    end_index = path[-1]


    print('------ path ------')
    print(path)
    print('------ path_name ------')
    print(path_name)
    print('------ path1 ------')
    print(path1)
    print('------ path2 ------')
    print(path2)
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    print('------ joint_name ------')
    print(joint_name)
    print('------ joint_parent ------')
    print(joint_parent)

    for i in range(len(joint_name)):
        print(i, joint_name[i], joint_parent[i])

    for _ in range(ITERATIONS):
        # reverse path
        for i in range(len(path) - 2, -1, -1):
            node_idex = path[i]
            end_pos = joint_positions[end_index]
            node_pos = joint_positions[node_idex]

            dir_to_target = target_pose - node_pos
            dir_to_end = end_pos - node_pos

            # rotate dir_to_end to dir_to_target
            R_rotate, rssd = R.align_vectors(dir_to_target.reshape(1, 3), dir_to_end.reshape(1, 3))
            R_rotate = R_rotate.as_matrix()
            # update joint_positions and orientations


            joint_flag = np.zeros(len(joint_positions))
            joint_flag[node_idex] = 1
            for j in range(joint_positions.shape[0]):
                update_index = j
                parent_index = joint_parent[update_index]
                if joint_flag[j] == 0 and joint_flag[joint_parent[j]] == 0:
                    continue

                # print('update_index:', update_index, 'parent_index:', parent_index, 'node_idex:', node_idex)
                joint_flag[update_index] = 1
                R_0 = R.from_quat(joint_orientations[update_index]).as_matrix()
                R_new = R_rotate @ R_0 
                joint_orientations[update_index] = R.from_matrix(R_new).as_quat()
                if update_index == node_idex:
                    continue
                offset = joint_positions[update_index] - node_pos
                joint_positions[update_index] = node_pos + R_rotate @ offset


            # for j in range(i, len(path)):
            #     update_index = path[j]
            #     R_0 = R.from_quat(joint_orientations[update_index]).as_matrix()
            #     R_new = R_rotate @ R_0 
            #     joint_orientations[update_index] = R.from_matrix(R_new).as_quat()


            #     offset = joint_positions[update_index] - node_pos
            #     joint_positions[update_index] = node_pos + R_rotate @ offset

            # print('------ update ------\n')

    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations