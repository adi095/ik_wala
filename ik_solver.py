import numpy as np
import pybullet as pb
import matplotlib.pyplot as pt
import evaluation as ev
from simulation import SimulationEnvironment

class IKSolver:
    def __init__(self, env):
        self.env = env
        self.robot_id = env.robot_id

    def calculate_ik(self, position, orientation_euler):
        orientation_quat = pb.getQuaternionFromEuler(orientation_euler)
        joint_angles = pb.calculateInverseKinematics(
            self.robot_id,
            self.env.num_joints - 1,
            position,
            targetOrientation=orientation_quat,
        )
        return joint_angles

    def set_gripper(self, close=True):
        pos = -0.1 if close else 0.1
        gripper_index = self.env.joint_index["m6"]
        pb.setJointMotorControl2(
            self.robot_id, gripper_index, pb.POSITION_CONTROL, targetPosition=pos
        )

    def move_arm(self, position, orientation_euler, duration=2):
        joint_angles = self.calculate_ik(position, orientation_euler)
        joint_dict = {
            self.env.joint_name[i]: joint_angles[i] * 180 / np.pi
            for i in range(len(joint_angles))
        }
        self.env.goto_position(joint_dict, duration)
        ee_pos = self.get_end_effector_position()
        print("\n--- joint Check ---")
        print(f"Target:   {np.round(position, 4)}")
        print(f"Achieved: {np.round(ee_pos, 4)}")
        print(f"Error:    {np.linalg.norm(np.array(position) - np.array(ee_pos)):.4f}")

    def get_end_effector_position(self):
    
       link_state = pb.getLinkState(self.robot_id, self.env.num_joints - 1)
       position = link_state[4]  # World position of link frame (com)
       return position    

    def execute_ik(self, block_label, goal_position):
        lift_offset = 0.01
        block_pos, block_ori = self.env.get_block_pose(block_label)
        block_euler = pb.getEulerFromQuaternion(block_ori)

        # Move above block
        above_block = (block_pos[0], block_pos[1], block_pos[2] + lift_offset)
        self.move_arm(above_block, block_euler)

        # Move down to block
        self.move_arm(block_pos, block_euler)

        # Close gripper
        self.set_gripper(close=True)
        self.env.settle(1)

        # Lift block
        self.move_arm(above_block, block_euler)

        # Move above goal
        above_goal = (goal_position[0], goal_position[1], goal_position[2] + lift_offset)
        self.move_arm(above_goal, block_euler)

        # Move down to goal
        self.move_arm(goal_position, block_euler)

        # Open gripper
        self.set_gripper(close=False)
        self.env.settle(1)

        # Retreat
        self.move_arm(above_goal, block_euler)

if __name__ == "__main__":
    env = SimulationEnvironment(show=True)
    ik_solver = IKSolver(env)

    tower_poses = ev.get_tower_base_poses(half_spots=3)

    # Add a test block
    loc, quat = tower_poses[1]
    loc = loc[:2] + (0.011,)
    label = env._add_block(loc, quat, side=0.02)
    env.settle(1.)

    # Set goal
    goal_loc, goal_quat = tower_poses[0]
    goal_loc = goal_loc[:2] + (0.01,)
    goal_poses = {label: (goal_loc, goal_quat)}

    # Run IK pick and place
    ik_solver.execute_ik(label, goal_loc)

    # Evaluate
    accuracy, loc_errors, rot_errors = ev.evaluate(env, goal_poses)
    env.close()

    # Print results
    print(f"\n{int(100 * accuracy)}% of blocks near correct goal positions")
    print(f"mean|max location error = {np.mean(loc_errors):.3f}|{np.max(loc_errors):.3f}")
    print(f"mean|max rotation error = {np.mean(rot_errors):.3f}|{np.max(rot_errors):.3f}")
