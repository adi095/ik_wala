import cv2
import numpy as np
import ikpy.chain
import transformations as tf
import tempfile
from simulation import SimulationEnvironment

# === PATCH URDF IN MEMORY (continuous -> revolute) ===
with open("poppy_ergo_jr.pybullet.urdf", "r") as f:
    urdf_text = f.read()
urdf_text = urdf_text.replace('type="continuous"', 'type="revolute"')
with tempfile.NamedTemporaryFile(delete=False, suffix=".urdf", mode='w') as tmp_urdf:
    tmp_urdf.write(urdf_text)
    patched_urdf_path = tmp_urdf.name

# === Load IK chain ===
ik_chain = ikpy.chain.Chain.from_urdf_file(patched_urdf_path)

# === Initialize simulation ===
env = SimulationEnvironment()  # Removed debug=True
joint_names = [info[0] for info in env.get_joint_info() if info[4] is not None]

# === Add a block and let it settle ===
block_pos = (0.0, -0.15, 0.015)  # Slightly above ground
block_quat = (1, 0, 0, 0)       # No rotation
block_label = env._add_block(loc=block_pos, quat=block_quat)
env.settle(1.0)

# === Get the block pose ===
target_pos, target_quat = env.get_block_pose(block_label)
target_rot = tf.quaternion_matrix(target_quat)[:3, :3]

# === Step 1: Position above the block ===
target_pos_above = np.array(target_pos)
target_pos_above[2] += 0.0  # 1 cm above the block

# Solve IK to position above the block
ik_angles = ik_chain.inverse_kinematics(
    target_position=target_pos_above,
    target_orientation=target_rot
)

# Map IK angles to simulation format
sim_joint_angles = {
    name: np.degrees(ik_angles[i + 1])
    for i, name in enumerate(joint_names) if i + 1 < len(ik_angles)
}
#sim_joint_angles["m6"] = 0  # Keep gripper open

print("Target (above block):", target_pos_above)
print("Joint angles (deg):", sim_joint_angles)

# Move to position above the block
env.goto_position(sim_joint_angles, duration=3.0)

# === Step 2: Lower the arm to grasp the block ===
target_pos_grasp = np.array(target_pos)  # At the block's height (no Z offset)
ik_angles_grasp = ik_chain.inverse_kinematics(
    target_position=target_pos_grasp,
    target_orientation=target_rot
)

# Map IK angles for grasping position
sim_joint_angles_grasp = {
    name: np.degrees(ik_angles_grasp[i + 1])
    for i, name in enumerate(joint_names) if i + 1 < len(ik_angles_grasp)
}
#sim_joint_angles_grasp["m6"] = 0  # Gripper still open

print("Target (grasp block):", target_pos_grasp)
print("Joint angles (deg):", sim_joint_angles_grasp)

# Lower the arm to the block
env.goto_position(sim_joint_angles_grasp, duration=2.0)

# === Step 3: Close the gripper to pick up the block ===
sim_joint_angles_grasp["m6"] = -20  # Close the gripper (adjust value based on your gripper's range)
env.goto_position(sim_joint_angles_grasp, duration=10.0)

# === Step 4: Lift the arm with the block ===
target_pos_lift = np.array(target_pos)
target_pos_lift[2] += 0.05  # Lift 5 cm above the initial block position
ik_angles_lift = ik_chain.inverse_kinematics(
    target_position=target_pos_lift,
    target_orientation=target_rot
)

# Map IK angles for lifting
sim_joint_angles_lift = {
    name: np.degrees(ik_angles_lift[i + 1])
    for i, name in enumerate(joint_names) if i + 1 < len(ik_angles_lift)
}
sim_joint_angles_lift["m6"] = -20  # Keep gripper closed

print("Target (lift block):", target_pos_lift)
print("Joint angles (deg):", sim_joint_angles_lift)

# Lift the arm with the block
env.goto_position(sim_joint_angles_lift, duration=3.0)

# === Capture frame ===
rgba, _, _ = env.get_camera_image()
frame_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
cv2.imshow("Result", frame_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === Cleanup ===
env.close()