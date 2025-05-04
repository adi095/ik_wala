import os
import tempfile
import ikpy.chain
import numpy as np
import pybullet as pb
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
env = SimulationEnvironment(show=True)  # Set show=True for visualization

# === Get joint information from simulation ===
joint_info = env.get_joint_info()
joint_names = [info[0] for info in joint_info if info[4] is not None]  # Exclude fixed joints

# === Print simulation joint information ===
print("\n=== Simulation Joint Information (from env.get_joint_info()) ===")
for info in joint_info:
    joint_name, parent_idx, translation, orientation, axis = info
    print(f"Joint: {joint_name}")
    print(f"  Parent Index: {parent_idx}")
    print(f"  Translation: {translation}")
    print(f"  Orientation: {orientation}")
    print(f"  Axis: {axis}")
    # Get joint limits from PyBullet
    pb_info = pb.getJointInfo(env.robot_id, env.joint_index.get(joint_name, -1))
    if pb_info[1].decode('UTF-8') == joint_name:
        lower_limit, upper_limit = pb_info[8], pb_info[9]
        print(f"  Limits: ({lower_limit:.3f}, {upper_limit:.3f}) radians")
    print()

# === Print IK chain information ===
print("\n=== IK Chain Information (from ikpy) ===")
for i, link in enumerate(ik_chain.links):
    print(f"Link {i}: {link.name}")
    if link.name == "Base link" or isinstance(link, ikpy.chain.OriginLink):
        print(f"  Type: OriginLink (base, no transformation)")
        print(f"  Joint Type: {link.joint_type}")
        print(f"  Bounds: {link.bounds}")
    else:
        print(f"  Translation: {link.translation}")
        print(f"  Rotation Axis: {link.rotation}")
        print(f"  Joint Type: {link.joint_type}")
        print(f"  Bounds: {link.bounds}")
    print()

# === Compare joint names ===
print("\n=== Joint Name Comparison ===")
sim_joint_names = [info[0] for info in joint_info]
ik_joint_names = [link.name for link in ik_chain.links if link.name]
print(f"Simulation joint names: {sim_joint_names}")
print(f"IK chain joint names: {ik_joint_names}")
missing_in_ik = set(sim_joint_names) - set(ik_joint_names)
missing_in_sim = set(ik_joint_names) - set(sim_joint_names)
print(f"Joint names missing in IK chain: {missing_in_ik}")
print(f"Joint names missing in simulation: {missing_in_sim}")

# === Cleanup ===
os.unlink(patched_urdf_path)  # Remove temporary URDF file
env.close()