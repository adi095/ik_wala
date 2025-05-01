"""
RRT controller to illustrate API methods and conventions
"""
import numpy as np
import matplotlib.pyplot as pt
from simulation import SimulationEnvironment
import evaluation as ev

class ExampleController:
    def __init__(self):
        # no optimized model data in this example
        pass

    def run(self, env, goal_poses):
        # run the controller in the environment to achieve the goal
        # this example ignores goal_poses
        # just follows a hand-coded trajectory for sake of example

        # get starting angles for trajectory
        init_angles = env.get_current_angles()
        trajectory = []

        # move arm down around block
        stage_angles = dict(init_angles)
        stage_angles["m2"] = 20. # second motor angle (degrees)
        stage_angles["m3"] = 35. # second motor angle (degrees)
        trajectory.append(stage_angles)

        # close the gripper (sixth motor)
        close_angles = dict(stage_angles)
        close_angles["m6"] = -20
        trajectory.append(close_angles)

        # lift arm
        lift_angles = dict(close_angles)
        lift_angles["m3"] = 20
        trajectory.append(lift_angles)

        # rotate arm
        rotate_angles = dict(lift_angles)
        rotate_angles["m1"] = -33
        trajectory.append(rotate_angles)

        # lower arm
        lower_angles = dict(rotate_angles)
        lower_angles["m3"] = 35
        trajectory.append(lower_angles)

        # release and lift arm
        release_angles = dict(lower_angles)
        release_angles["m3"] = 20
        release_angles["m6"] = 0
        trajectory.append(release_angles)

        # runs the trajectory
        duration = 5. # duration of each waypoint transition (in seconds)
        for waypoint in trajectory:
            env.goto_position(waypoint, duration)

if __name__ == "__main__":

    # initialize controller class
    controller = ExampleController()

    # launch environment, show=True visualizes the simulation
    # show=False is substantially faster for when you are training/testing
    env = SimulationEnvironment(show=True)

    # joint info is available for you to program forward/inverse kinematics
    joint_info = env.get_joint_info()
    for info in joint_info: print(info)
    input("[Enter] to continue ...")    

    # get the tower base positions used for evaluation
    # there are two concentric rings of towers
    # each ring has half_spots many towers
    # this example uses 2*3 towers, but real evaluation will have more
    tower_poses = ev.get_tower_base_poses(half_spots = 3)

    # add a block right in front of the robot
    # _add_block is a private method, you can you can use it here for testing
    # but your controller should not add blocks during evaluation
    loc, quat = tower_poses[1]
    loc = loc[:2] + (.011,) # increase z coordinate above floor with slight gap
    label = env._add_block(loc, quat, side=.02) # .02 is cube side length
    env.settle(1.) # run simulation for 1 second to let the block settle on the floor

    # you can get a synthetic camera image if you want to use it (not required)
    rgba, _, _ = env.get_camera_image()
    pt.imshow(rgba)
    pt.show()

    # a validation trial will have a dict of goal poses, one for each block
    # setup a goal one spot to the left
    loc, quat = tower_poses[0]
    loc = loc[:2] + (.01,) # increase z coordinate above floor
    goal_poses = {label: (loc, quat)}

    # run the controller on the trial
    controller.run(env, goal_poses)

    # evaluation metrics
    accuracy, loc_errors, rot_errors = ev.evaluate(env, goal_poses)

    # close any environment you instantiate to avoid memory leaks
    env.close()

    # display the metrics
    print(f"\n{int(100*accuracy)}% of blocks near correct goal positions")
    print(f"mean|max location error = {np.mean(loc_errors):.3f}|{np.max(loc_errors):.3f}")
    print(f"mean|max rotation error = {np.mean(rot_errors):.3f}|{np.max(rot_errors):.3f}")

    input("[Enter] to continue ...")

    # real evaluation should use randomly sampled trials
    # hand-coded trajectory will not work on these

    # sample a validation trial
    env, goal_poses = ev.sample_trial(num_blocks=5, num_swaps=1, show=True)

    # run the controller on the trial
    # copies goal_poses in case your controller modifies it (but you shouldn't)
    controller.run(env, dict(goal_poses))

    # evaluate success
    accuracy, loc_errors, rot_errors = ev.evaluate(env, goal_poses)

    env.close()

    print(f"\n{int(100*accuracy)}% of blocks near correct goal positions")
    print(f"mean|max location error = {np.mean(loc_errors):.3f}|{np.max(loc_errors):.3f}")
    print(f"mean|max rotation error = {np.mean(rot_errors):.3f}|{np.max(rot_errors):.3f}")

import random
import math
import numpy as np
import copy
from simulation import SimulationEnvironment

class Node:
    def __init__(self, config):
        self.config = config  # Joint angles (configurations)
        self.cost = 0.0  # Cost to reach this node
        self.parent = None  # Parent node

class RRTStarPlanner:
    def __init__(self, start_config, goal_config, obstacles, random_area, stepsize, sample_rate, env):
        self.start = Node(start_config)
        self.end = Node(goal_config)
        self.area_min = random_area[0]
        self.area_max = random_area[1]
        self.stepsize = stepsize
        self.sample_rate = sample_rate
        self.obstacle = obstacles
        self.tree_nodes = [self.start]
        self.env = env

    def planning(self):
        while True:
            random_point = self.get_random_point()

            # Find nearest node in the tree
            nearest_node_index = self.nearest_node(random_point)
            nearest_node = self.tree_nodes[nearest_node_index]
            new_node = self.steer(nearest_node, random_point)

            if self.is_collision_free(new_node):
                self.tree_nodes.append(new_node)
                # Check if the goal is reached
                if self.reached_goal(new_node):
                    path = self.reconstruct_path(new_node)
                    return path  # Return the final path

    def get_random_point(self):
        """ Sample random points in the joint space or goal-biased sampling """
        if random.randint(0, 100) > self.sample_rate:
            random_point = [random.uniform(self.area_min, self.area_max) for _ in range(len(self.start.config))]
        else:
            random_point = self.end.config  # Bias towards goal
        return random_point

    def nearest_node(self, random_point):
        """ Find the nearest node to the random sample """
        dlist = [np.linalg.norm(np.array(node.config) - np.array(random_point)) for node in self.tree_nodes]
        return dlist.index(min(dlist))  # Return the index of the nearest node

    def steer(self, nearest_node, random_point):
        """ Steer from the nearest node towards the random point """
        direction = np.array(random_point) - np.array(nearest_node.config)
        distance = np.linalg.norm(direction)
        direction = direction / distance if distance > 0 else direction
        new_config = np.array(nearest_node.config) + direction * self.stepsize
        new_node = Node(new_config)
        new_node.parent = self.tree_nodes.index(nearest_node)
        new_node.cost = nearest_node.cost + self.stepsize
        return new_node

    def is_collision_free(self, node):
        """ Check if the new node is collision-free """
        # You should use the `env` to check if the configuration leads to a collision
        return self.env.check_collision(node.config)  # Modify this to use PyBullet for collision checking

    def reached_goal(self, node):
        """ Check if the node is within a threshold distance from the goal """
        return np.linalg.norm(np.array(node.config) - np.array(self.end.config)) <= self.stepsize

    def reconstruct_path(self, node):
        """ Reconstruct the path from the goal to the start by following the parent nodes """
        path = [node.config]
        while node.parent is not None:
            node = self.tree_nodes[node.parent]
            path.append(node.config)
        path.reverse()
        return path

# Usage example:
if __name__ == "__main__":
    env = SimulationEnvironment(show=True)
    start_config = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Example joint angles (start)
    goal_config = [np.pi/4, -np.pi/4, np.pi/6, -np.pi/6, 0.0, 0.0]  # Example joint angles (goal)
    obstacles = [(5, 5, 1), (7, 7, 1)]  # Obstacles in the workspace (adjust this)
    random_area = [-np.pi, np.pi]  # Joint space limits
    stepsize = 0.05
    sample_rate = 5  # 5% goal-biased sampling
    
    planner = RRTStarPlanner(start_config, goal_config, obstacles, random_area, stepsize, sample_rate, env)
    path = planner.planning()

    # Visualize or execute the path with your environment
    print("Path found:", path)
    # Execute the path in the simulation environment
    for config in path:
        env.goto_position(config)
