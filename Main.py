import numpy as np
import matplotlib.pyplot as pt
from simulation import SimulationEnvironment
import evaluation as ev
from RRT_star import RRTStarPlanner


class ExampleController:
    def __init__(self):
        # Uses sampling-based RRT for motion planning
        pass

    def run(self, env, goal_poses):
        # run the controller in the environment to achieve the goal
        # using RRT* planner to generate the trajectory

        # get starting angles for trajectory
        init_angles = env.get_current_angles()
        trajectory = []

        # Assuming goal_poses is a dictionary, we take the first goal pose
        # Extract the position of the first goal from goal_poses
        goal_config = list(goal_poses.values())[0][0]

        # Set up obstacles and environment
        obstacles = [(5, 5, 1), (7, 7, 1)]  # You can adjust this based on your actual environment setup
        random_area = [-np.pi, np.pi]  # Joint space limits
        stepsize = 0.05  # Step size for RRT*
        sample_rate = 5  # 5% goal-biased sampling

        # Initialize the RRT* Planner
        planner = RRTStarPlanner(init_angles, goal_config, obstacles, random_area, stepsize, sample_rate, env)
        
        # Generate the path using the RRT* planner
        path = planner.planning()

        # Runs the trajectory
        duration = 5.  # duration of each waypoint transition (in seconds)
        for waypoint in path:
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
    tower_poses = ev.get_tower_base_poses(half_spots=3)

    # add a block right in front of the robot
    # _add_block is a private method, you can use it here for testing
    # but your controller should not add blocks during evaluation
    loc, quat = tower_poses[1]
    loc = loc[:2] + (.011,)  # increase z coordinate above floor with slight gap
    label = env._add_block(loc, quat, side=.02)  # .02 is cube side length
    env.settle(1.)  # run simulation for 1 second to let the block settle on the floor

    # you can get a synthetic camera image if you want to use it (not required)
    rgba, _, _ = env.get_camera_image()
    pt.imshow(rgba)
    pt.show()

    # a validation trial will have a dict of goal poses, one for each block
    # setup a goal one spot to the left
    loc, quat = tower_poses[0]
    loc = loc[:2] + (.01,)  # increase z coordinate above floor
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
