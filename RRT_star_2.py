import numpy as np
import pybullet as pb
from simulation import SimulationEnvironment

class Node:
    def __init__(self, q):
        self.q = q
        self.path_q = []
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, joint_limits, expand_dis=0.2, path_resolution=0.05,
                 goal_sample_rate=10, max_iter=500, connect_circle_dist=1.0):
        self.start = Node(start)
        self.end = Node(goal)
        self.joint_limits = joint_limits
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist
        self.node_list = []

    def planning(self, env):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node)

            if not self.check_collision(env, new_node):
                continue

            near_inds = self.find_near_nodes(new_node)
            new_node = self.choose_parent(env, new_node, near_inds)

            if new_node:
                self.node_list.append(new_node)
                self.rewire(env, new_node, near_inds)

            if self.calc_dist_to_goal(new_node.q) <= self.expand_dis:
                final_node = self.steer(new_node, self.end)
                if self.check_collision(env, final_node):
                    return self.generate_final_course(final_node)

        return None

    def steer(self, from_node, to_node):
        from_q = np.array(from_node.q)
        to_q = np.array(to_node.q)
        dist = np.linalg.norm(to_q - from_q)
        if dist > self.expand_dis:
            to_q = from_q + (to_q - from_q) / dist * self.expand_dis
        new_q = np.clip(to_q, [lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits])
        new_node = Node(new_q.tolist())
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.calc_dist(from_node.q, new_q.tolist())
        return new_node

    def get_random_node(self):
        if np.random.randint(0, 100) > self.goal_sample_rate:
            q = [np.random.uniform(*lim) for lim in self.joint_limits]
        else:
            q = self.end.q
        return Node(q)

    def get_nearest_node_index(self, rnd_node):
        return np.argmin([self.calc_dist(node.q, rnd_node.q) for node in self.node_list])

    def find_near_nodes(self, new_node):
        n = len(self.node_list)
        r = self.connect_circle_dist * np.sqrt((np.log(n) / n))
        dlist = [self.calc_dist(node.q, new_node.q) for node in self.node_list]
        return [i for i, d in enumerate(dlist) if d <= r]

    def choose_parent(self, env, new_node, near_inds):
        if not near_inds:
            return new_node if self.check_collision(env, new_node) else None
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            temp_node = self.steer(near_node, new_node)
            if self.check_collision(env, temp_node):
                costs.append(near_node.cost + self.calc_dist(near_node.q, temp_node.q))
            else:
                costs.append(float('inf'))

        min_cost = min(costs)
        if min_cost == float('inf'):
            return None
        best_ind = near_inds[np.argmin(costs)]
        best_node = self.steer(self.node_list[best_ind], new_node)
        best_node.cost = min_cost
        return best_node

    def rewire(self, env, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            temp_node = self.steer(new_node, near_node)
            if not self.check_collision(env, temp_node):
                continue
            cost = new_node.cost + self.calc_dist(new_node.q, near_node.q)
            if cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = cost

    def generate_final_course(self, goal_node):
        path = []
        node = goal_node
        while node:
            path.append(node.q)
            node = node.parent
        return path[::-1]

    def calc_dist(self, q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))

    def calc_dist_to_goal(self, q):
        return self.calc_dist(q, self.end.q)

    def check_collision(self, env, q_node):
        env.goto_position(env._angle_dict_from(q_node.q, convert=False), duration=0.01)
        return len(pb.getContactPoints(bodyA=env.robot_id)) == 0


class Controller:
    def __init__(self):
        self.joint_limits = [(-3.14, 3.14)] * 6

    def run(self, env, goal_poses):
        start = list(env._get_position())
        for label, (goal_loc, _) in goal_poses.items():
            goal = start.copy()
            goal[0] -= 0.5  # example adjustment to reach goal area

            planner = RRTStar(start, goal, self.joint_limits)
            path = planner.planning(env)

            if path:
                print(f"Path found for {label}, executing...")
                for q in path:
                    env.goto_position(env._angle_dict_from(q, convert=False), duration=0.2)
                env.settle(1.0)
            else:
                print(f"No path found for {label}")
