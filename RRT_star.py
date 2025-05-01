import random
import numpy as np

# Node class to store the configuration, cost, and parent
class Node:
    def __init__(self, config):
        self.config = config  # Joint angles (configurations)
        self.cost = 0.0  # Cost to reach this node
        self.parent = None  # Parent node


# RRT* Planner class
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
