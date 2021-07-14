import random
import numpy as np
import scipy
from scipy import spatial
import yaml
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import rospy
import sensor_msgs.msg
import rospkg
import argparse
from argparse import RawTextHelpFormatter

import matplotlib.pyplot as plt
from utils import BodyPart

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

MODEL_NAME = "upper_body"


class Space:
    def __init__(self, n_particles, body_part):
        self.n_particles = n_particles
        self.body_part = body_part

        self.joint_names = [self.body_part + "_axis" + str(i) for i in range(3)]
        if self.body_part == BodyPart.SHOULDER_LEFT:
            self.joint_names += ["elbow_left_axis0"]
        elif self.body_part == BodyPart.SHOULDER_RIGHT:
            self.joint_names += ["elbow_right_axis0"]

        self.n_axis = len(self.joint_names)

        self.particles_visited = np.zeros((n_particles, 4), dtype=np.float32)
        self.best_particle = 0
        self.gbest_value = int(1000)
        self.gbest_position = np.array([0, 0, 0, 0], dtype=np.float32)
        self.pbest_value = np.ones(n_particles) * int(1000)
        self.pbest_position = np.zeros((n_particles, 4), dtype=np.float32)
        self.vel = np.random.rand(n_particles, 4) * 0.01
        self.vel_towards_global_best = np.random.rand(n_particles, 4) * 0.01
        self.vel_towards_personal_best = np.random.rand(n_particles, 4) * 0.1
        self.visited = np.array([[0, 0, 0, 0]], dtype=np.float32)
        self.visited_colored = np.array([[0, 0, 0, 0, float(255 << 24 | 255 < 16 | 255 < 8)]], dtype=np.float32)
        self.last_visited = np.array([0, 0, 0, 0], dtype=np.float32)
        self.neighbors = np.ones(n_particles) * float('inf')
        self.colors = np.zeros(n_particles, dtype=np.float32)
        self.minima = np.zeros((4, 1), dtype=np.float32)
        self.maxima = np.zeros((4, 1), dtype=np.float32)
        self.minima[2] = -0.6 # -0.1
        self.maxima[2] = 0.6 # 0.1
        self.minima[3] = 0.0
        self.maxima[3] = 1.5
        self.receiving_data = False

        for i in range(self.colors.size):
            self.colors[i] = float(
                random.randint(50, 255) << 16 | random.randint(50, 255) << 8 | random.randint(50, 255) << 0)

        self.joint_targets_pub = rospy.Publisher("/roboy/pinky/simulation/joint_targets", sensor_msgs.msg.JointState,
                                                 queue_size=1)
        self.joint_targets_msg = sensor_msgs.msg.JointState()
        self.joint_targets_msg.name = self.joint_names
        self.joint_targets_msg.velocity = [0] * len(self.joint_names)
        self.joint_targets_msg.effort = [0] * len(self.joint_names)

        self.last_target = np.array([0, 0, 0, 0], dtype=np.float32)

        self.rate = rospy.Rate(100)

        self.received_messages = 0
        self.global_attraction = 0.01
        self.personal_attraction = 0.1
        self.random_speed = 0.5

        robot_path = rospkg.RosPack().get_path('robots') + '/' + MODEL_NAME
        with open(robot_path + "/joint_limits.yaml", 'r') as stream:
            try:
                joint_limits = yaml.safe_load(stream)
                polygon = []
                for i in range(len(joint_limits[self.joint_names[0]])):
                    polygon.append((joint_limits[self.joint_names[0]][i], joint_limits[self.joint_names[1]][i]))
                    if joint_limits[self.joint_names[0]][i] < self.minima[0]:
                        self.minima[0] = joint_limits[self.joint_names[0]][i]
                    if joint_limits[self.joint_names[1]][i] < self.minima[1]:
                        self.minima[1] = joint_limits[self.joint_names[1]][i]
                    if joint_limits[self.joint_names[0]][i] > self.maxima[0]:
                        self.maxima[0] = joint_limits[self.joint_names[0]][i]
                    if joint_limits[self.joint_names[1]][i] > self.maxima[1]:
                        self.maxima[1] = joint_limits[self.joint_names[1]][i]

                self.joint_limits = Polygon(polygon)
            except yaml.YAMLError as exc:
                print(exc)

        self.particles = np.array([random.uniform(self.minima[0], self.maxima[0])[0],
                                   random.uniform(self.minima[1], self.maxima[1])[0],
                                   random.uniform(self.minima[2], self.maxima[2])[0],
                                   random.uniform(self.minima[3], self.maxima[3])[0]],
                                  dtype=np.float32)

        for i in range(self.n_particles - 1):
            self.particles = np.vstack([self.particles, np.array([random.uniform(self.minima[0], self.maxima[0])[0],
                                                                  random.uniform(self.minima[1], self.maxima[1])[0],
                                                                  random.uniform(self.minima[2], self.maxima[2])[0],
                                                                  random.uniform(self.minima[3], self.maxima[3])[0]],
                                                                 dtype=np.float32)])

    def fitness(self):
        if self.visited.shape[0] > 4:
            particles_tree = scipy.spatial.KDTree(self.particles)
            visited_tree = scipy.spatial.KDTree(self.visited)
            neighbors = particles_tree.query_ball_tree(visited_tree, 0.5)
            for i in range(len(neighbors)):
                self.neighbors[i] = len(neighbors[i])

    def move(self):
        for i in range(self.particles.shape[0]):
            random_movement = (np.random.rand(1, 4) - 0.5) * self.random_speed
            self.vel_towards_global_best[i] = (self.gbest_position - self.particles[i]) * self.global_attraction
            self.vel_towards_personal_best[i] = (self.pbest_position[i] - self.particles[i]) * self.personal_attraction
            if self.joint_limits.contains(Point(self.particles[i][0], self.particles[i][1])) and \
                    self.minima[2] < self.particles[i][2] < self.maxima[2]:
                self.vel[i] = self.vel_towards_global_best[i] + self.vel_towards_personal_best[i] + random_movement
                self.particles[i] = self.particles[i] + self.vel[i]
            else:
                self.particles[i] = np.array([random.uniform(self.minima[0], self.maxima[0])[0],
                                              random.uniform(self.minima[1], self.maxima[1])[0],
                                              random.uniform(self.minima[2], self.maxima[2])[0],
                                              random.uniform(self.minima[3], self.maxima[3])[0]], dtype=np.float32)

    def set_pbest(self):
        for particle in range(self.particles.shape[0]):
            if self.neighbors[particle] <= self.pbest_value[particle]:
                self.pbest_value[particle] = self.neighbors[particle]
                self.pbest_position[particle] = self.particles[particle]

    def set_gbest(self):
        new_global_best = False
        for particle in range(self.particles.shape[0]):
            if (self.neighbors[particle] <= self.gbest_value) and self.best_particle != particle:
                self.gbest_value = self.neighbors[particle]
                self.gbest_position = self.particles[particle]
                self.best_particle = particle
                new_global_best = True
        if not new_global_best:
            random_particle = random.randint(0, self.n_particles - 1)
            self.best_particle = random_particle
            self.gbest_value = self.neighbors[random_particle]
            self.gbest_position = self.particles[random_particle]
            rospy.loginfo_throttle(1, "choosing random particle %d with %f neighbors at (%f %f %f %f)" % (
                random_particle, self.gbest_value,
                self.gbest_position[0], self.gbest_position[1],
                self.gbest_position[2], self.gbest_position[3]))
        else:
            rospy.loginfo_throttle(1, "choosing particle %d with %f neighbors at (%f %f %f %f)" % (
                self.best_particle, self.gbest_value,
                self.gbest_position[0], self.gbest_position[1],
                self.gbest_position[2], self.gbest_position[3]))

        target = np.array(
            [self.gbest_position[0], self.gbest_position[1], self.gbest_position[2], self.gbest_position[3]])
        target_dist = np.linalg.norm(self.last_target - target)
        n_sample = int(target_dist * 200)
        lin_target = np.linspace(self.last_target, target, n_sample)

        for t in lin_target:
            self.joint_targets_msg.position = t[:self.n_axis]
            self.joint_targets_pub.publish(self.joint_targets_msg)
            self.rate.sleep()

        self.last_target = target
        self.visited = np.vstack([self.visited, target])
        self.visited_colored = np.vstack([self.visited_colored, np.hstack([target, [self.colors[self.best_particle]]])])

    def run(self):
        self.receiving_data = True
        search_space.move()
        search_space.fitness()
        search_space.set_pbest()
        search_space.set_gbest()

        x_input = self.visited_colored[:, 0]
        y_input = self.visited_colored[:, 1]
        z_input = self.visited_colored[:, 2]
        z_points = self.visited_colored[:, -1]

        ax.scatter3D(x_input, y_input, z_input, c=z_points, cmap='hsv')
        plt.draw()
        plt.pause(0.02)
        ax.cla()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Particle swarm, use python particle_swarm.py -h for more information',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--body_part', type=BodyPart, choices=list(BodyPart),
                        help='Body part: head, shoulder_left, shoulder_right, hand_left, hand_right', required=True)
    parser.add_argument('--n_neighbors', type=int,
                        help='Number of neighbors', default=20)
    args = parser.parse_args()

    rospy.init_node(args.body_part + '_particle_swarm')
    search_space = Space(args.n_neighbors, args.body_part)
    while not rospy.is_shutdown():
        search_space.run()
