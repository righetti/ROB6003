import numpy as np
import pinocchio as pin
import os
import time
import meshcat
import sys

from pinocchio.robot_wrapper import RobotWrapper

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess


class PinRobotEnvMeshcat:
    def __init__(self):
        self._robots = []
        self.dt = 0.001

    def add_robot(self, robot):
        self._robots.append(robot)

    def step(self, sleep=False):
        for robot in self._robots:
            robot.step(self.dt)

class PinHeadRobot:
    def __init__(self, pin_env, pin_robot, viewer):
        self.pin_env = pin_env
        self.pin_robot = pin_robot 

        self._model = pin_robot.model
        self._data = pin_robot.data 

        # Setup the visualizer
        self.viz = viz = pin.visualize.MeshcatVisualizer(
            pin_robot.model, pin_robot.collision_model, pin_robot.visual_model
        )
        viz.initViewer(viewer)
        viz.loadViewerModel()
        q0 = pin.neutral(self._model)
        viz.display(q0)

        # Robot data
        self._q = np.zeros(self._model.nq)
        self._dq = np.zeros(self._model.nv)

        self.useFixedBase = self._model.nq == self._model.nv 

    def step(self, dt):
        ddq = pin.aba(self._model, self._data, self._q, self._dq, self._last_tau)

        dqMean = self._dq + self._data.ddq * .5 * dt
        self._q = pin.integrate(self._model, self._q, dqMean * dt);
        self._dq += self._data.ddq * dt;

        self.viz.display(self._q)

    def send_joint_command(self, tau):
        self._last_tau = tau

    def reset_state(self, q, dq):
        self._q = q.copy()
        self._dq = dq.copy()

        self.viz.display(self._q)

    def get_state(self):
        return self._q.copy(), self._dq.copy()


def start_robot_visualizer():
    server_args = []
    if 'google.colab' in sys.modules:
        server_args = ['--ngrok_http_tunnel']
        package_dirs = '/content/ROB6003/Project2/urdf/'
    else:
        package_dirs = './urdf/'

    from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
    proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=server_args)

    viewer = meshcat.Visualizer(zmq_url=zmq_url)

    # Build the simulator.
    sim_env = PinRobotEnvMeshcat()
    
    # Create a robot instance. This adds the robot to the simulator as well.
    urdf_file = 'iiwa.urdf'
    global END_EFF_FRAME_ID
    END_EFF_FRAME_ID = 17

    urdf = package_dirs + urdf_file
    pin_robot = pin.RobotWrapper.BuildFromURDF(urdf, package_dirs)

    global robot
    robot = PinHeadRobot(sim_env, pin_robot, viewer)

    robot.viz.viewer['ball'].set_object(meshcat.geometry.Sphere(0.05), 
                              meshcat.geometry.MeshLambertMaterial(
                             color=0xff22dd,
                             reflectivity=0.8))
    robot.viz.viewer['ball2'].set_object(meshcat.geometry.Sphere(0.05), 
                              meshcat.geometry.MeshLambertMaterial(
                             color=0xffdd22,
                             reflectivity=0.8))
    robot.viz.viewer['target_frame'].set_object(meshcat.geometry.triad())

    # Add the robot to the bullet_env to update the internal structure of the robot
    # ate every simulation steps.
    sim_env.add_robot(robot)

    print('You should see the Kuka iiwa robot now when going to this page:', web_url)


def display_robot(q):
    robot.viz.display(q)

def display_ball(pos):
    robot.viz.viewer['ball'].set_transform(meshcat.transformations.translation_matrix(pos))

def display_ball2(pos):
    robot.viz.viewer['ball2'].set_transform(meshcat.transformations.translation_matrix(pos))
    
    
def simulate_robot(robot_controller, T=10.):
    t = 0.
    dt = 0.001
    q = np.zeros([7,1])
    dq = np.zeros([7,1])
    display_robot(q)
    t_visual = 0
    dt_visual = 0.01
    while(t<T):
        M = robot.pin_robot.mass(q)
        nle = robot.pin_robot.nle(q, dq)
        tau = robot_controller(t, q.reshape((7,1)), dq.reshape((7,1)))
        g = robot.pin_robot.gravity(q.reshape((7,1)))
        ddq = np.linalg.inv(M) @ (tau - nle.reshape((7,1)) + g.reshape((7,1)))
        dq += dt * ddq
        q += dt*dq
        if t_visual == 10:
            display_robot(q)
            time.sleep(dt_visual)
            t_visual = 0
        t_visual += 1
        t += dt
