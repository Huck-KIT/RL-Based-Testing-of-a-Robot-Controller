import os
import pathlib
import subprocess
import time
from datetime import datetime

import numpy as np
from mpc_planner.path_advisor.local_path_plan import LocalPathPlanner
from mpc_planner.test_maps.mmc_graph_agv import Graph
from mpc_planner.trajectory_generator import TrajectoryGenerator
from mpc_planner.util.config import Configurator

from util import utils_geo
from zmqRemoteApi import RemoteAPIClient

# ###################
# ### PLEASE EDIT ###
# ###################
SCENARIO = 1  # crossing = 1, head-on = 2, overtaking = 3  # PLEASE EDIT
LASER_FIELD_CONFIG = 1  # 0 = none, 1 = small fields, 2 = big fields # PLEASE EDIT
PORT = 23100  # PLEASE EDIT
REPORT_NUM = 13  # PLEASE EDIT
RANDOM_SEED = 3  # PLEASE EDIT

REPORT_NAME = "report" + str(REPORT_NUM) + ".csv"
CURRENT_REPORT_NAME = "current_" + REPORT_NAME
CONFIG_FN = 'mpc_default.yaml'
BUILD_MPC = False
RENDER = True
SPAWN = True
PATH_TO_COPPELIASIM = '/home/kaiser_uyjg/CoppeliaSim/' #'C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/'
PATH_TO_SCENEFOLDER = '/home/kaiser_uyjg/multiagent-falsification/Scenario_Tom/MRK/'

MAX_STEPS_PER_EPISODE = 15
WORKER_SUB_STEPS = 10  # sets how many steps the worker can perform before he gets new values from the rl-algorithm
CRITICAL_DISTANCE = 0.3  # sets when to log an episode
MAX_AGV_RETARDATION = -7.5

class CoppeliaSim:
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, render, spawn, port_num=23000):
        subprocess.run(["xset", "-dpms"])
        if SCENARIO == 1:
            scenario_folder = "Crossing_Baseline/"
            if LASER_FIELD_CONFIG == 0:
                scenario_file = "crossing_baseline_none.ttt"
            elif LASER_FIELD_CONFIG == 1:
                scenario_file = "crossing_baseline_1m.ttt"
            elif LASER_FIELD_CONFIG == 2:
                scenario_file = "crossing_baseline_2m.ttt"
            else:
                sys.exit("Invalid laser field parameter")
        elif SCENARIO == 2:
            scenario_folder = "Head-on_Baseline/"
            if LASER_FIELD_CONFIG == 0:
                scenario_file = "head-on_baseline_none.ttt"
            elif LASER_FIELD_CONFIG == 1:
                scenario_file = "head-on_baseline_1m.ttt"
            elif LASER_FIELD_CONFIG == 2:
                scenario_file = "head-on_baseline_2m.ttt"
            else:
                sys.exit("Invalid laser field parameter")
        elif SCENARIO == 3:
            scenario_folder = "Overtaking_Baseline/"
            if LASER_FIELD_CONFIG == 0:
                scenario_file = "overtaking_baseline_none.ttt"
            elif LASER_FIELD_CONFIG == 1:
                scenario_file = "overtaking_baseline_1m.ttt"
            elif LASER_FIELD_CONFIG == 2:
                scenario_file = "overtaking_baseline_2m.ttt"
            else:
                sys.exit("Invalid laser field parameter")
        else:
            sys.exit("Invalid Scenario#")

        path_to_scene = PATH_TO_SCENEFOLDER + scenario_folder + scenario_file

        if spawn:
            if render:
                subprocess.Popen([PATH_TO_COPPELIASIM + 'coppeliaSim.sh', '-vscripterrors',
                                  '-GzmqRemoteApi.rpcPort=' + str(port_num), path_to_scene])
            else:
                subprocess.Popen([PATH_TO_COPPELIASIM + 'coppeliaSim.sh', '-vscripterrors', '-h',
                                  '-GzmqRemoteApi.rpcPort=' + str(port_num), path_to_scene])
            time.sleep(5)  # wait for CoppeliaSim to start up
            self._client = RemoteAPIClient(port=port_num)
        else:
            self._client = RemoteAPIClient()
        self._sim = self._client.getObject('sim')
        self._client.setStepping(True)
        self._sim.startSimulation()
        self._sim.setInt32Param(self._sim.intparam_speedmodifier, 64)
        time.sleep(2)  # wait for CoppeliaSim to process

        workerHandle = self._sim.getObject("/worker")
        self._workerScriptHandle = self._sim.getScript(self._sim.scripttype_childscript, workerHandle)
        agvsHandle = self._sim.getObject("/AGVs")
        self._agvsScriptHandle = self._sim.getScript(self._sim.scripttype_childscript, agvsHandle)
        obstaclesHandle = self._sim.getObject("/Obstacles")
        self._obstaclesScriptHandle = self._sim.getScript(self._sim.scripttype_childscript, obstaclesHandle)

        self._client.step()  # initial "warm up" phase
        self._sim.callScriptFunction("setAGVretardation@worker", self._workerScriptHandle, MAX_AGV_RETARDATION)
        self._sim.callScriptFunction("setRandomSeed@worker", self._workerScriptHandle, RANDOM_SEED)
        self.waypoints = self._sim.callScriptFunction("getAGVWps@AGVs", self._agvsScriptHandle, 1)
        self.obstacles = self._sim.callScriptFunction("getObstacles@obstacles", self._obstaclesScriptHandle)
        self.agv = AGVDT(CONFIG_FN, self._sim.callScriptFunction("getAGVstate@worker", self._workerScriptHandle, 1),
                         self.obstacles, self.waypoints)
        self.worker = WorkerDS(CONFIG_FN,
                               self._sim.callScriptFunction("getWorkerState@worker", self._workerScriptHandle))
        self.sampling_time = self.agv.config.ts
        self.simStepsPerControllerStep = int(self.sampling_time / round(self._sim.getSimulationTimeStep(), 2))
        self.episode_count = 0
        self.last_saved_report = 0
        self.steps_this_episode = 0
        self.total_steps_taken = 0
        self.emergency_steps_taken = 0
        self.save_report = False
        with open(REPORT_NAME, "w") as rep:
            rep.write("Timestamp" + ", " +
                      "Episode#" + ", " +
                      "Eps since last event" + ", " +
                      "Worker x-pos" + ", " +
                      "Worker y-pos" + ", " +
                      "Worker heading" + ", " +
                      "Worker visible" + ", " +
                      "Worker velo" + ", " +
                      "AGV x-pos" + ", " +
                      "AGV y-pos" + ", " +
                      "AGV heading" + ", " +
                      "AGV velo" + ", " +
                      "Is emergency stop" + ", " +
                      "Is collision" + ", " +
                      "Collision direction left/right/front/rear" + ", " +
                      "Distance" + ", " +
                      "Reward" + ", " +
                      "Reward was passed to rl" + "\n")

    def step(self):
        is_done = False
        step_reward = 0
        self.total_steps_taken += 1
        print("Step number: ", self.total_steps_taken)
        # perform some steps with the given velo and heading from RL to reduce train-load
        for i in range(WORKER_SUB_STEPS):
            self.worker.set_state(self._sim.callScriptFunction("getWorkerState@worker", self._workerScriptHandle),
                                  self._sim.callScriptFunction("getWorkerVelo@worker", self._workerScriptHandle),
                                  (not self._sim.callScriptFunction("isCovered@worker", self._workerScriptHandle)))
            self.agv.set_state(self._sim.callScriptFunction("getAGVstate@worker", self._workerScriptHandle),
                               self._sim.callScriptFunction("getAGVvelo@worker", self._workerScriptHandle))

            if (self._sim.callScriptFunction("isCollision@worker", self._workerScriptHandle)
                    or self._sim.callScriptFunction("isContact@worker", self._workerScriptHandle)):
                # collision or contact occurred, grab reward and end the episode
                is_done = True
            elif (self._sim.callScriptFunction("isEmergencyStop@worker", self._workerScriptHandle)
                    or
                    self._sim.callScriptFunction("isPerformingEmergStop@worker", self._workerScriptHandle)):
                # emergency stop situation
                # track how man steps were done while having an emergency stop situation to prevent deadlocks
                self.emergency_steps_taken += 1
                if self.emergency_steps_taken > 5*WORKER_SUB_STEPS:
                    # simulation is stuck, reset it
                    is_done = True
            else:
                # not an emergency stop situation
                if self.emergency_steps_taken > 0:
                    # continue after an emergency stop
                    self.emergency_steps_taken = 0
                    # prepare new path to next waypoint
                    if not self.agv.preparePath(waypoints=self.waypoints, waypoint_reached=False):
                        # path planning failed, cancel the episode
                        is_done = True
                        break
                is_done = is_done or self.agv.policy(self._sim.getSimulationTime(), self.waypoints,
                                                     self.worker.predict_path())
                self._sim.callScriptFunction("move@AGVs", self._agvsScriptHandle,
                                             self.agv.next_action[0], self.agv.next_action[1], 1)

            if self._sim.callScriptFunction("getDistToAGV@worker", self._workerScriptHandle) < CRITICAL_DISTANCE:
                # distance fell below critical value, save the report
                self.save_report = True

            current_reward = self.calc_reward()
            step_reward += current_reward
            # log only the current reward
            self.report(current_reward, was_passed_to_rl=0)

            self._client.step()

            if is_done:
                break

        # check if the episode exceeded steps allowed per episode
        self.steps_this_episode += 1
        if self.steps_this_episode > MAX_STEPS_PER_EPISODE:
            is_done = True

        # log the accumulated reward that was acquired during the loop
        self.report(step_reward, was_passed_to_rl=1)
        return is_done

    def reset(self):
        # save the current report if it's relevant
        if self.save_report:
            self.save_current_report()
            self.last_saved_report = self.episode_count
            self.save_report = False

        # create new current report
        open(CURRENT_REPORT_NAME, 'w').close()
        self.emergency_steps_taken = 0
        self.steps_this_episode = 0
        self.episode_count += 1
        self.print_cyan("Episode number: " + str(self.episode_count))
        self._sim.callScriptFunction("resetAll@worker", self._workerScriptHandle)
        self.agv.set_state(self._sim.callScriptFunction("getAGVstate@worker", self._workerScriptHandle),
                           self._sim.callScriptFunction("getAGVvelo@worker", self._workerScriptHandle))
        self.agv.preparePath(waypoints=self.waypoints, waypoint_reached=True)
        return np.array(self._get_sim_state(), dtype=np.float32)  # reward, done, info can't be included

    def _get_sim_state(self):
        return (self._sim.callScriptFunction("getWorkerState@worker", self._workerScriptHandle) +
                [int(not self._sim.callScriptFunction("isCovered@worker", self._workerScriptHandle))] +
                [self._sim.callScriptFunction("getWorkerVelo@worker", self._workerScriptHandle)] +
                self._sim.callScriptFunction("getAGVstate@worker", self._workerScriptHandle) +
                [self._sim.callScriptFunction("getAGVvelo@worker", self._workerScriptHandle)] +
                [int(self._sim.callScriptFunction("isEmergencyStop@worker", self._workerScriptHandle))] +
                [int(self._sim.callScriptFunction("isCollision@worker", self._workerScriptHandle))])

    def calc_reward(self):
        is_collision = self._sim.callScriptFunction("isCollision@worker", self._workerScriptHandle)
        current_distance = self._sim.callScriptFunction("getDistToAGV@worker", self._workerScriptHandle)
        current_agv_velo = self._sim.callScriptFunction("getAGVvelo@worker", self._workerScriptHandle)
        # if current_distance > 0:
        if not is_collision:
            if ((current_agv_velo > 0.1)
                    and
                    (not self._sim.callScriptFunction("isContact@worker", self._workerScriptHandle))):
                return np.exp(-current_distance)
            else:
                return 0
        else:
            if current_agv_velo > 0:
                current_worker_velo = self._sim.callScriptFunction("getWorkerVelo@worker", self._workerScriptHandle)
                return 100 * (1 + (current_agv_velo * current_worker_velo) / (1.2 * 1.5))
            else:
                return 0

    def report(self, reward, was_passed_to_rl):
        with open(CURRENT_REPORT_NAME, "a") as rep:
            rep.write(str(datetime.now()) + ", ")
            rep.write(str(self.episode_count) + ", ")
            rep.write(str(self.episode_count - self.last_saved_report) + ", ")
            sim_state = self._get_sim_state()
            rep.write(", ".join(str(value) for value in sim_state) + ", ")
            rep.write("".join(str(x) for x in self._sim.callScriptFunction("collisionDirection@worker",
                                                                           self._workerScriptHandle)) + ", ")
            rep.write(str(self._sim.callScriptFunction("getDistToAGV@worker", self._workerScriptHandle)) + ", ")
            rep.write(str(reward) + ", ")
            rep.write(str(was_passed_to_rl) + "\n")

    def save_current_report(self):
        with open(REPORT_NAME, "a") as all_rep:
            with open(CURRENT_REPORT_NAME, "r") as cur_rep:
                all_rep.write(cur_rep.read())

    def render(self, pred_state, mode="human"):
        self._sim.callScriptFunction("setAGVState@AGVs", self._agvsScriptHandle, pred_state[0][0], pred_state[0][1],
                                     pred_state[0][2], 2)

    def close(self):
        self._sim.stopSimulation()
        # self._sim.quitSimulator()

    @staticmethod
    def print_cyan(text):
        print("\033[96m {}\033[00m".format(text))


class AGVDT(object):

    def __init__(self, config_fn, state, static_obstacles, waypoints):
        yaml_fp = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', config_fn)
        self.config = Configurator(yaml_fp)

        graph = Graph(inflate_margin=(self.config.vehicle_width + 1.0 * self.config.vehicle_margin),
                      obstacles_sim=static_obstacles, scenario=SCENARIO)
        self._lpp = LocalPathPlanner(graph)

        self._traj_gen = TrajectoryGenerator(self.config, build=BUILD_MPC, use_tcp=False, verbose=True)
        self._traj_gen.set_obstacle_weights(1e4, 1e4)

        # handle static obstacles
        self._stc_constraints = [0.0] * self.config.Nstcobs * self.config.nstcobs  # num_of_stc_obs*vars_per_stc_obs

        for i, obstacle in enumerate(static_obstacles):
            b, a0, a1 = utils_geo.polygon_halfspace_representation(np.array(obstacle))
            self._stc_constraints[i * self.config.nstcobs: (i + 1) * self.config.nstcobs] = (b + a0 + a1)

        self.nextWP = 0  # AGV is supposed to start on the 0th waypoint
        self.idx = 0
        self.start_time_pause = np.inf
        self.emergencyStop = False
        self.mode = 'aligning'
        self.state = state
        self.velo = 0
        self.ang_velo = 0
        self.pred_state = state
        self.next_action = [0, 0]
        self.ref_path = []
        self.ref_traj = []
        self.preparePath(waypoints, waypoint_reached=True)

    def set_state(self, state, velo):
        self.ang_velo = (state[2] - self.state[2]) / self.config.ts
        self.state = state  # x-pos, y-pos, heading
        self._traj_gen.set_current_state(np.array([self.state[0], self.state[1], self.state[2]]))
        self.velo = velo

    # returns if the operation was successful, must cancel the episode if not
    def preparePath(self, waypoints, waypoint_reached=True):
        self.mode = 'aligning'
        self.idx = 0
        if waypoint_reached:
            self.nextWP = (self.nextWP + 1) % len(waypoints)
        try:
            self.ref_path = self._lpp.get_ref_path(self.state, waypoints[self.nextWP])
            self.ref_traj = self._lpp.get_ref_traj(self.config.ts, self.config.high_speed * self.config.lin_vel_max,
                                                   self.state)
            self._traj_gen.load_init_states(np.array([self.state[0], self.state[1], self.state[2]]),
                                            np.array([waypoints[self.nextWP][0], waypoints[self.nextWP][1],
                                                      waypoints[self.nextWP][2]]),
                                            np.array([waypoints[self.nextWP][0], waypoints[self.nextWP][1],
                                                      waypoints[self.nextWP][2]]))
            return True
        except (IndexError, ValueError):
            self.ref_path = []
            self.ref_traj = []
            return False

    # returns if the episode is done
    def policy(self, t, waypoints, dynObs):
        # print(self.idx)
        if self.idx >= len(self.ref_traj) - 1:
            # waypoint reached
            if waypoints[self.nextWP][3] and t < self.start_time_pause + 1:
                # handle pause point
                if self.start_time_pause == np.inf:
                    self.start_time_pause = t
                self.next_action = [0, 0]
                return False
            else:
                # plan path to next wp
                self.start_time_pause = np.inf
                if self.nextWP + 1 == len(waypoints):
                    # final waypoint reached
                    self.nextWP = 0
                    self.next_action = [0, 0]
                    return True
                # not the final waypoint, prepare path to next waypoint
                if not self.preparePath(waypoints, waypoint_reached=True):
                    # error occurred while planning trajectory, cancel the episode
                    return True

        [self.next_action], self.pred_state, self.idx, cost, refs, self.mode \
            = self._traj_gen.run_step(self.idx, self._stc_constraints, dynObs, self.ref_traj, self.mode)
        return False


class WorkerDS(object):

    def __init__(self, config_fn, initial_state):
        yaml_fp = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', config_fn)
        self.config = Configurator(yaml_fp)
        self.state = initial_state  # x-pos, y-pos, heading
        self.last_state = initial_state
        self.velocity = 0
        self.is_visible = True

    def set_state(self, state, velocity, visible=True):
        self.last_state = self.state
        self.state = state
        self.velocity = velocity
        self.is_visible = visible

    def predict_path(self):
        # number_of_dyn_obs*variables_per_dyn_ob*prediction_horizon (15*6*20)
        dynObs = [0.0] * self.config.Ndynobs * self.config.ndynobs * self.config.N_hor

        if self.is_visible:
            # worker is visible, handle him as dynamic object
            dx = self.state[0] - self.last_state[0]
            dy = self.state[1] - self.last_state[1]
            dh = self.state[2] - self.last_state[2]

            # handle dynamic obstacles, simple linear prediction for now
            for i in range(self.config.N_hor):
                dynObs[self.config.ndynobs * i] = self.state[0] + i * dx  # worker's x-pos
                dynObs[self.config.ndynobs * i + 1] = self.state[1] + i * dy  # worker's y-pos
                dynObs[self.config.ndynobs * i + 2] = 0.5
                dynObs[self.config.ndynobs * i + 3] = 0.5
                dynObs[self.config.ndynobs * i + 4] = self.state[2]  # 0 #worker's heading
                dynObs[self.config.ndynobs * i + 5] = 1

        return dynObs


if __name__ == "__main__":

    start = time.time()
    try:
        env = CoppeliaSim(RENDER, SPAWN, port_num=PORT)
        while env.episode_count <= 20000:
            if env.step():
                env.reset()
        env.close()

    except Exception as e:
        env.close()
        print('Runtime:', time.time() - start)
        raise e

    print('Runtime:', time.time() - start)
