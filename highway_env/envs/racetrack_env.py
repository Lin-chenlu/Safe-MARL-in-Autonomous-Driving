from itertools import repeat, product
from typing import Tuple, Dict, Text

import numpy as np

from Highway_env import utils
from Highway_env.envs.common.abstract import AbstractEnv
from Highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from Highway_env.road.road import Road, RoadNetwork
from Highway_env.vehicle.behavior import IDMVehicle


class RacetrackEnv(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
                "target_speeds": [0, 5, 10]
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 300,
            "collision_reward": -1,
            "lane_centering_cost": 4,
            "lane_centering_reward": 1,
            "action_reward": -0.3,
            "controlled_vehicles": 2,
            "other_vehicles": 1,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }

    def leader_is_terminal(self, vehicle):
        return vehicle.crashed or (not vehicle.on_road) or self.time >= self.config["duration"]

    def follower_is_terminal(self, vehicle):
        return vehicle.crashed or (not vehicle.on_road) or self.time >= self.config["duration"]

    def leader_agend_reward(self, vehicle, action):
        _, lateral = vehicle.lane.local_coordinates(vehicle.position)
        rewards = {
            "lane_centering_reward": 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": vehicle.crashed,
            "on_road_reward": vehicle.on_road,
        }
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def follower_agend_reward(self, vehicle, action):
        _, lateral = vehicle.lane.local_coordinates(vehicle.position)
        rewards = {
            "lane_centering_reward": 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": vehicle.crashed,
            "on_road_reward": vehicle.on_road,
        }
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or self.time >= self.config["duration"]

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        # Bug 修复：检查 action 数组长度，确保不会越界
        if len(action) < len(self.controlled_vehicles):
            action = np.pad(action, (0, len(self.controlled_vehicles) - len(action)), 'constant', constant_values=0)

        # simulation
        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        # observation
        obs = self.observation_type.observe()

        truncated = self._is_truncated()

        # terminate
        leader_terminated = self.leader_is_terminal(self.controlled_vehicles[0])
        follower_terminated = self.follower_is_terminal(self.controlled_vehicles[1])
        terminated = [leader_terminated, follower_terminated]

        # reward
        leader_reward = self.leader_agend_reward(self.controlled_vehicles[0], action[0])
        follower_reward = self.follower_agend_reward(self.controlled_vehicles[1], action[1])
        reward = [leader_reward, follower_reward]

        # info
        info = self._info(obs, action)
        info["crash"] = self.controlled_vehicles[0].crashed or self.controlled_vehicles[1].crashed
        # info["leader_arrived"] = self.has_arrived_target(self.controlled_vehicles[0])
        # info["follower_arrived"] = self.has_arrived_target(self.controlled_vehicles[1])

        # cost
        cost = np.zeros(2)
        cost[0] += self.controlled_vehicles[0].crashed + 0.5 * (not self.controlled_vehicles[0].on_road)
        cost[1] += self.controlled_vehicles[1].crashed + 0.5 * (not self.controlled_vehicles[1].on_road)
        info["cost"] = cost

        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                            speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b",
                     StraightLane([42, 5], [100, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                  speed_limit=speedlimits[1]))

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1 + 5, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 3 - Vertical Straight
        net.add_lane("c", "d", StraightLane([120, -20], [120, -30],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([125, -20], [125, -30],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2 + 5, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane("e", "f",
                     CircularLane(center3, radii3 + 5, np.deg2rad(0), np.deg2rad(136), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                     CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # 6 - Slant
        net.add_lane("f", "g", StraightLane([55.7, -15.7], [35.7, -35.7],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([59.3934, -19.2], [39.3934, -39.2],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[6]))

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane("g", "h",
                     CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("g", "h",
                     CircularLane(center4, radii4 + 5, np.deg2rad(315), np.deg2rad(165), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4 + 5, np.deg2rad(170), np.deg2rad(58), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane("i", "a",
                     CircularLane(center5, radii5 + 5, np.deg2rad(240), np.deg2rad(270), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("i", "a",
                     CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[8]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", i)
            # lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
            #     self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=35)
            #  longitudinal=rng.uniform(20, 50))

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        # vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
        #                                   longitudinal=rng.uniform(
        #                                       low=0,
        #                                       high=self.road.network.get_lane(("b", "c", 0)).length
        #                                   ),
        #                                   speed=6+rng.uniform(high=3))
        # self.road.vehicles.append(vehicle)

        # # Other vehicles
        # for i in range(rng.integers(self.config["other_vehicles"])):
        #     random_lane_index = self.road.network.random_lane_index(rng)
        #     vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
        #                                       longitudinal=rng.uniform(
        #                                           low=0,
        #                                           high=self.road.network.get_lane(random_lane_index).length
        #                                       ),
        #                                       speed=6+rng.uniform(high=3))
        #     # Prevent early collisions
        #     for v in self.road.vehicles:
        #         if np.linalg.norm(vehicle.position - v.position) < 20:
        #             break
        #     else:
        #         self.road.vehicles.append(vehicle)
