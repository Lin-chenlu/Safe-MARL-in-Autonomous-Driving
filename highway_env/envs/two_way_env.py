from typing import Dict, Text,Tuple

import numpy as np

from Highway_env import utils
from Highway_env.envs.common.abstract import AbstractEnv
from Highway_env.envs.common.action import Action
from Highway_env.road.lane import LineType, StraightLane
from Highway_env.road.road import Road, RoadNetwork
from Highway_env.utils import near_split
from Highway_env.vehicle.controller import ControlledVehicle,MDPVehicle
from Highway_env.vehicle.kinematics import Vehicle


class TwoWayEnv(AbstractEnv):

    """
    A risk management task: the agent is driving on a two-way lane with icoming traffic.

    It must balance making progress by overtaking and ensuring safety.

    These conflicting objectives are implemented by a reward signal and a constraint signal,
    in the CMDP/BMDP framework.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "horizon": 5
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "collision_reward": 0,
            "left_lane_constraint": 1,
            "left_lane_reward": 0.2,
            "high_speed_reward": 0.8,
        })
        return config

    def _reward(self, action: int, vehicle: Vehicle) -> float:
        """
        The vehicle is rewarded for driving with high speed
        :param action: the action performed
        :param vehicle: the vehicle to compute reward for
        :return: the reward of the state-action transition
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config.get("normalize_reward", False):
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["left_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "left_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self, vehicle=None) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        vehicle = vehicle if vehicle is not None else self.vehicle
        return (vehicle.crashed or
                self.config.get("offroad_terminal", False) and not vehicle.on_road or
                self.time >= self.config["duration"])

    def _is_truncated(self) -> bool:
        return False
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        # simulation
        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        # observation
        obs = self.observation_type.observe()

        truncated = self._is_truncated()

        # terminate
        terminated = [self._is_terminated(vehicle) for vehicle in self.controlled_vehicles]

        # reward
        reward = [self._reward(action, vehicle) for vehicle in self.controlled_vehicles]

        # info
        info = self._info(obs, action)
        info["leader_arrived"] = False
        info["follower_arrived"] = False
        info["crash"] = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        
        # cost
        cost = np.zeros(len(self.controlled_vehicles))
        for i, vehicle in enumerate(self.controlled_vehicles):
            cost[i] += 5 * vehicle.crashed
        info["cost"] = cost

        return obs, reward, terminated, truncated, info

    def _reset(self) -> np.ndarray:
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=800):
        """
        Make a road composed of a two-way road.

        :return: the road
        """
        net = RoadNetwork()

        # Lanes
        net.add_lane("a", "b", StraightLane([0, 0], [length, 0],
                                            line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED)))
        net.add_lane("a", "b", StraightLane([0, StraightLane.DEFAULT_WIDTH], [length, StraightLane.DEFAULT_WIDTH],
                                            line_types=(LineType.NONE, LineType.CONTINUOUS_LINE)))
        net.add_lane("b", "a", StraightLane([length, 0], [0, 0],
                                            line_types=(LineType.NONE, LineType.NONE)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the road

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", 1)).position(30, 0),
                                                     speed=30)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(3):
            self.road.vehicles.append(
                vehicles_type(road,
                              position=road.network.get_lane(("a", "b", 1))
                              .position(70+40*i + 10*self.np_random.normal(), 0),
                              heading=road.network.get_lane(("a", "b", 1)).heading_at(70+40*i),
                              speed=24 + 2*self.np_random.normal(),
                              enable_lane_change=False)
            )
        for i in range(2):
            v = vehicles_type(road,
                              position=road.network.get_lane(("b", "a", 0))
                              .position(200+100*i + 10*self.np_random.normal(), 0),
                              heading=road.network.get_lane(("b", "a", 0)).heading_at(200+100*i),
                              speed=20 + 5*self.np_random.normal(),
                              enable_lane_change=False)
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)
