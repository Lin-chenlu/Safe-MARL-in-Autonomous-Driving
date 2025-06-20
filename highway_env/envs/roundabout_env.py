from typing import Tuple, Dict, Text

import numpy as np

from Highway_env import utils
from Highway_env.envs.common.abstract import AbstractEnv
from Highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from Highway_env.road.road import Road, RoadNetwork
from Highway_env.vehicle.controller import MDPVehicle


class RoundaboutEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "absolute": True,
                "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-15, 15], "vy": [-15, 15]},
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [0, 8, 16]
            },
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 11,
            "normalize_reward": True
        })
        return config

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["high_speed_reward"]], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward":
                 MDPVehicle.get_speed_index(self.vehicle) / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1),
            "lane_change_reward": action in [0, 2],
            "on_road_reward": self.vehicle.on_road
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or self.time >= self.config["duration"]
    
    def leader_is_terminal(self, vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (vehicle.crashed or
                self.has_arrived_target(vehicle) or
                self.time >= self.config["duration"])
    
    def follower_is_terminal(self, vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (vehicle.crashed or
                self.has_arrived_target(vehicle) or
                self.time >= self.config["duration"])
    
    def leader_agend_reward(self, vehicle):
        """Per-agent per-objective reward signal."""

        reward = 0

        if vehicle.crashed:
            reward -= 5

        if vehicle.speed>=10 and vehicle.speed<=15:
            reward += 2
        
        if self.has_arrived_target(vehicle):
            reward += 5
        
        return reward
    
    def follower_agend_reward(self, vehicle):
        """Per-agent per-objective reward signal."""
        reward = 0

        if vehicle.crashed:
            reward -= 5

        if vehicle.speed>=10 and vehicle.speed<=15:
            reward += 2
        
        if self.has_arrived_target(vehicle):
            reward += 5
        
        return reward

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

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
        leader_terminated = self.leader_is_terminal(self.controlled_vehicles[0])
        follower_terminated = self.follower_is_terminal(self.controlled_vehicles[1])
        terminated = [leader_terminated, follower_terminated]

        # reward
        leader_reward = self.leader_agend_reward(self.controlled_vehicles[0])
        follower_reward = self.follower_agend_reward(self.controlled_vehicles[1])   
        reward = [leader_reward, follower_reward]       

        # info
        info = self._info(obs, action)
        info["leader_arrived"] = self.has_arrived_target(self.controlled_vehicles[0])
        info["follower_arrived"] = self.has_arrived_target(self.controlled_vehicles[1])
        info["crash"] = self.controlled_vehicles[0].crashed or self.controlled_vehicles[1].crashed

        # cost
        cost = np.zeros(2)
        cost[0] += 5*self.controlled_vehicles[0].crashed
        cost[1] += 5*self.controlled_vehicles[1].crashed
        info["cost"] = cost

        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius+4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("se", "ex",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee",
                         CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx",
                         CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne",
                         CircularLane(center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx",
                         CircularLane(center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we",
                         CircularLane(center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx",
                         CircularLane(center, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se",
                         CircularLane(center, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha),
                                      clockwise=False, line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2*dev  # [m]

        delta_en = dev-delta_st
        w = 2*np.pi/dev
        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev/2], line_types=(s, c)))
        net.add_lane("ses", "se", SineLane([2+a, dev/2], [2+a, dev/2-delta_st], a, w, -np.pi/2, line_types=(c, c)))
        net.add_lane("sx", "sxs", SineLane([-2-a, -dev/2+delta_en], [-2-a, dev/2], a, w, -np.pi/2+w*delta_en, line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))

        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)))
        net.add_lane("ees", "ee", SineLane([dev / 2, -2-a], [dev / 2 - delta_st, -2-a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("ex", "exs", SineLane([-dev / 2 + delta_en, 2+a], [dev / 2, 2+a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))

        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
        net.add_lane("nes", "ne", SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))

        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
        net.add_lane("wes", "we", SineLane([-dev / 2, 2+a], [-dev / 2 + delta_st, 2+a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("wx", "wxs", SineLane([dev / 2 - delta_en, -2-a], [-dev / 2, -2-a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        position_deviation = 2
        speed_deviation = 2

        self.controlled_vehicles = []
        # Ego-vehicle 1
        ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(85, 0),
                                                     speed=8,
                                                     heading=ego_lane.heading_at(140))
        try:
            ego_vehicle.plan_route_to("nxs")
            # ego_vehicle.speed_index = ego_vehicle.speed_to_index(10)
            # ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        
        # self.vehicle = ego_vehicle

        # Ego-vehicle 2
        ego_lane_2 = self.road.network.get_lane(("wer", "wes", 0))
        ego_vehicle_2 = self.action_type.vehicle_class(self.road,
                                                     ego_lane_2.position(125, 0),
                                                     speed=8,
                                                     heading=ego_lane_2.heading_at(140))
        try:
            ego_vehicle_2.plan_route_to("nxs")
            # ego_vehicle_2.speed_index = ego_vehicle_2.speed_to_index(10)
            # ego_vehicle_2.target_speed = ego_vehicle_2.index_to_speed(ego_vehicle_2.speed_index)
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle_2)
        self.controlled_vehicles.append(ego_vehicle_2)
        self.controlled_vehicles.append(ego_vehicle)
        # self.vehicle_2 = ego_vehicle_2

        # Incoming vehicle
        # destinations = ["exr", "sxr", "nxr"]
        # other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                            ("we", "sx", 1),
        #                                            longitudinal=5 + self.np_random.normal()*position_deviation,
        #                                            speed=16 + self.np_random.normal() * speed_deviation)

        # if self.config["incoming_vehicle_destination"] is not None:
        #     destination = destinations[self.config["incoming_vehicle_destination"]]
        # else:
        #     destination = self.np_random.choice(destinations)
        # vehicle.plan_route_to(destination)
        # vehicle.randomize_behavior()
        # self.road.vehicles.append(vehicle)

        # Other vehicles
        # for i in list(range(1, 2)) + list(range(-1, 0)):
        #     vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                                ("we", "sx", 0),
        #                                                longitudinal=20*i + self.np_random.normal()*position_deviation,
        #                                                speed=16 + self.np_random.normal() * speed_deviation)
        #     vehicle.plan_route_to(self.np_random.choice(destinations))
        #     vehicle.randomize_behavior()
        #     self.road.vehicles.append(vehicle)

        # # Entering vehicle
        # vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                            ("eer", "ees", 0),
        #                                            longitudinal=50 + self.np_random.normal() * position_deviation,
        #                                            speed=16 + self.np_random.normal() * speed_deviation)
        # vehicle.plan_route_to(self.np_random.choice(destinations))
        # vehicle.randomize_behavior()
        # self.road.vehicles.append(vehicle)

    def has_arrived_target(self, vehicle, exit_distance: float = 20) -> bool:
        #print(vehicle.lane.local_coordinates(vehicle.position)[0])
        return "nxs" in vehicle.lane_index[0] \
               and "nxr" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
