from typing import Tuple, Dict, Text

import numpy as np


from Highway_env import utils
from Highway_env.envs.common.abstract import AbstractEnv
from Highway_env.road.lane import LineType, StraightLane, CircularLane
from Highway_env.road.road import Road, RoadNetwork
from Highway_env.vehicle.controller import MDPVehicle


class UTurnEnv(AbstractEnv):

    """
    U-Turn risk analysis task: the agent overtakes vehicles that are blocking the
    traffic. High speed overtaking must be balanced with ensuring safety.
    """

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
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 40,
            "collision_reward": -1,
            "high_speed_reward": 0.4,
            "right_lane_reward": 0.1,
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
            "on_road_reward": self.vehicle.on_road
        }
        
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

    def leader_is_terminal(self, vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.time >= self.config["duration"]
    
    def follower_is_terminal(self, vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.time >= self.config["duration"]

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

    def _reset(self) -> np.ndarray:
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=128):
        """
        Making double lane road with counter-clockwise U-Turn.
        :return: the road
        """
        net = RoadNetwork()

        # Defining upper starting lanes after the U-Turn.
        # These Lanes are defined from x-coordinate 'length' to 0.
        net.add_lane("c", "d", StraightLane([length, StraightLane.DEFAULT_WIDTH], [0, StraightLane.DEFAULT_WIDTH],
                                            line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED)))
        net.add_lane("c", "d", StraightLane([length, 0], [0, 0],
                                            line_types=(LineType.NONE, LineType.CONTINUOUS_LINE)))

        # Defining counter-clockwise circular U-Turn lanes.
        center = [length, StraightLane.DEFAULT_WIDTH + 20]  # [m]
        radius = 20  # [m]
        alpha = 0  # [deg]

        radii = [radius, radius+StraightLane.DEFAULT_WIDTH]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("b", "c",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(-90+alpha),
                                      clockwise=False, line_types=line[lane]))

        offset = 2*radius

        # Defining lower starting lanes before the U-Turn.
        # These Lanes are defined from x-coordinate 0 to 'length'.
        net.add_lane("a", "b", StraightLane([0, ((2 * StraightLane.DEFAULT_WIDTH + offset) - StraightLane.DEFAULT_WIDTH)],
                                            [length, ((2 * StraightLane.DEFAULT_WIDTH + offset) - StraightLane.DEFAULT_WIDTH)],
                                            line_types=(LineType.CONTINUOUS_LINE,
                                                        LineType.STRIPED)))
        net.add_lane("a", "b", StraightLane([0, (2 * StraightLane.DEFAULT_WIDTH + offset)],
                                            [length, (2 * StraightLane.DEFAULT_WIDTH + offset)],
                                            line_types=(LineType.NONE,
                                                        LineType.CONTINUOUS_LINE)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Strategic addition of vehicles for testing safety behavior limits
        while performing U-Turn manoeuvre at given cruising interval.

        :return: the ego-vehicle
        """

        # These variables add small variations to the driving behavior.
        position_deviation = 2
        speed_deviation = 2

        ego_lane = self.road.network.get_lane(("a", "b", 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(0, 0),
                                                     speed=16)
        # Stronger anticipation for the turn
        ego_vehicle.PURSUIT_TAU = MDPVehicle.TAU_HEADING
        try:
            ego_vehicle.plan_route_to("d")
        except AttributeError:
            pass

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Note: randomize_behavior() can be commented out if more randomized
        # vehicle interactions are deemed necessary for the experimentation.

        # Vehicle 1: Blocking the ego vehicle
        vehicle = vehicles_type.make_on_lane(self.road,
                                                   ("a", "b", 0),
                                                   longitudinal=25 + self.np_random.normal()*position_deviation,
                                                   speed=13.5 + self.np_random.normal() * speed_deviation)
        vehicle.plan_route_to('d')
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Vehicle 2: Forcing risky overtake
        vehicle = vehicles_type.make_on_lane(self.road,
                                                   ("a", "b", 1),
                                                   longitudinal=56 + self.np_random.normal()*position_deviation,
                                                   speed=14.5 + self.np_random.normal() * speed_deviation)
        vehicle.plan_route_to('d')
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Vehicle 3: Blocking the ego vehicle
        vehicle = vehicles_type.make_on_lane(self.road,
                                                   ("b", "c", 1),
                                                   longitudinal=0.5 + self.np_random.normal()*position_deviation,
                                                   speed=4.5 + self.np_random.normal() * speed_deviation)
        vehicle.plan_route_to('d')
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Vehicle 4: Forcing risky overtake
        vehicle = vehicles_type.make_on_lane(self.road,
                                                   ("b", "c", 0),
                                                   longitudinal=17.5 + self.np_random.normal()*position_deviation,
                                                   speed=5.5 + self.np_random.normal() * speed_deviation)
        vehicle.plan_route_to('d')
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Vehicle 5: Blocking the ego vehicle
        vehicle = vehicles_type.make_on_lane(self.road,
                                                   ("c", "d", 0),
                                                   longitudinal=1 + self.np_random.normal()*position_deviation,
                                                   speed=3.5 + self.np_random.normal() * speed_deviation)
        vehicle.plan_route_to('d')
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Vehicle 6: Forcing risky overtake
        vehicle = vehicles_type.make_on_lane(self.road,
                                                   ("c", "d", 1),
                                                   longitudinal=30 + self.np_random.normal()*position_deviation,
                                                   speed=5.5 + self.np_random.normal() * speed_deviation)
        vehicle.plan_route_to('d')
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)



