import gym

import highway_env

import json

highway_env.register_highway_envs()
env = gym.make("highway-v0")
# # env.configure({
# #   "manual_control": True,
# #   "real_time_rendering": True,
# #   "screen_width": 1000,
# #   "screen_height": 1000,
# #   "duration": 20,
# #   "observation": {
# #       "type": "MultiAgentObservation",
# #         "observation_config": {
# #             "type": "Kinematics",
# #             "flatten": True,
# #             "absolute": True,
# #             "see_behind": True,
# #             "normalize": False,
# #             "features": ["x", "y", "vx", "vy"],
# #             "vehicles_count": 2
# #             }
# #   },
# #   "action": {
# #     "type": "MultiAgentAction",
# #         "action_config": {
# #             "type": "DiscreteMetaAction"
# #         }
# #   }
# # })


env.reset()
done = False
while not done:
    act = env.action_space.sample()

    # obs, reward, done, _, _ = env.step(act)

    # print(env.controlled_vehicles[0].target_speeds)
    # print(env.controlled_vehicles[1].target_speeds)
    # print(".......")
    # done = np.all(done)
    env.step(act)
    # 渲染并显示图像
    env.render()
    # 如果环境结束，停止循环
    done = env.done
env.close()  # 关闭环境


# 定义配置文件路径列表
config_files = [
    "merge_env_result/exp1/env_config.json",
    "merge_env_result/exp1_bilevel/env_config.json",
    "merge_env_result/exp2/env_config.json",
    "highway_env_result/exp1/env_config.json"
]

# 循环加载不同的配置文件并模拟环境
for config_file in config_files:
    env = gym.make("merge-v0")
    with open(config_file, 'r') as f:
        env.configure(json.load(f))
    env.reset()
    done = False
    while not done:
        act = env.action_space.sample()
        env.step(act)
        env.render()
        done = env.done
    env.close()