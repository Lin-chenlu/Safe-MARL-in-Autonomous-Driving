from runner_bilevel import Runner_Stochastic, Runner_C_Bilevel
from common.arguments import get_args
from common.utils import make_Highway_env
import numpy as np
import json
import time
from flask import Flask, request
import threading; import os


if __name__ == '__main__':
  # 模拟进度条
    app = Flask(__name__)

    @app.route('/get_progress', methods=['GET'])
    def get_progress():
        # 模拟进度条，这里可以替换为实际的进度获取逻辑
        progress = 0
        while progress <= 100:
            time.sleep(1)
            progress += 10
            yield json.dumps({'progress': progress})

    # 启动Flask应用
    threading.Thread(target=app.run, kwargs={'host': '127.0.0.1', 'port': 5000}).start()
    
    # 保留原有训练逻辑
    args = get_args()

    # 循环遍历不同的环境路径
    env_paths = ["./u_turn_env_result/exp1",
        "./two_way_env_result/exp1",
        
        "./merge_env_result/exp1",
        "./roundabout_env_result/exp1",
        "./highway_env_result/exp1",
        "./racetrack_env_result/exp1",
        "./intersection_env_result/exp1",
    ]

    # 为每个环境路径创建独立训练流程
    for env_path in env_paths:
        args.file_path = env_path
        # 生成随机种子
        args.seed = np.random.randint(0, 100000)
        args.save_dir = os.path.join(args.file_path, "seed_" + str(args.seed))
        
        # 创建保存目录
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
            
        # 加载环境配置文件
        config_path = os.path.join(args.file_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            vars(args).update(config)

        # set env
        env, eval_env, args = make_Highway_env(args)
        
        np.random.seed(args.seed)
        
        # choose action type and algorithm
        if args.action_type == "continuous":
            # constrained stackelberg maddpg
            runner = Runner_C_Bilevel(args, env, eval_env)
        elif args.action_type == "discrete":
            # constrained stackelberg Q learning
            runner = Runner_Stochastic(args, env, eval_env)
        
        # train or evaluate
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
        else:
            runner.run()
        
        # record video
        if args.record_video:
            video_path = runner.record_video()
            print(video_path)
            # 发送视频路径到后端
            
            response = request.post('http://127.0.0.1:5000/receive_video_path', json={'video_path': video_path})
            print(response.text)

        # 新增日志目录创建
        log_dir = os.path.join(os.getcwd(), "static/training_logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = open(os.path.join(log_dir, "latest.log"), "w")


        

