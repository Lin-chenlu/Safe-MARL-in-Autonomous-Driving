from runner_bilevel import Runner_Bilevel, Runner_Stochastic, Runner_C_Bilevel
from common.arguments import get_args
from common.utils import make_highway_env
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

    # 原有的训练逻辑
    # get the params
    args = get_args()

    # set train params
    # 循环遍历不同的环境路径
    paths = ["./merge_env_result/exp1", "./roundabout_env_result/exp1", "./intersection_env_result/exp1", "./racetrack_env_result/exp1"]
    seed = [0, 1, 2]
    for path in paths:
        for i in seed:
            args.seed = i
            args.save_dir = args.file_path + "/seed_" + str(args.seed)
        
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        with open(args.file_path+'/config.json','r') as f:
            vars(args).update(json.load(f))
        
        # set env
        env, eval_env, args = make_highway_env(args)
        
        np.random.seed(args.seed)
        
        # choose action type and algorithm
        if args.action_type == "continuous":
            # unconstrained stackelberg maddpg
            if args.version == "bilevel":
                runner = Runner_Bilevel(args, env, eval_env)
            # constrained stackelberg maddpg
            elif args.version == "c_bilevel":
                runner = Runner_C_Bilevel(args, env, eval_env)
        elif args.action_type == "discrete":
            # constrained or unconstrained(by setting extreme high cost threshold) stackelberg Q learning
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
            # 发送视频路径到后端
            import requests
            response = requests.post('http://127.0.0.1:5000/receive_video_path', json={'video_path': video_path})
            print(response.text)

        

