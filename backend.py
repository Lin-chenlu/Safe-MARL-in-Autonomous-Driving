import json
import os
from flask import *
from flask_cors import CORS
import subprocess
import threading

app = Flask(__name__)
CORS(app)

# 全局变量，用于存储子进程的状态和输出
subprocess_output = []
subprocess_running = False


def run_main_bilevel(env):
    global subprocess_output, subprocess_running
    subprocess_running = True
    subprocess_output = []  # 清空之前的输出
    path = [f'./{env}/exp1']
    try:
        # 使用 subprocess.Popen 来启动子进程，并实时捕获输出
        process = subprocess.Popen(['python', 'main_bilevel.py'] + path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                subprocess_output.append(output.strip())
        # 子进程结束后，标记为未运行
        subprocess_running = False
    except Exception as e:
        subprocess_output.append(f"Error: {str(e)}")
        subprocess_running = False

@app.route('/')
def index():
    return send_file('static\\index.html')

@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()
    env = data.get('env')
    car_num = data.get('car_num')

    # 替换env_config中的vehicles_count
    env_config_path = f'./{env}/exp1/env_config.json'
    if os.path.exists(env_config_path):
        with open(env_config_path, 'r') as f:
            env_config = json.load(f)
        env_config['vehicles_count'] = car_num
        with open(env_config_path, 'w') as f:
            json.dump(env_config, f, indent=4)

        return jsonify({'message': 'Data received successfully', 'status': 'config_updated'})
    else:
        return jsonify({'message': 'Environment config not found', 'status': 'error'}), 404


@app.route('/get_main_bilevel_status', methods=['GET'])
def get_main_bilevel_status():
    global subprocess_output, subprocess_running
    if not subprocess_running:
        # 修改为使用 request.args.get() 获取查询参数
        env = request.args.get('env')
        threading.Thread(target=run_main_bilevel, args=(env,)).start()
    # 返回当前的子进程状态和输出
    status = 'running' if subprocess_running else 'finished'
    return jsonify({
        'status': status,
      'subprocess_result': '\n'.join(subprocess_output)  # 将输出列表转换为字符串
    })


@app.route('/receive_video_path', methods=['POST'])
def receive_video_path():
    data = request.get_json()
    video_path = data.get('video_path')
    # 这里可以添加对视频路径的处理逻辑，例如保存到数据库等
    return jsonify({'message': 'Video path received successfully'})


if __name__ == '__main__':
    app.run(debug=True)