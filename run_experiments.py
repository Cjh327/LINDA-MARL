import os
import datetime
import time
from argparse import ArgumentParser

alg_config = '--config={alg}'
env_config = '--env-config={env}'
map_config = 'with use_tensorboard=True env_args.map_name={map}'

output_folder = './output'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', type=str, help='decide which environment to run', default='sc2')
    parser.add_argument('-m', '--map', nargs='+', help='decide maps to run if choose sc2 env', default=['MMM2'])
    parser.add_argument('-a', '--alg', nargs='+', help='decide algorithms to run', default=['linda'])
    parser.add_argument('-s', '--seed', type=int, help='specify given seed', default=None)
    parser.add_argument('-r', '--repeat', type=int, help='repeat n times for a given algorithm', default=2)
    parser.add_argument('-c', '--cuda', type=int, help='cuda id', default=0)

    args = parser.parse_args()

    if str(args.env).startswith('sc2'):
        for map_name in args.map:
            for alg_name in args.alg:
                for r in range(args.repeat):
                    time.sleep(3)
                    log_name = '{alg}-{map}-{time}.out'.format(alg=alg_name, map=map_name, time=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
                    command = 'CUDA_VISIBLE_DEVICES={} nohup python src/main.py {} {} {} > {} 2>&1 &'.format(args.cuda, alg_config.format(alg=alg_name),
                        env_config.format(env=args.env), map_config.format(map=map_name), os.path.join(output_folder, log_name))
                    print(command)
                    os.system(command)
    else:
        for alg_name in args.alg:
            for _ in range(args.repeat):
                time.sleep(3)
                log_name = '{alg}-{env}-{time}.out'.format(alg=alg_name, env=args.env, time=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
                command = 'nohup python src/main.py {} {} > {} 2>&1 &'.format(alg_config.format(alg=alg_name), env_config.format(env=args.env),
                    os.path.join(output_folder, log_name))
                os.system(command)