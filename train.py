import json
import os
from datetime import datetime
import pandas as pd
from time import *
from cityflow_env import CityFlowEnvM
from utility import parse_roadnet, plot_data_lists
from simple_dqn_phase import MDQNAgent_phase
from simple_dqn_duration import MDQNAgent_duration
import torch
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  

class DQNConfig_phase:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon_start = 0.2  
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.001  
        self.memory_capacity = 10000 
        self.batch_size = 32
        self.target_update = 5 
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  
        self.hidden_dim = 20  

class DQNConfig_duration:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon_start = 0.2  
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.001  
        self.memory_capacity = 10000  
        self.batch_size = 32
        self.target_update = 5 
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  
        self.hidden_dim = 20 

def main():
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    parser=argparse.ArgumentParser()
    parser.add_argument('--d',type=str,default='Syn1',help='dataset')
    args=parser.parse_args()

    print(str(args.d))

    dataset_path="Datasets/"+str(args.d)+"/"

    cityflow_config = {
        "interval": 1,
        "seed": 0,
        "laneChange": False,
        "dir": dataset_path,
        "roadnetFile": "roadnet.json",
        "flowFile": "flow.json",
        "rlTrafficLight": True,
        "saveReplay": False,
        "roadnetLogFile": "replayRoadNet.json",
        "replayLogFile": "replayLogFile.txt"
    }

    with open(os.path.join(dataset_path, "cityflow.config"), "w") as json_file:
        json.dump(cityflow_config, json_file)

    config = {
        'cityflow_config_file': dataset_path+"cityflow.config",
        'epoch': 200,
        'num_step': 3600,  
        'save_freq': 1,
        'model': 'DQN',
        'batch_size': 32
    }

    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)

    intersection_id = list(config['lane_phase_info'].keys())  # all intersections
    config["intersection_id"] = intersection_id
    phase_list = {id_: config["lane_phase_info"][id_]["phase"] for id_ in intersection_id}
    config["phase_list"] = phase_list

    model_dir = "model/{}_{}".format(config['model'], date)
    result_dir = "result/{}_{}".format(config['model'], date)
    config["result_dir"] = result_dir

    if not os.path.exists("model"):
        os.makedirs("model")
    if not os.path.exists("result"):
        os.makedirs("result")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    env = CityFlowEnvM(config["lane_phase_info"],
                       intersection_id,
                       num_step=config["num_step"],
                       thread_num=8,
                       cityflow_config_file=config["cityflow_config_file"]
                       )

    config["state_size"] = env.state_size

    cfg_phase=DQNConfig_phase()
    cfg_duration=DQNConfig_duration()

    Magent_phase = MDQNAgent_phase(intersection_id,
                        state_size=config["state_size"],
                        cfg=cfg_phase,
                        phase_list=config["phase_list"]
                        )
    Magent_duration = MDQNAgent_duration(intersection_id,
                        state_size=config["state_size"],
                        cfg=cfg_duration,
                        phase_list=config["phase_list"],
                        action_dim=25
                        )

    EPISODES = config['epoch']
    episode_rewards = {id_: [] for id_ in intersection_id}
    episode_travel_time = []
    vehicles_waiting_time = {}
    vehicles_travel_time = {}

    i=0
    while i<=EPISODES:
        env.reset()
        state_phase = {}
        state_duration = {}
        action_phase = {}
        phase = {}
        duration = {}
        reward = {id_: 0 for id_ in intersection_id}
        rest_timing = {id_: 0 for id_ in intersection_id}
        timing_choose = {id_: [] for id_ in intersection_id}
        for id_ in intersection_id: 
            state_phase[id_] = env.get_state_phase_(id_)
            state_duration[id_] = env.get_state_phase_(id_)

        episode_reward = {id_: 0 for id_ in intersection_id}  
        episode_length=0

        if i==200:
            config['num_step']=7200
        while episode_length <= config['num_step']:
            for id_, t in rest_timing.items():
                if t == 0:
                    if episode_length != 0:
                        reward[id_] = env.get_reward_(id_)
                        episode_reward[id_] += reward[id_]
                        done=0
                        if episode_length==3600:
                            done=1
                        Magent_phase.agents[id_].remember(state_phase[id_], phase[id_], reward[id_],
                                                          next_state_phase[id_],done)
                        Magent_phase.agents[id_].update()
                        Magent_duration.agents[id_].remember(state_duration[id_], duration[id_], reward[id_],
                                                             next_state_duration[id_],done)
                        Magent_duration.agents[id_].update()
                        state_phase[id_] = next_state_phase[id_]
                        state_duration[id_] = next_state_duration[id_]

                    action_phase[id_] = Magent_phase.agents[id_].choose_action(state_phase[id_])
                    phase[id_] = phase_list[id_][action_phase[id_]]

                    duration[id_] = Magent_duration.agents[id_].choose_action(state_duration[id_])
                    rest_timing[id_] = duration[id_]
                    timing_choose[id_].append(duration[id_])

            next_state_phase, next_state_duration, reward_ = env.step(phase)  

            vehicles_id_list = env.eng.get_vehicles(include_waiting=False)
            for id_ in vehicles_id_list:
                vehicle_info = env.eng.get_vehicle_info(id_)

                if id_ not in vehicles_waiting_time:
                    vehicles_waiting_time[id_] = 0
                if (vehicle_info['running'] == '1' and float(vehicle_info['speed']) <= 0.1) or (
                    vehicle_info['running'] == '0'):
                    if id_ in vehicles_waiting_time:
                        vehicles_waiting_time[id_] += 1
                    else:
                        vehicles_waiting_time[id_] = 1

                if id_ in vehicles_travel_time:
                    vehicles_travel_time[id_] += 1
                else:
                    vehicles_travel_time[id_] = 1
            for id_ in intersection_id:
                episode_reward[id_] += reward_[id_]

            episode_length+=1

            if i==200:
                if episode_length%100==0:
                    print('{} Average Travel Time: {}'.format(episode_length,env.eng.get_average_travel_time()))

            for id_ in rest_timing:
                rest_timing[id_] -= 1

        print('\n')
        print('Episode: {},travel time: {}'.format(i, env.eng.get_average_travel_time()))
        i+=1
        episode_travel_time.append(env.eng.get_average_travel_time())
        for id_ in intersection_id:
            episode_rewards[id_].append(episode_reward[id_])

    Magent_phase.save(model_dir + "/{}-{}.h5".format(config['model'] + '_phase', i + 1))
    Magent_duration.save(model_dir + "/{}-{}.h5".format(config['model'] + '_duration', i + 1))


    df = pd.DataFrame(episode_rewards)
    df.to_csv(result_dir + '/rewards.csv', index=False)

    df = pd.DataFrame({"travel time": episode_travel_time})
    df.to_csv(result_dir + '/travel time.csv', index=False)


    plot_data_lists([episode_rewards[id_] for id_ in intersection_id], intersection_id,
                    figure_name=result_dir + '/rewards.pdf')
    plot_data_lists([episode_travel_time], ['travel time'], figure_name=result_dir + '/travel time.pdf')


    df = pd.DataFrame(vehicles_waiting_time, index=[1])
    df.to_csv('vehicles_waiting_time.csv', index=False)


    df = pd.DataFrame(vehicles_travel_time, index=[1])
    df.to_csv('vehicles_travel_time.csv', index=False)


    waiting_travel_time = {}
    for id_ in vehicles_travel_time:
        waiting_travel_time[id_] = vehicles_waiting_time[id_] / (vehicles_travel_time[id_] - vehicles_waiting_time[id_])
    df = pd.DataFrame(waiting_travel_time, index=[1])
    df.to_csv('waiting_travel_time.csv', index=False)

if __name__ == '__main__':
    main()
