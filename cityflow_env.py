import cityflow
import os
import math
import numpy as np


class CityFlowEnvM(object):
    def __init__(self,
                 lane_phase_info,
                 intersection_id,
                 num_step=2000,
                 thread_num=1,
                 cityflow_config_file='',
                 path_to_log='result'
                 ):
        self.eng = cityflow.Engine(cityflow_config_file, thread_num=thread_num)
        self.num_step = num_step
        self.intersection_id = intersection_id  
        self.state_size = None
        self.lane_phase_info = lane_phase_info  

        self.path_to_log = path_to_log
        self.current_phase = {}
        self.current_phase_time = {}
        self.start_lane = {}
        self.end_lane = {}
        self.phase_list = {}
        self.phase_startLane_mapping = {}
        self.intersection_lane_mapping = {}  
        self.vehicles_waiting_time = {}
        self.vehicles_travel_time = {}
        self.vehicles_distance = {}

        self.w1 = 0
        self.w2 = 1

        for id_ in self.intersection_id:
            self.start_lane[id_] = self.lane_phase_info[id_]['start_lane']
            self.end_lane[id_] = self.lane_phase_info[id_]['end_lane']
            self.phase_startLane_mapping[id_] = self.lane_phase_info[id_]["phase_startLane_mapping"]

            self.phase_list[id_] = self.lane_phase_info[id_]["phase"]
            self.current_phase[id_] = self.phase_list[id_][0]
            self.current_phase_time[id_] = 0
        self.get_state_phase() 

    def reset(self):
        self.vehicles_waiting_time = {}
        self.vehicles_travel_time = {}
        self.vehicles_distance = {}
        self.eng.reset()

    def step(self, action):
        for id_, a in action.items():
            if self.current_phase[id_] == a:
                self.current_phase_time[id_] += 1
            else:
                self.current_phase[id_] = a
                self.current_phase_time[id_] = 1
            self.eng.set_tl_phase(id_, self.current_phase[id_]) 
        self.eng.next_step()

        vehicles_id_list = self.eng.get_vehicles(include_waiting=False)
        for id_ in vehicles_id_list:
            vehicle_info = self.eng.get_vehicle_info(id_)
  
            if id_ not in self.vehicles_waiting_time:
                self.vehicles_waiting_time[id_] = 0
            if (vehicle_info['running'] == '1' and float(vehicle_info['speed']) <= 0.1) or (
                    vehicle_info['running'] == '0'):
                if id_ in self.vehicles_waiting_time:
                    self.vehicles_waiting_time[id_] += 1
                else:
                    self.vehicles_waiting_time[id_] = 1

            if id_ in self.vehicles_travel_time:
                self.vehicles_travel_time[id_] += 1
            else:
                self.vehicles_travel_time[id_] = 1
 
            if (vehicle_info['running'] == '1'):
                self.vehicles_distance[id_] = float(vehicle_info['distance'])

        return self.get_state_phase(),self.get_state_duration(phase=action), self.get_reward()


    def get_state_phase(self):
        state_phase = {id_: self.get_state_phase_(id_) for id_ in self.intersection_id}
        return state_phase


    def get_state_phase_(self, id_):
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        end_vehicle_count = [-state['end_lane_vehicle_count'][lane] for lane in self.end_lane[id_]]

        pressure = []

        for i in range(len(self.start_lane[id_])):
            pressure.append((start_vehicle_count[i] - end_vehicle_count[i]))

        waiting_time = []
        for lane in self.start_lane[id_]:
            vehicle_list = state['start_lane_vehicles'][lane]
            sum = 0
            for vehicle in vehicle_list:
                sum += self.vehicles_waiting_time[vehicle] / self.vehicles_travel_time[vehicle]
            waiting_time.append(sum)
        pressure = list(np.array(pressure) * self.w1 + np.array(waiting_time) * self.w2)

        return_state = pressure + [state['current_phase']]
        return self.preprocess_state(return_state)


    def get_state_duration(self, phase):
        state_duration = {id_: self.get_state_duration_(id_, phase) for id_ in self.intersection_id}
        return state_duration


    def get_state_duration_(self, id_, phase):
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        end_vehicle_count = [-state['end_lane_vehicle_count'][lane] for lane in self.end_lane[id_]]

        pressure = []

        for i in range(len(self.start_lane[id_])):
            pressure.append((start_vehicle_count[i] - end_vehicle_count[i]))

        waiting_time = []
        for lane in self.start_lane[id_]:
            vehicle_list = state['start_lane_vehicles'][lane]
            sum = 0
            for vehicle in vehicle_list:
                sum += self.vehicles_waiting_time[vehicle] / self.vehicles_travel_time[vehicle]
            waiting_time.append(sum)
        pressure = list(np.array(pressure) * self.w1 + np.array(waiting_time) * self.w2)

        return_state = pressure + [phase[id_]]
        return self.preprocess_state(return_state)

    def intersection_info(self, id_):
        state = {}
        state['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()
        state['lane_vehicles'] = self.eng.get_lane_vehicles()
        state['start_lane_vehicle_count'] = {lane: state['lane_vehicle_count'][lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicle_count'] = {lane: state['lane_vehicle_count'][lane] for lane in self.end_lane[id_]}
        state['start_lane_vehicles'] = {lane: state['lane_vehicles'][lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicles'] = {lane: state['lane_vehicles'][lane] for lane in self.end_lane[id_]}

        state['current_phase'] = self.current_phase[id_]
        state['current_phase_time'] = self.current_phase_time[id_]
        return state

    def preprocess_state(self, state):
        return_state = np.array(state)
        if self.state_size is None:
            self.state_size = len(return_state.flatten())
        return_state = np.reshape(return_state, [1, self.state_size])
        return return_state

    def get_reward(self):
        reward = {id_: self.get_reward_(id_) for id_ in self.intersection_id}
        return reward

    def get_reward_(self, id_):
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        end_vehicle_count = [state['end_lane_vehicle_count'][lane] for lane in self.end_lane[id_]]

        waiting_time = []
        for lane in self.start_lane[id_]:
            vehicle_list = state['start_lane_vehicles'][lane]
            temp_sum = 0
            for vehicle in vehicle_list:
                temp_sum += self.vehicles_waiting_time[vehicle] / self.vehicles_travel_time[vehicle]
            waiting_time.append(temp_sum)

        pressure = (sum(start_vehicle_count) - sum(end_vehicle_count)) * self.w1 + sum(waiting_time) * self.w2
        reward = -abs(pressure)
        return reward

    def bulk_log(self):
        self.eng.set_replay_file((os.path.join(self.path_to_log, "replay.txt")))

    def get_timing_(self, id_, phase):
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        end_vehicle_count = [state['end_lane_vehicle_count'][lane] for lane in self.end_lane[id_]]
        start_vehicle_count_cop = [] 
        for i in range(len(start_vehicle_count)):
            if i % 3 != 2:
                start_vehicle_count_cop.append(start_vehicle_count[i])

        phase_lane = [[1, 7], [3, 5], [0, 6], [2, 4], [0, 1], [6, 7], [2, 3], [4, 5]]
        w1 = 1
        w2 = 2
        max_count = max(start_vehicle_count_cop[phase_lane[phase - 1][0]],
                        start_vehicle_count_cop[phase_lane[phase - 1][1]])
        min_count = min(start_vehicle_count_cop[phase_lane[phase - 1][0]],
                        start_vehicle_count_cop[phase_lane[phase - 1][1]])
        vehicle_count = (min_count * w1 + max_count * w2) / (w1 + w2)
        timing = math.ceil(vehicle_count / 2) * 5
        if timing > 25:
            timing = 25
        if timing < 5:
            timing = 5
        return vehicle_count, timing
