# communication.py
import time
import random
import threading
import copy
import numpy as np
from vehicle_ccn import Vehicle
from queue import Queue
from scipy.stats import zipf

class Communication:
    def __init__(self, satellites, base_stations, vehicles, uavs, ground_station, a, n, time_slots):
        self.satellites = satellites
        self.base_stations = base_stations
        self.vehicles = vehicles
        self.uavs = uavs
        self.ground_station= ground_station
        self.entity_type = ['satellite', 'UAV', 'grid']  # entity
        self.content_time_delays = {'satellite':20 , 'UAV': 10, 'grid': 5}  # Time delays for each content category
        # Initialize the content request queue
        self.content_request_queue = Queue()
        self.content_received_time = None  # Initialize the time value as None
        self.content_received_time_uav= None
        self.content_received_time_bs= None
        self.content_hit =0
        self.spatio_temporal_flag=0
        self.probabilities={}
        self.round=n
        self.alpha = a
        self.time_slots=time_slots
    def get_time_delay(self, entity):
            return self.content_time_delays.get(entity, 0)

    def broadcast_loop(self, vehicles, uavs, base_stations, satellites, grid_size):
        while True:
            # Retrieve content requests from the shared queue and process them in separate threads
            while not self.content_request_queue.empty():
                content_request = self.content_request_queue.get()
                broadcast_thread = threading.Thread(target=self.broadcast_content_request, args=(content_request, vehicles, uavs, base_stations, satellites, grid_size))
                broadcast_thread.start()
                time.sleep(0.5)

    def generate_probabilities(self, satellites, uavs, grid_size, time_slots):
        s = time_slots+1 // 360
        uav_ranks = len(uavs)
        satellite_ranks = len(satellites)
        grid_ranks = (grid_size * grid_size)

        #self.probabilities = {f"Region{i}": np.random.dirichlet(np.ones(satellite_ranks + uav_ranks + grid_ranks), size=s) for i in range(1, 5)}
        #self.probabilities /= np.sum(self.probabilities)
        self.probabilities = {f"Region{i}": np.random.dirichlet(np.ones(satellite_ranks + uav_ranks), size=s) for i in range(1, 5)}
        # self.probabilities /= np.sum(self.probabilities)
        #print(self.probabilities)
        return

    '''def send_content_request(self, vehicle, vehicles, uavs, base_stations, satellites, grid_size, current_time, slot, no_of_content_each_category):
        # Generate unique random ID
        unique_id = random.randint(1, 999999)
        # Divide the grid into four regions
        grid_rows = grid_size
        grid_columns = grid_size
        region_size_row = grid_rows // 2
        region_size_col = grid_columns // 2

        # Determine the region based on the vehicle's current location
        current_row, current_col = vehicle.current_location  # Assuming current_location is a tuple (row, col)
        region_row = current_row // region_size_row
        region_col = current_col // region_size_col

        # print(f" Results for: {current_row} {current_col} {region_row} and {region_col}")

        index = slot // 360
        #ranks = list(range(len(uavs) + len(satellites) + (grid_size * grid_size)))
        ranks = list(range(len(uavs) + len(satellites)))
        normalized_prob = []

        # Choose the region-specific content based on the specified distribution
        if region_row == 0 and region_col == 0:  # Region 1
            normalized_prob = self.probabilities['Region1'][index]
            normalized_prob /= np.sum(normalized_prob)
            element_index = np.random.choice(ranks, p=normalized_prob) + 1
        elif region_row == 0 and region_col == 1:  # Region 2
            normalized_prob = self.probabilities['Region2'][index]
            normalized_prob /= np.sum(normalized_prob)
            element_index = np.random.choice(ranks, p=normalized_prob) + 1
        elif region_row == 1 and region_col == 0:  # Region 3
            normalized_prob = self.probabilities['Region3'][index]
            normalized_prob /= np.sum(normalized_prob)
            element_index = np.random.choice(ranks, p=normalized_prob) + 1
        elif region_row == 1 and region_col == 1:  # Region 4
            normalized_prob = self.probabilities['Region4'][index]
            normalized_prob /= np.sum(normalized_prob)
            element_index = np.random.choice(ranks, p=normalized_prob) + 1
        else:
            return {'unique_id': 0}

        if 1 <= element_index <= len(satellites):
            entity_type = 'satellite'
            satellite = satellites[f"Satellite{element_index}"]
            # Step 3: Choose a category from the specific satellite based on Zipf distribution
            category_ranks = list(range(len(satellite.content_categories)))
            category_probabilities = np.array([1 / (rank + 1) for rank in category_ranks])
            category_probabilities /= np.sum(category_probabilities)
            category_index = np.random.choice(category_ranks, p=category_probabilities)
            content_category = satellite.content_categories[category_index]

            # Step 4: Choose the content number based on Zipf distribution
            content_no = np.random.zipf(
                2) % no_of_content_each_category + 1  # Assuming content numbers range from 1 to 10
            request_id = f"{vehicle.vehicle_id}_{int(current_time)}_{satellite.satellite_id}_{content_category}_{content_no}"
            cord = satellite.satellite_id
        elif len(satellites) < element_index <= (len(satellites) + len(uavs)):
            entity_type = 'UAV'
            uav = uavs[element_index - len(satellites) - 1]
            # Step 3: Choose a category from the specific UAV based on Zipf distribution
            category_ranks = list(range(len(uav.content_categories)))
            category_probabilities = np.array([1 / (rank + 1) for rank in category_ranks])
            category_probabilities /= np.sum(category_probabilities)
            category_index = np.random.choice(category_ranks, p=category_probabilities)
            content_category = uav.content_categories[category_index]

            # Step 4: Choose the content number based on Zipf distribution
            content_no = np.random.zipf(
                2) % no_of_content_each_category + 1  # Assuming content numbers range from 1 to 10
            request_id = f"{vehicle.vehicle_id}_{current_time}_{uav.uav_id}_{content_category}_{content_no}"
            cord = uav.uav_id
        else:

            entity_type = 'grid'
            grid = (element_index - (len(uavs) + len(satellites))) - 1
            # Step 3: Choose a category from the specific grid based on Zipf distribution
            category_ranks = list(range(len(vehicle.content_categories)))
            category_probabilities = np.array([1 / (rank + 1) for rank in category_ranks])
            category_probabilities /= np.sum(category_probabilities)
            category_index = np.random.choice(category_ranks, p=category_probabilities)
            content_category = vehicle.content_categories[category_index]

            # Step 4: Choose the content number based on Zipf distribution
            content_no = np.random.zipf(
                2) % no_of_content_each_category + 1  # Assuming content numbers range from 1 to 10
            request_id = f"{vehicle.vehicle_id}_{current_time}_grid{grid}_{content_category}_{content_no}"
            cord = grid
        #print(request_id)
        return {
            'unique_id': unique_id,
            'request_id': request_id,
            'requesting_vehicle': vehicle,
            'g_time': current_time,
            'type': entity_type,
            'category': content_category,
            'coord': cord,
            'no': content_no,
            'RDB': self.get_time_delay(entity_type),
            'hop_count': 0,
            'time_spent': 0
        }'''
    import numpy as np

    def custom_zipf(self, alpha, n):
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")
        if alpha > 1:
            return (np.random.zipf(alpha) % n) + 1
        else:
            # For alpha < 1, use a power-law distribution
            xk = np.arange(1, n + 1, dtype=float)
            pk = xk ** (-alpha) / np.sum(xk ** (-alpha))
            x = np.random.choice(xk, p=pk)
            return int(round(x))

    def send_content_request(self, vehicle, vehicles, uavs, base_stations, satellites, grid_size, current_time, slot, no_of_content_each_category, zipf_value):
        # Generate unique random ID
        unique_id = random.randint(1, 999999)
        alpha = zipf_value
        # Combine lists of satellites, UAVs, and grid elements
        #all_entities = len(satellites) + len(uavs)
        #print(all_entities)
        all_entities = len(satellites) + len(uavs) + (grid_size * grid_size)

        # Choose the entity based on Zipf distribution

        #element_index = (np.random.zipf(3) % all_entities) +1
        #print(element_index)
        #element_index = (np.random.zipf(2) % all_entities) + 1


        # Define parameters
        #shape_parameter = 0.5  # You can adjust this based on your requirement
        #size = 1
        # Generate Pareto-distributed values
        #pareto_values = np.random.pareto(shape_parameter, size)
        # Map the values to the desired integer range
        #element_index = int(np.round(pareto_values * (all_entities - 1) + 1))
        # Clip values to make sure they are within the desired range
        #element_index = int(np.clip(element_index, 1, all_entities))
        #print(element_index)

        element_index=self.custom_zipf(alpha, all_entities)
        #print(f"element{element_index}")

        if 1 <= element_index <= len(satellites):
            # Selected entity is a satellite
            entity_type = 'satellite'
            satellite = satellites[f"Satellite{element_index}"]
            # Choose a category from the specific satellite based on Zipf distribution
            category_index = self.custom_zipf(alpha, len(satellite.content_categories))
            content_category = satellite.content_categories[category_index-1]
            # Choose the content number based on Zipf distribution
            content_no = self.custom_zipf(alpha, no_of_content_each_category )
            #print(content_no)
            request_id = f"{vehicle.vehicle_id}_{int(current_time)}_{satellite.satellite_id}_{content_category}_{content_no}"
            cord = satellite.satellite_id
            vehicle_list = [] #to track which vehicle has already processed it

        elif len(satellites) < element_index <= (len(satellites) + len(uavs)):
            entity_type = 'UAV'
            uav = uavs[element_index - len(satellites) - 1]
            # Choose a category from the specific UAV based on Zipf distribution
            category_index = self.custom_zipf(alpha, len(uav.content_categories))
            #print(category_index)
            #category_index = np.random.zipf(2) % len(uav.content_categories)
            content_category = uav.content_categories[category_index-1]
            # Choose the content number based on Zipf distribution
            #content_no = np.random.zipf(2) % no_of_content_each_category + 1
            content_no = self.custom_zipf(alpha, no_of_content_each_category )
            #print(content_no)
            request_id = f"{vehicle.vehicle_id}_{current_time}_{uav.uav_id}_{content_category}_{content_no}"
            cord = uav.uav_id
            vehicle_list = [] #to track which vehicle has already processed it
        else:
            #print('here grid')
            entity_type = 'grid'
            grid = max((element_index - (len(uavs) + len(satellites))) - 1, 0)  # Avoid negative values
            # Choose a category from the specific grid based on Zipf distribution
            category_index = self.custom_zipf(alpha, len(vehicle.content_categories))
            content_category = vehicle.content_categories[category_index-1]
            # Choose the content number based on Zipf distribution
            content_no = self.custom_zipf(alpha, no_of_content_each_category)
            #print(content_no)

            request_id = f"{vehicle.vehicle_id}_{current_time}_gird{grid}_{content_category}_{content_no}"
            cord = grid
            vehicle_list = []  # to track which vehicle has already processed it

        #print(request_id)
        return {'unique_id': unique_id, 'request_id': request_id, 'requesting_vehicle': vehicle, 'g_time': current_time,
                'type': entity_type, 'category': content_category, 'coord': cord, 'no': content_no,
                'RDB': self.get_time_delay(entity_type), 'hop_count': 0, 'time_spent': 0,  'vehicle_list': vehicle_list}

    def get_coordinates_from_index(self, grid_index, grid_size):
        # Calculate the x and y coordinates from the grid_index
        #print('here1', grid_index, grid_size)
        x = grid_index % grid_size
        y = grid_index // grid_size
        return x, y

    # Previous code...
    def get_connected_satellites(self, target_slot, communication_schedule, satellites):
        in_range_satellites = []
        if target_slot in communication_schedule:
            satellite_ids = communication_schedule[target_slot]
            for satellite_id in satellite_ids:
                satellite_key = f"Satellite{satellite_id}"
                if satellite_key in satellites:
                    in_range_satellites.append(satellites[satellite_key])
                    # print("Connected Satellites:", [satellite.satellite_id for satellite in in_range_satellites])
        return in_range_satellites
    def broadcast_request(self, requesting_vehicle, content_request, vehicles, uavs, satellites, base_stations,  grid_size, current_time, slot, communication_schedule, ground_station):
        element_type = content_request['type']
        content_coord = content_request['coord']
        hop_count = content_request['hop_count']
        requested_entity_type = content_request['type']
        requested_coord = content_request['coord']
        requested_category = content_request['category']
        requested_content_no = content_request['no']
        # if the requesting vehicle can generate the content
        #print('from broadcast_request:', content_coord)

        flag1 = 0
        if hop_count==0: #when the request is generated
            if element_type == "satellite" or element_type== "UAV":
                for uav in uavs: #all the generated request would be received by UAVs and update the record
                    if uav.is_within_coverage(requesting_vehicle.current_location[0],requesting_vehicle.current_location[1]):
                        content_request['time_spent']=random.uniform(0.007, 0.01)
                        content=uav.process_content_request(self, requesting_vehicle, content_request, vehicles, uavs, satellites, slot, communication_schedule)
                        if content:
                            flag1 = 1 #if we get content from uav
                            for v in vehicles: #when UAV receive the content , it broadcasts to the region and vehicle and BS takes decision to cache it.
                                if uav.is_within_coverage(v.current_location[0], v.current_location[1]):
                                    v.update_action_space(content, slot, federated_update=False)

                            for bs in base_stations:
                                if uav.is_within_coverage(bs.current_location[0], bs.current_location[1]):
                                    bs.update_action_space(content, slot, federated_update=False)

            if requesting_vehicle.process_content_request(self, content_request, slot, grid_size, flag1): #check for the requesting vehicle
                if element_type == "satellite":
                        if requested_category in ["I", "II", "III"]:
                            with open(f'receiving_time_satellite_{requested_category}_content_{self.alpha}_{self.time_slots}_{self.round}.txt',
                                'a') as file:
                                        file.write(str(self.content_received_time) + '\n')
                if element_type == "UAV":
                    if requested_category in ["II", "III", "IV"]:
                        with open(
                                f'receiving_time_uav_{requested_category}_content_{self.alpha}_{self.time_slots}_{self.round}.txt',
                                'a') as file:
                                        file.write(str(self.content_received_time) + '\n')
                return #if content is avaialble we return becyse this is the minimum'''

        if hop_count < 4:
            new_content_request = copy.deepcopy(content_request)
            new_content_request['hop_count'] += 1
            flag2=0
            for vehicle in vehicles:
                if (vehicle != requesting_vehicle and vehicle.is_within_range(requesting_vehicle.current_location) and
                        vehicle.vehicle_id not in new_content_request['vehicle_list']): #checking whether this request has already processed by this vehicle
                        if vehicle.process_content_request(self, new_content_request, slot, grid_size, flag1):
                            flag1=1
                            break
            for bs in base_stations:
                if bs.check_within_range(requesting_vehicle.current_location):
                      if bs.process_content_request(self, new_content_request, slot, grid_size, ground_station, flag1):
                          flag1=1
            if flag1 == 0 and element_type == "grid":
                if vehicle != requesting_vehicle and vehicle.is_within_range(requesting_vehicle.current_location):
                    requesting_vehicle=vehicle
                    self.broadcast_request(requesting_vehicle, new_content_request, vehicles, uavs, satellites, base_stations,  grid_size, current_time, slot, communication_schedule, ground_station)
            else:
                if element_type == "satellite":
                    if requested_category in ["I", "II", "III"]:
                        with open(
                                f'receiving_time_satellite_{requested_category}_content_{self.alpha}_{self.time_slots}_{self.round}.txt',
                                'a') as file:
                                        file.write(str(self.content_received_time) + '\n')
                if element_type == "UAV":
                    if requested_category in ["II", "III", "IV"]:
                        with open(
                                f'receiving_time_uav_{requested_category}_content_{self.alpha}_{self.time_slots}_{self.round}.txt',
                                'a') as file:
                                        file.write(str(self.content_received_time) + '\n')

                return
        else:
            if element_type == "satellite":
                if requested_category in ["I", "II", "III"]:
                    with open(
                            f'receiving_time_satellite_{requested_category}_content_{self.alpha}_{self.time_slots}_{self.round}.txt',
                            'a') as file:
                                    file.write(str(self.content_received_time) + '\n')
            if element_type == "UAV":
                if requested_category in ["II", "III", "IV"]:
                    with open(
                            f'receiving_time_uav_{requested_category}_content_{self.alpha}_{self.time_slots}_{self.round}.txt',
                            'a') as file:
                                  file.write(str(self.content_received_time) + '\n')
            return

    def run(self, vehicles, uavs, base_stations, satellites, grid_size, current_time, slot, communication_schedule, time_slots, ground_station):
        while not self.content_request_queue.empty(): #this is the loop of request processing
            print(slot)
            content_request = self.content_request_queue.get()
            self.content_received_time= 1000
            self.content_received_time_uav =None
            self.content_received_time_bs= None
            #time_delay_bound = content_request['RDB'] + content_request['g_time']
            requesting_vehicle = content_request['requesting_vehicle']
            self.broadcast_request(requesting_vehicle, content_request, vehicles, uavs, satellites, base_stations, grid_size, current_time, slot, communication_schedule, ground_station) #broadcast to vehicles and UAV
            '''self.content_received_event.wait()
            if self.content_received_time is not None:
                if self.content_received_time <= time_delay_bound:
                    category = content_request['category']
                    if category in self.content_received_count_per_category:
                        self.content_received_count_per_category[category] += 1'''

