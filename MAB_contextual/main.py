from federated_mab import FederatedAggregator
from vehicle_ccn import Vehicle
from uav_ccn import UAV
from satellite_ccn import Satellite
from bs_ccn import BaseStation
from communication import Communication
from gs_ccn import GroundStation
import time
import random
import math
import numpy as np


# Function to run the simulation for a specific time slot value
def run_simulation(alpha, time_slots, num_simulations):
    cache_hit_list = []
    cache_hit_optimal_list = []
    total_hit_for_sagin_links_list_uav= []
    total_hit_for_sagin_links_list_sat = []
    total_hit_from_gs_list = []
    total_hit_from_direct_uav_list = []
    cache_1st_hop_list = []
    cache_2nd_hop_list = []
    cache_hit_uav_list =[]
    total_request_received_list = []
    total_request_for_caching_list = []


    cache_hit_v_list = []
    source_hit_v_list =[]
    request_v_list =[]
    request_v_cache_list=[]

    cache_hit_b_list = []
    source_hit_b_list = []
    request_b_list = []
    request_b_cache_list = []
    no=0

    # Initialize Federated Learning Server
    aggregator = FederatedAggregator()

    for _ in range(num_simulations):
        # Set grid size and UAV grid size
        #grid_size = 1000  # each grid 10 meter in our simulation (10km * 10km)
        grid_size = 100  # each grid 100 meter in our simulation (10km * 10km)
        uav_grid_size = 20 # 2000 meter range //50
        vehicle_range = 10  # 1000 meter
        bs_range = 15  # 1.5 km range
        vehicle_count = 50 #varibale
        satellite_count = 9 #varible
        bs_count = 3 #varible
        no_of_content_each_category= 10

        uav_content_generation_period=10
        no_of_request_genertaed_in_each_timeslot=5
        epsilon=0.1

        # Calculate the number of rows and columns to evenly distribute the vehicles
        num_rows = int(math.sqrt(vehicle_count))
        num_cols = vehicle_count // num_rows

        # Calculate the step size for distributing vehicles within rows and columns
        row_step = grid_size // num_rows
        col_step = grid_size // num_cols

        # Create vehicles with evenly distributed initial positions
        vehicles = []

        for i in range(num_rows):
            for j in range(num_cols):
                # Calculate the initial position for the vehicle
                x = j * col_step + col_step // 2
                y = i * row_step + row_step // 2

                # Create the vehicle and add it to the list
                vehicle = Vehicle(vehicle_id="V" + str(len(vehicles) + 1), grid_size=grid_size, vehicle_range=vehicle_range, vehicle_speed=random.randint(1, 5), aggregator=aggregator)
                vehicle.current_location = (x, y)
                vehicles.append(vehicle)


        # Calculate UAV count
        uav_count = (grid_size // uav_grid_size) ** 2

        # Create UAVs //change for spatio_temporal_request_generation
        uavs = [UAV(uav_id=f"UAV{i}", grid_size=grid_size, uav_grid_size=uav_grid_size, aggregator=aggregator,
                    current_zipf=(random.uniform(0.25, 2.0))) for i in range(uav_count)]

        # Add neighbors to UAVs
        for i in range(uav_count):
            # Calculate the neighboring UAVs based on the grid size and UAV grid size
            neighboring_uavs = []
            row = i // (grid_size // uav_grid_size)
            col = i % (grid_size // uav_grid_size)

            # Top neighbor
            if row > 0:
                neighboring_uavs.append(uavs[i - (grid_size // uav_grid_size)])

            # Bottom neighbor
            if row < (grid_size // uav_grid_size) - 1:
                neighboring_uavs.append(uavs[i + (grid_size // uav_grid_size)])

            # Left neighbor
            if col > 0:
                neighboring_uavs.append(uavs[i - 1])

            # Right neighbor
            if col < (grid_size // uav_grid_size) - 1:
                neighboring_uavs.append(uavs[i + 1])

            # Add neighbors to the UAV
            uavs[i].neighbors = neighboring_uavs

        # Create Satellites and store them in a dictionary
        satellites = {f"Satellite{i+1}": Satellite(satellite_id=f"Satellite{i+1}") for i in range(satellite_count)}
        # Create Base Stations
        base_stations = [BaseStation(bs_id=f"BS{i}", bs_range=bs_range, grid_size=grid_size, aggregator=aggregator) for i in range(bs_count)]

        # Create Ground Station
        ground_station = GroundStation(gs_id="GroundStation")

        no += 1
        # Create Communication instance
        communication = Communication(satellites, base_stations, vehicles, uavs, ground_station, alpha, no, time_slots)


        consecutive_slots_g = 10  # Number of consecutive time slots a satellite stays connected to GS
        cooldown_slots = 20 # Number of consecutive time slots a satellite remains unavailable after disconnection
        output_file = 'output.txt'
        # Initialize communication_schedule as a dictionary of lists
        communication_schedule = {slot: [] for slot in range(1, time_slots + 1)}

        # Initialize a list of available satellites
        available_satellites = list(range(1, satellite_count + 1))

        # Dictionary to track when each satellite becomes available again
        satellite_cooldown = {}
        connected_satellite = None

        for slot in range(1, time_slots+1):
            # Check if it's time to connect a new satellite
            if (slot - 1) % consecutive_slots_g == 0:
                communication_schedule[slot] = []

                # If there are available satellites and we need to connect more, do so
                while len(communication_schedule[slot]) < 3 and available_satellites:
                    # Select a random satellite from available ones
                    connected_satellite = random.choice(available_satellites)
                    # Set the cooldown end slot for this satellite
                    satellite_cooldown[connected_satellite] = (slot-1) + consecutive_slots_g + cooldown_slots
                    # Remove the connected satellite from the list of available ones
                    available_satellites.remove(connected_satellite)
                    # Add the connected satellite to the set of up satellites for the current slot
                    communication_schedule[slot].append(connected_satellite)

                    # For consecutive_slots_g - 1 slots, assign the same connected satellites
                for i in range(1, consecutive_slots_g):
                    communication_schedule[slot + i] = set(communication_schedule[slot])

            # Check if any satellite becomes available
            for satellite, cooldown_end_slot in satellite_cooldown.copy().items():
                if slot >= cooldown_end_slot:
                     available_satellites.append(satellite)
                     del satellite_cooldown[satellite]

        # Print communication_schedule vertically
        '''for slot, sl in communication_schedule.items():
            print(f"Time Slot {slot}: {', '.join([f'Satellite{s}' for s in sl])}")'''

        # Run the simulation
        # Function to check the communication_schedule and retrieve connected satellites
        '''for satellite_id, satellite_obj in satellites.items():
        print(f"Satellite ID: {satellite_id}, Satellite Object: {satellite_obj}")'''

# running the other files one by one
        for slot in range(1, time_slots+1):
            #print('slot', slot)
            current_time = (slot-1) * 60
            # Check if it's time to perform actions for satellite
            if (slot - 1) % consecutive_slots_g == 0:
                for satellite_id in satellites:
                        satellites[satellite_id].run(satellites, communication, communication_schedule, current_time, slot, no_of_content_each_category, ground_station)
                ground_station.run(current_time, satellites)

            for uav in uavs:
                    uav.run(current_time, communication, communication_schedule, slot, satellites, no_of_content_each_category, uav_content_generation_period, epsilon, uavs) # clean cache and caching

            for vehicle in vehicles:
                    vehicle.run(current_time, slot, time_slots, vehicles, uavs, base_stations, satellites, communication, grid_size, no_of_request_genertaed_in_each_timeslot, no_of_content_each_category)

            for bs in base_stations:
                    bs.run(current_time, slot)
                    # Aggregate updates periodically (e.g., every 5 rounds)

            if slot > 10:
                if ((slot - 1) % 10 == 0):
                    aggregator.aggregate_updates()


            communication.run(vehicles, uavs, base_stations, satellites, grid_size, current_time, slot, communication_schedule, time_slots, ground_station)

        cache_hit=0

        hit_for_sagin_links_uav=0
        hit_for_sagin_links_sat =0
        hit_from_direct_uav=0
        optimal_hit = 0
        cache_hit_u=0
        cache_1st_hop = 0
        cache_2nd_hop = 0
        total_request_received=0
        total_request_from_other_sources = 0

        b_cache_hit = 0
        b_source_hit = 0
        b_request = 0
        b_request_caching = 0
        bs_content_hit_from_gs = 0

        v_cache_hit = 0
        v_source_hit = 0
        v_request = 0
        v_request_caching = 0
        avg_decision_latency=0
        avg_decision_latency_v=0

        cache_hit += communication.content_hit #total cache

        for uav in uavs:
            cache_hit_u += uav.content_hit  #total_cache_hit
            optimal_hit += uav.optimal_content_hit #optimal_hit
            cache_1st_hop += uav.content_hit_cache_1st_hop
            cache_2nd_hop += uav.content_hit_cache_2nd_hop
            hit_for_sagin_links_uav += uav.content_hit_source_uav_2nd_hop
            hit_for_sagin_links_sat += uav.content_hit_source_sat_2nd_hop
            hit_from_direct_uav  +=  uav.content_hit_source_uav_1st_hop
            total_request_received += uav.global_request_receive_count
            total_request_from_other_sources += uav.request_receive_from_other_source
            avg_decision_latency = uav.avg_decision_latency / uav.decision_count if uav.decision_count > 0 else 0.0

        for bs in base_stations:
            b_cache_hit += bs.content_cache_hit
            b_source_hit += bs.content_hit_from_source
            b_request += bs.request_receive
            b_request_caching += bs.request_receive_cache
            bs_content_hit_from_gs += bs.content_hit_from_gs

        for v in vehicles:
            v_cache_hit += v.content_cache_hit
            v_source_hit += v.content_hit_from_source
            v_request += v.request_receive
            v_request_caching += v.request_receive_cache
            avg_decision_latency_v = v.avg_decision_latency / v.decision_count if v.decision_count > 0 else 0.0

        #````````````````````````````````````````````````````````````````````````````````````````````

        cache_hit_list.append(cache_hit)  #total cache

        total_hit_from_direct_uav_list.append(hit_from_direct_uav)  # direct UAV
        total_hit_for_sagin_links_list_uav.append(hit_for_sagin_links_uav)  # sagin links or 2nd hop
        total_hit_for_sagin_links_list_sat.append(hit_for_sagin_links_sat)  # sagin links or 2nd hop
        total_request_received_list.append(total_request_received)  # total request receive by UAV
        cache_hit_optimal_list.append(optimal_hit) #optimal cache
        cache_hit_uav_list.append(cache_hit_u) #vehicle_cache
        cache_1st_hop_list.append(cache_1st_hop)
        cache_2nd_hop_list.append(cache_2nd_hop)
        total_request_for_caching_list.append(total_request_from_other_sources) #total request for other source

        cache_hit_v_list.append(v_cache_hit)
        source_hit_v_list.append(v_source_hit)
        request_v_list.append(v_request)
        request_v_cache_list.append(v_request_caching)

        cache_hit_b_list.append(b_cache_hit)
        source_hit_b_list.append(b_source_hit)
        request_b_list.append(b_request)
        request_b_cache_list.append(b_request_caching)
        total_hit_from_gs_list.append(bs_content_hit_from_gs)  # gs_cache

        print(f"values for total caching: total cache hit:{cache_hit}")

        print(f"hit values for uav: hit_from_direct_hit: {hit_from_direct_uav}, hit from GS: {bs_content_hit_from_gs}, hit for SAGIN links_uav: {hit_for_sagin_links_uav}, hit for SAGIN links_sat: {hit_for_sagin_links_sat}"
              f"values cache_hit_details: total hit from uav: {cache_hit_u}, optimal_hit :{optimal_hit}, cache_1st_hop: {cache_1st_hop}, cache_2nd_hop: {cache_2nd_hop}"
              f"values for requests: total request received : {total_request_received}, total request for caching: {total_request_from_other_sources}")

        print(f"values for vehicles: hit from source: {v_source_hit}, hit from cache: {v_cache_hit}, total request: {v_request}, request for cache:{v_request_caching}")
        print(f"values for bs: hit from source: {b_source_hit}, hit from cache: {b_cache_hit}, total request: {b_request}, request for cache:{b_request_caching}")
        print(f"average decision latency:{avg_decision_latency}")
        print(f"average decision latency:{avg_decision_latency_v}")
        file.write(f"values for total caching: total cache hit:{cache_hit} \n")
        file.write(f"hit values for uav: hit_from_direct_hit: {hit_from_direct_uav}, hit from GS: {bs_content_hit_from_gs}, hit for SAGIN links_uav: {hit_for_sagin_links_uav}, hit for SAGIN links_sat: {hit_for_sagin_links_sat}"
            f"values cache_hit_details: total hit from uav: {cache_hit_u}, optimal_hit :{optimal_hit}, cache_1st_hop: {cache_1st_hop}, cache_2nd_hop: {cache_2nd_hop}"
            f"values for requests: total request receved : {total_request_received}, total request for caching: {total_request_from_other_sources} \n")

        file.write(
            f"values for vehicles: hit from source: {v_source_hit}, hit from cache: {v_cache_hit}, total request: {v_request}, request for cache:{v_request_caching} \n")
        file.write(
            f"values for bs: hit from source: {b_source_hit}, hit from cache: {b_cache_hit}, total request: {b_request}, request for cache:{b_request_caching} \n")

    avg_cache_hit=np.mean(cache_hit_list) # total cache

    avg_hit_from_gs = np.mean(total_hit_from_gs_list)
    avg_hit_from_direct_hit = np.mean(total_hit_from_direct_uav_list)
    avg_hit_for_SAGIN_links_uav = np.mean(total_hit_for_sagin_links_list_uav)
    avg_hit_for_SAGIN_links_sat = np.mean(total_hit_for_sagin_links_list_sat)
    avg_total_request_received = np.mean(total_request_received_list)
    avg_total_request_received_for_other_sources = np.mean(total_request_for_caching_list)
    avg_cache_hit_1st_hop = np.mean(cache_1st_hop_list)
    avg_cache_hit_2nd_hop = np.mean(cache_2nd_hop_list)
    avg_cache_hit_u = np.mean(cache_hit_uav_list)
    avg_cache_hit_optimal = np.mean(cache_hit_optimal_list)


    avg_cache_hit_v=np.mean(cache_hit_v_list)
    avg_source_hit_v=np.mean(source_hit_v_list)
    avg_request_v=np.mean(request_v_list)
    avg_request_cache_v=np.mean(request_v_cache_list)

    avg_cache_hit_b=np.mean(cache_hit_b_list)
    avg_source_hit_b=np.mean(source_hit_b_list)
    avg_request_b=np.mean(request_b_list)
    avg_request_cache_b=np.mean(request_b_cache_list)



    return avg_cache_hit, avg_hit_from_gs, avg_hit_from_direct_hit, avg_hit_for_SAGIN_links_uav, avg_hit_for_SAGIN_links_sat, avg_total_request_received,\
        avg_total_request_received_for_other_sources, avg_cache_hit_1st_hop, avg_cache_hit_2nd_hop, avg_cache_hit_u, \
        avg_cache_hit_optimal, avg_cache_hit_v, avg_source_hit_v, avg_request_v, avg_request_cache_v, avg_cache_hit_b, \
        avg_source_hit_b, avg_request_b, avg_request_cache_b

# Run simulations for different time slot values
time_slot_values = [100]
alphas= [0.5]
num_simulations = 1

# Open a file for writing
for time_slots in time_slot_values:
    for alpha in alphas:
        with open(f'simulation_results_{time_slots}_{alpha}.txt', 'a') as file:
            avg_cache_hit, avg_hit_from_gs, avg_hit_from_direct_hit, avg_hit_for_SAGIN_links_uav, avg_hit_for_SAGIN_links_sat, avg_total_request_received, \
            avg_total_request_received_for_other_sources, avg_cache_hit_1st_hop, avg_cache_hit_2nd_hop, avg_cache_hit_u, \
            avg_cache_hit_optimal, avg_cache_hit_v, avg_source_hit_v, avg_request_v, avg_request_cache_v, avg_cache_hit_b, \
            avg_source_hit_b, avg_request_b, avg_request_cache_b = run_simulation(alpha, time_slots, num_simulations)

            # Write the results to the file
            print(f"\nResults for value {alpha} for {time_slots} time slots (averaged over {num_simulations} simulations):\n")
            print(f"Average total content hit: {avg_cache_hit}\n")
            file.write(f"\nResults for value {alpha} for {time_slots} time slots (averaged over {num_simulations} simulations):\n")
            file.write(f"Average total content hit: {avg_cache_hit}\n")
            file.write(f"Average total content hit: {avg_cache_hit/avg_total_request_received_for_other_sources}\n")
            file.write(f"Average decision latency_UAV: {avg_decision_latency}\n")
            file.write(f"Average decision latency_vehicle: {avg_decision_latency_v}\n")


            file.write(f"Average direct_hit_uav: {avg_hit_from_direct_hit}\n")
            file.write(f"Average direct_hit_for_SAGIN_UAV_links: {avg_hit_for_SAGIN_links_uav}\n")
            file.write(f"Average direct_hit_for_SAGIN_SAT_links: {avg_hit_for_SAGIN_links_sat}\n")

            #file.write(f"Average optimal content Hit_uav: {avg_cache_hit_optimal}\n")
            file.write(f"Cache hit ratio_uav: {avg_cache_hit_u}\n")
            file.write(f"Cache hit ratio_uav: {avg_cache_hit_u / avg_total_request_received_for_other_sources}\n")
            file.write(f"Optimal Cache hit ratio_uav: {avg_cache_hit_optimal / avg_total_request_received_for_other_sources}\n")
            file.write(f" Cache hit 1st hop: {avg_cache_hit_1st_hop  / avg_total_request_received_for_other_sources}\n")
            file.write(f"Cache hit 2nd hop: {avg_cache_hit_2nd_hop / avg_total_request_received_for_other_sources}\n")

            file.write(f"Average source hit from vehicle: {avg_source_hit_v}\n")
            file.write(f"Average cache hit from vehicle: {avg_cache_hit_v}\n")
            file.write(f"Average request for cache to vehicle: {avg_request_cache_v}\n")
            file.write(f"Average request to vehicle: {avg_request_v}\n")

            file.write(f"Average source hit from BS: {avg_source_hit_b}\n")
            file.write(f"Average cache hit from BS: {avg_cache_hit_b}\n")
            file.write(f"Average request for cache to BS: {avg_request_cache_b}\n")
            file.write(f"Average request to BD: {avg_request_b}\n")
