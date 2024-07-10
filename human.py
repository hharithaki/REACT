import csv
import json
import time
from simulation.unity_simulator import comm_unity
import utils
import sklweka.jvm as jvm

human_counter = 32
ah_counter = 28

# initiate the simluator
comm = comm_unity.UnityCommunication(port='8080')

jvm.start()

# select the environment
env_id = 1
comm.reset(env_id)

# Get the state observation
success, graph = comm.environment_graph()

# remove unnecessary objects and prepare the domain
success1, message, success2, graph = utils.clean_graph(comm, graph, ['chicken'])

# Get nodes for differnt objects
kitchen_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchen'][0]
kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
stove_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
bench_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'bench'][0]
breadslice_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'breadslice'][0]
cutlets_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cutlets'][0]
fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
poundcake_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'poundcake'][0]
waterglass_id = 103 # bathroom otherwise [node['id'] for node in graph['nodes'] if node['class_name'] == 'waterglass'][0]

# collect and process data
id_dict = {
    'kitchentable': str(kitchentable_id),
    'stove': str(stove_id),
    'microwave': str(microwave_id),
    'bench': str(bench_id),
    'breadslice': str(breadslice_id),
    'cutlets': str(cutlets_id),
    'fryingpan': str(fryingpan_id),
    'poundcake': str(poundcake_id),
    'waterglass': str(waterglass_id)
}

# Add human
comm.add_character('Chars/Female1', initial_room='kitchen')
# Add ad hoc agent
comm.add_character('Chars/Male1', initial_room='kitchen')
# Select an initial state
initialscript, human_act, ah_act = utils.select_initialstate(id_dict, '<char0>', '<char1>')
# Prepare domain
for script_instruction in initialscript:
    act_success, human_success, ah_success, message = comm.render_script([script_instruction], recording=False, skip_animation=True)
    print(act_success, human_success, ah_success, message)
    # Get the state observation
    success, graph = comm.environment_graph()

step = 0
prev_human_actions = ['None', 'None']
prev_ah_actions = ['None', 'None']
goal = False
human_success = False
ah_success = False
current_script = initialscript # since the ids are usually same to all envs lets skip retriving ids part for now

# ---------------------------------- START: AD HOC TEAMWORK ---------------------------------- #

while (not goal) and (step != 40):
    # Write to human ASP and Get human action
    human_actions, ah_fluents, common_fluents, human_counter = utils.run_ASP_human(graph, prev_human_actions.copy(), prev_ah_actions.copy(), human_success, ah_success, human_counter)
    ah_actions, ah_counter = utils.run_ASP_ahagent(graph, ah_fluents, common_fluents, prev_human_actions.copy(), prev_ah_actions.copy(), ah_counter, env_id, id_dict, current_script.copy(), step)

    script = utils.generate_script(human_actions, ah_actions, id_dict, '<char0>', '<char1>')
    current_script.append(script[0])
    if '|' in script[0]:
        script_split = script[0].split('|')
        act1 = script_split[0]
        act2 = script_split[1]
        act1_split = (act1.split(' '))[1:]
        act2_split = (act2.split(' '))[1:]
        if act1_split == act2_split and act1_split[0] == '[grab]':
            prev_human_actions.pop(0)
            prev_human_actions.append('None')
            prev_ah_actions.pop(0)
            prev_ah_actions.append(script_split[0])
        else:
            prev_human_actions.pop(0)
            prev_human_actions.append(act1)
            prev_ah_actions.pop(0)
            prev_ah_actions.append(act2)
    else:
        prev_human_actions.pop(0)
        prev_human_actions.append(script[0])
        prev_ah_actions.pop(0)
        prev_ah_actions.append('None')

    # for script_instruction in script:
    act_success, human_success, ah_success, message = comm.render_script([script[0]], recording=False, skip_animation=True)
    
    if not act_success:
        print('#########################################ACTION FAIL###################################################')

    # get the graph
    success, graph = comm.environment_graph()

    # check whether goal is achived and set goal
    goal = utils.get_goal_achived(graph)
    # Previous action of the agent (include multiple actions in their particular order?)
    step = step+1

jvm.stop()
