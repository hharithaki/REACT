import csv
from simulation.unity_simulator import comm_unity
import utils

# initiate the simluator
comm = comm_unity.UnityCommunication()

# select the environment
env_id = 1
comm.reset(env_id)

# get the graph
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

step = 0
prev_human_actions = ['None', 'None']
prev_ah_actions = ['None', 'None']
goal = False
act_success = False
# Add human
comm.add_character('Chars/Female1', initial_room='kitchen')
# Add ad hoc agent
comm.add_character('Chars/Male1', initial_room='kitchen')
# Get the state observation
success, graph = comm.environment_graph()

while not goal:
    # Write to human ASP and Get human action
    human_actions, ah_fluents, common_fluents = utils.run_ASP_human(graph, prev_human_actions, prev_ah_actions, act_success)
    print(human_actions)
    ah_actions = utils.run_ASP_ahagent(graph, ah_fluents, common_fluents, prev_human_actions)
    print(ah_actions)

    human_script = utils.generate_script(human_actions, id_dict, '<char0>')[:-4]
    prev_human_actions.pop(0)
    prev_human_actions.append(human_script[0])
    print(human_script[0])

    # ah_script = utils.generate_script(actions, id_dict, '<char1>')
    # prev_ah_actions.pop(0)
    # prev_ah_actions.append(ah_script[0])
    
    # for script_instruction in script:
    act_success, message = comm.render_script([human_script[0]], recording=False, skip_animation=True)
    print(act_success)

    # get the graph
    success, graph = comm.environment_graph()
    # check whether goal is achived and set goal
    goal = utils.get_goal_achived(graph)
    # Previous action of the agent (include multiple actions in their particular order?)
    step = step+1