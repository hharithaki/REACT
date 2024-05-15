import time
from simulation.unity_simulator import comm_unity
from unity_simulator import utils_viz

# remove objects
def remove_obj_from_environment(obj, comm, graph):
    ids = [node['id'] for node in graph['nodes']]
    obj_ids = [node['id'] for node in graph['nodes'] if node['class_name'] == obj]
    for obj_id in obj_ids:
        if obj_id in ids:
            edges_to_remove = [edge for edge in graph['edges'] if edge['to_id'] == obj_id or edge['from_id'] == obj_id]
            for edge in edges_to_remove:
                graph['edges'].remove(edge)

            nodes_to_remove = [node for node in graph['nodes'] if node['id'] == obj_id]
            for node in nodes_to_remove:
                graph['nodes'].remove(node)
            success, message = comm.expand_scene(graph)

def clean_graph(comm, graph, objects):
    for obj in objects:
        remove_obj_from_environment(obj, comm, graph)
    utils_viz.clean_graph(graph)
    success1, message = comm.expand_scene(graph)
    success2, graph = comm.environment_graph()
    return success1, message, success2, graph

def add_missing_objects(comm, graph):
    # Get nodes for differnt objects
    kitchen_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchen'][0]
    kitchencounter_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchencounter'][0]
    kitchencabinet_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchencabinet'][0]
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]

    # add new objects and update env
    # utils_viz.add_node(graph, {'id': 1000, 'category': 'Props', 'class_name': 'cuttingboard', 'properties': [], 'states': []})
    # utils_viz.add_edge(graph, 1000, 'ON', kitchencounter_id)
    # success, message = comm.expand_scene(graph, ignore_placing_obstacles=True)
    # success, graph = comm.environment_graph()

    utils_viz.add_node(graph, {'id': 1002, 'category': 'Food', 'class_name': 'bread', 'properties': [], 'states': []})
    utils_viz.add_edge(graph, 1002, 'INSIDE', kitchencabinet_id)
    success, message = comm.expand_scene(graph, ignore_placing_obstacles=True)
    success, graph = comm.environment_graph()

    return success, graph

# initiate the simluator
comm = comm_unity.UnityCommunication()

# select the environment
env_id = 1
comm.reset(env_id)

# get the graph
success, graph = comm.environment_graph()

# remove unnecessary objects and prepare the domain
success1, message, success2, graph = clean_graph(comm, graph, ['chicken'])

# add missing objects
sucess, graph = add_missing_objects(comm, graph)

with open("graph1.txt", 'w') as f:  
    for key, value in graph.items():  
        f.write('%s:%s\n' % (key, value))

# Get nodes for differnt objects
kitchen_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchen'][0]
kitchencounter_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchencounter'][0]
kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
stove_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
bench_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'bench'][0]
dishwasher_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'dishwasher'][0]
dishwashingliquid_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'dishwashingliquid'][0]
# coffeemaker_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'coffeemaker'][0]
# coffeepot_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'coffeepot'][0]
# mug_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'mug'][0]
# fridge_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fridge'][0]
breadslice_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'breadslice'][0]
# cutting_board_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cuttingboard'][0]
knife_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cutleryknife'][0]
plate_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'plate'][0]
# bread_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'bread'][0]
cutlets_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cutlets'][0]
fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
condimentshaker_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'condimentshaker'][0]
poundcake_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'poundcake'][0]
chocolatesyrup_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'chocolatesyrup'][0]
# apple_id = 259 # [node['id'] for node in graph['nodes'] if node['class_name'] == 'apple'][0]
waterglass_id = 103 # bathroom otherwise [node['id'] for node in graph['nodes'] if node['class_name'] == 'waterglass'][0]
# Add agents
comm.add_character('Chars/Female1', initial_room='kitchen')
# Add agents
comm.add_character('Chars/Male1', initial_room='kitchen')
# action script
script = [
    # '<char0> [walk] <kitchen> ({})'.format(kitchen_id),
    # '<char0> [find] <plate> ({})'.format(plate_id),
    # '<char0> [grab] <plate> ({})'.format(plate_id),
    # '<char0> [find] <breadslice> ({})'.format(breadslice_id),
    # '<char0> [grab] <breadslice> ({})'.format(breadslice_id),
    # '<char0> [putback] <plate> ({}) <kitchentable> ({})'.format(plate_id, kitchentable_id),
    # '<char0> [find] <plate> ({})'.format(plate_id),
    # '<char0> [putback] <breadslice> ({}) <kitchentable> ({})'.format(breadslice_id,  kitchentable_id),
    # '<char0> [find] <cutlets> ({})'.format(cutlets_id),
    # '<char0> [grab] <cutlets> ({})'.format(cutlets_id),    
    # '<char0> [find] <fryingpan> ({})'.format(fryingpan_id),
    # '<char0> [putback] <cutlets> ({}) <fryingpan> ({})'.format(cutlets_id,  fryingpan_id),
    # '<char0> [switchon] <stove> ({})'.format(stove_id),
    # '<char0> [find] <condimentshaker> ({})'.format(condimentshaker_id),
    # '<char0> [grab] <condimentshaker> ({})'.format(condimentshaker_id),
    # '<char0> [putback] <condimentshaker> ({}) <fryingpan> ({})'.format(condimentshaker_id,  fryingpan_id),
    # '<char0> [grab] <condimentshaker> ({})'.format(condimentshaker_id),
    # '<char0> [putback] <condimentshaker> ({}) <kitchencounter> ({})'.format(condimentshaker_id,  kitchencounter_id),
    # '<char0> [find] <cutleryknife> ({})'.format(knife_id),
    # '<char0> [grab] <cutleryknife> ({})'.format(knife_id),
    # '<char0> [putback] <cutleryknife> ({}) <kitchentable> ({})'.format(knife_id,  kitchentable_id),
    # '<char0> [switchoff] <stove> ({})'.format(stove_id),
    # '<char0> [find] <plate> ({})'.format(plate_id+1),
    # '<char0> [grab] <plate> ({})'.format(plate_id+1),
    # '<char0> [putback] <plate> ({}) <kitchentable> ({})'.format(plate_id+1, kitchentable_id),
    # '<char0> [find] <cutlets> ({})'.format(cutlets_id),
    # '<char0> [grab] <cutlets> ({})'.format(cutlets_id),
    # '<char0> [find] <plate> ({})'.format(plate_id+1),
    # '<char0> [putback] <cutlets> ({}) <kitchentable> ({})'.format(cutlets_id, kitchentable_id),
    # '<char0> [find] <poundcake> ({})'.format(poundcake_id),
    # '<char0> [grab] <poundcake> ({})'.format(poundcake_id),
    # '<char0> [find] <microwave> ({})'.format(microwave_id),
    # '<char0> [open] <microwave> ({})'.format(microwave_id),
    # '<char0> [putin] <poundcake> ({}) <microwave> ({})'.format(poundcake_id, microwave_id),
    # '<char0> [close] <microwave> ({})'.format(microwave_id),
    # '<char0> [switchon] <microwave> ({})'.format(microwave_id),
    # '<char0> [find] <bench> ({})'.format(bench_id),
    # '<char0> [sit] <bench> ({})'.format(bench_id), # ASP eat action will be mapped to here together with sit; pretending to be eating?
    # '<char0> [find] <cutleryknife> ({})'.format(knife_id),
    # '<char0> [grab] <cutleryknife> ({})'.format(knife_id),
    # '<char0> [find] <breadslice> ({})'.format(breadslice_id),
    # '<char0> [grab] <breadslice> ({})'.format(breadslice_id),
    # '<char0> [eat] <breadslice> ({})'.format(breadslice_id),
    # '<char0> [putback] <breadslice> ({}) <kitchentable> ({})'.format(breadslice_id, kitchentable_id),
    # '<char0> [putback] <cutleryknife> ({}) <kitchentable> ({})'.format(cutleryknife_id, kitchentable_id),
    # '<char0> [standup]',
    # '<char0> [find] <microwave> ({})'.format(microwave_id),
    # '<char0> [switchoff] <microwave> ({})'.format(microwave_id),
    # '<char0> [open] <microwave> ({})'.format(microwave_id),
    # '<char0> [find] <poundcake> ({})'.format(poundcake_id),
    # '<char0> [grab] <poundcake> ({})'.format(poundcake_id),
    # '<char0> [close] <microwave> ({})'.format(microwave_id),
    # '<char0> [find] <plate> ({})'.format(plate_id+2),
    # '<char0> [grab] <plate> ({})'.format(plate_id+2),
    # '<char0> [putback] <plate> ({}) <kitchentable> ({})'.format(plate_id+2, kitchentable_id),
    # '<char0> [putback] <poundcake> ({}) <kitchentable> ({})'.format(poundcake_id, kitchentable_id),
    # '<char0> [find] <chocolatesyrup> ({})'.format(chocolatesyrup_id),
    # '<char0> [grab] <chocolatesyrup> ({})'.format(chocolatesyrup_id),
    '<char0> [find] <poundcake> ({}) | <char1> [find] <poundcake> ({})'.format(poundcake_id, poundcake_id),
    '<char0> [grab] <poundcake> ({}) | <char1> [grab] <poundcake> ({})'.format(poundcake_id, poundcake_id)
    # '<char0> [putback] <chocolatesyrup> ({}) <kitchentable> ({})'.format(chocolatesyrup_id, kitchentable_id), # supposedly on poundcake
    # '<char0> [find] <apple> ({})'.format(apple_id),
    # '<char0> [grab] <apple> ({})'.format(apple_id),
    # '<char0> [putback] <apple> ({}) <kitchentable> ({})'.format(apple_id, kitchentable_id), # eat
    # '<char0> [standup]',
    # '<char0> [find] <mug> ({})'.format(mug_id),
    # '<char0> [grab] <mug> ({})'.format(mug_id),
    # '<char0> [putback] <mug> ({}) <kitchencounter> ({})'.format(mug_id, kitchencounter_id),
    # '<char0> [find] <coffeemaker> ({})'.format(coffeemaker_id),
    # '<char0> [switchon] <coffeemaker> ({})'.format(coffeemaker_id),
    # '<char0> [find] <coffeepot> ({})'.format(coffeepot_id),
    # '<char0> [grab] <coffeepot> ({})'.format(coffeepot_id),
    # '<char0> [putback] <coffeepot> ({}) <mug> ({})'.format(coffeepot_id, mug_id)
    # '<char0> [find] <waterglass> ({})'.format(waterglass_id),
    # '<char0> [grab] <waterglass> ({})'.format(waterglass_id),
    # '<char0> [drink] <waterglass> ({})'.format(waterglass_id)
    # '<char0> [find] <dishwasher> ({})'.format(dishwasher_id),
    # '<char0> [open] <dishwasher> ({})'.format(dishwasher_id),
    # '<char0> [putback] <waterglass> ({}) <dishwasher> ({})'.format(waterglass_id, dishwasher_id),
    # '<char0> [find] <plate> ({})'.format(plate_id),
    # '<char0> [grab] <plate> ({})'.format(plate_id),
    # '<char0> [putback] <plate> ({}) <dishwasher> ({})'.format(plate_id, dishwasher_id),
    # '<char0> [find] <plate> ({})'.format(plate_id+1),
    # '<char0> [grab] <plate> ({})'.format(plate_id+1),
    # '<char0> [putback] <plate> ({}) <dishwasher> ({})'.format(plate_id+1, dishwasher_id),
    # '<char0> [find] <plate> ({})'.format(plate_id+2),
    # '<char0> [grab] <plate> ({})'.format(plate_id+2),
    # '<char0> [putback] <plate> ({}) <dishwasher> ({})'.format(plate_id+2, dishwasher_id),
    # '<char0> [find] <cutleryknife> ({})'.format(knife_id),
    # '<char0> [grab] <cutleryknife> ({})'.format(knife_id),
    # '<char0> [putback] <cutleryknife> ({}) <dishwasher> ({})'.format(knife_id, dishwasher_id),
    # '<char0> [find] <fryingpan> ({})'.format(fryingpan_id),
    # '<char0> [grab] <fryingpan> ({})'.format(fryingpan_id),
    # '<char0> [putback] <fryingpan> ({}) <dishwasher> ({})'.format(fryingpan_id, dishwasher_id),
    # '<char0> [find] <dishwashingliquid> ({})'.format(dishwashingliquid_id),
    # '<char0> [grab] <dishwashingliquid> ({})'.format(dishwashingliquid_id),
    # '<char0> [putback] <dishwashingliquid> ({}) <dishwasher> ({})'.format(dishwashingliquid_id, dishwasher_id),
    # '<char0> [close] <dishwasher> ({})'.format(dishwasher_id),
    # '<char0> [switchon] <dishwasher> ({})'.format(dishwasher_id)
]
    # '<char0> [open] <kitchencabinet> ({})'.format(kitchen_cabinet_id),
    # '<char0> [find] <bread> ({})'.format(bread_id),
    # '<char0> [grab] <bread> ({})'.format(bread_id),
    # '<char0> [close] <kitchencabinet> ({})'.format(kitchen_cabinet_id),
    # '<char0> [putback] <bread> ({}) <cuttingboard> ({})'.format(bread_id, cutting_board_id)
    # '<char0> [find] <bread> ({})'.format(bread_id)

# for script_instruction in script:
#     print(script_instruction)
#     act_sucess, msg = comm.render_script([script_instruction], recording=False, skip_animation=True)
#     print(act_sucess)
#     # get the graph, observation
#     success, graph = comm.environment_graph()

# View the sumulation
success, response = comm.render_script(script, recording=True, frame_rate=20, camera_mode=["PERSON_FROM_BACK"])
print(success, response)
# success, graph = comm.environment_graph()
with open("graph2.txt", 'w') as f:  
    for key, value in graph.items():  
        f.write('%s:%s\n' % (key, value))
    