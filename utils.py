from itertools import permutations
import random
import re
import subprocess
import time
import numpy as np
from simulation.unity_simulator import utils_viz, comm_unity
from sklweka.classifiers import Classifier
from sklweka.dataset import to_instance
delimeters = ['(', ')', ',']
human_asp_pre = 'ASP/human_pre.sp'
human_asp = 'ASP/human.sp'
ah_asp_pre = 'ASP/ahagent_pre.sp'
ah_asp_new = 'ASP/ahagent.sp'
display_marker = 'display'
human_model = 'human_model.model'
interacted_items = ['None', 'None']
categorya_food = ['cutlets']
categoryb_food = ['poundcake']
food = ['breadslice'] + categorya_food + categoryb_food
drinks = ['waterglass']
sittable = ['bench']
electricals = ['microwave']
appliances = electricals + ['stove']
containers = ['fryingpan']
graspable = food + drinks + containers
objects = appliances + graspable + sittable
heated_ = [[obj,False] for obj in categoryb_food]
cooked_ = [[obj,False] for obj in categorya_food]

def process_graph(graph, prev_actions):

    state = []
    # Previous action of the agent (include multiple actions in their particular order?)
    # state.append(prev_actions) -- TODO
    act = prev_actions[0].split()
    if len(act) == 4:
        state.append('_'.join([act[1][1:-1],act[2][1:-1]]))
    elif len(act) == 6:
        state.append('_'.join([act[1][1:-1],act[2][1:-1],act[4][1:-1]]))
    else:
        state.append('find_watergalss')

    # Item interactions (immediately previous interaction item or multiple items?)
    script_split = prev_actions[1].split()
    if len(script_split) == 4:
        state.append('_'.join([script_split[1][1:-1],script_split[2][1:-1]]))
        interacted_items.pop(0)
        interacted_items.append(script_split[2][1:-1])
    elif len(script_split) == 6:
        state.append('_'.join([script_split[1][1:-1],script_split[2][1:-1],script_split[4][1:-1]]))
        interacted_items.pop(0)
        # interacted_items.pop(0)
        interacted_items.append(script_split[2][1:-1])
        # interacted_items.append(script_split[4][1:-1])
    else:
        state.append('find_waterglass')
        interacted_items.pop(0)
        interacted_items.append('waterglass')
        interacted_items.pop(0)
        interacted_items.append('waterglass')
    state.append(interacted_items[0])
    state.append(interacted_items[1])
    
    # Location of the agent
    human_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
    state.append(human_pose['position'][0]) # x
    state.append(human_pose['position'][1]) # y
    state.append(human_pose['position'][2]) # z
    state.append(human_pose['rotation'][0]) # x
    state.append(human_pose['rotation'][1]) # y
    state.append(human_pose['rotation'][2]) # z

    # Proximity to the kitchen table
    kitchentable_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    prox_kitchentable = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(kitchentable_pose['position']))    
    state.append(prox_kitchentable)

    # Proximity to the kitchen counter
    kitchentable_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'kitchencounter'][0]
    prox_kitchencounter = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(kitchentable_pose['position']))
    state.append(prox_kitchencounter)

    # Status of microwave (on/off/open/closed).
    microwave_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    state.append(microwave_status[0])
    state.append(microwave_status[1])

    #  Items inside microwave
    microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == microwave_id and edge['relation_type'] == 'INSIDE']
    item_name = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        item_name.append(names)
    item_name = [item for idx, item in enumerate(item_name) if item != 'plate']
    state.append(item_name[0] if len(item_name) > 0 else 'None')
    
    # Status of Stove (on/off/open/closed).
    stove_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    state.append(stove_status[0])
    state.append(stove_status[1])

    #  Items on stove (fryingpan) - since it cannot be grabbed
    fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == fryingpan_id and edge['relation_type'] == 'ON']
    item_name = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        item_name.append(names)
    state.append(item_name[0] if len(item_name)>0 else 'None')

    # Items currently on the dinning table
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    edges = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == kitchentable_id and edge['relation_type'] == 'ON']
    table_items = [node['class_name']  for edge in edges for node in graph['nodes'] if node['id'] == edge]

    # Number of items on the dinning table.
    no_table_items = len(table_items)
    state.append(no_table_items)

    return state

def process_observation(script_instruction, graph, prev_actions):
    state = []
    # Previous action of the agent (include multiple actions in their particular order?)
    # state.append(prev_actions) -- TODO
    act = prev_actions[0].split()
    if len(act) == 4:
        state.append('_'.join([act[1][1:-1],act[2][1:-1]]))
    elif len(act) == 6:
        state.append('_'.join([act[1][1:-1],act[2][1:-1],act[4][1:-1]]))
    else:
        state.append(prev_actions[0])

    # Item interactions (immediately previous interaction item or multiple items?)
    script_split = prev_actions[1].split()
    if len(script_split) == 4:
        state.append('_'.join([script_split[1][1:-1],script_split[2][1:-1]]))
        interacted_items.pop(0)
        interacted_items.append(script_split[2][1:-1])
    elif len(script_split) == 6:
        state.append('_'.join([script_split[1][1:-1],script_split[2][1:-1],script_split[4][1:-1]]))
        interacted_items.pop(0)
        # interacted_items.pop(0)
        interacted_items.append(script_split[2][1:-1])
        # interacted_items.append(script_split[4][1:-1])
    else:
        state.append(prev_actions[1])
    state.append(interacted_items[0])
    state.append(interacted_items[1])
    
    # Location of the agent
    human_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
    state.append(human_pose['position'][0]) # x
    state.append(human_pose['position'][1]) # y
    state.append(human_pose['position'][2]) # z
    state.append(human_pose['rotation'][0]) # x
    state.append(human_pose['rotation'][1]) # y
    state.append(human_pose['rotation'][2]) # z

    # Proximity to the kitchen table
    kitchentable_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    prox_kitchentable = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(kitchentable_pose['position']))    
    state.append(prox_kitchentable)

    # Proximity to the kitchen counter
    kitchentable_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'kitchencounter'][0]
    prox_kitchencounter = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(kitchentable_pose['position']))
    state.append(prox_kitchencounter)

    # Status of microwave (on/off/open/closed).
    microwave_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    state.append(microwave_status[0])
    state.append(microwave_status[1])

    #  Items inside microwave
    microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == microwave_id and edge['relation_type'] == 'INSIDE']
    item_name = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        item_name.append(names)
    item_name = [item for idx, item in enumerate(item_name) if item != 'plate']
    state.append(item_name[0] if len(item_name) > 0 else 'None')
    
    # Status of Stove (on/off/open/closed).
    stove_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    state.append(stove_status[0])
    state.append(stove_status[1])

    #  Items on stove (fryingpan) - since it cannot be grabbed
    fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == fryingpan_id and edge['relation_type'] == 'ON']
    item_name = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        item_name.append(names)
    state.append(item_name[0] if len(item_name)>0 else 'None')

    # Items currently on the dinning table
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    edges = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == kitchentable_id and edge['relation_type'] == 'ON']
    table_items = [node['class_name']  for edge in edges for node in graph['nodes'] if node['id'] == edge]
    # state.append(tuple(table_items))

    # Number of items on the dinning table.
    no_table_items = len(table_items)
    state.append(no_table_items)

    # Inventory changes such as number of breadslices in the toaster, number of plates on the kitchen counter.
    # toaster_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'toaster'][0]
    # breadslice_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'breadslice']
    # toaster_bread = 0
    # table_bread = 0
    # for id in breadslice_id:
    #     edges = [edge for edge in graph['edges'] if edge['from_id'] == id and edge['to_id'] == toaster_id and edge['relation_type'] == 'INSIDE']
    #     toaster_bread += len(edges)
    #     edges = [edge for edge in graph['edges'] if edge['from_id'] == id and edge['to_id'] == kitchentable_id and edge['relation_type'] == 'ON']
    #     table_bread += len(edges)
    # state.append(toaster_bread)
    # state.sppend(table_bread)

    # action of the agent
    act = script_instruction.split()
    if len(act) == 4:
        state.append('_'.join([act[1][1:-1],act[2][1:-1]]))
    elif len(act) == 6:
        state.append('_'.join([act[1][1:-1],act[2][1:-1],act[4][1:-1]]))
    return state

# Depending on the accuracy of the model we can decide to expand the number of features by including:
#     - proximity to the microwave
#     - proximity to the stove
#     - multiple previous actions of the agent
#     - sepreate interacted_items list for food, appliances, furniture and surfaces
#     - bench
#     - Time since last action (might need to include a clock) - usually the actions are considerably fast; hence this might not help much.

def get_effects(action):
    # action and effect
    effect_dict = {
        'find': 'found',
        'grab': 'in_hand',
        'put': 'on',
        'put_in': 'inside',
        'eat': 'ate',
        'drink': 'drank',
        'sit': 'sat',
        'stand': 'stood',
        'open': 'opened',
        'close': 'closed',
        'switch_on': 'switched_on',
        'switch_off': 'switched_off'
    }
    for delimeter in delimeters:
        action = " ".join(action.split(delimeter))
    action = action.split()
    effect = 'holds(' + effect_dict[action[1]] + '('
    if action[1] in (['find', 'grab', 'eat', 'drink', 'sit', 'stand']):
        effect = effect + action[2] + ',' + action[3] + '),' + str(int(action[4])+1) + ').'
    elif action[1] in (['open', 'close', 'switch_on', 'switch_off']):
       effect = effect + action[3] + '),' + str(int(action[4])+1) + ').'
    elif action[1] in (['put', 'put_in']):
        effect = effect + action[3] + ',' + action[4] + '),' + str(int(action[5])+1) + ').'
    return effect

def get_goal_achived(graph):
    table_goal = False 
    # all items on the dinning table
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    edges = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == kitchentable_id and edge['relation_type'] == 'ON']
    table_items = [node['class_name']  for edge in edges for node in graph['nodes'] if node['id'] == edge]
    goal_list = ['poundcake', 'cutlets', 'waterglass', 'breadslice']
    if (all(item in table_items for item in goal_list)):
        table_goal = True
    # two characters -> first human
    character_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
    # first bench id
    bench_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'bench'][0]
    # human sitting at the table
    item_id = [edge for edge in graph['edges'] if edge['from_id'] == character_id and edge['to_id'] == bench_id and edge['relation_type'] == 'SITTING']
    if len(item_id) > 0 and table_goal:
        return True
    else:
        return False

def convert_state(graph, prev_human_actions, prev_ah_actions, human_success, ah_success, timestep):
    human_fluents = []
    ah_fluents = []
    fluents = []
    human_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 1 and edge['relation_type'] == 'HOLDS_RH']
    ah_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 2 and edge['relation_type'] == 'HOLDS_RH']
    human_hand_objects = []
    ah_hand_objects = []

    for item in human_object_ids: # objects in human hand
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        human_hand_objects.append(names)
    for item in ah_object_ids: # objects in ah agent hand
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        ah_hand_objects.append(names)

    # % --------------- % find
    if 'find' in prev_human_actions[-1] and human_success:
        action = (prev_human_actions[-1].split())[2][1:-1] # <char0> [find] <poundcake> (248) True
        for obj in objects:
            if obj == action:
                human_fluents.append('holds(found(human,' + action + '),' + timestep + ').')
                ah_fluents.append('holds(agent_found(human,' + action + '),' + timestep + ').')
            else:
                human_fluents.append('-holds(found(human,' + obj + '),' + timestep + ').') 
    else:
        human_found_set = False
        for obj in objects:
            if (obj in appliances or obj in sittable) and prev_human_actions[0] != 'None' and not human_found_set:
                if obj == (prev_human_actions[0].split())[2][1:-1]:
                    human_fluents.append('holds(found(human,' + obj + '),' + timestep + ').')
                    ah_fluents.append('holds(agent_found(human,' + obj + '),' + timestep + ').')
                    human_found_set = True
                else:
                    human_fluents.append('-holds(found(human,' + obj + '),' + timestep + ').')
            elif obj in human_hand_objects and not human_found_set:
                human_fluents.append('holds(found(human,' + obj + '),' + timestep + ').')
                ah_fluents.append('holds(agent_found(human,' + obj + '),' + timestep + ').')
                human_found_set = True
            else:
                human_fluents.append('-holds(found(human,' + obj + '),' + timestep + ').')
    if 'find' in prev_ah_actions[-1] and ah_success:
        action = (prev_ah_actions[-1].split())[2][1:-1] # <char0> [find] <poundcake> (248) True
        for obj in objects:
            if obj == action:
                ah_fluents.append('holds(found(ahagent,' + action + '),' + timestep + ').')
                human_fluents.append('holds(agent_found(ahagent,' + action + '),' + timestep + ').')
            else:
                ah_fluents.append('-holds(found(ahagent,' + obj + '),' + timestep + ').') 
    else:
        ah_found_set = False
        for obj in objects:
            if (obj in appliances or obj in sittable) and prev_ah_actions[0] != 'None' and not ah_found_set:
                if obj == (prev_ah_actions[0].split())[2][1:-1]:
                    ah_fluents.append('holds(found(ahagent,' + obj + '),' + timestep + ').')
                    human_fluents.append('holds(agent_found(ahagent,' + obj + '),' + timestep + ').')
                    ah_found_set = True
                else:
                    ah_fluents.append('-holds(found(ahagent,' + obj + '),' + timestep + ').')
            elif obj in ah_hand_objects and not ah_found_set:
                ah_fluents.append('holds(found(ahagent,' + obj + '),' + timestep + ').')
                human_fluents.append('holds(agent_found(ahagent,' + obj + '),' + timestep + ').')
                ah_found_set = True
            else:
                ah_fluents.append('-holds(found(ahagent,' + obj + '),' + timestep + ').')

    # % --------------- % grab
    if 'grab' in prev_human_actions[-1] and human_success:
        action = (prev_human_actions[-1].split())[2][1:-1]
        for obj in graspable:
            if obj == action:
                human_fluents.append('holds(in_hand(human,' + action + '),' + timestep + ').')
                ah_fluents.append('holds(agent_hand(human,' + action + '),' + timestep + ').')
                new_fluent = 'holds(found(human,' + obj + '),' + timestep + ').'
                human_fluents = [fluent for fluent in human_fluents if fluent != '-'+new_fluent]
                human_fluents.append(new_fluent)
                ah_fluents.append('holds(agent_found(human,' + obj + '),' + timestep + ').')
                human_found_set = True
            else:
                human_fluents.append('-holds(in_hand(human,' + obj + '),' + timestep + ').')
    else:
        for obj in graspable:
            if obj in human_hand_objects:
                human_fluents.append('holds(in_hand(human,' + obj + '),' + timestep + ').')
                ah_fluents.append('holds(agent_hand(human,' + obj + '),' + timestep + ').')
            else:
                human_fluents.append('-holds(in_hand(human,' + obj + '),' + timestep + ').')
    if 'grab' in prev_ah_actions[-1] and ah_success:
        action = (prev_ah_actions[-1].split())[2][1:-1]
        for obj in graspable:
            if obj == action:
                ah_fluents.append('holds(in_hand(ahagent,' + action + '),' + timestep + ').')
                human_fluents.append('holds(agent_hand(ahagent,' + action + '),' + timestep + ').')
                new_fluent = 'holds(found(ahagent,' + obj + '),' + timestep + ').'
                ah_fluents = [fluent for fluent in ah_fluents if fluent != '-'+new_fluent]
                ah_fluents.append(new_fluent)
                human_fluents.append('holds(agent_found(ahagent,' + obj + '),' + timestep + ').')
                ah_found_set = True
            else:
                ah_fluents.append('-holds(in_hand(ahagent,' + obj + '),' + timestep + ').')
    else:
        for obj in graspable:
            if obj in ah_hand_objects:
                ah_fluents.append('holds(in_hand(ahagent,' + obj + '),' + timestep + ').')
                human_fluents.append('holds(agent_hand(ahagent,' + obj + '),' + timestep + ').')
            else:
                ah_fluents.append('-holds(in_hand(ahagent,' + obj + '),' + timestep + ').')
    # % --------------- % put
    # Items on dinning table
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    edges = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == kitchentable_id and edge['relation_type'] == 'ON']
    table_items = [node['class_name']  for edge in edges for node in graph['nodes'] if node['id'] == edge]
    for obj in graspable:
        if obj in table_items:
            fluents.append('holds(on(' + obj + ',kitchentable),' + timestep + ').')
        else:
            fluents.append('-holds(on(' + obj + ',kitchentable),' + timestep + ').')
    # Items inside microwave
    microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == microwave_id and edge['relation_type'] == 'INSIDE']
    microwave_item = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        microwave_item.append(names)
    for obj in graspable:
        if obj in microwave_item:
            fluents.append('holds(on('+ obj + ',microwave),' + timestep + ').')
        else:
            fluents.append('-holds(on('+ obj + ',microwave),' + timestep + ').')
    # Items on stove
    fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
    stove_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    pan_on_stove = [True for edge in graph['edges'] if edge['from_id'] == fryingpan_id and edge['to_id'] == stove_id and edge['relation_type'] == 'ON']
    pan_on_stove = pan_on_stove[0] if pan_on_stove else False
    for obj in graspable:
        if obj == 'fryingpan' and pan_on_stove:
            fluents.append('holds(on('+ obj + ',stove),' + timestep + ').')
        else:
            fluents.append('-holds(on('+ obj + ',stove),' + timestep + ').')
    # temp assumption => nothing on stove
    # default_fluents = ['-holds(on('+ obj + ',stove),0).' for obj in graspable]
    # fluents = fluents + default_fluents

    # % --------------- % open/close
    # Status of microwave
    microwave_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    print('-------------', microwave_status)
    if 'CLOSED' in microwave_status:
        fluents.append('-holds(opened(microwave),' + timestep + ').')
    elif 'OPEN' in microwave_status:
        fluents.append('holds(opened(microwave),' + timestep + ').')

    # % --------------- % switch on/off
    # Status of microwave
    if 'OFF' in microwave_status:
        fluents.append('-holds(switched_on(microwave),' + timestep + ').')
    elif 'ON' in microwave_status:
        fluents.append('holds(switched_on(microwave),' + timestep + ').')
    # Status of Stove
    stove_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    if 'OFF' in stove_status:
        fluents.append('-holds(switched_on(stove),' + timestep + ').')
    elif 'ON' in stove_status:
        fluents.append('holds(switched_on(stove),' + timestep + ').')

    # % --------------- % heated
    for obj in categoryb_food: # poundcake
        heated_idx = [idx for idx, item in enumerate(heated_) if item[0] == obj][0]
        if (obj in microwave_item and 'ON' in microwave_status) or heated_[heated_idx][1]:
            fluents.append('holds(heated(' + obj + '),' + timestep + ').')
            heated_[heated_idx][1] = True
        else:
            fluents.append('-holds(heated(' + obj + '),' + timestep + ').')

    # % --------------- % cooked
    # Items on fryingpan
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == fryingpan_id and edge['relation_type'] == 'ON']
    pan_items = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        pan_items.append(names)
    for obj in categorya_food: # cutlets
        cooked_idx = [idx for idx, item in enumerate(cooked_) if item[0] == obj][0]
        if (obj in pan_items and 'ON' in stove_status and pan_on_stove) or cooked_[cooked_idx][1]:
            fluents.append('holds(cooked(' + obj + '),' + timestep + ').')
            cooked_[cooked_idx][1] = True
        else:
            fluents.append('-holds(cooked(' + obj + '),' + timestep + ').')
    # % --------------- % ate => once we reach the eat action we will stop running the prg since goal has been reached hence adding some defaults here.
    if 'eat' in prev_human_actions[-1] and human_success:
        obj1 = (prev_human_actions[-1].split())[2][1:-1]
        if 'eat' in prev_human_actions[0]:
            obj2 = (prev_human_actions[0].split())[2][1:-1]
            for obj in food:
                if obj == obj1 or obj == obj2:
                    human_fluents.append('holds(ate(human,' + obj + '),' + timestep + ').')
                else:
                    human_fluents.append('-holds(ate(human,' + obj + '),' + timestep + ').')
        else:
            for obj in food:
                if obj == obj1:
                    human_fluents.append('holds(ate(human,' + obj + '),' + timestep + ').')
                else:
                    human_fluents.append('-holds(ate(human,' + obj + '),' + timestep + ').')
    else:
        default_fluents = ['-holds(ate(human,' + obj + '),' + timestep + ').' for obj in food]
        human_fluents = human_fluents + default_fluents
    # % --------------- % drink => same
    default_fluents = ['-holds(drank(human,' + obj + '),' + timestep + ').' for obj in drinks]
    human_fluents = human_fluents + default_fluents
    # % --------------- % put_in
    for obj in categorya_food:
        if obj in pan_items:
            fluents.append('holds(inside(' + obj + ',fryingpan),' + timestep + ').')
        else:
            fluents.append('-holds(inside(' + obj + ',fryingpan),' + timestep + ').')
    # % --------------- % sit
    if 'sit' in prev_human_actions[-1] and human_success:
        action = (prev_human_actions[-1].split())[2][1:-1]
        for obj in sittable:
            if obj == action:
                human_fluents.append('holds(sat(human,' + action + '),' + timestep + ').')
            else:
                human_fluents.append('-holds(sat(human,' + obj + '),' + timestep + ').')
    else:
        default_fluents = ['-holds(sat(human,' + obj + '),' + timestep + ').' for obj in sittable]
        human_fluents = human_fluents + default_fluents
    return human_fluents, ah_fluents, fluents

def answer_set_finder(expression, answer):
    if not re.search('[\>\<\=\!]',expression):
        expression = re.sub('I\+1', 'I', expression)
        expression = re.sub('\(', '\(', expression)
        expression = re.sub('\)', '\)', expression)
        expression = re.sub('[A-Z][0-9]?', '[a-z0-9_]+(?:\(.+?\))?', expression)
        literal = re.findall("(?<!-)"+expression, answer)
    else:
        literal = [expression]
    return literal

def process_answerlist(answer):
    answer_list = answer_set_finder('occurs(A,I)', answer)
    action_list = []
    for i in range(len(answer_list)):
        for element in answer_list:
            if re.search(rf',{i}\)$',element) != None:
                action_list.insert(i, element)
    return action_list

def generate_script(human_act, ah_act, id_dict, human_character, ah_character):
    script = []
    # select the agent with the longest plan
    plan_len = len(human_act) if len(human_act) > len(ah_act) else len(ah_act)
    for action_index in range(plan_len):
        # either of the agents may or may not have an act at the last steps
        if len(human_act) > action_index:
            human_action = human_act[action_index]
            for delimeter in delimeters:
                human_action = " ".join(human_action.split(delimeter))
            human_action_split = human_action.split()
            if human_action_split[1] in ['put', 'put_in']:
                if human_action_split[1] == 'put' and human_action_split[4] in ['microwave']:
                    human_script_instruction = human_character + ' [putin] <' + human_action_split[3] + '> (' + id_dict[human_action_split[3]] + ') <' + human_action_split[4] + '> (' + id_dict[human_action_split[4]] + ')'
                else:
                    human_script_instruction = human_character + ' [putback] <' + human_action_split[3] + '> (' + id_dict[human_action_split[3]] + ') <' + human_action_split[4] + '> (' + id_dict[human_action_split[4]] + ')'
            elif human_action_split[1] in ['eat', 'drink']:
                human_script_instruction = None
            else:
                human_script_instruction = human_character + ' [' + human_action_split[1].replace('_','') + '] <' + human_action_split[3] + '> (' + id_dict[human_action_split[3]] + ')'
        else:
            human_script_instruction = None
        if len(ah_act) > action_index:
            ah_action = ah_act[action_index]
            for delimeter in delimeters:
                ah_action = " ".join(ah_action.split(delimeter))
            ah_action_split = ah_action.split()
            if ah_action_split[1] in ['put', 'put_in']:
                if ah_action_split[1] == 'put' and ah_action_split[4] in ['microwave']:
                    ah_script_instruction = ah_character + ' [putin] <' + ah_action_split[3] + '> (' + id_dict[ah_action_split[3]] + ') <' + ah_action_split[4] + '> (' + id_dict[ah_action_split[4]] + ')'
                else:
                    ah_script_instruction = ah_character + ' [putback] <' + ah_action_split[3] + '> (' + id_dict[ah_action_split[3]] + ') <' + ah_action_split[4] + '> (' + id_dict[ah_action_split[4]] + ')'
            else:
                ah_script_instruction = ah_character + ' [' + ah_action_split[1].replace('_','') + '] <' + ah_action_split[3] + '> (' + id_dict[ah_action_split[3]] + ')'
        else:
            ah_script_instruction = None
        script_instruction = (human_script_instruction + '|' + ah_script_instruction) if human_script_instruction and ah_script_instruction else (human_script_instruction if human_script_instruction else ah_script_instruction)
        script.append(script_instruction)
    return script

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

def predict_next_action(graph, prev_human_actions):
    values = process_graph(graph, prev_human_actions)
    if (any(word in values[0] for word in ['bench','eat','drink'])) or (any(word in values[1] for word in ['bench','eat','drink'])):
        # no trianing data
        return None
    model, header = Classifier.deserialize(human_model)
    # create new instance
    inst = to_instance(header,values) # Instance.create_instance(values)
    inst.dataset = header
    # make prediction
    index = model.classify_instance(inst)
    return header.class_attribute.value(int(index))

# block: cannot keep a single env since it is impossible to revert the predicted action execution effects and revert back to original script
def get_future_state(script_time, graph, ah_fluents, common_fluents, prev_human_actions, prev_ah_actions, env_id, id_dict, current_script):
    future_action = predict_next_action(graph, prev_human_actions)
    print(future_action)
    if not future_action:
        return graph, prev_human_actions, prev_ah_actions, current_script, False, False, script_time
    if 'grab' in future_action:
        ah_positive_in_hand = [item for item in ah_fluents if item.startswith('holds(in_hand(ahagent,')]
        action_split = future_action.split('_')
        obj = action_split[1]
        if ('holds(in_hand(ahagent,'+ obj + '),0).' in ah_positive_in_hand):
            return graph, prev_human_actions, prev_ah_actions, current_script, False, False, script_time
    # assume ad hoc agent does nothing; only human action added to script
    future_action = future_action.split('_')
    if len(future_action) == 2:
        future_action = '<char0> [' + future_action[0] + '] <' + future_action[1] + '> (' + id_dict[future_action[1]] + ')'
    else:
        future_action = '<char0> [' + future_action[0] + '] <' + future_action[1] + '> (' + id_dict[future_action[1]] + ') <' + future_action[2] + '> (' + id_dict[future_action[2]] + ')'
    current_script.append(future_action)
    
    start_time = time.time()
    # initiate a second env
    comm_dummy = comm_unity.UnityCommunication(port='8082')
    comm_dummy.reset(env_id)
    success_dummy, graph_dummy = comm_dummy.environment_graph()
    success1_dummy, message_dummy, success2_dummy, graph_dummy = clean_graph(comm_dummy, graph_dummy, ['chicken'])

    # Add human
    comm_dummy.add_character('Chars/Female1', initial_room='kitchen')
    # Add ad hoc agent
    comm_dummy.add_character('Chars/Male1', initial_room='kitchen')


    for script_instruction in current_script:
        act_success, human_success, ah_success, message = comm_dummy.render_script([script_instruction], recording=False, skip_animation=True)
    script_time = script_time + (time.time()-start_time)
    # Get the state observation
    success, graph = comm_dummy.environment_graph()
    if human_success:
        prev_human_actions.pop(0)
        prev_human_actions.append(future_action)
    else:
        del current_script[-1]
    # next_future_action = predict_next_action(graph, prev_human_actions)
    return graph, prev_human_actions, prev_ah_actions, current_script, human_success, ah_success, script_time

def refine_fluents(script_time, literal_time, graph, ah_fluents, common_fluents, prev_human_actions, prev_ah_actions, env_id, id_dict, current_script):
    all_ah_fluents = ah_fluents.copy()
    all_common_fluents = common_fluents.copy()
    for i in range(2):
        print('-----------------------------------', i)
        graph, prev_human_actions, prev_ah_actions, current_script, human_success, ah_success, script_time = get_future_state(script_time, graph, ah_fluents, common_fluents, prev_human_actions, prev_ah_actions, env_id, id_dict, current_script)
        # process state to fluents
        start = time.time()
        human_fluents, ah_fluents, common_fluents = convert_state(graph, prev_human_actions, prev_ah_actions, human_success, ah_success, str(i))
        # find and merge human fluents
        if i == 0:
            # remove
            tem_ah_fluents = [item for item in ah_fluents if 'agent_found' in item or 'agent_hand' in item]
            all_ah_fluents = [item for item in all_ah_fluents if item not in tem_ah_fluents and '-'+item not in tem_ah_fluents and item[1:] not in tem_ah_fluents]
            all_ah_fluents = all_ah_fluents + tem_ah_fluents
            tem_common_fluents = [item for item in common_fluents if item not in all_common_fluents]
            all_common_fluents = [item for item in all_common_fluents if item not in tem_common_fluents and '-'+item not in tem_common_fluents and item[1:] not in tem_common_fluents]
            all_common_fluents = all_common_fluents + tem_common_fluents
        else:
            tem_ah_fluents = [item for item in ah_fluents if 'agent_found' in item or 'agent_hand' in item]
            all_ah_fluents = all_ah_fluents + tem_ah_fluents
            tem_common_fluents = [item for item in common_fluents if item.replace('),'+str(i)+').','),0).') not in all_common_fluents]
            all_common_fluents = all_common_fluents + tem_common_fluents
        end = time.time()
        literal_time = literal_time + (end-start)
    return all_ah_fluents, all_common_fluents, script_time, literal_time

# return answer sets for the new ASP file
def run_ASP_human(ASP_time, graph, prev_human_actions, prev_ah_actions, human_success, ah_success, human_counter):
    found_solution = False
    answer_split = None
    counter = human_counter
    positive_counter = True
    reader = open(human_asp_pre, 'r')
    pre_asp = reader.read()
    reader.close()
    pre_asp_split = pre_asp.split('\n')
    display_marker_index = pre_asp_split.index(display_marker)
    human_fluents, ah_fluents, common_fluents = convert_state(graph, prev_human_actions, prev_ah_actions, human_success, ah_success, '0')
    start_time = time.time()
    while (not found_solution) or counter == 0:
        const_term = ['#const n = ' + str(counter) + '.']
        asp_split = const_term + pre_asp_split[:display_marker_index] + human_fluents + common_fluents + pre_asp_split[display_marker_index:]
        asp = '\n'.join(asp_split)
        f1 = open(human_asp, 'w')
        f1.write(asp)
        f1.close()
        try:
            sub_start_time = time.time()
            answer = subprocess.check_output('java -jar ASP/sparc.jar ' +human_asp+' -A -n 1',shell=True, timeout=10)
            sub_end_time = time.time()
        except subprocess.TimeoutExpired as exec:
            print('command timed out')
            counter = counter-1
            continue
        answer_split = (answer.decode('ascii'))
        if len(answer_split) > 1:
            found_solution = True
            human_counter = counter
            end_time = time.time()
            ASP_time = ASP_time + (end_time-start_time) - (sub_end_time-sub_start_time)
        if counter > 0 and positive_counter:
            counter = counter-1 # in case
        else:
            counter = counter+1
            positive_counter = False
    actions = process_answerlist(answer_split)
    return actions, ah_fluents, common_fluents, human_counter, ASP_time

# return answer sets for the new ASP file
def run_ASP_ahagent(ASP_time, script_time, literal_time, total_time, graph, ah_fluents, common_fluents, prev_human_actions, prev_ah_actions, ah_counter, env_id, id_dict, current_script, step):
    goal = False
    found_solution = False
    answer_split = None
    counter = ah_counter
    start_time = time.time()
    ah_fluents, common_fluents, script_time, literal_time = refine_fluents(script_time, literal_time, graph, ah_fluents, common_fluents, prev_human_actions, prev_ah_actions, env_id, id_dict, current_script)
    end_time = time.time()
    total_time = total_time + (end_time-start_time)
    positive_counter = True
    start_time = time.time()
    while (not found_solution) or counter == 0:
        const_term = ['#const n = ' + str(counter) + '.']
        reader = open(ah_asp_pre, 'r')
        pre_asp = reader.read()
        reader.close()
        pre_asp_split = pre_asp.split('\n')
        display_marker_index = pre_asp_split.index(display_marker)
        asp_split = const_term + pre_asp_split[:display_marker_index] + ah_fluents + common_fluents + pre_asp_split[display_marker_index:]
        asp = '\n'.join(asp_split)
        f1 = open(ah_asp_new, 'w')
        f1.write(asp)
        f1.close()
        try:
            sub_start_time = time.time()
            answer = subprocess.check_output('java -jar ASP/sparc.jar ' +ah_asp_new+' -A -n 1',shell=True, timeout=10)
            sub_end_time = time.time()
        except subprocess.TimeoutExpired as exec:
            print('command timed out')
            counter = counter-1
            continue
        answer_split = (answer.decode('ascii'))
        if len(answer_split) > 1:
            found_solution = True
            ah_counter = counter
            end_time = time.time()
            ASP_time = ASP_time + (end_time-start_time) - (sub_end_time-sub_start_time)
        if counter > 0 and positive_counter:
            counter = counter-1 # in case
        else:
            counter = counter+1
            positive_counter = False
    actions = process_answerlist(answer_split)
    # if step == 2:
    #     with open("asp_404_2.sp", 'w') as f:    
    #         f.write(asp)
    #     with open("script_404.txt", 'w') as f2:    
    #         f2.write(str(current_script))
    return actions, ah_counter, ASP_time, script_time, literal_time, total_time

def generate_initialscript(human_act, ah_act, id_dict, human_character, ah_character):
    script = []
    # select the agent with the longest plan
    plan_len = len(human_act) if len(human_act) > len(ah_act) else len(ah_act)
    for action_index in range(plan_len):
        # either of the agents may or may not have an act at the last steps
        if len(human_act) > action_index:
            human_action = human_act[action_index]
            for delimeter in delimeters:
                human_action = " ".join(human_action.split(delimeter))
            human_action_split = human_action.split()
            if human_action_split[1] in ['put']:
                human_script_instruction = human_character + ' [putback] <' + human_action_split[3] + '> (' + id_dict[human_action_split[3]] + ') <' + human_action_split[4] + '> (' + id_dict[human_action_split[4]] + ')'
            else:
                human_script_instruction = human_character + ' [' + human_action_split[1].replace('_','') + '] <' + human_action_split[3] + '> (' + id_dict[human_action_split[3]] + ')'
        else:
            human_script_instruction = None
        if len(ah_act) > action_index:
            ah_action = ah_act[action_index]
            for delimeter in delimeters:
                ah_action = " ".join(ah_action.split(delimeter))
            ah_action_split = ah_action.split()
            ah_script_instruction = ah_character + ' [' + ah_action_split[1].replace('_','') + '] <' + ah_action_split[3] + '> (' + id_dict[ah_action_split[3]] + ')'
        else:
            ah_script_instruction = None
        script_instruction = human_script_instruction + '|' + ah_script_instruction if human_script_instruction and ah_script_instruction else (human_script_instruction if human_script_instruction else ah_script_instruction)
        script.append(script_instruction)
    return script

def select_initialstate(id_dict, human_character, ah_character):
    human_act = []
    ah_act = []
    items = ['breadslice', 'poundcake', 'waterglass', 'cutlets', 'empty', 'empty']
    surfaces = ['kitchentable', 'microwave', 'default', 'kitchentable', 'ah_hand', 'human_hand']
    arrangements = list(permutations(items))
    selected = arrangements[633] # random.choice(arrangements)
    object_surfaces = [[i,s] for i, s in zip(selected, surfaces)]
    print(object_surfaces)
    # drop the default and empty elements
    hand_items = object_surfaces[4:]
    object_surfaces = [ob_su for ob_su in object_surfaces if ob_su[1] != 'default' and ob_su[1] != 'ah_hand' and ob_su[1] != 'human_hand' and ob_su[0] != 'empty']
    microwave_door_state = ['open','close'][0] # random.choice(['open','close'])
    microwave_status = ['off','on'][1] # random.choice(['off','on'])
    stove_status = ['off','on'][1] # random.choice(['off','on'])
    print(microwave_status, microwave_door_state, stove_status)
    # human will place the object, ad_agent will only perform in_hand
    for ob_sr in object_surfaces:
        human_act.append('occurs(find(human,'+ ob_sr[0] +'),I)')
        human_act.append('occurs(grab(human,'+ ob_sr[0] +'),I)')
        human_act.append('occurs(find(human,'+ ob_sr[1] +'),I)')
        human_act.append('occurs(put(human,' + ob_sr[0] + ',' + ob_sr[1]  + '),I)')
    if microwave_door_state == 'open':
        human_act.append('occurs(find(human,microwave),I)')
        human_act.append('occurs(open(human,microwave),I)') # execut condition if the door is opened cannot be on
    elif microwave_door_state == 'close':
        human_act.append('occurs(find(human,microwave),I)')
        human_act.append('occurs(close(human,microwave),I)')
        if microwave_status == 'on': # default - off
            human_act.append('occurs(find(human,microwave),I)')
            human_act.append('occurs(switch_on(human,microwave),I)')
    if stove_status == 'on': # default - off
        human_act.append('occurs(find(human,stove),I)')
        human_act.append('occurs(switch_on(human,stove),I)')
    for ha_it in hand_items:
        if ha_it[0] != 'empty' and ha_it[1] == 'ah_hand':
            ah_act.append('occurs(find(ah_agent,'+ ha_it[0] +'),I)')
            ah_act.append('occurs(grab(ah_agent,'+ ha_it[0] +'),I)')
        if ha_it[0] != 'empty' and ha_it[1] == 'human_hand':
            human_act.append('occurs(find(human,'+ ha_it[0] +'),I)')
            human_act.append('occurs(grab(human,'+ ha_it[0] +'),I)')
    script = generate_initialscript(human_act, ah_act, id_dict, human_character, ah_character)
    return script, human_act, ah_act
