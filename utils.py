import re
import subprocess
import numpy as np
from simulation.unity_simulator import utils_viz
from sklweka.classifiers import Classifier
from sklweka.dataset import Instance, missing_value
delimeters = ['(', ')', ',']
human_asp_pre = 'ASP/human_pre.sp'
human_asp = 'ASP/human.sp'
ah_asp_pre = 'ASP/ahagent_pre.sp'
ah_asp_new = 'ASP/ahagent.sp'
display_marker = 'display'
human_model = 'human_model.model'
interacted_items = ['None', 'None', 'None']
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
human_coutner = 35
ah_counter = 30

def process_graph(graph, prev_actions):
    state = []
    # Previous action of the agent
    act = prev_actions[0].split()
    if len(act) == 4:
        state.append(''.join([act[1],act[2]]))
    elif len(act) == 6:
        state.append(''.join([act[1],act[2],act[4]]))
    else:
        state.append(prev_actions[0])

    # Item interactions (immediately previous interaction item or multiple items?)
    script_split = prev_actions[1].split()
    if len(script_split) == 4:
        state.append(''.join([script_split[1],script_split[2]]))
        interacted_items.pop(0)
        interacted_items.append(script_split[2])
    elif len(script_split) == 6:
        state.append(''.join([script_split[1],script_split[2],script_split[4]]))
        interacted_items.pop(0)
        interacted_items.pop(0)
        interacted_items.append(script_split[2])
        interacted_items.append(script_split[4])
    else:
        state.append(prev_actions[1])
    state.append(tuple(interacted_items)) # TODO - seperate to two features an only use two interactive items.
    
    # Location of the agent
    human_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
    state.append(tuple(human_pose['position'])) # TODO - seperate x,y,z coordinates
    state.append(tuple(human_pose['rotation'])) # TODO - seperate x,y,z coordinates

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
        state.append(''.join([act[1],act[2]]))
    elif len(act) == 6:
        state.append(''.join([act[1],act[2],act[4]]))
    else:
        state.append(prev_actions[0])

    # Item interactions (immediately previous interaction item or multiple items?)
    script_split = prev_actions[1].split()
    if len(script_split) == 4:
        state.append(''.join([script_split[1],script_split[2]]))
        interacted_items.pop(0)
        interacted_items.append(script_split[2])
    elif len(script_split) == 6:
        state.append(''.join([script_split[1],script_split[2],script_split[4]]))
        interacted_items.pop(0)
        interacted_items.pop(0)
        interacted_items.append(script_split[2])
        interacted_items.append(script_split[4])
    else:
        state.append(prev_actions[1])
    state.append(tuple(interacted_items)) # TODO - seperate to two features an only use two interactive items.
    
    # Location of the agent
    human_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
    state.append(tuple(human_pose['position'])) # TODO - seperate x,y,z coordinates
    state.append(tuple(human_pose['rotation'])) # TODO - seperate x,y,z coordinates

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
        state.append(''.join([act[1],act[2]]))
    elif len(act) == 6:
        state.append(''.join([act[1],act[2],act[4]]))
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

def convert_state(graph, prev_human_actions, prev_ah_actions, act_success):
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
    if 'find' in prev_human_actions[-1] and act_success:
        action = (prev_human_actions[-1].split())[2][1:-1] # <char0> [find] <poundcake> (248) True
        for obj in objects:
            if obj == action:
                human_fluents.append('holds(found(human,' + action + '),0).')
                ah_fluents.append('holds(agent_found(human,' + action + '),0).')
            else:
                human_fluents.append('-holds(found(human,' + obj + '),0).') 
    else:
        human_found_set = False
        for obj in objects:
            if (obj in appliances or obj in sittable) and prev_human_actions[0] != 'None' and not human_found_set:
                if obj == (prev_human_actions[0].split())[2][1:-1]:
                    human_fluents.append('holds(found(human,' + obj + '),0).')
                    ah_fluents.append('holds(agent_found(human,' + obj + '),0).')
                    human_found_set = True
                else:
                    human_fluents.append('-holds(found(human,' + obj + '),0).')
            elif obj in human_hand_objects and not human_found_set:
                human_fluents.append('holds(found(human,' + obj + '),0).')
                ah_fluents.append('holds(agent_found(human,' + obj + '),0).')
                human_found_set = True
            else:
                human_fluents.append('-holds(found(human,' + obj + '),0).')
    if 'find' in prev_ah_actions[-1] and act_success:
        action = (prev_ah_actions[-1].split())[2][1:-1] # <char0> [find] <poundcake> (248) True
        for obj in objects:
            if obj == action:
                ah_fluents.append('holds(found(ahagent,' + action + '),0).')
                human_fluents.append('holds(agent_found(ahagent,' + action + '),0).')
            else:
                ah_fluents.append('-holds(found(ahagent,' + obj + '),0).') 
    else:
        ah_found_set = False
        for obj in objects:
            if (obj in appliances or obj in sittable) and prev_ah_actions[0] != 'None' and not ah_found_set:
                if obj == (prev_ah_actions[0].split())[2][1:-1]:
                    ah_fluents.append('holds(found(ahagent,' + obj + '),0).')
                    human_fluents.append('holds(agent_found(ahagent,' + obj + '),0).')
                    found_set = True
                else:
                    ah_fluents.append('-holds(found(ahagent,' + obj + '),0).')
            elif obj in ah_hand_objects and not found_set:
                ah_fluents.append('holds(found(ahagent,' + obj + '),0).')
                human_fluents.append('holds(agent_found(ahagent,' + obj + '),0).')
                ah_found_set = True
            else:
                ah_fluents.append('-holds(found(ahagent,' + obj + '),0).')

    # % --------------- % grab
    if 'grab' in prev_human_actions[-1] and act_success:
        action = (prev_human_actions[-1].split())[2][1:-1]
        for obj in graspable:
            if obj == action:
                human_fluents.append('holds(in_hand(human,' + action + '),0).')
                ah_fluents.append('holds(agent_hand(human,' + action + '),0).')
            else:
                human_fluents.append('-holds(in_hand(human,' + obj + '),0).')
    else:
        for obj in graspable:
            if obj in human_hand_objects:
                human_fluents.append('holds(in_hand(human,' + obj + '),0).')
                ah_fluents.append('holds(agent_hand(human,' + obj + '),0).')
            else:
                human_fluents.append('-holds(in_hand(human,' + obj + '),0).')
    if 'grab' in prev_ah_actions[-1] and act_success:
        action = (prev_ah_actions[-1].split())[2][1:-1]
        for obj in graspable:
            if obj == action:
                ah_fluents.append('holds(in_hand(ahagent,' + action + '),0).')
                human_fluents.append('holds(agent_hand(ahagent,' + action + '),0).')
            else:
                ah_fluents.append('-holds(in_hand(ahagent,' + obj + '),0).')
    else:
        for obj in graspable:
            if obj in ah_hand_objects:
                ah_fluents.append('holds(in_hand(ahagent,' + obj + '),0).')
                human_fluents.append('holds(agent_hand(ahagent,' + obj + '),0).')
            else:
                ah_fluents.append('-holds(in_hand(ahagent,' + obj + '),0).')
    # % --------------- % put
    # Items on dinning table
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    edges = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == kitchentable_id and edge['relation_type'] == 'ON']
    table_items = [node['class_name']  for edge in edges for node in graph['nodes'] if node['id'] == edge]
    for obj in graspable:
        if obj in table_items:
            fluents.append('holds(on(' + obj + ',kitchentable),0).')
        else:
            fluents.append('-holds(on(' + obj + ',kitchentable),0).')
    # Items inside microwave
    microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == microwave_id and edge['relation_type'] == 'INSIDE']
    microwave_item = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        microwave_item.append(names)
    for obj in graspable:
        if obj in microwave_item:
            fluents.append('holds(on('+ obj + ',microwave),0).')
        else:
            fluents.append('-holds(on('+ obj + ',microwave),0).')
    # Items on stove
    # fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
    # stove_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    # pan_on_stove = [True for edge in graph['edges'] if edge['from_id'] == fryingpan_id and edge['to_id'] == stove_id and edge['relation_type'] == 'ON'][0]
    # for obj in graspable:
    #     if obj == 'fryingpan' and pan_on_stove:
    #         fluents.append('holds(on('+ obj + ',stove),0).')
    #     else:
    #         fluents.append('-holds(on('+ obj + ',stove),0).')
    # temp assumption => nothing on stove
    default_fluents = ['-holds(on('+ obj + ',stove),0).' for obj in graspable]
    fluents = fluents + default_fluents

    # % --------------- % open/close
    # Status of microwave
    microwave_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    if 'CLOSED' in microwave_status:
        fluents.append('-holds(opened(microwave),0).')
    elif 'OPEN' in microwave_status:
        fluents.append('holds(opened(microwave),0).')

    # % --------------- % switch on/off
    # Status of microwave
    if 'OFF' in microwave_status:
        fluents.append('-holds(switched_on(microwave),0).')
    elif 'ON' in microwave_status:
        fluents.append('holds(switched_on(microwave),0).')
    # Status of Stove
    stove_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    if 'OFF' in stove_status:
        fluents.append('-holds(switched_on(stove),0).')
    elif 'ON' in stove_status:
        fluents.append('holds(switched_on(stove),0).')

    # % --------------- % heated
    for obj in categoryb_food: # poundcake
        heated_idx = [idx for idx, item in enumerate(heated_) if item[0] == obj][0]
        if (obj in microwave_item and 'ON' in microwave_status) or heated_[heated_idx][1]:
            fluents.append('holds(heated(' + obj + '),0).')
            heated_[heated_idx][1] = True
        else:
            fluents.append('-holds(heated(' + obj + '),0).')

    # % --------------- % cooked
    # Items on fryingpan
    fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == fryingpan_id and edge['relation_type'] == 'ON']
    stove_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    pan_on_stove = [True for edge in graph['edges'] if edge['from_id'] == fryingpan_id and edge['to_id'] == stove_id and edge['relation_type'] == 'ON'][0]
    pan_items = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        pan_items.append(names)
    for obj in categorya_food: # cutlets
        cooked_idx = [idx for idx, item in enumerate(cooked_) if item[0] == obj][0]
        if (obj in pan_items and 'ON' in stove_status and pan_on_stove) or cooked_[cooked_idx][1]:
            fluents.append('holds(cooked(' + obj + '),0).')
            cooked_[cooked_idx][1] = True
        else:
            fluents.append('-holds(cooked(' + obj + '),0).')
    # % --------------- % ate => once we reach the eat action we will stop running the prg since virtaul home does not support eat hence adding default values below
    default_fluents = ['-holds(ate(human,' + obj + '),0).' for obj in food]
    human_fluents = human_fluents + default_fluents
    # % --------------- % drink => same
    default_fluents = ['-holds(drank(human,' + obj + '),0).' for obj in drinks]
    human_fluents = human_fluents + default_fluents
    # % --------------- % put_in
    for obj in categorya_food:
        if obj in pan_items:
            fluents.append('holds(inside(' + obj + ',fryingpan),0).')
        else:
            fluents.append('-holds(inside(' + obj + ',fryingpan),0).')
    # % --------------- % sit
    if 'sit' in prev_human_actions[-1] and act_success:
        action = (prev_human_actions[-1].split())[2][1:-1]
        for obj in sittable:
            if obj == action:
                human_fluents.append('holds(sat(human,' + action + '),0).')
            else:
                human_fluents.append('-holds(sat(human,' + obj + '),0).')
    else:
        default_fluents = ['-holds(sat(human,' + obj + '),0).' for obj in sittable]
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

def generate_script(answer, id_dict, character):
    script = []
    for action in answer:
        for delimeter in delimeters:
            action = " ".join(action.split(delimeter))
        action_split = action.split()
        if action_split[1] in ['put', 'put_in']:
            if action_split[1] == 'put' and action_split[4] in ['microwave']:
                script_instruction = character + ' [putin] <' + action_split[3] + '> (' + id_dict[action_split[3]] + ') <' + action_split[4] + '> (' + id_dict[action_split[4]] + ')'
            else:
                script_instruction = character + ' [putback] <' + action_split[3] + '> (' + id_dict[action_split[3]] + ') <' + action_split[4] + '> (' + id_dict[action_split[4]] + ')'
        else:
            script_instruction = character + ' [' + action_split[1].replace('_','') + '] <' + action_split[3] + '> (' + id_dict[action_split[3]] + ')'
        if script_instruction.startswith('<char0> [grab] <fryingpan>') or script_instruction.startswith('<char0> [putback] <fryingpan> (161) <stove>'):
            continue
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
    model, header = Classifier.deserialize(human_model)
    # create new instance
    inst = Instance.create_instance(values)
    inst.dataset = header
    # make prediction
    index = model.classify_instance(inst)
    return header.class_attribute.value(int(index))
    
def refine_fluents(graph, ah_fluents, common_fluents, prev_human_actions):
    action = predict_next_action(graph, prev_human_actions)
    print('TODO')
    return ah_fluents, common_fluents

# return answer sets for the new ASP file
def run_ASP_human(graph, prev_human_actions, prev_ah_action, act_success):
    found_solution = False
    answer_split = None
    counter = human_coutner
    reader = open(human_asp_pre, 'r')
    pre_asp = reader.read()
    reader.close()
    pre_asp_split = pre_asp.split('\n')
    display_marker_index = pre_asp_split.index(display_marker)
    human_fluents, ah_fluents, common_fluents = convert_state(graph, prev_human_actions, prev_ah_action, act_success)
    while (not found_solution) or counter == 0:
        const_term = ['#const n = ' + str(counter) + '.']
        asp_split = const_term + pre_asp_split[:display_marker_index] + human_fluents + common_fluents + pre_asp_split[display_marker_index:]
        asp = '\n'.join(asp_split)
        f1 = open(human_asp, 'w')
        f1.write(asp)
        f1.close()
        try:
            answer = subprocess.check_output('java -jar ASP/sparc.jar ' +human_asp+' -A -n 1',shell=True, timeout=60)
        except subprocess.TimeoutExpired as exec:
            print('command timed out')
            counter = counter-1
            continue
        answer_split = (answer.decode('ascii'))
        if len(answer_split) > 1:
            found_solution = True
            human_coutner = counter
        counter = counter-1 # in case
    actions = process_answerlist(answer_split)
    return actions, ah_fluents, common_fluents

# return answer sets for the new ASP file
def run_ASP_ahagent(graph, ah_fluents, common_fluents, prev_human_actions):
    found_solution = False
    answer_split = None
    counter = ah_coutner
    ah_fluents, common_fluents = refine_fluents(graph, ah_fluents, common_fluents, prev_human_actions)
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
            answer = subprocess.check_output('java -jar ASP/sparc.jar ' +ah_asp_new+' -A -n 1',shell=True, timeout=60)
        except subprocess.TimeoutExpired as exec:
            print('command timed out')
            counter = counter-1
            continue
        answer_split = (answer.decode('ascii'))
        if len(answer_split) > 1:
            found_solution = True
            ah_coutner = counter
        counter = counter-1 # in case
    actions = process_answerlist(answer_split)
    return actions



# graph, ah_fluents, common_fluents, prev_human_actions

#     human_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 1 and edge['relation_type'] == 'HOLDS_RH']
#     ah_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 2 and edge['relation_type'] == 'HOLDS_RH']

#     for item in human_object_ids: # objects in human hand
#         names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
#         human_hand_objects.append(names)
#     for item in ah_object_ids: # objects in ah agent hand
#         names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
#         ah_hand_objects.append(names)
# find(agent,objects) -> found(#agent,#objects)
# grab(agent,graspable) -> in_hand(#agent,#graspable)
# put(agent,graspable,surfaces) -> on(#graspable,#surfaces)
# open(agent,electricals) -> opened(#electricals)
# close(agent,electricals) -> -opened(#electricals)
# switch_on(agent,appliances) -> switched_on(#appliances)
# switch_off(agent,appliances) -> -switched_on(#appliances)
# put_in(agent,categorya_food,cook_containers) -> inside(#categorya_food,#cook_containers)

# heated(#categoryb_food)
# cooked(#categorya_food)

# [find]<bench>
#     # negative found
#     human_negative_found = [item for item in ah_fluents if item.startswith('-holds(found(human,')]
#     ah_negative_found = [item for item in ah_fluents if item.startswith('-holds(found(ah_agent,')]
#     # postivie found
#     human_positive_found = [item for item in ah_fluents if item.startswith('holds(found(human,')]
#     ah_positive_found = [item for item in ah_fluents if item.startswith('holds(found(ah_agent,')]
#     # negative in_hand
#     human_negative_in_hand = [item for item in ah_fluents if item.startswith('-holds(in_hand(human,')]
#     ah_negative_in_hand = [item for item in ah_fluents if item.startswith('-holds(in_hand(ah_agent,')]
#     # postivie in_hand
#     human_positive_in_hand = [item for item in ah_fluents if item.startswith('holds(in_hand(human,')]
#     ah_positive_in_hand = [item for item in ah_fluents if item.startswith('holds(in_hand(ah_agent,')]
#     # negative on
#     negative_on = [item for item in ah_fluents if item.startswith('-holds(on(')]
#     # postivie on
#     positive_on = [item for item in ah_fluents if item.startswith('holds(on(')]
#     # negative opened
#     negative_opened = [item for item in ah_fluents if item.startswith('-holds(opened(')]
#     # postivie opened
#     positive_opened = [item for item in ah_fluents if item.startswith('holds(opened(')]
#     # negative switched_on
#     negative_switched_on = [item for item in ah_fluents if item.startswith('-holds(switched_on(')]
#     # postivie switched_on
#     positive_switched_on = [item for item in ah_fluents if item.startswith('holds(switched_on(')]
#     # negative inside
#     negative_inside = [item for item in ah_fluents if item.startswith('-holds(inside(')]
#     # postivie inside
#     positive_inside = [item for item in ah_fluents if item.startswith('holds(inside(')]
#     found_fluents, in_hand_fluents, on_fluents, opened_fluents, switched_on_fluents, inside_fluents = ([] for i in range(6))
#     if len(re.findall(r'\[.*?\]|\<.*?\>', future_action)) == 2:
#         # % --------------- % find, grab, open, switchoff, close, switchon
#         action = re.findall(r'\[(.*?)\]', future_action) # [find]<poundcake>
#         obj = re.findall(r'\<(.*?)\>', future_action)
#         if action in ['eat','sit','drink'] or obj in ['bench']:
#             return ah_fluents, common_fluents
#         elif action == 'find':
#             # for fluent in ah_fluents:
#             # if the human/ad hoc agent has already found the object ignore the action
#             if ('holds(found(human,' + obj + '),0).' not in human_positive_found) and ('holds(found(ah_agent,' + obj + '),0).' not in ah_positive_found):
#                 # if the human has another found but the immediate prev action is not a find then
#                 # replace that with negative, add new found fluent literal, remove the founds negative
#                 if len(human_positive_found) > 0 and ('find' not in prev_human_actions[-1]):
#                     positive_found = [item.replace('holds(found', '-holds(found') for item in human_positive_found]
#                     new_fluent = 'holds(found(human,' + obj + '),0).')
#                     negative_found = [item for item in human_negative_found if item ! = '-holds(found(human,' + obj + '),0)']
#                     found_fluents = negative_found + ah_negative_found + positive_found + ah_positive_found + new_fluent
#         elif action == 'grab':
#             if ('holds(found(human,'+ obj + '),0).' not in human_positive_found)
#     else:
#         # % --------------- % putin, putback
#         action = re.findall(r'\[.*?\]|\<.*?\>', future_action))[0][1:-1] # [putin]<poundcake><microwave>
#         obj1 = re.findall(r'\[.*?\]|\<.*?\>', future_action))[1][1:-1]
#         obj2 = re.findall(r'\[.*?\]|\<.*?\>', future_action))[2][1:-1]

#     else:
#         human_found_set = False
#         for obj in objects:
#             if (obj in appliances or obj in sittable) and prev_human_actions[0] != 'None' and not human_found_set:
#                 if obj == (prev_human_actions[0].split())[2][1:-1]:
#                     human_fluents.append('holds(found(human,' + obj + '),0).')
#                     ah_fluents.append('holds(agent_found(human,' + obj + '),0).')
#                     human_found_set = True
#                 else:
#                     human_fluents.append('-holds(found(human,' + obj + '),0).')
#             elif obj in human_hand_objects and not human_found_set:
#                 human_fluents.append('holds(found(human,' + obj + '),0).')
#                 ah_fluents.append('holds(agent_found(human,' + obj + '),0).')
#                 human_found_set = True
#             else:
#                 human_fluents.append('-holds(found(human,' + obj + '),0).')



#     if found_fluent
#     if 'find' in prev_ah_actions[-1] and act_success:
#         action = (prev_ah_actions[-1].split())[2][1:-1] # <char0> [find] <poundcake> (248) True
#         for obj in objects:
#             if obj == action:
#                 ah_fluents.append('holds(found(ahagent,' + action + '),0).')
#                 human_fluents.append('holds(agent_found(ahagent,' + action + '),0).')
#             else:
#                 ah_fluents.append('-holds(found(ahagent,' + obj + '),0).') 
#     else:
#         ah_found_set = False
#         for obj in objects:
#             if (obj in appliances or obj in sittable) and prev_ah_actions[0] != 'None' and not ah_found_set:
#                 if obj == (prev_ah_actions[0].split())[2][1:-1]:
#                     ah_fluents.append('holds(found(ahagent,' + obj + '),0).')
#                     human_fluents.append('holds(agent_found(ahagent,' + obj + '),0).')
#                     found_set = True
#                 else:
#                     ah_fluents.append('-holds(found(ahagent,' + obj + '),0).')
#             elif obj in ah_hand_objects and not found_set:
#                 ah_fluents.append('holds(found(ahagent,' + obj + '),0).')
#                 human_fluents.append('holds(agent_found(ahagent,' + obj + '),0).')
#                 ah_found_set = True
#             else:
#                 ah_fluents.append('-holds(found(ahagent,' + obj + '),0).')

#     # % --------------- % grab
#     if 'grab' in prev_human_actions[-1] and act_success:
#         action = (prev_human_actions[-1].split())[2][1:-1]
#         for obj in graspable:
#             if obj == action:
#                 human_fluents.append('holds(in_hand(human,' + action + '),0).')
#                 ah_fluents.append('holds(agent_hand(human,' + action + '),0).')
#             else:
#                 human_fluents.append('-holds(in_hand(human,' + obj + '),0).')
#     else:
#         for obj in graspable:
#             if obj in human_hand_objects:
#                 human_fluents.append('holds(in_hand(human,' + obj + '),0).')
#                 ah_fluents.append('holds(agent_hand(human,' + obj + '),0).')
#             else:
#                 human_fluents.append('-holds(in_hand(human,' + obj + '),0).')
#     if 'grab' in prev_ah_actions[-1] and act_success:
#         action = (prev_ah_actions[-1].split())[2][1:-1]
#         for obj in graspable:
#             if obj == action:
#                 ah_fluents.append('holds(in_hand(ahagent,' + action + '),0).')
#                 human_fluents.append('holds(agent_hand(ahagent,' + action + '),0).')
#             else:
#                 ah_fluents.append('-holds(in_hand(ahagent,' + obj + '),0).')
#     else:
#         for obj in graspable:
#             if obj in ah_hand_objects:
#                 ah_fluents.append('holds(in_hand(ahagent,' + obj + '),0).')
#                 human_fluents.append('holds(agent_hand(ahagent,' + obj + '),0).')
#             else:
#                 ah_fluents.append('-holds(in_hand(ahagent,' + obj + '),0).')
#     # % --------------- % put
#     # Items on dinning table
#     kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
#     edges = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == kitchentable_id and edge['relation_type'] == 'ON']
#     table_items = [node['class_name']  for edge in edges for node in graph['nodes'] if node['id'] == edge]
#     for obj in graspable:
#         if obj in table_items:
#             fluents.append('holds(on(' + obj + ',kitchentable),0).')
#         else:
#             fluents.append('-holds(on(' + obj + ',kitchentable),0).')
#     # Items inside microwave
#     microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
#     item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == microwave_id and edge['relation_type'] == 'INSIDE']
#     microwave_item = []
#     for item in item_id:
#         names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
#         microwave_item.append(names)
#     for obj in graspable:
#         if obj in microwave_item:
#             fluents.append('holds(on('+ obj + ',microwave),0).')
#         else:
#             fluents.append('-holds(on('+ obj + ',microwave),0).')
#     # Items on stove
#     # fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
#     # stove_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
#     # pan_on_stove = [True for edge in graph['edges'] if edge['from_id'] == fryingpan_id and edge['to_id'] == stove_id and edge['relation_type'] == 'ON'][0]
#     # for obj in graspable:
#     #     if obj == 'fryingpan' and pan_on_stove:
#     #         fluents.append('holds(on('+ obj + ',stove),0).')
#     #     else:
#     #         fluents.append('-holds(on('+ obj + ',stove),0).')
#     # temp assumption => nothing on stove
#     default_fluents = ['-holds(on('+ obj + ',stove),0).' for obj in graspable]
#     fluents = fluents + default_fluents

#     # % --------------- % open/close
#     # Status of microwave
#     microwave_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
#     if 'CLOSED' in microwave_status:
#         fluents.append('-holds(opened(microwave),0).')
#     elif 'OPEN' in microwave_status:
#         fluents.append('holds(opened(microwave),0).')

#     # % --------------- % switch on/off
#     # Status of microwave
#     if 'OFF' in microwave_status:
#         fluents.append('-holds(switched_on(microwave),0).')
#     elif 'ON' in microwave_status:
#         fluents.append('holds(switched_on(microwave),0).')
#     # Status of Stove
#     stove_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
#     if 'OFF' in stove_status:
#         fluents.append('-holds(switched_on(stove),0).')
#     elif 'ON' in stove_status:
#         fluents.append('holds(switched_on(stove),0).')

#     # % --------------- % heated
#     for obj in categoryb_food: # poundcake
#         heated_idx = [idx for idx, item in enumerate(heated_) if item[0] == obj][0]
#         if (obj in microwave_item and 'ON' in microwave_status) or heated_[heated_idx][1]:
#             fluents.append('holds(heated(' + obj + '),0).')
#             heated_[heated_idx][1] = True
#         else:
#             fluents.append('-holds(heated(' + obj + '),0).')

#     # % --------------- % cooked
#     # Items on fryingpan
#     fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
#     item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == fryingpan_id and edge['relation_type'] == 'ON']
#     stove_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
#     pan_on_stove = [True for edge in graph['edges'] if edge['from_id'] == fryingpan_id and edge['to_id'] == stove_id and edge['relation_type'] == 'ON'][0]
#     pan_items = []
#     for item in item_id:
#         names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
#         pan_items.append(names)
#     for obj in categorya_food: # cutlets
#         cooked_idx = [idx for idx, item in enumerate(cooked_) if item[0] == obj][0]
#         if (obj in pan_items and 'ON' in stove_status and pan_on_stove) or cooked_[cooked_idx][1]:
#             fluents.append('holds(cooked(' + obj + '),0).')
#             cooked_[cooked_idx][1] = True
#         else:
#             fluents.append('-holds(cooked(' + obj + '),0).')
#     # % --------------- % ate => once we reach the eat action we will stop running the prg since virtaul home does not support eat hence adding default values below
#     default_fluents = ['-holds(ate(human,' + obj + '),0).' for obj in food]
#     human_fluents = human_fluents + default_fluents
#     # % --------------- % drink => same
#     default_fluents = ['-holds(drank(human,' + obj + '),0).' for obj in drinks]
#     human_fluents = human_fluents + default_fluents
#     # % --------------- % put_in
#     for obj in categorya_food:
#         if obj in pan_items:
#             fluents.append('holds(inside(' + obj + ',fryingpan),0).')
#         else:
#             fluents.append('-holds(inside(' + obj + ',fryingpan),0).')
#     # % --------------- % sit
#     if 'sit' in prev_human_actions[-1] and act_success:
#         action = (prev_human_actions[-1].split())[2][1:-1]
#         for obj in sittable:
#             if obj == action:
#                 human_fluents.append('holds(sat(human,' + action + '),0).')
#             else:
#                 human_fluents.append('-holds(sat(human,' + obj + '),0).')
#     else:
#         default_fluents = ['-holds(sat(human,' + obj + '),0).' for obj in sittable]
#         human_fluents = human_fluents + default_fluents
#     return human_fluents, ah_fluents, fluents
