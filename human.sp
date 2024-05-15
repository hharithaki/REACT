#const n = 35.
sorts
#agent = {human}.
#other_agents = {ahagent}.
#all_agents = #agent + #other_agents.
#categorya_food = {cutlets}.
#categoryb_food = {poundcake}.
#food = {breadslice} + #categorya_food + #categoryb_food.
#drinks = {waterglass}.
#sittable = {bench}.
#cooking_electricals = {microwave}.
#electricals = #cooking_electricals.
#cooking_appliances = {stove}.
#appliances = #electricals + #cooking_appliances.
#cook_containers = {fryingpan}.
#containers = #cook_containers.
#tables = {kitchentable}.
#surfaces = #appliances + #tables. % remove tables from here later when you add the condition of plates
#graspable = #food + #drinks + #containers. 
#objects = #appliances + #graspable + #sittable.
#step = 0..n.
#value = 0..5.
#sum_val = 0..10.
#boolean = {true, false}.

#agent_actions = find(#agent,#objects) + grab(#agent,#graspable) + put(#agent,#graspable,#surfaces) + open(#agent,#electricals) + close(#agent,#electricals)
                 + switch_on(#agent,#appliances) + switch_off(#agent,#appliances) + eat(#agent,#food) + put_in(#agent,#categorya_food,#cook_containers) 
                 + drink(#agent,#drinks) + sit(#agent,#sittable).
#exogenous_actions = exo_find(#agent, #objects).
#action = #agent_actions + #exogenous_actions.

#inertial_f = found(#agent,#objects) + in_hand(#agent,#graspable) + on(#graspable,#surfaces) + opened(#electricals) + switched_on(#appliances)
              + heated(#categoryb_food) + ate(#agent,#food) + inside(#categorya_food,#cook_containers) + cooked(#categorya_food) + drank(#agent,#drinks)
              + sat(#agent,#sittable).
#defined_f = agent_found(#other_agents,#objects) + agent_hand(#other_agents,#graspable).
#fluents = #inertial_f + #defined_f.

predicates
holds(#fluents, #step).
occurs(#action, #step).

%%history
hpd(#agent_actions, #step).
obs(#fluents, #boolean, #step).
unobserved(#exogenous_actions, #step).
diagnosing(#step).

%% planning
success().
something_happened(#step).
goal(#step).
planning(#step).
current_step(#step).

cost(#agent_actions,#step,#value).
cost_defined(#agent_actions).
total(#sum_val).

rules
% -------------------------- casual laws --------------------------%

% find causes the agent to find that object. This includes walking and turning.
holds(found(R,O),I+1) :- occurs(find(R,O),I).

% grab causes an object to be in the hand of the agent; as well it to be removed form any previous surfaces.
holds(in_hand(R,O),I+1) :- occurs(grab(R,O),I).

% put causes an object to be placed in the relevant surface.
holds(on(O,S),I+1) :- occurs(put(R,O,S),I), O != S.

% open causes the agent to open a door of a appliance.
holds(opened(S),I+1) :- occurs(open(R,S),I).

% close causes the agent to close a door of a appliance.
-holds(opened(S),I+1) :- occurs(close(R,S),I).

% switch on causes the agent to switch on an appliance.
holds(switched_on(A),I+1) :- occurs(switch_on(R,A),I).

% switch off causes the agent to switch off an appliance.
-holds(switched_on(A),I+1) :- occurs(switch_off(R,A),I).

% eat/drink causes the agent to eat/drink.
holds(ate(R,F),I+1) :- occurs(eat(R,F),I).
holds(drank(R,D),I+1) :- occurs(drink(R,D),I).

% put_in causes an food to be placed inside that container.
holds(inside(F,C),I+1) :- occurs(put_in(R,F,C),I).

% sit causes the agent to sit.
holds(sat(R,S),I+1) :- occurs(sit(R,S),I).

% ----------------------- state constraints -----------------------%

% agent cannot find two objects at the same time.
-holds(found(R,O1),I) :- holds(found(R,O2),I), O1 != O2.

% agent cannot have two objects in its hand at the same time.
-holds(in_hand(R,O1),I) :- holds(in_hand(R,O2),I), O1 != O2.

% object cannot be in hand and on surface the same time.
-holds(on(O,S),I) :- holds(in_hand(R,O),I).
-holds(in_hand(R,O),I) :- holds(on(O,S),I).

% object cannot be in hand and inside continer the same time.
-holds(inside(F,C),I) :- holds(in_hand(R,F),I).
-holds(in_hand(R,F),I) :- holds(inside(F,C),I).

% ready_made food is heated if it was inside a cooking electrical and the electrical was switched on.
holds(heated(M),I) :- holds(on(M,S),I), holds(switched_on(S),I), #cooking_electricals(S).

% raw_food is cooked if it was on a cooking appliance and the appliance was switched on.
holds(cooked(F),I) :- holds(inside(F,C),I), holds(on(C,S),I), holds(switched_on(S),I), #cook_containers(C), #cooking_appliances(S).

holds(agent_found(T,O),I+1) :- holds(agent_found(T,O),I), not holds(agent_found(T,O1),I+1), I >= 0, O1!=O.
holds(agent_hand(T,O),I+1) :- holds(agent_hand(T,O),I), not holds(agent_hand(T,O1),I+1), I >= 0, O1!=O.

% define cost for actions - default cost for agent_actions is 0
cost(A,I,0) :- #agent_actions(A), #step(I), not cost_defined(A).
cost_defined(A) :- cost(A,I,V), V != 0.

cost(put(H,breadslice,kitchentable),I,1) :- holds(on(cutlets,kitchentable),I).
cost(put(H,breadslice,kitchentable),I,2) :- holds(on(poundcake,kitchentable),I).
cost(put(H,breadslice,kitchentable),I,3) :- holds(on(waterglass,kitchentable),I).

cost(put(H,cutlets,kitchentable),I,1) :- holds(on(poundcake,kitchentable),I).
cost(put(H,cutlets,kitchentable),I,2) :- holds(on(waterglass,kitchentable),I).

cost(put(H,poundcake,kitchentable),I,1) :- holds(on(waterglass,kitchentable),I).

cost(eat(H,breadslice),I,1) :- holds(ate(H,cutlets),I).
cost(eat(H,breadslice),I,2) :- holds(ate(H,poundcake),I).

cost(eat(H,cutlets),I,1) :- holds(ate(H,poundcake),I).

% -------------------- executability conditions -------------------%

% impossible to find something if the agent is sitting in the bench; unless that thing is on the kitchen table.
-occurs(find(R,O),I):- holds(sat(R,bench),I), not holds(on(O,kitchentable),I).

% impossible to find appliance if the agent is sitting in the bench.
-occurs(find(R,O),I):- holds(sat(R,bench),I), #appliances(O).

% impossible to find if already found.
-occurs(find(R,O),I) :- holds(found(R,O),I).

% impossible to find if the other agent has found it. This is to prevent the agents from going after the same object.
-occurs(find(R,O),I) :- holds(agent_found(T,O),I), #other_agents(T).

% impossible to grab something before finding it.
-occurs(grab(R,O),I):- not holds(found(R,O),I).

% impossible to grab something if that object is already in the hand of the agent.
-occurs(grab(R,O),I):- holds(in_hand(R,O),I).

% impossible to grab something if that object is in the hand of the other agent.
-occurs(grab(R,O),I):- holds(agent_hand(T,O),I), #other_agents(T).

% impossible to grab something from inside an electrical if the door is closed.
-occurs(grab(R,O),I):- not holds(opened(S),I), holds(on(O,S),I).

% impossible to grab something from an appliance if it is switched_on.
-occurs(grab(R,O),I):- holds(switched_on(A),I), holds(inside(O,C),I), holds(on(C,A),I).

% impossible to put down something if that object is not in the hand of the agent.
-occurs(put(R,O,S),I) :- not holds(in_hand(R,O),I).

% impossible to put an object down if the surface location is not found.
-occurs(put(R,O,S),I) :- not holds(found(R,S),I).

% impossible to put something inside an electrical if the door is closed.
-occurs(put(R,O,S),I) :- not holds(opened(S),I).

% impossible to put food on a contianer if the food is not in the hand of the agent.
-occurs(put_in(R,F,C),I) :- not holds(in_hand(R,F),I).

% impossible to put_in food unless the agent found the container.
-occurs(put_in(R,F,C),I) :- not holds(found(R,C),I).

% impossible to open the door of an electrical if it is not switched_off.
-occurs(open(R,S),I) :- holds(switched_on(S),I).

% impossible to open the door of an electrical before finding it.
-occurs(open(R,O),I):- not holds(found(R,O),I).

% impossible to open a door if it is already opened.
-occurs(open(R,S),I):- holds(opened(S),I).

% impossible to switch_on an electrical unless the door is closed.
-occurs(switch_on(R,E),I) :- holds(opened(E),I).

% impossible to switch on an appliance before finding it.
-occurs(switch_on(R,A),I):- not holds(found(R,A),I).

% impossible to switch_on an appliance if it is already switched_on.
-occurs(switch_on(R,A),I):- holds(switched_on(A),I).

% impossible to close the foor of an electrical before finding it.
-occurs(close(R,O),I):- not holds(found(R,O),I).

% impossible to close a door if it is already closed.
-occurs(close(R,S),I):- not holds(opened(S),I).

% impossible to switch off an appliance before finding it.
-occurs(switch_off(R,A),I):- not holds(found(R,A),I).

% impossible to switch_off an appliance if it is already switched_off.
-occurs(switch_off(R,A),I):- not holds(switched_on(A),I).

% impossible to eat ready_made food unless they are heated.
-occurs(eat(R,M),I) :- not holds(heated(M),I).

% impossible to eat food unless they are on the table.
-occurs(eat(R,F),I) :- not holds(on(F,kitchentable),I).

% impossible to eat raw_food unless they are cooked.
-occurs(eat(R,F),I) :- not holds(cooked(F),I).

% impossible to drink unless they are on the table.
-occurs(drink(R,F),I) :- not holds(on(F,kitchentable),I).

% impossible to eat food unless the agent is sitting in the bench.
-occurs(eat(R,F),I) :- not holds(sat(R,bench),I).

% impossible to drink unless the agent is sitting in the bench.
-occurs(drink(R,F),I) :- not holds(sat(R,bench),I).

% impossible to putdown something on the kitchen table (not microwave) when you have found something else.
-occurs(put(H,F2,kitchentable),I) :- holds(found(H,F1),I), F1 != F2.

% impossible to sit on a sittable before finding it.
-occurs(sit(R,S),I):- not holds(found(R,S),I).

% ------------------------ inertial axioms ------------------------%

holds(F,I+1) :- #inertial_f(F), holds(F,I), not -holds(F,I+1).
-holds(F,I+1) :- #inertial_f(F), -holds(F,I), not holds(F,I+1).

% ------------------------------ CWA ------------------------------%

-occurs(A,I) :- not occurs(A,I).
-holds(F,I) :- #defined_f(F), not holds(F,I).

% ---------------------------- history ----------------------------%
% guarantees that agent takes all fluents in the system into consideration
holds(F,0) | -holds(F,0) :- #inertial_f(F).

% record all the actions that happened
occurs(A,I) :- hpd(A,I), current_step(I1), I < I1.

% reality check axioms. gurantees that the agents expectations agrees with its observations
:- current_step(I1), I <= I1, obs(F,true,I), -holds(F,I).
:- current_step(I1), I <= I1, obs(F,false,I), holds(F,I).

% record unobserved exogenous action occurrences
occurs(A,I) :- unobserved(A,I), #exogenous_actions(A).

% generate minimal explanations.
unobserved(A,I0) :+ current_step(I1), diagnosing(I1), I0 < I1, not hpd(A,I0), #exogenous_actions(A).

% --------------------------- planning ---------------------------%

% to achieve success the system should satisfies the goal. Failure is not acceptable
success :- goal(I), I <= n.
:- not success, current_step(I0), planning(I0).

% consider the occurrence of exogenous actions when they are absolutely necessary for resolving a conflict
occurs(A,I) :+ #agent_actions(A), #step(I), current_step(I0), planning(I0), I0 <= I.

% agent can not execute parallel actions
-occurs(A1,I) :- occurs(A2,I), A1 != A2, #agent_actions(A1), #agent_actions(A2).

% an action should occur at each time step until the goal is achieved
something_happened(I1) :- current_step(I0), planning(I0), I0 <= I1, occurs(A,I1), #agent_actions(A).
:- not something_happened(I), something_happened(I+1), I0 <= I, current_step(I0), planning(I0).

total(S) :- S = #sum{C, A:occurs(A,I), cost(A,I,C)}.
#minimize {V@2, V:total(V)}.

%%%--------------------------------------------------------------%%%

planning(I) | diagnosing(I):- current_step(I).

planning(0).
current_step(0).

%%%--------------------------------------------------------------%%%

% --------------- %

goal(I) :- holds(ate(human,poundcake),I), holds(ate(human,breadslice),I), holds(ate(human,cutlets),I), holds(on(waterglass,kitchentable),I).

-holds(found(human,microwave),0).
-holds(found(human,stove),0).
-holds(found(human,breadslice),0).
-holds(found(human,cutlets),0).
-holds(found(human,poundcake),0).
-holds(found(human,waterglass),0).
-holds(found(human,fryingpan),0).
-holds(found(human,bench),0).
-holds(in_hand(human,breadslice),0).
-holds(in_hand(human,cutlets),0).
-holds(in_hand(human,poundcake),0).
-holds(in_hand(human,waterglass),0).
-holds(in_hand(human,fryingpan),0).
-holds(on(breadslice,kitchentable),0).
-holds(on(cutlets,kitchentable),0).
-holds(on(poundcake,kitchentable),0).
-holds(on(waterglass,kitchentable),0).
-holds(on(fryingpan,kitchentable),0).
-holds(on(breadslice,microwave),0).
-holds(on(cutlets,microwave),0).
-holds(on(poundcake,microwave),0).
-holds(on(waterglass,microwave),0).
-holds(on(fryingpan,microwave),0).
-holds(on(breadslice,stove),0).
-holds(on(cutlets,stove),0).
-holds(on(poundcake,stove),0).
-holds(on(waterglass,stove),0).
-holds(on(fryingpan,stove),0).
-holds(opened(microwave),0).
-holds(switched_on(microwave),0).
-holds(switched_on(stove),0).
-holds(heated(poundcake),0).
-holds(cooked(cutlets),0).
-holds(ate(human,breadslice),0).
-holds(ate(human,cutlets),0).
-holds(ate(human,poundcake),0).
-holds(drank(human,waterglass),0).
-holds(inside(cutlets,fryingpan),0).
-holds(sat(human,bench),0).
display
occurs.
