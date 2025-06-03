




# i want to load the action space of grid2op, bus_14, difficulty 0, and print each action in the action space
# and then i want a statistical analysis of the action space, i want to know
# that for each action, what object does it impact (what substation)
# just print everything in a readable format
# how to create and load action space is in the env/utils.py file, pls do it like that
# but simpler, i only care about the action space of bus_14, difficulty 0
# how to know its impact 
'''
an action is an object of class BaseAction
in the docs, it says:
__str__()→ str[source]
This utility allows printing in a human-readable format what objects will be impacted by the action.

Returns
:
str – The string representation of an BaseAction in a human-readable format.

Return type
:
str

Examples

It is simply the “print” function:

action = env.action_space(...)
print(action)
_subs_impacted
This attributes is either not initialized (set to None) or it tells, for each substation, if it is impacted by the action (in this case BaseAction._subs_impacted[sub_id] is True) or not (in this case BaseAction._subs_impacted[sub_id] is False)

Type
:
numpy.ndarray, dtype:bool

_lines_impacted
This attributes is either not initialized (set to None) or it tells, for each powerline, if it is impacted by the action (in this case BaseAction._lines_impacted[line_id] is True) or not (in this case BaseAction._subs_impacted[line_id] is False)

Type
:
numpy.ndarray, dtype:bool
'''