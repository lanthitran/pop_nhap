import grid2op
env_name = "l2rpn_case14_sandbox"  # or any other name

env = grid2op.make(env_name)  # again: redo this step each time you customize "..."
# for example if you change the `action_class` or the `backend` etc.

#env.generate_classes()

