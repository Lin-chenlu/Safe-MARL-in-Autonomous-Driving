# Hide pygame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


from gym.envs.registration import register


def register_Highway_envs():
    """Import the envs module so that envs register themselves."""

    

    # highway_env.py
    register(
        id='highway-v0',
        entry_point='Highway_env.envs:HighwayEnv',
    )


    # intersection_env.py
    register(
        id='intersection-v0',
        entry_point='Highway_env.envs:IntersectionEnv',
    )

    register(
        id='intersection-v1',
        entry_point='Highway_env.envs:ContinuousIntersectionEnv',
    )

    register(
        id='intersection-multi-agent-v0',
        entry_point='Highway_env.envs:MultiAgentIntersectionEnv',
    )

    register(
        id='intersection-multi-agent-v1',
        entry_point='Highway_env.envs:TupleMultiAgentIntersectionEnv',
    )


    # merge_env.py
    register(
        id='merge-v0',
        entry_point='Highway_env.envs:MergeEnv',
    )



    # racetrack_env.py
    register(
        id='racetrack-v0',
        entry_point='Highway_env.envs:RacetrackEnv',
    )

    # roundabout_env.py
    register(
        id='roundabout-v0',
        entry_point='Highway_env.envs:RoundaboutEnv',
    )

    # two_way_env.py
    register(
        id='two-way-v0',
        entry_point='Highway_env.envs:TwoWayEnv',
        max_episode_steps=15
    )

    # u_turn_env.py
    register(
        id='u-turn-v0',
        entry_point='Highway_env.envs:UTurnEnv'
    )

