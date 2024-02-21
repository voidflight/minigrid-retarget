from gymnasium import register
from minigrid.core.world_object import Lava, Wall
from minigrid.envs import CrossingEnv, DynamicObstaclesEnv, MultiRoomEnv

from src.ppo.my_probe_envs import (
    Probe1,
    Probe2,
    Probe3,
    Probe4,
    Probe5,
    Probe6,
)

from .memory import MemoryEnv
from .fetchobstacles import FetchObstaclesEnv
from .multienvironments import MultiEnvSampler


def get_dynamic_obstacles_multi_env(render_mode="rgb_array", max_steps=1000):
    envs = []
    env = DynamicObstaclesEnv(
        size=6,
        n_obstacles=0,
        agent_start_pos=None,
        render_mode=render_mode,
        max_steps=max_steps,
    )
    envs.append(env)
    for size in range(6, 10):
        for num_obstacles in range(5, 7):
            env = DynamicObstaclesEnv(
                size=size,
                n_obstacles=num_obstacles,
                agent_start_pos=None,
                render_mode=render_mode,
                max_steps=max_steps,
            )
            envs.append(env)

    return MultiEnvSampler(envs)

def get_dynamic_obstacles_rand_goal(render_mode="rgb_array", max_steps=1000):
    env = DynamicObstaclesEnv(
        size=5, 
        n_obstacles=3, 
        rand_goal=True,
        render_mode=render_mode
    )
    return env

def get_fetch_obstacles(render_mode = "rgb_array", max_steps = 1000):
    env = FetchObstaclesEnv(
        size = 5,
        n_obstacles = 1,
        n_objs = 3,
        render_mode = render_mode
    )
    return env

def get_fetch_obstacles_7x7(render_mode = "rgb_array", max_steps = 1000):
    env = FetchObstaclesEnv(
        size = 7,
        n_obstacles = 3,
        n_objs = 3,
        render_mode = render_mode
    )
    return env

def get_fetch_obstacles_6x6(render_mode = "rgb_array", max_steps = 1000):
    env = FetchObstaclesEnv(
        size = 6,
        n_obstacles = 1,
        n_objs = 3,
        render_mode = render_mode
    )
    return env

def get_fetch_obstacles_8x8(render_mode = "rgb_array", max_steps = 1000):
    env = FetchObstaclesEnv(
        size = 8,
        n_obstacles = 4,
        n_objs = 3,
        render_mode = render_mode
    )
    return env
    
def get_crossing_multi_env(render_mode="rgb_array", max_steps=1000):
    envs = []
    for size in range(5, 14, 2):
        for num_crossings in range(0, 7):
            env = CrossingEnv(
                size=size,
                num_crossings=num_crossings,
                obstacle_type=Lava,
                render_mode=render_mode,
                max_steps=max_steps,
            )
            envs.append(env)

    for size in range(5, 14, 2):
        for num_crossings in range(0, 7):
            env = CrossingEnv(
                size=size,
                num_crossings=num_crossings,
                obstacle_type=Wall,
                render_mode=render_mode,
                max_steps=max_steps,
            )
            envs.append(env)

    return MultiEnvSampler(envs)


def get_multi_room_env(render_mode="rgb_array", max_steps=1000):
    envs = []
    for min_rooms in range(1, 5):
        for max_rooms in range(min_rooms, 5):
            for max_room_size in range(5, 10):
                env = MultiRoomEnv(
                    minNumRooms=min_rooms,
                    maxNumRooms=max_rooms,
                    maxRoomSize=max_room_size,
                    render_mode=render_mode,
                    max_steps=max_steps,
                )
                envs.append(env)

    return MultiEnvSampler(envs)


def get_memory_env_random_direction(render_mode="rgb_array", max_steps=1000):
    env = MemoryEnv(
        size=7,
        random_length=False,
        random_direction=True,
        max_steps=max_steps,
        render_mode=render_mode,
    )

    return env


def get_memory_env_fixed_start(render_mode="rgb_array", max_steps=1000):
    env = MemoryEnv(
        size=7,
        random_length=False,
        random_direction=False,
        random_start_pos=False,
        max_steps=max_steps,
        render_mode=render_mode,
    )

    return env


print("Registering DynamicObstaclesMultiEnv-v0")
print("Registering CrossingMultiEnv-v0")
print("Registering Probe Envs")


def register_envs():
    register(
        id="DynamicObstaclesMultiEnv-v0",
        entry_point="environments.registration:get_dynamic_obstacles_multi_env",
    )

    register(
        id="CrossingMultiEnv-v0",
        entry_point="environments.registration:get_crossing_multi_env",
    )

    register(
        id="MultiRoomMultiEnv-v0",
        entry_point="environments.registration:get_multi_room_env",
    )

    probes = [Probe1, Probe2, Probe3, Probe4, Probe5, Probe6]
    for i in range(6):
        register(id=f"Probe{i+1}-v0", entry_point=probes[i])

    register(
        id="MiniGrid-MemoryS7RandomDirection-v0",
        entry_point="src.environments.registration:get_memory_env_random_direction",
    )

    register(
        id="MiniGrid-MemoryS7FixedStart-v0",
        entry_point="src.environments.registration:get_memory_env_fixed_start",
    )
    
    register(
        id="DynamicObstaclesRandGoal-v0",
        entry_point="src.environments.registration:get_dynamic_obstacles_rand_goal")
        
    register(
        id = "FetchObstacles-8x8-v0",
        entry_point="src.environments.registration:get_fetch_obstacles_8x8")
    
    register(
        id = "FetchObstacles-7x7-v0",
        entry_point="src.environments.registration:get_fetch_obstacles_7x7")
    
    register(
        id = "FetchObstacles-6x6-v0",
        entry_point="src.environments.registration:get_fetch_obstacles_6x6")
    
    register(
        id = "FetchObstacles-v0",
        entry_point="src.environments.registration:get_fetch_obstacles")
