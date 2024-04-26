#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 23:40:38 2023

@author: catgoddess
"""

from __future__ import annotations

from operator import add

from gymnasium.spaces import Discrete

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Ball, Box

from .key import Key # This version of Key is slightly different from the minigrid one

class FetchObstaclesEnv(MiniGridEnv):
    def __init__(
        self,
        size=8,
        n_obstacles=4,
        n_objs=3,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.n_objs = n_objs
        self.obj_types = ["key"]
        
        self.target_types = ["key", "key"]
        self.target_colors = ["blue", "purple"]
        self.target_type = "key"
        self.target_color = "blue"
        
        MISSION_SYNTAX = [
            "get a",
            "go get a",
            "fetch a",
            "go fetch a",
            "you must fetch a",
        ]
        self.size = size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[MISSION_SYNTAX, COLOR_NAMES, self.obj_types],
        )
        
        if max_steps is None:
            max_steps = 5 * size**2
        
        if n_obstacles <= size / 2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size / 2)
        
        self.gen_count = 0
        
        self.memory = None
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        
    
    @staticmethod
    def _gen_mission(syntax: str, color: str, obj_type: str):
        return f"{syntax} {color} {obj_type}"
    
    def switch_target(self, target_type, target_color):
        self.target_type = target_type
        self.target_color = target_color 
    
    def switch_target_set(self, target_types = None, target_colors = None):
        if target_types:
            self.target_types = target_types
        if target_colors:
            self.target_colors = target_colors
            
    def _gen_grid(self, width, height):
        
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        self.place_agent()
        
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball("grey"))
            self.place_obj(self.obstacles[i_obst], max_tries=100)
        
        objs = []
        
        obj = None
        
        for i in range(len(self.target_types)):
            
            if self.target_types[i] == "key":
                obj = Key(self.target_colors[i])
            elif self.target_types[i] == "box":
                obj = Box(self.target_colors[i])
            self.place_obj(obj)
            objs.append(obj)
        
        
        reduced_color_names = []
        for color in COLOR_NAMES:
            if color not in self.target_colors:
                reduced_color_names.append(color)
        
        while len(objs) < self.n_objs:
            obj_type = self._rand_elem(self.obj_types)
            obj_color = self._rand_elem(reduced_color_names)

            if obj_type == "key":
                obj = Key(obj_color)
            elif obj_type == "box":
                obj = Box(obj_color)
            else:
                raise ValueError(
                    "{} object type given. Object type can only be of values key and box.".format(
                        obj_type
                    )
                )

            self.place_obj(obj)
            objs.append(obj)

        descStr = f"{self.target_color} {self.target_type}"

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = "get a %s" % descStr
        elif idx == 1:
            self.mission = "go get a %s" % descStr
        elif idx == 2:
            self.mission = "fetch a %s" % descStr
        elif idx == 3:
            self.mission = "go fetch a %s" % descStr
        elif idx == 4:
            self.mission = "you must fetch a %s" % descStr
        assert hasattr(self, "mission")
        
    
    def step(self, action):
        
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type not in self.obj_types

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            try:
                self.place_obj(
                    self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
                )
                self.grid.set(old_pos[0], old_pos[1], None)
             
            except Exception:
                pass
            
        obs, reward, terminated, truncated, info = super().step(action)
        
        if action == self.actions.forward and not_clear:
            # print("hit obstacle")
            reward = -1
            terminated = True
            
            return obs, reward, terminated, truncated, info
        
        if self.carrying:
            if (self.carrying.type == self.target_type):
                if (self.carrying.color == self.target_color):
                    reward = self._reward()
                    terminated = True
                else:
                    # print(self.carrying.color)
                    reward = 0.1
                    terminated = True
            else:
                # print(self.carrying.type)
                reward = -1
                terminated = True

        return obs, reward, terminated, truncated, info
