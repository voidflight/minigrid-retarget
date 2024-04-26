#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:31:17 2024

@author: catgoddess
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)

from minigrid.core.world_object import WorldObj

from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)

if TYPE_CHECKING:
    from minigrid.minigrid_env import MiniGridEnv

Point = Tuple[int, int]

class Key(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("key", color)

    def can_pickup(self):
        return True

    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return True
    
    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))