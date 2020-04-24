# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:26:30 2020

"""
import enum

MAX_HEALTH = 5
MIN_HEALTH = 0
DHEALTH = 1

class Status(enum.Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    DEAD = 3

class Player:
    def __init__(self,x,y,recovery_prob):
        self.health = MAX_HEALTH
        self.status = Status.SUSCEPTIBLE
        self.x = x
        self.y = y
        self.recovery_prob = recovery_prob
        self.infection_duration = 0


