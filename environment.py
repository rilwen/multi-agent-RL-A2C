# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:56:32 2020

"""
import random as rd
import player
import numpy as np
import enum


OLD = 0.3
OLD_REC_PROB = 0
YOUNG_REC_PROB = 0.1
NB_INFECTED = 2
INFECTION_DURATION = 3
WASHING_PROTECTION = 1
VIEWPORT_SIZE = 3 #side of a square with player inside

class Action(enum.Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    WASH_HANDS = 4
    IDLE = 5

class Environment:
    def __init__(self, width=100, height=100, population = 5000):
        self.width = width
        self.height = height
        self.population = population
        self.players = []
        self.viewport_size = VIEWPORT_SIZE
        self.last_actions = [Action.IDLE]*self.population
        #self.pub = (self.width//2,self.height//2)
        self.pub = (0,0)
        for _ in range(self.population):
            x = rd.randint(0,self.width-1)
            y = rd.randint(0,self.height-1)
            self.players.append(player.Player(x,y,YOUNG_REC_PROB))
        for p in range(int(OLD*self.population)):
            self.players[p].recovery_prob = OLD_REC_PROB
            
        rd.shuffle(self.players)
        for i in range(NB_INFECTED):
            self.players[i].status = player.Status.INFECTED
            
    def step(self, actions):
        infection_map = np.zeros((self.width,self.height),dtype=int)
        self.last_actions = actions
        
        for p in self.players:
            if p.status is player.Status.INFECTED:
                infection_map[p.x,p.y] = 1
#                if p.y > 0:
#                    infection_map[p.x,p.y-1] = 1cx
#                if p.y < self.height -1:
#                    infection_map[p.x,p.y+1] = 1
#                if p.x > 0:
#                    infection_map[p.x-1,p.y] = 1
#                    if p.y > 0:
#                        infection_map[p.x-1,p.y-1] = 1
#                    if p.y < self.height -1:
#                        infection_map[p.x-1,p.y+1] = 1
#                if p.x < self.width-1:
#                    infection_map[p.x+1,p.y] = 1
#                    if p.y > 0:
#                        infection_map[p.x+1,p.y-1] = 1
#                    if p.y < self.height -1:
#                        infection_map[p.x+1,p.y+1] = 1
                    
#                infection_map[p.x,p.y+1] = 1
#                infection_map[p.x,p.y-1] = 1
#                infection_map[p.x+1,p.y] = 1
#                infection_map[p.x+1,p.y+1] = 1
#                infection_map[p.x+1,p.y-1] = 1
#                infection_map[p.x-1,p.y] = 1
#                infection_map[p.x-1,p.y+1] = 1
#                infection_map[p.x-1,p.y-1] = 1
                                
        for p,a in zip(self.players, actions): # infection spread
            if p.status is player.Status.SUSCEPTIBLE and infection_map[p.x,p.y]:
                washing_protection = WASHING_PROTECTION
                if a is not Action.WASH_HANDS:
                    washing_protection = 0
                if rd.random() >= washing_protection:
                    p.status = player.Status.INFECTED
                    
        for p in self.players:
            if p.status is player.Status.INFECTED:
                p.infection_duration += 1
                p.health -= player.DHEALTH
                if p.health == player.MIN_HEALTH:
                    p.status = player.Status.DEAD
                elif p.infection_duration > INFECTION_DURATION:
                    if rd.random() < p.recovery_prob:
                        p.status = player.Status.RECOVERED
                        p.infection_duration = 0
                        p.health = player.MAX_HEALTH

        rewards = []
        for p,a in zip(self.players, actions):
            
            reward = 0
            if p.status is not player.Status.DEAD:                
#                if a is Action.LEFT:
#                    p.x = (p.x - 1)%self.width
#                    reward =2
#                elif a is Action.RIGHT:
#                    p.x = (p.x + 1)%self.width
#                    reward =1
#                elif a is Action.UP:
#                    reward =1
#                    p.y = (p.y - 1)%self.height
#                elif a is Action.DOWN:
#                    p.y = (p.y + 1)%self.height
#                    reward =1
                if a is Action.LEFT and p.x > 0:
                    p.x = p.x - 1
                    reward =1
                elif a is Action.LEFT and p.x == 0:
                    reward =-1
                elif a is Action.RIGHT and p.x < self.width-1:
                    p.x = p.x + 1
                    reward =1
                elif a is Action.RIGHT and p.x == self.width-1:
                    reward =-1
                elif a is Action.UP and p.y < self.height -1:
                    p.y = p.y + 1
                    reward =1
                elif a is Action.UP and p.y == self.height -1:
                    reward =-1
                elif a is Action.DOWN and p.y > 0:
                    p.y = p.y - 1
                    reward =1
                elif a is Action.DOWN and p.y == 0:
                    reward =-1
                
                #if p.health < player.MAX_HEALTH//2:
                #    reward /= 2
                    
                if (p.x,p.y) == self.pub:
                    reward += 5

#                if p.health > player.MAX_HEALTH*0.7:
#                    reward += 5
#                elif p.health < player.MAX_HEALTH*0.4:
#                    reward += 0
#                else:
#                    reward -= 5

            rewards.append(reward)
            
        return rewards

    def create_tpv(self):
        tpv = np.zeros((self.width,self.height))
        for p in self.players:
            if p.status is not player.Status.DEAD:
                tpv[p.x,p.y] += 40
        tpv = np.clip(tpv,0,255)
        #tpv[self.pub[0],self.pub[1]] = -100
        tpv[self.pub[0],self.pub[1]] = 1000
        return tpv
    
    def observe(self):
        tpv = np.full((3*self.width,3*self.height),-2000)
        tpv_board = self.create_tpv()
        tpv[self.width:2*self.width,self.height:2*self.height] = tpv_board
        observations = []
        vp2 = self.viewport_size//2
        
        #tpvtile = np.tile(tpv,(3,3)) # hendle edges with torus topology
        for p in self.players:
            #playerview = tpvtile[p.x+self.width-vp2:p.x+self.width+vp2+1,p.y+self.height-vp2:p.y+self.height+vp2+1]
            playerview = tpv[p.x+self.width-vp2:p.x+self.width+vp2+1,p.y+self.height-vp2:p.y+self.height+vp2+1]
            observations.append((playerview,p.health))
        return observations
            
            
    def reset(self):
        for p in self.players:
            p.status = player.Status.SUSCEPTIBLE
            p.health = player.MAX_HEALTH
            p.x = rd.randint(0,self.width-1)
            p.y = rd.randint(0,self.height-1)
            p.infection_duration = 0
        for p in np.random.choice(self.players, NB_INFECTED, replace=False):
            p.status = player.Status.INFECTED
            