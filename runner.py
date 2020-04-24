# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:56:44 2020

"""
import numpy as np
import matplotlib.pyplot as plt
import player
import environment
import agent
import pandas as pd
import visualisation as vis

def run_episode(env,n_steps,agents,agent_indices,gamma):
    obs = env.observe()
    all_rewards = np.empty((n_steps,len(env.players)))
    all_actions = np.empty((n_steps,len(env.players)))
    all_bitmaps = [vis.make_empty_bitmap(env)]*3
    all_bitmaps.append(vis.make_bitmap(env))
    all_pub_visits = np.zeros(len(agents))
    for i in range(n_steps):
        are_dead = [p.status is player.Status.DEAD for p in env.players]
        for pi,(o,ai,is_dead) in enumerate(zip(obs,agent_indices,are_dead)): # enumerate gives me a player index required by add_observation()
            if not is_dead:
                agents[ai].add_observation(o,pi)
        actions = [environment.Action.IDLE]*len(env.players)
        #epsilon = 0.1# if i<4 else 0
        for ag in agents:
            for a,pi in ag.get_actions(0):
                actions[pi] = a
        all_actions[i] = [a.value for a in actions] # logging actions performed
        rewards = env.step(actions)
        all_bitmaps.append(vis.make_bitmap(env))
        all_rewards[i] = rewards
        next_obs = env.observe()
        for pi,(ob,a,r,next_ob,ai,is_dead,p) in enumerate(zip(obs,actions,rewards,next_obs,agent_indices,are_dead,env.players)):            
            if not is_dead:
                agents[ai].add_sars(pi,ob,a,r,next_ob) # add experience from each player            
                if (p.x,p.y)==(env.pub[0],env.pub[1]):
                    all_pub_visits[ai] +=1
        if i%7 == 6:
            for ag in agents:
                ag.update(gamma)
    
    return all_rewards, all_actions, all_bitmaps, all_pub_visits

def run_training(env,n_steps,n_episodes,gamma, learning_rate):
    agent_old = agent.Agent(learning_rate)
    agent_young = agent.Agent(learning_rate)
    agents = [agent_old, agent_young]
    agent_indices = []
    for p in env.players:
        if p.recovery_prob == environment.OLD_REC_PROB:
            agent_indices.append(0)            
        else:
            agent_indices.append(1)
            
    number_olds = int(environment.OLD*len(agent_indices))
    number_youngs = len(agent_indices) - number_olds
    list_mean_reward_agent_young = []
    list_mean_reward_agent_old = []
    list_mean_young_washed_hands = []
    list_mean_old_washed_hands = []
    list_survivors_old = []
    list_survivors_young =[]
    list_pub_young =[]
    list_pub_old =[]
    frame_counter = 0
    for i in range(n_episodes):
        env.reset()
        all_rewards, all_actions, all_bitmaps, all_pub_visits = run_episode(env,n_steps,agents,agent_indices,gamma)
        for bitmap in all_bitmaps:
            ax=vis.make_frame(bitmap)
            vis.save_frame(ax,"training15/training_rep-clip40_lr1e-3_%d.png"%frame_counter)
            frame_counter += 1
            plt.close('all')
        episode_rewards = np.sum(all_rewards,axis=0)
        total_reward_agent_young = 0
        total_reward_agent_old = 0
        did_wash_hands = all_actions == environment.Action.WASH_HANDS.value
        n_washed_hands = np.sum(did_wash_hands,0)
        tot_young_washed_hands = 0
        tot_old_washed_hands = 0
        for e,ai,nwh in zip(episode_rewards,agent_indices,n_washed_hands):
            if ai == 1:
                total_reward_agent_young += e
                tot_young_washed_hands += nwh
            else:
                total_reward_agent_old += e
                tot_old_washed_hands += nwh
        mean_reward_agent_young = total_reward_agent_young/number_youngs
        mean_reward_agent_old = total_reward_agent_old/number_olds
        mean_young_washed_hands = tot_young_washed_hands/number_youngs
        mean_old_washed_hands = tot_old_washed_hands/number_olds
        mean_pub_young = all_pub_visits[1]/number_youngs
        mean_pub_old = all_pub_visits[0]/number_olds
        
        are_dead = [p.status is player.Status.DEAD for p in env.players]
        survivors_old = [ai for d,ai in zip(are_dead,agent_indices) if not d and ai == 0]
        survivors_young = [ai for d,ai in zip(are_dead,agent_indices) if not d and ai == 1]        
                
        print(i, mean_reward_agent_young, mean_reward_agent_old,mean_young_washed_hands,mean_old_washed_hands)
        
        list_mean_reward_agent_young.append(mean_reward_agent_young)
        list_mean_reward_agent_old.append(mean_reward_agent_old)        
        list_mean_young_washed_hands.append(mean_young_washed_hands)
        list_mean_old_washed_hands.append(mean_old_washed_hands)
        list_survivors_old.append(len(survivors_old))
        list_survivors_young.append(len(survivors_young))
        list_pub_young.append(mean_pub_young)
        list_pub_old.append(mean_pub_old)
        
        output = {'Mean reward young': list_mean_reward_agent_young, 
                  'Mean reward old': list_mean_reward_agent_old, 
                  'Young washing': list_mean_young_washed_hands, 
                  'Old washing': list_mean_old_washed_hands, 
                  'Young survivors':list_survivors_young, 
                  'Old survivors': list_survivors_old, 
                  'Pub young': list_pub_young, 'Pub old':list_pub_old}
        df = pd.DataFrame(output, columns = ['Mean reward young', 'Mean reward old', 
                                             'Young washing', 'Old washing','Young survivors',
                                             'Old survivors','Pub young','Pub old'])
        df.to_csv("training15/training_rep-clip40_lr2e-4_players%d_steps%d_episodes%d_wxh%d_lr%g.csv"%(len(env.players),n_steps,n_episodes,env.width,learning_rate))
    
    return df

