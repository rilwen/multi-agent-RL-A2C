# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:45:14 2020

"""

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import environment as env
import player
	
# import regularizer
from tensorflow.keras.regularizers import l2
# instantiate regularizer


class Agent:
    def __init__(self,learning_rate):
        self._experience = []
        self._observations = []
        self._learning_rate = learning_rate
        self._viewport_input = keras.Input(shape=(env.VIEWPORT_SIZE,env.VIEWPORT_SIZE), dtype=tf.float32, name='viewport')
        self._health_input = keras.Input(shape=(), dtype=tf.int32, name='health')
        flat_viewport = layers.Flatten()(self._viewport_input)
        health_onehot = tf.one_hot(self._health_input, player.MAX_HEALTH-player.MIN_HEALTH+1)
        flat_input = layers.concatenate([flat_viewport,health_onehot])
        bias_init = tf.constant_initializer(1)
        
        #extract features from inputs
        reg = l2(0.001)
        pl1 = layers.Dense(100, activation='relu',bias_initializer = bias_init, kernel_regularizer=reg,bias_regularizer=reg)(flat_input)
        pl2 = layers.Dense(100, activation='relu',bias_initializer = bias_init, kernel_regularizer=reg,bias_regularizer=reg)(pl1)
        self._policy_features = layers.Dense(50, activation='relu',bias_initializer = bias_init, kernel_regularizer=reg,bias_regularizer=reg)(pl2)
        
        vl1 = layers.Dense(100, activation='relu',bias_initializer = bias_init, kernel_regularizer=reg,kernel_initializer='zeros',bias_regularizer=reg)(flat_input)
        vl2 = layers.Dense(100, activation='relu',bias_initializer = bias_init, kernel_regularizer=reg,kernel_initializer='zeros',bias_regularizer=reg)(vl1)
        self._value_features = layers.Dense(50, activation='relu',bias_initializer = bias_init,kernel_initializer='zeros', kernel_regularizer=reg,bias_regularizer=reg)(vl2)

        #extract features from inputs
#        reg = l2(0.001)
#        w1 = layers.Dense(100, activation='relu',bias_initializer = bias_init, kernel_regularizer=reg,bias_regularizer=reg)(flat_input)
#        w2 = layers.Dense(100, activation='relu',bias_initializer = bias_init, kernel_regularizer=reg,bias_regularizer=reg)(w1)
#        w3 = layers.Dense(100, activation='relu',bias_initializer = bias_init, kernel_regularizer=reg,bias_regularizer=reg)(w2)
        
        #self._policy_logits = layers.Dense(len(env.Action), kernel_regularizer=reg,bias_regularizer=reg)(w3)
        self._policy_logits = layers.Dense(len(env.Action), kernel_regularizer=reg,bias_regularizer=reg)(self._policy_features)
        
        #self._value_fun = layers.Dense(1, kernel_regularizer=reg,kernel_initializer='zeros',bias_regularizer=reg)(w3) # linear activation so that value fun can be negative
        self._value_fun = layers.Dense(1, kernel_regularizer=reg,kernel_initializer='zeros',bias_regularizer=reg)(self._value_features) # linear activation so that value fun can be negative
        
        self.inputs = {'viewport': self._viewport_input, 'health': self._health_input}
        self.outputs = {'policy_logits': self._policy_logits, 'value_fun': self._value_fun}
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs)
        
        #self.optimiser = tf.keras.optimizers.SGD(lr=learning_rate,clipvalue=0.5)
        self.optimiser = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        
    def add_sars(self,pi,ob,a,r,next_ob):
        self._experience.append((pi,ob,a.value,r,next_ob))
        
    def add_observation(self,ob,player_index):
        self._observations.append((ob,player_index))
        
    def get_actions(self,epsilon):
        if not self._observations: return []
        vps = []
        healths = []
        for (vp,h),_ in self._observations:
            vps.append(vp)
            healths.append(h)
        healths = np.array(healths,dtype=np.int32)
        vps = np.array(vps,dtype=np.float32)
        inputs = {'viewport':vps,'health':healths}
        outputs = self.model(inputs)
        distr = tf.nn.softmax(tf.cast(outputs['policy_logits'],tf.float64)).numpy()
        if epsilon>0:
            uniform_distr= np.full(distr.shape,1/len(env.Action))
            distr = distr*(1-epsilon)+uniform_distr*epsilon
        #print(distr)
        #distr = distr/np.sum(distr,1) # distribution of actions (columns) for each player (rows)
        actions = []
        for d,(_,pi) in zip(distr,self._observations):
            #print(pi, d)
            action = env.Action(np.random.choice(len(env.Action),p=d))
            actions.append((action,pi))
        self._observations = []
        return actions
        
    def get_action(self,ob): # not used
        viewport,health = ob
        inputs = {'viewport':np.array([viewport],dtype=np.float32),'health':np.array([health],dtype=np.int32)}
        outputs = self.model(inputs)
        distr = tf.cast(tf.nn.softmax(outputs['policy_logits'][0]),tf.float64)
        distr /= sum(distr)
        action_index = np.random.choice(len(env.Action), p=distr)
        return env.Action(action_index)

    def gradients_from_trajectory(self,trajectory,gamma):
        if not trajectory: return
        vps = []
        healths = []
        actions = []
        
        last_next_health = trajectory[-1][4][1]
        last_next_vp = trajectory[-1][4][0]
        is_terminal = last_next_health == player.MIN_HEALTH
        next_inputs = {'viewport':np.array([last_next_vp],dtype=np.float32),'health':np.array([last_next_health],dtype=np.int32)}
        q_value = 0 if is_terminal else self.model(next_inputs)['value_fun'].numpy()[0]
        q_values = []
        for _,(vp,h),a,r,(next_vp,next_h) in reversed(trajectory):
            vps.append(vp)
            healths.append(h)
            actions.append(a)
            q_value = r+ gamma*q_value
            q_values.append(q_value)
        
        vps = np.array(vps,dtype=np.float32)
        healths = np.array(healths,dtype=np.int32)
        actions = np.array(actions,dtype=np.int32)
        q_values = np.array(q_values,dtype=np.float32)
                
        
        inputs = {'viewport':vps,'health':healths}
        with tf.GradientTape() as tape:
            outputs = self.model(inputs)
            values = outputs['value_fun']
            #print("values=",values)
            policy_logits = outputs['policy_logits']
            policy = tf.nn.softmax(policy_logits)
            #print(policy, actions)
            actions_onehot = tf.one_hot(actions,len(env.Action))
            probs = tf.reduce_sum(actions_onehot*policy,axis=1)
            log_probs = tf.math.log(probs)
            advantages = q_values - values
            actor_loss = - tf.reduce_mean(log_probs*advantages.numpy()) # using mean makes the procedure batch size invariant
            critic_loss = 0.5*(tf.reduce_mean(advantages**2))
            #logits_reg_loss = 0.001*tf.reduce_mean(policy_logits*policy_logits)
            entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(policy,policy_logits))*0.01
            #total_loss = actor_loss + critic_loss + logits_reg_loss - entropy_loss
            total_loss = actor_loss + critic_loss - entropy_loss
            for loss in self.model.losses:
                total_loss += loss
            
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(total_loss,self.model.trainable_weights)        
        clipped_grads, _ = tf.clip_by_global_norm(grads, 0.5)
        #print(grads)
        
        #clipped_grads = [tf.clip_by_value(g,-1.,1.) for g in grads]
        
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        #self.optimiser.apply_gradients(zip(grads, self.model.trainable_weights))
        return clipped_grads
    
        
    def update(self,gamma):
        if not self._experience: return
        
        pis = set([t[0] for t in self._experience])        
        grads = []
        for pi in pis:
            # Trajectory of player with index pi. Preserves time ordering.
            trajectory = [t for t in self._experience if t[0]==pi]
            new_grads=self.gradients_from_trajectory(trajectory,gamma)
            grads = new_grads if not grads else [g+ng for g,ng in zip(grads,new_grads)]
        
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimiser.apply_gradients(zip(grads, self.model.trainable_weights))
        
        self._experience = []
        
        
    def update_CS(self,gamma):
        if not self._experience: return
        vps = []
        healths = []
        actions = []
        rewards = []
        next_vps = []
        next_healths = []
        
        for (vp,h),a,r,(next_vp,next_h) in self._experience:
            vps.append(vp)
            healths.append(h)
            actions.append(a)
            rewards.append(r)
            next_vps.append(next_vp)
            next_healths.append(next_h)
        vps = np.array(vps,dtype=np.float32)
        next_vps = np.array(next_vps,dtype=np.float32)
        healths = np.array(healths,dtype=np.int32)
        next_healths = np.array(next_healths,dtype=np.int32)
        actions = np.array(actions,dtype=np.int32)
        rewards = np.array(rewards,dtype=np.float32)
                
        #A2C
        inputs = {'viewport':vps,'health':healths}
        next_inputs = {'viewport':next_vps,'health':next_healths}
        
        next_outputs = self.model(next_inputs)
        next_values = next_outputs['value_fun'].numpy() # Get numpy array so that it can be modified in next line.
        next_values[next_healths==player.MIN_HEALTH] = 0 #zeroing all future states for dead players (terminal states )
        q_values = rewards + gamma*next_values
        #print("q_values=",rewards,'+gamma*',next_values)
        with tf.GradientTape() as tape:
            outputs = self.model(inputs)
            values = outputs['value_fun']
            #print("values=",values)
            policy_logits = outputs['policy_logits']
            policy = tf.nn.softmax(policy_logits)
            actions_onehot = tf.one_hot(actions,len(env.Action))
            probs = tf.reduce_sum(actions_onehot*policy,axis=1)
            log_probs = tf.math.log(probs)
            advantages = q_values - values
            actor_loss = - tf.reduce_mean(log_probs*advantages.numpy()) # using mean makes the procedure batch size invariant
            critic_loss = 0.5*(tf.reduce_mean(advantages**2))
            #logits_reg_loss = 0.001*tf.reduce_mean(policy_logits*policy_logits)
            entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(policy,policy_logits))*0.01
            #total_loss = actor_loss + critic_loss + logits_reg_loss - entropy_loss
            total_loss = actor_loss + critic_loss - entropy_loss
            for loss in self.model.losses:
                total_loss += loss
            
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(total_loss,self.model.trainable_weights)
        #grads, _ = tf.clip_by_global_norm(grads, 5.0)
        #print(grads)
        
        #clipped_grads = [tf.clip_by_value(g,-1.,1.) for g in grads]
        
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimiser.apply_gradients(zip(grads, self.model.trainable_weights))

        self._experience = []
        