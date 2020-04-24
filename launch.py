# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:13:05 2020

"""

#.\venv\Scripts\activate
import matplotlib
matplotlib.use('AGG')
import tensorflow as tf
import runner
import environment as env

def main():
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.experimental.list_physical_devices('GPU') 
    e = env.Environment(7,7,10)
    #tf.debugging.enable_check_numerics()
    df = runner.run_training(e,32,6000,0.99,0.001)
    

if __name__ == '__main__':
    main()
    
