# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:50:03 2020

"""

#import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import player
import environment

def save_movie(video_frames,movie_path,fps):
    #print('Movie will run for %g seconds' % total_movie_times_sec)
    frame_shape = video_frames[0].shape
    height, width, channels = frame_shape
    #n_frames = len(video_frames)
    #print('%d video frames from %s until %s with shape %dx%dx%d' % (
    #        n_frames, start_datetime, last_datetime, height, width, channels))
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    #fourcc = cv.VideoWriter_fourcc(*'X264')
    video = cv.VideoWriter(movie_path, fourcc, float(fps), (width, height))
    for frame in video_frames:
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        video.write(frame)
    video.release()
    
    
def make_frame(data, size=(10,10)):
    fig=plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig,[0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data,aspect='equal')
    #fig.tight_layout(pad=0)
    return ax

def make_bitmap(env):
    bitmap = np.zeros((env.height,env.width,3))
    bitmap[env.pub[1],env.pub[0]] = [150,75,0]
    drinkers = 0
    for p,a in zip(env.players, env.last_actions):
        if p.status is not player.Status.DEAD:
            if a is environment.Action.WASH_HANDS:
                pixel=[0,0,255]
            elif p.status is not player.Status.INFECTED:
                if p.recovery_prob == environment.OLD_REC_PROB:
                    pixel=[255,255,255]
                else:
                    pixel=[0,255,0]
            else:
                scaling_factor = p.health/player.MAX_HEALTH
                pixel = [255*scaling_factor,0,0]
            
            if (p.y,p.x)==(env.pub[1],env.pub[0]):
                drinkers += 255/env.population*4
                bitmap[env.pub[1],env.pub[0]]=[150,75,np.clip(drinkers,0,255)]
            else:
                bitmap[p.y,p.x]=pixel
    return bitmap/255

def make_empty_bitmap(env):
    bitmap = np.ones((env.height,env.width,3))
    return bitmap

def save_frame(ax, filename):
    ax.get_figure().savefig(filename, dpi=72)