#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT A METRIC PER CHANNEL 
@author: lauracarlton
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
def SQE_2Dplot_func(snirfObj, metric,savepath= None, colormap=plt.cm.jet):
    '''
    CREATE A 2D MONTAGE OF OPTODES WITH CHANNELS COLOURED ACCORDING TO A GIVEN METRIC
    
    Parameters
    ----------
    snirfObj : a snirf file that contains the measurement list 
    metric : an array of values that is used to create the colormap - one per channel
    savepath : a string that specifies where the figrue should be saved. if none given the figure won't save
    '''

    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r
    
    def pol2cart(theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y
    
    def convert_optodePos3D_to_circular2D(pos, tranformation_matrix, norm_factor):
        pos = np.append(pos, np.ones((pos.shape[0],1)), axis=1)
        pos_sphere = np.matmul(pos,tranformation_matrix)
        pos_sphere_norm = np.sqrt(np.sum(np.square(pos_sphere), axis=1))
        pos_sphere_norm= pos_sphere_norm.reshape(-1,1)
        pos_sphere = np.divide(pos_sphere,pos_sphere_norm)
        azimuth, elevation, r = cart2sph(pos_sphere[:,0], pos_sphere[:,1], pos_sphere[:,2])
        elevation = math.pi/2-elevation;
        x, y = pol2cart(azimuth,elevation)
        x = x/norm_factor
        y = y/norm_factor
        return x, y
    
    channels_df = pd.read_excel('10-5-System_Mastoids_EGI129.xlsx') 
    probe_landmark_pos3D = []
    circular_landmark_pos3D = []
    
    #%% find the landmarks in the probe
    for u in range(len(snirfObj.nirs[0].probe.landmarkLabels)):
        idx_list = channels_df.index[channels_df['Label']==snirfObj.nirs[0].probe.landmarkLabels[u]].tolist()
        if idx_list:
            circular_landmark_pos3D.append([channels_df['X'][idx_list[0]],channels_df['Y'][idx_list[0]], channels_df['Z'][idx_list[0]]])
            landmark_pos3D = snirfObj.nirs[0].probe.landmarkPos3D[u,0:3].tolist()
            landmark_pos3D.extend([1])
            probe_landmark_pos3D.append(landmark_pos3D)
            
    landmarks2D_theta = (channels_df['Theta']*2*math.pi/360).to_numpy()
    landmarks2D_phi = ((90-channels_df['Phi'])*2*math.pi/360).to_numpy()
    x,y = pol2cart(landmarks2D_theta, landmarks2D_phi)
    
    norm_factor = max(np.sqrt(np.add(np.square(x),np.square(y))))
    temp = np.linalg.inv(np.matmul(np.transpose(probe_landmark_pos3D),probe_landmark_pos3D))
    tranformation_matrix = np.matmul(temp,np.matmul(np.transpose(probe_landmark_pos3D),circular_landmark_pos3D))        
    tranformation_matrix = tranformation_matrix
    # tranformation_matrix = np.linalg.lstsq(probe_landmark_pos3D, circular_landmark_pos3D, rcond=None)
    
    #%% scale indices 
    sourcePos2DX , sourcePos2DY = convert_optodePos3D_to_circular2D(snirfObj.nirs[0].probe.sourcePos3D, tranformation_matrix, norm_factor)
    detectorPos2DX , detectorPos2DY = convert_optodePos3D_to_circular2D(snirfObj.nirs[0].probe.detectorPos3D, tranformation_matrix, norm_factor)
    
    scale = 1.3
    sourcePos2DX = sourcePos2DX*scale
    detectorPos2DX = detectorPos2DX*scale
    sourcePos2DY = sourcePos2DY*scale
    detectorPos2DY = detectorPos2DY*scale
        
    #%% plot the positions on the unit circle
    t = np.linspace(0, 2 * np.pi, 100)
    head_x = [math.sin(i) for i in t]
    head_y = [math.cos(i) for i in t]

    plt.figure(figsize=(12,12))
    plt.plot(head_x,head_y,'k')
    for u in range(len(sourcePos2DX)):
        plt.plot(sourcePos2DX[u] , sourcePos2DY[u], 'r.', markersize=8)
        
    for u in range(len(detectorPos2DX)):
        plt.plot(detectorPos2DX[u] , detectorPos2DY[u], 'b.',markersize=8)
    
    
    cmap = colormap
    norm  = matplotlib.colors.Normalize(vmin=min(metric[metric != -np.inf]),vmax= max(metric[metric != np.inf]))
    sm = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
    
    fontDict = dict(fontweight = 'bold', fontstretch= 'expanded',fontsize = 7)
    for u in range(len(snirfObj.nirs[0].data[0].measurementList)//2 - 1):
        sourceIndex =  snirfObj.nirs[0].data[0].measurementList[u].sourceIndex
        detectorIndex =  snirfObj.nirs[0].data[0].measurementList[u].detectorIndex
        
        x = [sourcePos2DX[sourceIndex-1], detectorPos2DX[detectorIndex-1]]
        y = [sourcePos2DY[sourceIndex-1], detectorPos2DY[detectorIndex-1]]
        
        if np.isnan(metric[u]) or np.isinf(metric[u]):
            color = 'gray'
            linestyle = '--'
        else:
            color = cmap(norm(metric[u]))
            linestyle = '-'
            
        plt.plot(x,y, color=color,linestyle=linestyle, linewidth = 2)
        plt.text(sourcePos2DX[sourceIndex-1], sourcePos2DY[sourceIndex-1],str(sourceIndex),fontdict=fontDict, bbox=dict(color = 'r',boxstyle = "round, pad=0.3", alpha=0.1))
        plt.text(detectorPos2DX[detectorIndex-1], detectorPos2DY[detectorIndex-1], str(detectorIndex),fontdict=fontDict, bbox=dict(color='b',boxstyle = "round, pad=0.3", alpha=0.05))
                                                                                                                                     
    plt.plot(0, 1 , marker="^",markersize=28)
    # sm = plt.cm.ScalarMappable(cmap=cmap)
    plt.colorbar(sm,shrink =0.25)
        
    plt.axis('equal')
    plt.axis('off')
    
    if savepath != None:
        plt.savefig(savepath, dpi=100)

    plt.show()

    pass
