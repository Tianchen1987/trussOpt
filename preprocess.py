
import numpy as np 
from math import pi, atan,sqrt

def ptGrid(n_in,l_in):
    nList = []
    for i in range(n_in[0]):
        for j in range(n_in[1]):
            for k in range(n_in[2]):
                nList.append([round(i*l_in[0],10),round(j*l_in[1],10),round(k*l_in[2],10)])
    return nList

def calAngle(pt0,pt1):
    xyL  = ((pt1[1]-pt0[1])**2 + (pt1[0]-pt0[0])**2)**0.5

    if pt1[2]-pt0[2] == 0:
        phi = 0
    elif xyL == 0:
        phi = pi/2
    else:
        phi   = atan((pt1[2]-pt0[2])/xyL)
    
    if pt1[1]-pt0[1] == 0:
        theta = pi/2
    elif pt1[0]-pt0[0] == 0:
        theta = 0
    else:
        theta = atan((pt1[1]-pt0[1])/(pt1[0]-pt0[0]))
    
    return [phi,theta]

def genLine(pt, a_i, E_i ,dMax):
    checkList = []
    crv = []
    for i, pt_i in enumerate(pt):
        for j,pt_j in enumerate(pt[i+1:]):
            [phi,theta] = calAngle(pt_j,pt_i)
            d = distPt(pt_j,pt_i)
            phi_ = round(phi,10)
            theta_ = round(theta,10)
            if [pt_i, phi_, theta_] not in checkList and d<dMax:
                checkList.append([pt_i, phi_, theta_])
                crv.append([i,i+j+1, a_i, E_i])
    return crv

def distPt(pt0,pt1):
    return sqrt(sum([(pt1[i]-pt0[i])**2 for i in [0,1,2]]))

def genBC(cPt,fPt,nList):
    cId = [nList.index(cPti['n']) for cPti in cPt]
    fId = [nList.index(fPti['n']) for fPti in fPt]
    cList = []
    fList = []

    for i, cIdi in enumerate(cId):
        cList.append([cIdi] + [cPt[i]['dof'][0]] + [cPt[i]['dof'][1]] + [cPt[i]['dof'][2]] + [cPt[i]['mag']])

    for i, fIdi in enumerate(fId):
        fList.append([fIdi] + [fPt[i]['dof'][0]] + [fPt[i]['dof'][1]] + [fPt[i]['dof'][2]] + [fPt[i]['mag']])
    return [cList,fList]

import os

def f(l):
    return [item for sublist in l for item in sublist]

def writeList(wDir, mName, m, l,fName):
    
    fPath = wDir+'/'+str(m)+mName
    # Check whether the specified path exists or not
    if not os.path.exists(fPath):
      # Create a new directory because it does not exist 
      os.makedirs(fPath)
      
    np.savetxt(wDir+'/'+str(m)+mName+'/'+fName+'.csv', l, delimiter=',')
           