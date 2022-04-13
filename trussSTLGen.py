'''
This converts a truss design (in terms of a list of coordinates and connection) into STL files (either ascii or binary) with a predefined polygon count
'''
import csv
import numpy as np
import math
import transforms3d as t3 #For generating transformation matrices
import struct #For writing to bin
import matplotlib as plt

def main():
    trussSTLGenObj=trussSTLGen('../sample_inputs/','coord_2.csv','conn_2.csv','test.stl',8, 'mm', False)

#----------------------------------------------
class trussSTLGen(object):
    def __init__(self,fPath,fCoord,fCon,fOut,iDiv, units, boolAscii):
        '''
            input:
            fPath: path of the input file
            fCoord: file name of the coordinate file
            fCon: file name of the connectivity file
            fOutL file name of the STL
            iDiv: number of polygon to approximate the circular cross section, 4 would mean a square
            boolAscii: True - ascii, false - binary (smaller file size but not human readable)
        '''
        if units == 'mm':
            uM = 1000
        else:
            uM = 1
        reader=np.genfromtxt(fPath+fCoord,delimiter=',')
        sN = reader
        # import csv node / elem file
        reader=np.genfromtxt(fPath+fCon,delimiter=',')

        sE=[[i for i in j[0:2]] for j in reader]
        sE=np.array(sE).astype('int')
        sE = reader[:, 0:2].astype('int')
        #sR=[[i for i in j[2:3]] for j in reader]
        #sR=np.array(sR).astype('float')
        sA = reader[:,2:3].flatten()
        sR = np.sqrt(sA/np.pi)*uM
        iNumBars=len(sE)
        x1=[]
        x2=[]
        objFacets=np.zeros([3,iDiv*32,iNumBars])
        for n in range(0,iNumBars):
            x1=[sN[sE[n,0]-1,0]*uM, sN[sE[n,0]-1,1]*uM, sN[sE[n,0]-1,2]*uM]
            x2=[sN[sE[n,1]-1,0]*uM, sN[sE[n,1]-1,1]*uM, sN[sE[n,1]-1,2]*uM]
            objFacets[:,:,n]=self.genFacets(x1,x2,iDiv,sR[n])
        self.writeFacets(boolAscii,fPath+fOut,objFacets)

    def genFacets(self,x1,x2,div,dR):
        if np.linalg.norm(x1)>np.linalg.norm(x2):
            x1,x2=x2,x1
        x1=np.array(x1)
        x2=np.array(x2)
        
        d=x2-x1
        dProjLen=(d[0]**2+d[2]**2)**0.5
        yrotate=np.array([0,1,0])
        
        if dProjLen!=0.0:
            dPhi=math.atan(d[1]/dProjLen)
            dTheta=math.acos(d[2]/dProjLen)
            if d[0]<0:
                dTheta=2*math.pi-dTheta
            M1=t3.axangles.axangle2aff(yrotate,dTheta)
        else:
            dPhi=math.pi/2
            M1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        x_p_axis=np.dot(M1,np.array([1,0,0,0]).T)
        M2=t3.axangles.axangle2aff(-x_p_axis[0:3],dPhi)    
        M3=np.eye(4)
        M3[0:3,3]=x1.T
        M4=np.eye(4)
        M4[0:3,3]=x2.T
        M_1=np.dot(np.dot(M3,M2),M1)
        M_2=np.dot(np.dot(M4,M2),M1)
        
        areaPoly=dR**2*np.pi
        sPoly=np.sqrt(4*areaPoly/(div*2)*np.tan(np.pi/(div*2)))
        rPoly=sPoly/np.sin(np.pi/(div*2))/2

        angle=np.linspace(0,2*np.pi-np.pi/div,2*div)
        x=[rPoly*math.cos(a) for a in angle]
        y=[rPoly*math.sin(a) for a in angle]
        x_coord=[[y1,y2,0,1] for y1 , y2 in zip(x, y)]
        
        x_trans_1=[np.dot(M_1,x_i) for x_i in x_coord]
        x_trans_2=[np.dot(M_2,x_i) for x_i in x_coord]
        listFacetNode=np.vstack([x_trans_1,x_trans_2,np.hstack([x1,1]),np.hstack([x2,1])])

        dElem=div*2
        n_start=dElem*2+1
        n_end=dElem*2+2
        listFacetElem=[]
        for n in np.arange(1,dElem+1):
            if n!=dElem:
                n1=n
                n2=n+1
                n3=n+dElem
                n4=n+dElem+1
            else:
                n1=n
                n2=n-dElem+1
                n3=n+dElem
                n4=n+1
            
            listFacetElem.append([n1,n2,n3])
            listFacetElem.append([n2,n4,n3])
            listFacetElem.append([n2,n1,n_start])
            listFacetElem.append([n3,n4,n_end])
            
            
        listFacetElem=np.array(listFacetElem,dtype='int')

        arrFacets=np.zeros([3,4,dElem*4])
        for n in np.arange(0,dElem*4,1):
            v1=listFacetNode[listFacetElem[n,1]-1,0:3].transpose()-listFacetNode[listFacetElem[n,0]-1,0:3].transpose()
            v2=listFacetNode[listFacetElem[n,2]-1,0:3]-listFacetNode[listFacetElem[n,0]-1,0:3]

            vNorm=np.cross(v1,v2)
            vNorm=vNorm/np.linalg.norm(vNorm)
            
            temp=np.vstack([vNorm,
                listFacetNode[listFacetElem[n,0]-1,0:3],
                listFacetNode[listFacetElem[n,1]-1,0:3],
                listFacetNode[listFacetElem[n,2]-1,0:3]])
            arrFacets[:,:,n]=temp.T
        arrFacets=np.squeeze(np.reshape(arrFacets,(3,-1,1),'F'))
        return arrFacets

    def writeFacets(self,boolAscii,strOutput,objFacets):
        strTitle='OBJECT'
        if boolAscii==False:
            f=open(strOutput,'w')
            print('Write '+str((objFacets.shape)[2])+' bars\n')
            print('...\n')
            f.write('solid '+strTitle+'\n')
            for n in np.arange(0,objFacets.shape[2]):
                indBar=np.squeeze(np.reshape(objFacets[:,:,n],[3,4,-1],"F"))
                for m in np.arange(0,indBar.shape[2]):
                    indTri=indBar[:,:,m].T
                    f.write(
                        "  facet normal " + '{:.17f}'.format(indTri[0,0]) + " " + '{:.17f}'.format(indTri[0,1]) + " " + '{:.17f}'.format(indTri[0,2]) + "\n"+
                        "    outer loop\n"+
                        "      vertex "+ '{:.17f}'.format(indTri[1,0]) +" "+ '{:.17f}'.format(indTri[1,1]) +" "+ '{:.17f}'.format(indTri[1,2]) +"\n"+
                        "      vertex "+ '{:.17f}'.format(indTri[2,0]) +" "+ '{:.17f}'.format(indTri[2,1]) +" "+ '{:.17f}'.format(indTri[2,2]) +"\n"+
                        "      vertex "+ '{:.17f}'.format(indTri[3,0]) +" "+ '{:.17f}'.format(indTri[3,1]) +" "+ '{:.17f}'.format(indTri[3,2]) +"\n"+
                        "    endloop\n"+
                        "  endfacet\n")
            f.write('endsolid '+strTitle)
        
        else:
            binFacets=np.reshape(objFacets,[12,-1,objFacets.shape[2]],"F")
            row13=np.ones([1, binFacets.shape[1],binFacets.shape[2]])
            binFacets=np.concatenate((binFacets, row13),0)
            intFacets=binFacets.shape[1]*binFacets.shape[2]
            binFacets=np.squeeze(np.reshape(binFacets,[-1,1,binFacets.shape[2]],"F"))
            binFacets=np.squeeze(np.reshape(binFacets,[-1,1],"F"))
            
            f=open(strOutput,'w+b')
            f.write(struct.pack("80s",strTitle))
            f.write(struct.pack("I",intFacets))
            for n in np.arange(1,len(binFacets)+1):
                if n%13!=0:
                    f.write(struct.pack("f",binFacets[n-1]))
                else:
                    f.write(struct.pack("H",binFacets[n-1]))
            f.close
            print('Done writing\n')
            return binFacets
            
if __name__=="__main__":
    main()