import numpy as np
import json
from scipy.sparse import coo_matrix


# plot 3D figure with the truss members in undeformed, and in deformed configurations
def plotStructFun(glb, elem, inputData, inputUnit, a, f, plotting):
    
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    
    deformation = inputData['nodes'] + np.reshape(glb['u'],(-1,3)) # from number of nodes x 3 to number of DOF x 1, in a [x0,y0,z0,x1,y1,z1]
    
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')

    
    if plotting['deformed'] == 0 or plotting['deformed'] == 2:
        for i,[n0,n1] in enumerate(zip(elem['n0'],elem['n1'])):
            if a[i]>10e-15:
                x = [[inputData['nodes'][n0][i], inputData['nodes'][n1][i]] for i in [0,1,2]]
                ax.plot3D(x[0],x[1],x[2], c='grey')
    
    if inputUnit == 'm':
        uM = 1
    else:
        uM = 0.001
        
    mtp = 0.5*2834.65 # meter to point
    
    if plotting['deformed'] == 1 or plotting['deformed'] == 2:

        c = []
        for fi in f:
            if fi>0:
                c.append('r')
            else:
                c.append('b')

        for i,[n0,n1] in enumerate(zip(elem['n0'],elem['n1'])):
            
            if a[i]>10e-15:
                x = [[deformation[n0][i], deformation[n1][i]] for i in [0,1,2]]
                ax.plot3D(x[0],x[1],x[2], c=c[i], linewidth=uM*mtp*2*np.sqrt(a[i]/np.pi))

    plt.show()
    
def importData(cwd, inputUnit):
    inputDataNames =['members', 'constraints', 'nodes', 'forces']
    
    inputData = dict()
    num = dict()
    
    for dataName in inputDataNames:
        inputData[dataName] = np.genfromtxt(cwd+'/'+dataName+'.csv', delimiter=',')
        if len(np.shape(inputData[dataName])) == 1:
            numDim = 1
        else:
            numDim = np.shape(inputData[dataName])[0]
        num[dataName[0]] = numDim

    # reading parameter data
    with open(cwd+'/'+'parameters.csv') as f:
        inputData['parameters'] = json.loads(f.read())
        
    if inputUnit == 'mm':
      uM = 1000
    else:
      uM = 1
      
    inputData['nodes'] = inputData['nodes']*uM
    inputData['members'][:,2] = inputData['members'][:,2]*(uM**2)
    inputData['members'][:,3] = inputData['members'][:,3]/(uM**2)

    inputData['constraints'][:,4] = inputData['constraints'][:,4]*uM
    inputData['parameters']['minA']        = inputData['parameters']['minA'] * (uM**2)
    inputData['parameters']['maxA']        = inputData['parameters']['maxA'] * (uM**2)
    inputData['parameters']['yieldStress'] = inputData['parameters']['yieldStress'] / (uM**2)
    return [inputData, num]
    
def initializeStruct(inputData,num):

    num['nDOF'] = 3 # nodal DOF
    num['lDOF'] = 2*num['nDOF'] # elemental DOF (there are two nodes per members)
    num['DOF']  = num['n']*num['nDOF'] # global DOF

    elem = dict()
    
    # Assemble global connectivity matrix
    elem['A']   = inputData['members'][:,2]
    elem['E']   = inputData['members'][:,3]

    elem['n0']  = np.array(inputData['members'][:,0], dtype=np.int32)
    elem['n1']  = np.array(inputData['members'][:,1], dtype=np.int32)
    
    ##
    # $$u_y = \frac{f_y}{E}$$
    
    glb = dict()
    ## Initialize global information storage
    glb['b_l'] = np.array([[1,0,0,-1,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [-1,0,0,1,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]])

    # initial force vector
    glb['F'] = assembleBC(inputData['forces'],num,'F')

    # initial displacement vector
    glb['u'] = assembleBC(inputData['constraints'],num,'u')
    
    # Find the indices of the u inits that are inf (free), or not
    glb['fDOF'] = np.where(glb['u'] == -np.inf)[0] # these dofs are free
    glb['pDOF'] = np.where(glb['u'] != -np.inf)[0] # these dofs are constrained

    # number of free and prescribed DOFs
    num['fDOF'] = len(glb['fDOF'])
    num['pDOF'] = len(glb['pDOF'])
    
    return [glb, elem, num]
    
    
def assembleA(num, nodes, n0, n1):

    data = np.hstack((-1*np.ones(num['m'], dtype=np.int8), np.ones(num['m'], dtype=np.int8))) # data -1 and 1 [2*m x 1]

    row  = np.arange(0,num['m'])
    row  = np.hstack((row,row))
    
    #cols = [n0,n1] # cols - node col # [2*n x 1]
    col  = np.hstack((n0,n1))

    C   = coo_matrix((data, (row, col)), shape=(num['m'],num['n'])).toarray()  # connectivity matrix [m x n]

    u    = np.matmul(C,nodes) # [m x n] x [n x 3] --> [m x 3]

    U    = coo_matrix((u[:,0],(np.arange(num['m']),np.arange(num['m']))), shape=(num['m'],num['m'])).toarray()  # [mxm]
    V    = coo_matrix((u[:,1],(np.arange(num['m']),np.arange(num['m']))), shape=(num['m'],num['m'])).toarray()  # [mxm]
    W    = coo_matrix((u[:,2],(np.arange(num['m']),np.arange(num['m']))), shape=(num['m'],num['m'])).toarray()  # [mxm]

    eL   = np.sum(u**2,axis=-1)**(1./2)# element length
    
    L    = np.diag(eL) # diagonalize eL
    
    [Dx,Dy,Dz] = [np.matmul(np.matmul(C.transpose(),x),np.linalg.inv(L))  for x in [U,V,W]]# [n x m][m x m][m x m] 
    
    AKm = np.zeros((num['DOF'],num['m']))

    AKm[0::3,:] = Dx
    AKm[1::3,:] = Dy
    AKm[2::3,:] = Dz
    
    AKm = -AKm
    
    return [C, AKm,eL]
    
def SolverLE(glb, elem, num):
    # FEM Analysis
    #
    # $$B_{i,i}=\frac{L_i^2}{E_it_i}\,\quad k=1,...,m$$
    #
    # $$K=AB^{-1}A^{T}$$
    #
    # $$Ku=F$$
    #
    # $$s=\frac{1}{2}u^TKu$$
    f_id  = np.ix_(glb['fDOF'])
    p_id  = np.ix_(glb['pDOF'])
    ff_id = np.ix_(glb['fDOF'],glb['fDOF'])
    fp_id = np.ix_(glb['fDOF'],glb['pDOF'])
    pf_id = np.ix_(glb['pDOF'],glb['fDOF'])
    pp_id = np.ix_(glb['pDOF'],glb['pDOF'])
    
    # Solve for the unknown displacements
    glb['u'][f_id]=np.linalg.solve(glb['Km'][ff_id], glb['F'][f_id]-np.matmul(glb['Km'][fp_id],glb['u'][p_id])) # K11 x u1 = F1 - K12 x u2
    # Solve for reaction
    glb['F'][p_id]=np.matmul(glb['Km'][pf_id], glb['u'][f_id]) + np.matmul(glb['Km'][pp_id], glb['u'][p_id])

    ## Post processing
    
    elem['f']=-1/2 * elem['B'] * (np.matmul(glb['A'][f_id].transpose(), glb['u'][f_id])+np.matmul(glb['A'][f_id].transpose(), glb['u'][f_id])) # member force
    
    elem['stress'] = elem['f'] / elem['A'] # member stress
    elem['strain'] = elem['stress'] / elem['E'] # member strain
    
    # calculate element wise displacement
    elem['d'] = -np.matmul(glb['A'][f_id].transpose(), glb['u'][f_id]) # displacement in member - A^T*u
    #elem['d'] = elem['strain'] * eL # displacement in member - epsilon * L
    
    # Check if the calculations were correct
    status = dict()
    status['equilibrium']   = sum((np.matmul(glb['A'][f_id],elem['f'])+glb['F'][f_id])**2)
    status['compatibility'] = sum((np.matmul(np.diag(elem['B']**-1),elem['f'])+np.matmul(glb['A'][f_id].transpose(),glb['u'][f_id]))**2)
    status['stiffness']     = sum((glb['F'][f_id]-np.matmul(glb['Km'][ff_id],glb['u'][f_id]))**2)
    
    return [glb,elem,status]
    
def assembleBC(table,num,BC):
    if BC == 'u':
        out   = np.zeros(num['DOF'])-np.inf
        numBC = num['c']
    elif BC == 'F':
        out   = np.zeros(num['DOF'])
        numBC = num['f']
    
    for i in np.arange(0,numBC):
        if numBC == 1:
            n  = table[0]
            d  = table[1:4]
            m  = table[4]
        else:
            n  = table[i,0]
            d  = table[i,1:4]
            m  = table[i,4]
        for row in np.arange(0,num['nDOF']):
                if d[row] == 1:
                    idx = int(num['nDOF']*n+row)
                    out[idx] = m
    return out

def assembleKm(AKm, BKm, num):
    # glb.Km=np.array((num['DOF'],num['DOF']))
    # for k in np.arange(0, num['m']-1):
        # km=elem(k).A*elem(k).Et/l(k)*glb.b_l
        # Km=elem(k).T.'*km*elem(k).T
        # [glb.Km]=assembleId(num.nDOF,n1(k),n2(k),glb.Km,Km)
        
    # [n x n] = [n x m] x [m x m] x [m x n]
    Km  = np.matmul(np.matmul(AKm, np.diag(BKm)), AKm.transpose())
    return Km

'''
function
   optTruss - this function should define c, A_eq, b_eq, A_ub, b_ub, bounds, and use 
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
input:
   glb
   elem
   num
   yieldStress
output:
   elem['a'] - optimized cross sectional area as an array [mx1]
   elem['f'] - element-wise forces
'''
def optTruss(glb, elem, num, yieldStress, minA, maxA):
    f_id  = np.ix_(glb['fDOF'])

    c    = np.hstack((np.zeros(num['m']), elem['L']))

    A_eq = np.hstack((glb['A'][f_id], np.zeros((len(glb['fDOF']),num['m']))))
    b_eq = -glb['F'][f_id]

    A_ub = np.zeros((num['m']*2, num['m']*2))
    b_ub = np.zeros(num['m']*2)
    
    lb   = np.zeros(num['m']*2)
    ub   = np.zeros(num['m']*2)

    for i in np.arange(num['m']):
        A_ub[i,          i]          = -1.
        A_ub[i,          i+num['m']] = -yieldStress
        A_ub[i+num['m'], i]          =  1.
        A_ub[i+num['m'], i+num['m']] = -yieldStress

        lb[i]          = None
        ub[i]          = None
        lb[i+num['m']] = minA
        ub[i+num['m']] = maxA

    bounds= [(ubi,lbi) for ubi,lbi in zip(lb,ub)]

    from scipy.optimize import linprog
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds = bounds)
    a_i = res['x'][num['m']:]
    f_i = res['x'][0:num['m']]
    return [a_i,f_i]