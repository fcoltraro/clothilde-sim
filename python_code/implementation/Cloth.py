import os
# Limit threads for pyKDTREE and CHOLMOD 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np 
import scipy.sparse as sp
from sksparse.cholmod import cholesky, cholesky_AAt
from pykdtree.kdtree import KDTree
import polyscope as ps
from line_profiler import profile

class Cloth:
    def __init__(self,verts,faces,name="clothilde"):
        #positions and velocities
        self.positions = np.array(verts, order = 'F') #current position of the vertices of the mesh
        assert self.positions.shape[1] == 3 and self.positions.ndim == 2, 'Something is wrong with the vertices dimensions'
        self.velocities = np.zeros(self.positions.shape, order = 'F') + 1e-12 #current velocities of the vertices of the mesh
        self.history_pos = [self.positions] #history of the vertices of the mesh 
        self.history_vel = [self.velocities] #history of the velocities of the vertices of the mesh 
        #self.positions += 0.0001*np.random.randn(self.positions.shape[0],3) #avoid singular flat case

        #topology of the mesh
        self.faces = np.array(faces) #quadrangulation of the vertices in positions (index based)
        assert self.faces.shape[1] == 4 and self.faces.ndim == 2, 'Current implementation only supports quad meshes'
        self.edges = [] #list of unoriented edges in set form
        self.edges_matrix = np.zeros([0,2]) #edges in matrix form for efficient computations
        self.n_verts = self.positions.shape[0]
        self.n_faces = self.faces.shape[0]
        self.n_edges = 0
        #TODO: check euler characteristic of each connected component and assert it should be contractible?
        self.A0 = None # adjacency matrix for edges-vertices
        self.A1 = None # adjacency matrix for faces-edges
        self.A2 = None # adjacency matrix for faces-vertices
        self.neighbors = None # neighbors dict wrt the edges of the mesh
        self.edges_bnd = np.zeros([0,2]) #edges corresponding to the boundary of the mesh in matrix form
        self.nodes_bnd = None #these are indices wrt vertices, not 3D positions

        #for self-collisions
        self.rad = None #radious of the balls
        self.last_check = np.array(verts, order = 'F') #for checking close self-collision pairs
        self.den_last = 1
        self.kn = 12 #get k nearest nodes to every node
        self.nodes = np.arange(self.n_verts) #needed for proximity detection
        self.empty = np.array([],dtype=int) #handy sometimes
        self.ni = np.repeat(np.arange(self.n_verts),self.kn)

        #for plotting with polyscope
        self.ps_frame = 0 #for making a movie: go through the history
        self.label = name #user given name 
        self.polyscoped = False
        
        # finite element matrices for computing forces
        self.reference_element = self.ReferenceElement(self.faces.shape[1])
        self.Fg = None # gravity force
        self.D = None # Rayleign damping
        self.M = None # mass matrix
        self.K = None # stiffness matrix
        self.M_lum = None # mass matrix for flattened positions in shape (3*n_verts,1)
        self.m_sqrt = None # 1/sqrt(m_i) for cholesky decompositions
        self.m_sqrt_mat = None # same as before but as a column matrix
        self.factor_E = None # factor of the cholesky matrix for fast implicit euler
        
        # default physical parameters of the cloth
        self.rho = None # density of the cloth
        self.delta = None # virtual mass for aerodynamics
        self.kappa = None # bending stiffness
        self.shr = None # shear elasticity
        self.str = None # stretch elasticity
        self.alpha = None # slow damping 
        self.beta = None # fast damping
        self.mu_floor = None #friction with to the floor
        self.mu_self = None #friction for self-collisions
        
        # solver variables
        self.dt = None #time step for simulating
        self.tol = None #solver tolerance in %
        self.total_iters = 0 #global number of iterations perfomed when calling simulate
        self.warning = False #for displaying the warning

        #controled nodes
        self.control = [] #for precomputing cholesky factorizations and only updating when necessary
        self.Iu = self.empty; self.Ju = self.empty; self.Ku = self.empty

        #compute all the necesary elements for simulation only once
        self.prepareSimulation()

    def restart(self):
        self.positions = self.history_pos[0]
        self.velocities = self.history_vel[0]
        self.history_pos = [self.positions] 
        self.history_vel = [self.velocities] 
        self.total_iters = 0
        self.warning = False

    def __repr__(self):
        return f"Cloth({self.n_verts} vertices, {self.faces.shape[0]} quads)"
    
    class ReferenceElement:
        def __init__(self, type):
            match type:
                case 3:
                    self.w = np.array([1, 1, 1])/6
                    self.nodesCoord = np.block([[0, 0], [1, 0], [0, 1]])
                    self.z = np.block([[0.5, 0], [0, 0.5], [0.5, 0.5]])
                    self.xi, self.eta = self.z[:, 0], self.z[:, 1]
                    self.N = np.block([[1-self.xi-self.eta], [self.xi], [self.eta]])
                    self.Nxi = np.block([[-np.ones(len(self.w))], 
                                         [np.ones(len(self.w))], 
                                         [np.zeros(len(self.w))]]).T
                    self.Neta = np.block([[-np.ones(len(self.w))], 
                                         [np.zeros(len(self.w))], 
                                         [np.ones(len(self.w))]]).T

                case 4:
                    self.w = np.array([1, 1, 1, 1])
                    self.nodesCoord = np.block([[-1, -1], [1, -1], [1, 1], [-1, 1]])
                    self.z = self.nodesCoord/np.sqrt(3)
                    self.xi, self.eta = self.z[:, 0], self.z[:, 1]
                    self.N = (1/4)*np.block([[np.multiply(1-self.xi, 1-self.eta)], 
                                             [np.multiply(1+self.xi, 1-self.eta)], 
                                             [np.multiply(1+self.xi, 1+self.eta)], 
                                             [np.multiply(1-self.xi, 1+self.eta)]])
                    self.Nxi = (1/4)*np.block([[self.eta - 1],
                                               [1 - self.eta], 
                                               [1 + self.eta], 
                                               [-1 - self.eta]]).T
                    self.Neta = (1/4)*np.block([[self.xi - 1], 
                                                [-1 - self.xi], 
                                                [1 + self.xi], 
                                                [1 - self.xi]]).T
    
    def prepareSimulation(self):
        # compute all auxiliar objects for fast simulation
        self.checkQuadMesh()
        self.computeEdges()
        self.buildShareEdgeMatrix()
        self.buildAdjacencyMatrices()
        self.computeBoundary()
        self.triangulateQuadMesh()
        self.prepareMatrices()
        self.computeStretchShear()

    def checkQuadMesh(self):
        pass
        #TODO check that every quad mesh element is well ordered

    def computeEdges(self):
        if self.n_edges == 0: #only do it once
            match self.faces.shape[1]:
                case 3:
                    #all unoriented edges with repetitions     
                    e1 = self.faces[:,[0,1]]; e2 = self.faces[:,[0,2]]; e3 = self.faces[:,[1,2]]
                    #we form a set to remove repeated edges
                    edges = set(map(frozenset, e1))
                    edges.update(set(map(frozenset, e2))); edges.update(set(map(frozenset, e3)))
                case 4:
                    #quads    
                    e1 = self.faces[:,[0,1]]; e2 = self.faces[:,[1,2]]; 
                    e3 = self.faces[:,[2,3]]; e4 = self.faces[:,[3,0]]
                    #we form a set to remove repeated edges
                    edges = set(map(frozenset, e1)); edges.update(set(map(frozenset, e2))); 
                    edges.update(set(map(frozenset, e3))); edges.update(set(map(frozenset, e4)))
            self.edges = list(edges) #list of unoriented edges in set form
            self.edges_matrix = np.array(list(map(list,self.edges))) #in matrix form, handy for some computations
            self.n_edges = len(self.edges)

    def buildAdjacencyMatrices(self):
        assert self.n_edges > 0, "Please compute first all edges"
        
        if self.A0 is None:
            row = np.array(range(self.n_edges)); row = np.concatenate((row, row))
            col = self.edges_matrix[:,0]; col = np.concatenate((col, self.edges_matrix[:,1]))
            data = np.ones_like(row)
            self.A0 = sp.coo_matrix((data, (row, col)), shape=(self.n_edges, self.n_verts)).tocsr()
            self.A0t = self.A0.T.tocsr()
        
        if self.A1 is None:
            #magic dict
            ind_edges = dict((k, i) for i, k in enumerate(self.edges))
            match self.faces.shape[1]:
                case 3:
                    row = np.array(range(self.n_faces)); row = np.concatenate((row,row,row))
                    #heavylifting
                    e1 = self.faces[:,[0,1]]; se1 = list(map(frozenset,e1))
                    e2 = self.faces[:,[1,2]]; se2 = list(map(frozenset,e2))
                    e3 = self.faces[:,[2,0]]; se3 = list(map(frozenset,e3))
                    #find indices in edges_matrix
                    ind1 = np.array([ind_edges[x] for x in se1])
                    ind2 = np.array([ind_edges[x] for x in se2])
                    ind3 = np.array([ind_edges[x] for x in se3])
                    col = np.concatenate((ind1,ind2,ind3))
                case 4:
                    row = np.array(range(self.n_faces)); row = np.concatenate((row,row,row,row))
                    #heavylifting
                    e1 = self.faces[:,[0,1]]; se1 = list(map(frozenset,e1))
                    e2 = self.faces[:,[1,2]]; se2 = list(map(frozenset,e2))
                    e3 = self.faces[:,[2,3]]; se3 = list(map(frozenset,e3))
                    e4 = self.faces[:,[3,0]]; se4 = list(map(frozenset,e4))
                    #find indices in edges_matrix
                    ind1 = np.array([ind_edges[x] for x in se1])
                    ind2 = np.array([ind_edges[x] for x in se2])
                    ind3 = np.array([ind_edges[x] for x in se3])
                    ind4 = np.array([ind_edges[x] for x in se4])
                    col = np.concatenate((ind1,ind2,ind3,ind4))                   
            #create sparse matrix
            data = np.ones_like(row)
            self.A1 = sp.coo_matrix((data, (row, col)), shape=(self.n_faces, self.n_edges)).tocsr()

        if self.A2 is None:
            row = np.array(range(self.n_faces)); row = np.concatenate((row, row, row, row))
            col = np.concatenate((self.faces[:,0],self.faces[:,1],self.faces[:,2],self.faces[:,3]))
            data = np.ones_like(row)
            self.A2 = sp.coo_matrix((data, (row, col)), shape=(self.n_faces, self.n_verts)).tocsr()
            self.A2t = self.A2.T.tocsr()
            self.nodes_faces_count = np.array(self.A2.sum(axis=0))[0]
            self.Am = sp.vstack([sp.eye(self.n_verts),0.25*self.A2]).tocsr() #for plotting

    def computeBoundary(self):
        sumCols = np.array(self.A1.T.sum(axis=1))
        index = np.where(sumCols == 1)[0] #an edge only contained in one face
        edges_bnd = self.edges_matrix[index,:] 
        self.nodes_bnd = np.unique(edges_bnd.reshape(2*edges_bnd.shape[0])) # indices of the nodes of the boundary
        self.edges_bnd = edges_bnd

    def triangulateQuadMesh(self):
        #triangulation of quad mesh
        k1, k2, k3, k4 = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2], self.faces[:, 3]
        self.f0 = k1; self.f1 = k2
        self.f2 = k3; self.f3 = k4
        n_tot = self.n_verts + self.n_faces
        k5 = np.arange(self.n_verts, n_tot)
        self.triangles = np.vstack([
            np.column_stack([k1, k2, k5]),
            np.column_stack([k2, k3, k5]),
            np.column_stack([k3, k4, k5]),
            np.column_stack([k4, k1, k5]),
        ])
        #computation of its edges
        edges = np.vstack([
            self.triangles[:, [0, 1]],
            self.triangles[:, [1, 2]],
            self.triangles[:, [2, 0]]
        ])     
        edges = np.sort(edges, axis=1)     
        self.edges_tri = np.unique(edges, axis=0)
        #computation of neighbors
        S = sp.lil_matrix((n_tot, n_tot)); alpha = 0.75
        for n in range(n_tot):
            aux = (self.edges_tri[:,0] == n) + (self.edges_tri[:,1] == n)
            edges_n = self.edges_tri[aux == True,:]
            neighs_n = np.setdiff1d(np.unique(edges_n),n)
            if n in self.nodes_bnd:
                S[n, n] = 1
            else:
                S[n, n] = alpha
                S[n,neighs_n] = (1 - alpha)/neighs_n.shape[0]
        self.S = S

    def prepareMatrices(self):
        if self.M is None: # compute matrices with reference element if not done before
            
            #mass matrix and laplacian
            M, L = self.precomputeMatrix(self.faces)
            # lumped mass matrices and inverses
            m_lum = M.sum(axis = 1)  #lumping the mass matrix in vector form
            m_inv = np.array([1./x for x in m_lum])  # inverse of the lumped mass matrix
            m_sqrt = np.array([1./np.sqrt(x) for x in m_lum])  # inverse of the root of the lumped mass matrix
            #save matrices
            M_lum = sp.diags(m_lum).tocsc() # diagonal matrix with the lumped mass matrix
            self.M = M_lum #use only the lumped version
            M_inv = sp.diags(m_inv).tocsc()
            self.K = L.T@ M_inv@ L # stiffness matrix from laplacian

            # save the results for three dimensions xyz
            #self.M_inv = sp.block_diag((M_inv, M_inv, M_inv))
            #self.m_inv = np.concatenate([m_inv, m_inv, m_inv]) #vector form
            self.m_inv = m_inv
            self.m_inv_mat = self.m_inv[:,np.newaxis] #column matrix
            self.M_lum = sp.block_diag((M_lum, M_lum, M_lum)).tocsc()
            self.m_lum = m_lum[:,np.newaxis] #vector form

            self.m_sqrt = np.concatenate([m_sqrt, m_sqrt, m_sqrt]) #3-vector form
            self.m_sqrt_mat = self.m_sqrt.reshape((-1,),order = 'F') #column matrix
            self.m_sqrt_inv_mat = 1./self.m_sqrt_mat

            #gravity
            Fg = sp.lil_matrix((self.n_verts,3)); Fg[:,2] = -m_lum
            self.Fg = Fg.tocsc()

            #floor constraints
            self.If = self.nodes
            self.Jf = self.nodes + 2*self.n_verts
            self.Kf = np.ones_like(self.If)

    def precomputeMatrix(self,faces):
        M = sp.lil_array(np.zeros((self.n_verts, self.n_verts)))
        L = sp.lil_array(np.zeros((self.n_verts, self.n_verts)))
        
        mat1 = [np.kron(self.reference_element.N[j:j+1].T, self.reference_element.N[j:j+1]) for j in range(faces.shape[1])]

        for i in range(faces.shape[0]):
            X_i = np.block([[self.positions[node]] for node in faces[i]])
            Me = np.zeros((faces.shape[1], faces.shape[1]))
            Le = np.zeros((faces.shape[1], faces.shape[1]))

            for j in range(faces.shape[1]):
                phi_xi, phi_eta = self.reference_element.Nxi[j] @ X_i, self.reference_element.Neta[j] @ X_i
                dphi = np.block([[phi_xi], [phi_eta]])
                E,F,G = phi_xi @ phi_xi.T, phi_xi @ phi_eta.T, phi_eta @ phi_eta.T
                m = np.block([[E, F], [F, G]])
                dS = np.sqrt(abs(E*G - F**2)) * self.reference_element.w[j]
                Nxyz_k = dphi.T @ np.linalg.solve(m, np.block([[self.reference_element.Nxi[j]], [self.reference_element.Neta[j]]]))
                Me += mat1[j]*dS
                Le += (Nxyz_k[0:1].T @ Nxyz_k[0:1] + Nxyz_k[1:2].T @ Nxyz_k[1:2] + Nxyz_k[2:3].T @ Nxyz_k[2:3])*dS
                
            for j in range(faces.shape[1]):
                for k in range(faces.shape[1]):
                    M[faces[i, j], faces[i, k]] += Me[j, k]
                    L[faces[i, j], faces[i, k]] += Le[j, k]
        return M.tocsc(), L.tocsc()
    
    
    def computeStretchShear(self):
        neighs_xi = {i: set() for i in range(self.n_verts)}
        neighs_eta = {i: set() for i in range(self.n_verts)}

        for face in self.faces:
            #direction xi
            neighs_xi[face[0]].add(face[1])
            neighs_xi[face[1]].add(face[0])
            neighs_xi[face[2]].add(face[3])
            neighs_xi[face[3]].add(face[2])
            #direction eta
            neighs_eta[face[0]].add(face[3])
            neighs_eta[face[3]].add(face[0])
            neighs_eta[face[1]].add(face[2])
            neighs_eta[face[2]].add(face[1])

        neighs_shear = []
        corners_shear = []
        for n in range(self.n_verts):
            if len(neighs_xi[n]) == 2 and len(neighs_eta[n]) == 2:       
                neighs_shear.append(list(neighs_xi[n]) + list(neighs_eta[n]))
            elif len(neighs_xi[n]) == 2 and len(neighs_eta[n]) == 1:
                neighs_shear.append([n] + list(neighs_eta[n]) + list(neighs_xi[n]))
            elif len(neighs_xi[n]) == 1 and len(neighs_eta[n]) == 2:
                neighs_shear.append([n] + list(neighs_xi[n]) + list(neighs_eta[n]))
            elif len(neighs_xi[n]) == 1 and len(neighs_eta[n]) == 1:
                corners_shear.append([n] + list(neighs_xi[n]) + [n] + list(neighs_eta[n]))

        bars = np.vstack([self.faces[:,[0,1]],self.faces[:,[1,2]],
                          self.faces[:,[2,3]],self.faces[:,[3,0]]])
        bars = np.unique(np.sort(bars, axis = 1),axis=0)

        #remove constraints from the seams
        shear_neighs = np.array(neighs_shear)
        shear_corners = np.array(corners_shear)
        if shear_corners.shape[0] == 0:
           shear_corners = np.zeros((0,4),dtype=int)

        #inititate the class    
        self.stretch = self.Stretch(bars, self.positions, self.n_verts, self.m_sqrt)
        self.shear = self.Shear(shear_neighs, shear_corners, self.positions, self.n_verts, self.m_sqrt)

    class Stretch:
        def __init__(self, bars, X, n_verts, m_sqrt):
            self.n_verts = n_verts
            self.bars = bars; 
            self.bars1 = self.bars[:,1]
            self.bars0 = self.bars[:,0]
            self.n_conds = bars.shape[0]
            self.I = np.tile(np.arange(self.n_conds), 6)
            v1 = bars[:, 0]; v2 = bars[:, 1]
            self.J = np.concatenate([v1,v1 + n_verts, v1 + 2 * n_verts,
                                     v2,v2 + n_verts, v2 + 2 * n_verts])
            #for the control u
            self.II = self.I.copy()
            self.JJ = self.J.copy()
            self.Ku = []
            #initial condition
            self.val0 = np.zeros((self.n_conds,))
            self.grad = sp.csc_matrix((np.arange(len(self.II)), (self.II, self.JJ)), 
                                       shape=(self.n_conds, 3*self.n_verts))
            self.gradT = sp.csr_matrix((np.arange(len(self.II)), (self.JJ, self.II)), 
                                       shape=(3*self.n_verts,self.n_conds))
            self.order = self.grad.data.astype(np.int64)
            self.orderT = self.gradT.data.astype(np.int64)
            self.m_sqrt = m_sqrt
            self.m_sqrt_JJ = self.m_sqrt[self.JJ]
            self.val0 = self.evaluate(X,np.zeros((0,)),[])
            self.abs_val0 = np.abs(self.val0)
            self.factor = None

        def update_u(self,I,J,K):
            self.Ku = K    
            if len(I) > 0:
               self.II = np.concatenate([self.I,I+self.n_conds])
               self.JJ = np.concatenate([self.J,J])  
            else:
               self.II = self.I.copy()  
               self.JJ = self.J.copy() 
            self.m_sqrt_JJ = self.m_sqrt[self.JJ]
            self.grad = sp.csc_matrix((np.arange(len(self.II)), (self.II, self.JJ)), 
                                       shape=(self.n_conds+len(I), 3*self.n_verts))
            self.order = self.grad.data.astype(np.int64)
            self.gradT = sp.csr_matrix((np.arange(len(self.II)), (self.JJ, self.II)), 
                                       shape=(3*self.n_verts,self.n_conds+len(I)))
            self.orderT = self.gradT.data.astype(np.int64)

        @profile    
        def evaluate(self,phi,u,control,grad=True):
            phi_mat = phi.reshape((self.n_verts, 3), order='F')
            vec = phi_mat[self.bars1,:] - phi_mat[self.bars0,:]; 
            longs = np.einsum('ij,ij->i', vec, vec); 
            val_str = longs - self.val0
            if grad:
                grad1 = 2*(vec).flatten(order='F')
                grad0 = - grad1
                K = np.concatenate([grad0,grad1,self.Ku])*self.m_sqrt_JJ + 1e-16
                self.grad.data = K[self.order]
                self.gradT.data = K[self.orderT]
                val_u = phi_mat[control,:].flatten(order='F') - u
                val = np.concatenate([val_str,val_u])
            else:
                val_u = phi_mat[control,:].flatten(order='F') - u
                val = np.concatenate([val_str,val_u])
            return val

    class Shear:
        def __init__(self, shear_neighs, shear_corners, X, n_verts, m_sqrt):
            self.n_verts = n_verts
            self.n_crn = shear_corners.shape[0]
            self.n_conds = shear_neighs.shape[0] + shear_corners.shape[0]
            In = np.tile(np.arange(shear_neighs.shape[0]), 12)
            v1 = shear_neighs[:, 0]; v2 = shear_neighs[:, 1]
            v3 = shear_neighs[:, 2]; v4 = shear_neighs[:, 3]
            Jn = np.concatenate([ v1,v1 + n_verts, v1 + 2 * n_verts,
                                v2,v2 + n_verts, v2 + 2 * n_verts,
                                v3,v3 + n_verts, v3 + 2 * n_verts,
                                v4,v4 + n_verts, v4 + 2 * n_verts])
            if self.n_crn > 0:
                Ic = np.tile(np.arange(self.n_crn), 9) + shear_neighs.shape[0]
                self.I = np.concatenate([In,Ic])
                w1 = shear_corners[:, 0] #repeated indices
                w2 = shear_corners[:, 1]; w3 = shear_corners[:, 3]
                Jc = np.concatenate([ w1,w1 + n_verts, w1 + 2 * n_verts,
                                      w2,w2 + n_verts, w2 + 2 * n_verts,
                                      w3,w3 + n_verts, w3 + 2 * n_verts])
                self.J = np.concatenate([Jn,Jc])
            else:
                self.I = In; self.J = Jn
            self.neighs = np.vstack([shear_neighs,shear_corners])
            self.neighs0 = self.neighs[:,0]
            self.neighs1 = self.neighs[:,1]
            self.neighs2 = self.neighs[:,2]
            self.neighs3 = self.neighs[:,3]

            #for the control u
            self.II = self.I.copy()
            self.JJ = self.J.copy()
            self.Ku = []
            #initial condition
            self.val0 = np.zeros((self.n_conds,))
            self.grad = sp.csc_matrix((np.arange(len(self.II)), (self.II, self.JJ)), 
                                      shape=(self.n_conds, 3*self.n_verts))
            self.order = self.grad.data.astype(np.int64)
            self.gradT = sp.csr_matrix((np.arange(len(self.II)), (self.JJ, self.II)), 
                                      shape=(3*self.n_verts,self.n_conds))
            self.orderT = self.gradT.data.astype(np.int64)
            self.m_sqrt = m_sqrt
            self.m_sqrt_JJ = self.m_sqrt[self.JJ]
            self.val0 = self.evaluate(X,np.zeros((0,)),[])
            self.abs_val0 = np.abs(self.val0)
            self.factor = None

        def update_u(self,I,J,K):
            self.Ku = K    
            if len(I) > 0:
               self.II = np.concatenate([self.I,I+self.n_conds])
               self.JJ = np.concatenate([self.J,J])  
            else:
               self.II = self.I.copy()  
               self.JJ = self.J.copy()         
            self.m_sqrt_JJ = self.m_sqrt[self.JJ]  
            self.grad = sp.csc_matrix((np.arange(len(self.II)), (self.II, self.JJ)), 
                                      shape=(self.n_conds+len(I), 3*self.n_verts))
            self.order = self.grad.data.astype(np.int64)
            self.gradT = sp.csr_matrix((np.arange(len(self.II)), (self.JJ, self.II)), 
                                      shape=(3*self.n_verts,self.n_conds+len(I)))
            self.orderT = self.gradT.data.astype(np.int64)

        @profile
        def evaluate(self,phi,u,control,grad=True):
            phi_mat = phi.reshape((self.n_verts, 3), order='F')
            vec1 = phi_mat[self.neighs1,:] - phi_mat[self.neighs0,:]; 
            vec2 = phi_mat[self.neighs3,:] - phi_mat[self.neighs2,:]; 
            dots = np.einsum('ij,ij->i', vec1, vec2)
            val_shr = dots - self.val0
            if grad:
                if self.n_crn > 0:
                    _grad1 = vec2[-self.n_crn:].flatten(order='F')
                    _grad2 = vec1[-self.n_crn:].flatten(order='F')
                    _grad0 = -_grad1 -_grad2
                    #all the grads minus the corners
                    grad1 = vec2[:-self.n_crn].flatten(order='F')
                    grad0 = -grad1
                    grad3 = vec1[:-self.n_crn].flatten(order='F')
                    grad2 = -grad3
                else:
                    grad1 = vec2.flatten(order='F')
                    grad0 = -grad1
                    grad3 = vec1.flatten(order='F')
                    grad2 = -grad3
                    _grad0 = []; _grad1 = []; _grad2 = []

                K = np.concatenate([grad0,grad1,grad2,grad3,
                                   _grad0,_grad1,_grad2,self.Ku])*self.m_sqrt_JJ + 1e-16  
                self.grad.data = K[self.order]
                self.gradT.data = K[self.orderT]
                val_u = phi_mat[control,:].flatten(order='F') - u
                val = np.concatenate([val_shr,val_u])
            else:
                val_u = phi_mat[control,:].flatten(order='F') - u
                val = np.concatenate([val_shr,val_u])
            return val
        
    def estimateTimeStep(self,L=1):
        h = np.sqrt(np.mean(self.stretch.abs_val0))
        dt = (1/3)*(h/np.sqrt(2*9.81*L))
        print("Based on your mesh, your best dt is:",dt)
        return dt
        
    def preparePolyscope(self):
        self.polyscoped = True
        ps.init()
        ps.remove_all_structures()
        ps.register_surface_mesh(self.label, self.Am@self.positions, self.triangles, smooth_shade=True, transparency=0.9, edge_width = 0)
        ps.register_point_cloud(self.label, self.positions, enabled = False)
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("tile_reflection")  # set +Z as up direction
        ps.set_ground_plane_height(-0.005) # adjust the plane height

    
    def plotMesh(self):    
        if self.polyscoped is False:
            self.preparePolyscope()
        """Plot the current mesh"""
        ps.get_surface_mesh(self.label).update_vertex_positions(self.Am@self.positions)
        ps.get_point_cloud(self.label).update_point_positions(self.positions)
        if self.rad is not None:
           ps.get_point_cloud(self.label).set_radius(rad=self.rad,relative=False)
        ps.show()

    def makeMovie(self, speed = 1, repeat = True, smooth = 0):
        if self.polyscoped is False:
            self.preparePolyscope()
        self.ps_frame = 0
        skip = speed
        ps.get_point_cloud(self.label).set_radius(rad=self.rad,relative=False)

        def goThroughHistory():
            # Update Polyscope visualization
            phi_mat = self.history_pos[self.ps_frame]
            phi_all = self.Am@phi_mat
            for _ in range(smooth):
                phi_all = self.S@phi_all
            ps.get_surface_mesh(self.label).update_vertex_positions(phi_all)
            ps.get_point_cloud(self.label).update_point_positions(phi_mat)

            # Advance simulation time by skipping frames accordingly
            self.ps_frame += skip
            if self.ps_frame >= len(self.history_pos):
                if repeat:
                   self.ps_frame = 0  # Loop back to start
                else:
                   #display last frame before stopping
                   phi_mat = self.history_pos[-1]
                   phi_all = self.Am@phi_mat
                   for _ in range(smooth):
                       phi_all = self.S@phi_all
                   ps.get_surface_mesh(self.label).update_vertex_positions(phi_all)
                   ps.get_point_cloud(self.label).update_point_positions(phi_mat)
                   ps.clear_user_callback()

        ps.set_user_callback(goThroughHistory)
        ps.show()
        ps.clear_user_callback()


    def saveFrames(self, width = 800, height = 600, speed = 1, smooth=2):
        if not self.polyscoped:
            self.preparePolyscope()

        os.makedirs("frames", exist_ok=True)
        ps.set_screenshot_extension(".png")
        ps.set_automatically_compute_scene_extents(True)
        ps.set_window_size(width,height)

        mesh = ps.get_surface_mesh(self.label)
        pc = ps.get_point_cloud(self.label)
        pc.set_radius(rad=self.rad,relative=False)

        for i, phi_mat in enumerate(self.history_pos[::speed]):
            # Compute smoothed positions
            phi_all = self.Am @ phi_mat
            for _ in range(smooth):
                phi_all = self.S @ phi_all

            # Update Polyscope geometry
            mesh.update_vertex_positions(phi_all)
            pc.update_point_positions(phi_mat)

            # Save frame
            ps.screenshot(f"frames/frame_{i:03d}.png", transparent_bg=False)
            print("Frame saved:", i)


    def computeRadiouses(self):
        #lenght of edges of the quad mesh
        e0 = self.edges_matrix[:,0]; e1 = self.edges_matrix[:,1]
        longs = self.computeNorm(self.positions[e1]-self.positions[e0])
        min_l = np.min(longs); max_l = np.max(longs)
        diff_rel = np.round(100*(max_l - min_l)/min_l,3)
        assert diff_rel <= 50, f"Relative difference between smallest and biggest edge is '{diff_rel}'% more than 50%, please re-define mesh"
        #take into account diagonals
        d0 = self.faces[:,0]; d1 = self.faces[:,1]; d2 = self.faces[:,2]; d3 = self.faces[:,3]; 
        diag0 = self.computeNorm(self.positions[d0]-self.positions[d2])
        diag1 = self.computeNorm(self.positions[d1]-self.positions[d3])
        #constant radious of the balls
        self.rad = self.thck*np.mean(longs)/2.05

        #matrix of radiouses
        matrix_rads = 2*self.rad*np.ones((self.n_verts,self.n_verts),dtype=float)
        #reduce in case it is too big
        sum_rads = np.minimum(2*self.rad,0.976*longs)
        matrix_rads[e0,e1] = sum_rads; matrix_rads[e1,e0] = sum_rads   
        #do the same for the diagonals
        sum_rads0 = np.minimum(2*self.rad,0.976*diag0)
        sum_rads1 = np.minimum(2*self.rad,0.976*diag1)
        matrix_rads[d0,d2] = sum_rads0; 
        matrix_rads[d1,d3] = sum_rads1
        #save matrix for fast indixing
        self.matrix_rads = matrix_rads

 
    def setSimulatorParameters(self, dt = 0.0025, tol = 0.0075, 
                               rho = 0.1, delta = 0.1, alpha = 0.2,
                               kappa = 0.5*1e-4, str = 0.01*1e-4, shr = 10*1e-4,
                               mu_f = 0.2, mu_s = 0.35, thck = 0.95):
        #solver parameters
        self.dt = dt #time step
        self.tol = tol #tolerance for constraints
        self.implicitEuler = False

        #physical parameters
        self.g = 9.8 #gravity acceleration in m/s**2
        self.rho = rho # density of the cloth
        self.delta = delta # virtual mass 
        self.alpha = alpha # slow damping 
        self.kappa = kappa # bending stiffness
        self.beta = 0.015*self.kappa # fast damping: do not change in general
        self.str = str/(dt**2) # stretch elasticity
        self.shr = shr/(dt**2) # shear elasticity
        self.mu_floor = mu_f #friction with to the floor
        self.mu_self = mu_s #friction for self-collisions

        #self-collision parameters
        self.thck = thck
        self.mov_tol = 0.02 #when some node moves 2% or more than its previous position, run computeClosePairs()
        self.computeRadiouses()
        self.eps_sus = 3.3*self.rad #threshold for detecting close balls in computeClosePairs()


        #factorize implicit step matrix E for fast unconstrained step
        D = self.alpha*self.M + self.beta*self.K 
        K = self.kappa*self.K; M = self.rho*self.M; 
        E = M + self.dt*D + (self.dt**2)*K 
        Et = M + 0.5*self.dt*D + 0.25*(self.dt**2)*K 

        #save the matrices
        self.factor_E = cholesky(E)
        self.factor_Et = cholesky(Et)
        self.D = D

        #precompute for fast unconstrained step matrix operations
        self.rho_M = M      
        dt_rho_M = (self.dt*self.rho_M).diagonal()
        self.dt_rho_M = dt_rho_M[:, np.newaxis]
        self.dt2_delta_Fg = (dt**2)*self.delta*self.g*self.Fg
        self.half_dt2_delta_Fg = 0.5*(dt**2)*self.delta*self.g*self.Fg  

        #aerodynamics    
        self.half_dt2_Fg = 0.5*(dt**2)*self.g*self.Fg
        self.F_z = self.half_dt2_Fg[:,2].toarray().flatten(order='F')
        self.rho_M_plus_dt_D = (self.rho_M + self.dt*self.D).tocsr()
        self.E_aux = (self.rho_M + 0.5*self.dt*self.D - 0.25*(self.dt**2)*K).tocsr()

    def unionMask(self,a,b):
        self.mask_col[:] = False         # reset without reallocating
        self.mask_col[a] = True
        self.mask_col[b] = True
        idx = np.nonzero(self.mask_col)[0] 
        return idx
    
    def innerProduct(self,u,v):
        return np.einsum('ij,ij->i',u,v)   
                              
    def normalize(self,w):
        norm_w = self.computeNorm(w) + 1e-12
        return w/norm_w[:,np.newaxis]
    
    def computeNorm(self,w):
        return np.sqrt(self.innerProduct(w,w))

    @profile
    def floorCollisions(self,phi):
        phi_mat = phi.reshape((self.n_verts, 3), order='F').copy()
        ind_col = np.nonzero(phi_mat[:,2] < 0)[0]
        self.flr = False #bookeeping if floor collisions occurred
        if ind_col.shape[0] > 0:
            self.flr = True
            #normal forces
            norm_Fn = self.nodes_faces_count[ind_col]*np.abs(phi_mat[ind_col,2]) #normal force 
            phi_mat[ind_col,2] = 0 #orthogonal projection to the floor          
            #friction
            vt = (self.positions[ind_col] - phi_mat[ind_col]) #tangent friction direction per node 
            vt[:,2] = 0; #project on the floor   
            #spread the forces         
            F_mu = self.frictionForce(self.mu_floor,norm_Fn,vt,cap=True)  
            phi_mat[ind_col] += F_mu  
            phi = phi_mat.flatten(order='F') #update positions
        return phi  
    
    def frictionForce(self,mu,Fn,vt,cap = True):
        norm_vt = np.sqrt(self.innerProduct(vt,vt)) 
        quotient = (mu*Fn)/(norm_vt + 1e-12)
        if cap:
           k = np.minimum(1,quotient) #cannot move more than where CCD computed the intersection
        else:
           k = quotient
        return k[:,np.newaxis]*vt
    
    @profile
    def computeFrictionCorrection(self,phi,dlt_phi):
        phi_mat = phi.reshape((self.n_verts, 3), order='F') 
        #friction: compute tangent direction
        nu_mat = dlt_phi.reshape((self.n_verts, 3), order='F')
        norm_Fn = self.computeNorm(nu_mat)
        nu = nu_mat/(norm_Fn[:,np.newaxis] + 1e-12)
        v = self.positions - phi_mat
        vt = v - (self.innerProduct(v,nu)[:,np.newaxis])*nu 
        #compute friction force vector
        F_mu = self.frictionForce(self.mu_self,norm_Fn,vt,cap = True)
        return F_mu.flatten(order='F')

    
    @profile
    def updateSelfCollisions(self,phi): 
        phi_mat = phi.reshape((self.n_verts, 3), order='F') 
        #simplified CCD for the balls
        xy = phi_mat[self.near_nn1] - phi_mat[self.near_nn0]
        #normal
        norm_xy = self.computeNorm(xy)
        normal_all = xy / norm_xy[:,np.newaxis]
        #orient normal
        res0 = self.innerProduct(self.xy0,normal_all); flip = (res0 < 0); 
        normal_all[flip] = -normal_all[flip]; norm_xy[flip] = -norm_xy[flip]             
        #evaluate the constraints
        self.vals_slf = norm_xy - self.rads
        self.normals_slf = normal_all 
        if self.vals_slf.shape[0] > 0:
           self.error_slf = np.min(self.vals_slf/self.rads)
        else:
           self.error_slf = 1

    @profile
    def solveLCP(self, max_iter = 50):
        #objects to compute only once
        normals = self.normals_slf[self.ind_slf]
        b0_col = self.near_nn0[self.ind_slf]; b1_col = self.near_nn1[self.ind_slf]
        b_col = np.concatenate([b1_col,b0_col])
        #counts to take average impulses
        count0 = np.bincount(b0_col, minlength=self.n_verts)
        count1 = np.bincount(b1_col, minlength=self.n_verts)
        count = count0 + count1; 
        #averages
        avg = 1/(count + 1e-12); avg[count == 0] = 0; 
        #mass inverses: set controled to zero
        w = self.m_inv.copy(); w[self.control] = 0
        wa  = (avg*w)[:,np.newaxis]      
        rads = self.rads[self.ind_slf]       

        #initial impulses
        num = -self.vals_slf[self.ind_slf]; 
        den = w[b0_col] + w[b1_col] 
        landa = np.maximum(0,num/den)
        #corrections
        dlt = landa[:,np.newaxis]*normals
        dlt2 = np.concatenate([+dlt,-dlt], axis = 0)
        #global correction
        dlt_tot = np.zeros((self.n_verts,3))
        np.add.at(dlt_tot,b_col,dlt2); 
        dlt_phi = wa*dlt_tot

        #iterative process
        error_l = -1; ii = 0
        while error_l < -self.tol and ii < max_iter:  
            dlt_xy = dlt_phi[b1_col] - dlt_phi[b0_col]
            dlt_vals = -self.innerProduct(normals,dlt_xy)
            #compute multipliers
            res = (num + dlt_vals)
            error_l = np.min(-res/rads)
            landa = np.maximum(0, landa + res/den)
            #corrections
            dlt = landa[:,np.newaxis]*normals
            dlt2 = np.concatenate([+dlt,-dlt], axis = 0)
            #global correction
            dlt_tot.fill(0.0)
            np.add.at(dlt_tot,b_col,dlt2); 
            dlt_phi = wa*dlt_tot
            ii += 1
        #print(ii)
        return dlt_phi.flatten(order='F')


    @profile
    def selfCollisions(self,phi,n_iter,max_iters=50):    
        if n_iter == 0:
            #precompute objects for selfcollisions
            self.prepareCollisions(phi)        
        #check for possible selfcollisions
        self.updateSelfCollisions(phi)

        if self.error_slf < -self.tol: #correct detected self-collisions
            #add new and previous selfcollisions
            ind_s = np.nonzero((self.vals_slf/self.rads) < self.tol)[0]
            self.ind_slf = self.unionMask(self.ind_slf,ind_s)
            #correct positions
            dlt_phi = self.solveLCP(max_iters)
            phi += dlt_phi
            #apply friction if needed
            if self.mu_self > 0:
                F_mu = self.computeFrictionCorrection(phi,dlt_phi)
                phi += F_mu
            #check for possible new selfcollisions
            #self.updateSelfCollisions(phi)
            
        return phi
    
    def buildShareEdgeMatrix(self):
        share_edge = np.zeros((self.n_verts, self.n_verts), dtype=bool)
        for edge in self.edges_matrix:  # each edge has 2 nodes
            a,b = edge
            share_edge[a, b] = True
            share_edge[b, a] = True
        self.share_edge = share_edge
    
    @profile
    def computeClosePairs(self,phi_mat):
        #build the tree only for the nodes
        tree_n = KDTree(phi_mat)

        #node-node close pairs
        dists, neighs = tree_n.query(phi_mat, k=self.kn+1) #query it for k nodes neighbors
        #reshape removing the first pair
        dist = dists[:,1:].reshape(-1)
        nj = neighs[:,1:].reshape(-1)
        #remove far away pairs and duplicates
        mask = (dist < self.eps_sus) & (self.ni < nj)
        ni = self.ni[mask]; nj = nj[mask]
        #second mask
        mask2 = ~self.share_edge[ni,nj]
        ni = ni[mask2]; nj = nj[mask2]
        #set radiouses to avoid jitering when the balls are too big
        self.rads = self.matrix_rads[ni,nj]
        #potential colliding nodes-nodes
        self.near_nn0 = ni; self.near_nn1 = nj

        #mask for indices
        self.mask_col = np.zeros(self.near_nn0.shape[0], dtype=bool)

    @profile
    def updateClosePairs(self,phi_mat):
        #check close pairs
        diff = phi_mat - self.last_check
        mov = np.sqrt(np.max(self.innerProduct(diff, diff)/self.den_last))
        if (mov > self.mov_tol) or (self.total_iters == 0): #only check when at least 1 node has moved more than mov_eps
            self.computeClosePairs(phi_mat) #update close pairs
            self.last_check = phi_mat #update last checked mesh
            self.den_last = self.innerProduct(self.last_check,self.last_check)
            #print("Close Nodes-Nodes")
            #print(np.vstack([self.near_nn0,self.near_nn1]).T)
    
    @profile
    def prepareCollisions(self,phi):
        phi_mat = phi.reshape((self.n_verts, 3), order='F') 
        self.updateClosePairs(phi_mat)
        #do costly indexing operations only once
        self.xy0 = self.positions[self.near_nn1] - self.positions[self.near_nn0]
        #store past collisions
        self.ind_slf = self.empty
        #store if floor collisions have happened
        self.flr = True
    
    def projectControl(self,phi,u_mat,control,n_ctr):
        if n_ctr > 0:
            phi_mat = phi.reshape((self.n_verts, 3), order='F')
            phi_mat[control] = u_mat
            phi = phi_mat.reshape((self.n_verts*3, ), order='F')
        return phi 

    @profile
    def projectConstraints(self,constraints,phi,u,control,landa,par,update_chol,den_error,n):
        #evaluate constraints
        if n == 0:
            val = constraints.evaluate(phi,u,control,grad=True)
            if update_chol or constraints.factor is None:
               constraints.factor = cholesky_AAt(constraints.grad, beta = par) 
            else:
               constraints.factor.cholesky_AAt_inplace(constraints.grad, beta = par)
        else:
            val = constraints.evaluate(phi,u,control,grad=False)
        b = - val - par*landa 
        #solve
        dlt_lambda = constraints.factor(b)
        #update
        landa += dlt_lambda
        phi += self.m_sqrt_mat*(constraints.gradT@dlt_lambda)

        #check errors 
        val = constraints.evaluate(phi,u,control,grad=False)
        if u.shape[0] > 0:
           aux_error = (val[:-u.shape[0]] + par*landa[:-u.shape[0]])/(constraints.abs_val0 + den_error)
        else:
           aux_error = (val + par*landa)/(constraints.abs_val0 + den_error)
        error = np.linalg.norm(aux_error,ord=np.inf) 

        return phi, landa, error
    
    def ImplicitEuler(self):
        q = self.dt2_delta_Fg + (self.dt_rho_M * self.velocities) + (self.rho_M_plus_dt_D @ self.positions)
        #solve the sistem with the cholesky factor
        x = self.factor_E(q)
        return x.reshape((3*self.n_verts,),order='F')

    @profile
    def TrapezoidalRule(self):
        q = self.half_dt2_delta_Fg + (self.dt_rho_M * self.velocities) + (self.E_aux @ self.positions)
        #solve the sistem with the cholesky factor
        x = self.factor_Et(q)
        return x.reshape((3*self.n_verts,),order='F')

    def unconstrainedStep(self, implicitEuler):
        if implicitEuler:
            return self.ImplicitEuler()
        return self.TrapezoidalRule()
    
    def processControlInputs(self,u,control):
        n_ctr = len(control)
        if n_ctr > 0:
           u[:,2] = np.maximum(0,u[:,2])
           u = u.reshape((3*n_ctr,),order='F')
           u_mat = u.reshape((n_ctr,3),order='F')
        else:
           u = np.zeros((0,))
           u_mat = np.zeros((0,3))
        #check if we need to update cholesky decomp. of constraints
        update_chol = False
        if self.control != control:
            #update internal variables
            self.control = control
            update_chol = True
            if n_ctr > 0:
                Iu = np.arange(3*n_ctr)
                Ju = np.concatenate((control, [x+self.n_verts for x in control], [x+2*self.n_verts for x in control]))
                Ku = np.ones_like(Iu)            
            else:
                Iu = self.empty; Ju = self.empty; Ku = self.empty
            self.shear.update_u(Iu,Ju,Ku)
            self.stretch.update_u(Iu,Ju,Ku)
        return u, u_mat, n_ctr, update_chol


    @profile
    def simulate(self, u, control):
        #current position of the cloth for self-collisions
        phi0 = self.positions.reshape((3*self.n_verts,),order = 'F')

        #process the control inputs
        u, u_mat, n_ctr, update_chol = self.processControlInputs(u,control)

        #unconstrained step to correct
        phi = self.unconstrainedStep(self.implicitEuler)

        #lagrange multipliers for the shear and stretch constraints
        lambda_shr = np.zeros((self.shear.n_conds + u.shape[0],)); 
        lambda_str = np.zeros((self.stretch.n_conds + u.shape[0],)); 

        #solver variables for inextensiblity 
        n_iter = 0; error_str = np.inf; error_shr = np.inf; self.error_slf = -np.inf

        while error_str > self.tol or error_shr > self.tol: #or self.error_slf < -self.tol:

            #shearing
            phi, lambda_shr, error_shr = self.projectConstraints(self.shear,phi,u,control,
                                                                lambda_shr,self.shr,
                                                                update_chol,0.01,n_iter%3)

            #stretching
            phi, lambda_str, error_str = self.projectConstraints(self.stretch,phi,u,control,
                                                                lambda_str,self.str,
                                                                update_chol,0,0)    
 
            #control constraints
            phi = self.projectControl(phi,u_mat,control,n_ctr)

            #self-collisions
            phi = self.selfCollisions(phi,n_iter); 

            #iteration count 
            n_iter += 1

        #floor collisions
        phi = self.floorCollisions(phi)

        #update internal cloth variables
        dphi = (phi-phi0)/self.dt
        self.positions = phi.reshape((self.n_verts, 3), order='F')
        self.velocities = dphi.reshape((self.n_verts, 3), order='F')
        self.history_pos.append(self.positions)
        self.history_vel.append(self.velocities)
        self.total_iters += n_iter

        #warnings
        if self.total_iters/(len(self.history_pos)-1) > 4 and self.warning == False:
           print("WARNING: average of more than 4 iterations taken, for better performance reduce dt")
           self.warning = True

