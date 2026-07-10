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
from scipy.spatial import cKDTree

class Cloth:
    def __init__(self,verts,quads,tris, seams=[],name="clothilde"):
        #positions and velocities
        self.positions = np.array(verts, order = 'F') #current position of the vertices of the mesh
        assert self.positions.shape[1] == 3 and self.positions.ndim == 2, 'Something is wrong with the vertices dimensions'
        self.velocities = np.zeros(self.positions.shape, order = 'F') + 1e-12 #current velocities of the vertices of the mesh
        self.history_pos = [self.positions] #history of the vertices of the mesh 
        self.history_vel = [self.velocities] #history of the velocities of the vertices of the mesh 
        #self.positions += 0.0001*np.random.randn(self.positions.shape[0],3) #avoid singular flat case

        #topology of the mesh
        self.quads = np.array(quads)
        self.faces = np.array(tris) #quadrangulation of the vertices in positions (index based)
        assert self.faces.shape[1] == 3 and self.faces.ndim == 2, 'Current implementation only supports quad meshes'
        self.triangles = tris
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

        #seams treatment
        self.seams = np.array(seams)
        self.n_seams = self.seams.shape[0]
        if self.n_seams > 0:
            self.Is = np.concatenate([np.arange(3*self.n_seams), np.arange(3*self.n_seams)])
            self.Js = np.concatenate([self.seams[:,0], self.seams[:,0]+self.n_verts, self.seams[:,0]+2*self.n_verts,
                                      self.seams[:,1], self.seams[:,1]+self.n_verts, self.seams[:,1]+2*self.n_verts])
            self.Ks = np.concatenate([np.ones(3*self.n_seams), -np.ones(3*self.n_seams)])
        else:
            self.seams = np.zeros((0,2),dtype=int)
            self.Is = np.array([],dtype=int); self.Js = np.array([],dtype=int); self.Ks = np.array([],dtype=int); 
        self.seams_IJK = [self.Is,self.Js,self.Ks]

        #for self-collisions
        self.rad = 0.003 #radious of the balls
        self.last_check = np.array(verts, order = 'F') #for checking close self-collision pairs
        self.den_last = 1
        self.ke = 10 #get k nearest nodes to every node
        self.kf = 12 #get k nearest nodes to every node
        self.nodes = np.arange(self.n_verts) #needed for proximity detection
        self.empty = np.array([],dtype=int) #handy sometimes
        self.table = False
        self.vals_ee = [0]

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
        self.share_control = np.zeros((self.n_verts, self.n_verts), dtype=bool)

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
        return f"Cloth({self.n_verts} vertices, {self.faces.shape[0]} tris)"
    
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
        self.buildAdjacencyMatrices()
        self.buildShareNodeMatrix()
        #self.buildShareEdgeMatrix()
        self.computeBoundary()
        self.computeSmoother()
        self.prepareMatrices()
        self.computeStretchShear()
        self.precomputeBoundaryBending()
        #self.assembleQuadFlatnessK()

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
            self.edges_matrix = np.array(list(map(list,self.edges))) #in matrix form, handy for many computations
            self.e0 = self.edges_matrix[:,0]; self.e1 = self.edges_matrix[:,1]
            self.n_edges = len(self.edges)
            self.ei = np.repeat(np.arange(self.n_edges),self.ke)
            self.fi = np.repeat(np.arange(self.n_faces),self.kf)


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
            row = np.array(range(self.n_faces))
            match self.faces.shape[1]:
                case 3:
                    row = np.concatenate((row, row, row))
                    col = np.concatenate((self.faces[:,0],self.faces[:,1],self.faces[:,2]))
                    self.f0 = self.faces[:,0]; self.f1 = self.faces[:,1]; self.f2 = self.faces[:,2]
                case 4:
                    row = np.concatenate((row, row, row, row))
                    col = np.concatenate((self.faces[:,0],self.faces[:,1],self.faces[:,2],self.faces[:,3]))
            data = np.ones_like(row)
            self.A2 = sp.coo_matrix((data, (row, col)), shape=(self.n_faces, self.n_verts)).tocsr()
            self.A2t = self.A2.T.tocsr()
            self.nodes_faces_count = np.array(self.A2.sum(axis=0))[0]
            #self.Am = sp.vstack([sp.eye(self.n_verts),0.25*self.A2]).tocsr() #for plotting

    def computeBoundary(self):
        sumCols = np.array(self.A1.T.sum(axis=1))
        index = np.where(sumCols == 1)[0] #an edge only contained in one face
        edges_bnd = self.edges_matrix[index,:] 
        self.nodes_bnd = np.unique(edges_bnd.reshape(2*edges_bnd.shape[0])) # indices of the nodes of the boundary
        self.edges_bnd = edges_bnd

    def computeSmoother(self):
        #computation of neighbors
        S = sp.lil_matrix((self.n_verts, self.n_verts)); alpha = 0.75
        for n in range(self.n_verts):
            aux = (self.edges_matrix[:,0] == n) + (self.edges_matrix[:,1] == n)
            edges_n = self.edges_matrix[aux == True,:]
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
            self.m_lum = m_lum[:,np.newaxis] #matrix form

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
    

    def precomputeBoundaryBending(self, eps_inv_mass=0.0):
        """
        Boundary-only Laplacian-style bending precompute.

        Builds:
        Mb : (n_verts x n_verts)  1D boundary mass matrix (assembled on boundary edges)
        Lb : (n_verts x n_verts)  1D boundary stiffness / Laplacian matrix (assembled on boundary edges)
        Saves:
        Kb = Lb.T @ Minv_b @ Lb, where Minv_b is a lumped (diagonal) inverse of Mb
            (with zero inverse on inactive vertices; optionally eps regularization in denominator)
        """
        n = self.n_verts
        pos = self.positions
        edges = np.asarray(self.edges_bnd, dtype=int)
        corners = np.asarray(self.corners, dtype=int)

        # --- Assemble Mb, Lb as global (n x n) sparse matrices ---
        Mb = sp.lil_array((n, n))
        Lb = sp.lil_array((n, n))

        for (a, b) in edges:
            xa = pos[a]
            xb = pos[b]
            ell = float(np.linalg.norm(xb - xa))
            if ell < 1e-12:
                continue

            # 1D linear FEM on segment: mass and stiffness
            Me = (ell / 6.0) * np.array([[2.0, 1.0],
                                        [1.0, 2.0]], dtype=float)
            Ke = (1.0 / ell) * np.array([[ 1.0, -1.0],
                                        [-1.0,  1.0]], dtype=float)

            # Assemble 2x2 into global
            idx = (a, b)
            for iL in range(2):
                I = idx[iL]
                for jL in range(2):
                    J = idx[jL]
                    Mb[I, J] += Me[iL, jL]
                    Lb[I, J] += Ke[iL, jL]

        Mb = Mb.tocsc()

        # --- Lumped inverse mass (safe with skipped corners / isolated verts) ---
        d = np.asarray(Mb.diagonal()).ravel()
        invd = np.zeros_like(d)
        mask = d > 0.0
        invd[mask] = 1.0 / (d[mask] + float(eps_inv_mass))
        Minv_b = sp.diags(invd, format="csc")
        Lb[corners,:] = 0
        Lb = Lb.tocsc()

        # Boundary bending stiffness/operator in same style as interior: Kb = Lb^T Minv Lb
        self.Kb = (Lb.T @ Minv_b @ Lb).tocsc()


    def assembleQuadFlatnessK(self, k_flat=1.0, use_mass_weight=True):
        """
        Assemble sparse stiffness matrix K for quad flatness energy:

            E = 1/2 sum_q w_q || x00 - x10 - x01 + x11 ||^2

        Assumes self.faces contains quads ordered as:

            [v00, v10, v11, v01]

        Then the stencil in face ordering is:

            [+1, -1, +1, -1]

        Parameters
        ----------
        k_flat : float
            Global flatness stiffness multiplier.

        use_mass_weight : bool
            If True, use a mass-normalized weight similar in spirit to L^T M^{-1} L.
            If False, every quad receives weight k_flat.

        Returns
        -------
        K : scipy.sparse.csc_matrix, shape (n_verts, n_verts)
            Scalar stiffness matrix. For positions x of shape (n_verts, 3),
            use:

                f_flat = -K @ x

        """

        faces = np.asarray(self.faces, dtype=np.int64)
        n_verts = self.n_verts

        assert faces.ndim == 2 and faces.shape[1] == 4

        # Face order: [v00, v10, v11, v01]
        signs = np.array([1.0, -1.0, 1.0, -1.0])

        rows = []
        cols = []
        data = []

        m_lum = np.asarray(self.m_lum, dtype=float)

        for f in faces:
            if use_mass_weight:
                # Approximate mass associated with this quad.
                # Since m_lum is vertex-lumped, summing the four vertex masses
                # is a reasonable local scale.
                m_q = np.sum(m_lum[f])

                if m_q <= 0.0:
                    continue

                w_q = k_flat / m_q
            else:
                w_q = k_flat

            # Local K_q = w_q * s^T s
            for a in range(4):
                ia = f[a]
                sa = signs[a]

                for b in range(4):
                    ib = f[b]
                    sb = signs[b]

                    rows.append(ia)
                    cols.append(ib)
                    data.append(w_q * sa * sb)

        self.Kflat = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(n_verts, n_verts)
        ).tocsc()    
    
    def computeStretchShear(self):
        neighs_xi = {i: set() for i in range(self.n_verts)}
        neighs_eta = {i: set() for i in range(self.n_verts)}

        for face in self.quads:
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
        self.corners = []
        for n in range(self.n_verts):
            if len(neighs_xi[n]) == 2 and len(neighs_eta[n]) == 2:       
                neighs_shear.append(list(neighs_xi[n]) + list(neighs_eta[n]))
            elif len(neighs_xi[n]) == 2 and len(neighs_eta[n]) == 1:
                neighs_shear.append([n] + list(neighs_eta[n]) + list(neighs_xi[n]))
            elif len(neighs_xi[n]) == 1 and len(neighs_eta[n]) == 2:
                neighs_shear.append([n] + list(neighs_xi[n]) + list(neighs_eta[n]))
            elif len(neighs_xi[n]) == 1 and len(neighs_eta[n]) == 1:
                corners_shear.append([n] + list(neighs_xi[n]) + [n] + list(neighs_eta[n]))
                self.corners.append(n)

        bars = np.vstack([self.quads[:,[0,1]],self.quads[:,[1,2]],
                          self.quads[:,[2,3]],self.quads[:,[3,0]]])
        bars = np.unique(np.sort(bars, axis = 1),axis=0)

        bars_t = np.vstack([self.faces[:,[0,1]],self.faces[:,[1,2]],self.faces[:,[2,0]]])
        bars_t = np.unique(np.sort(bars_t, axis = 1),axis=0)

        # Encode each edge (i, j) as a unique integer key
        keys_bars = bars[:, 0] * self.n_verts + bars[:, 1]
        keys_bars_t = bars_t[:, 0] * self.n_verts + bars_t[:, 1]

        # Keep triangle edges that are not quad edges
        mask_diag = ~np.isin(keys_bars_t, keys_bars)

        diag_bars = bars_t[mask_diag]

        #remove constraints from the seams
        shear_neighs = np.array(neighs_shear)
        shear_corners = np.array(corners_shear)
        if shear_corners.shape[0] == 0:
           shear_corners = np.zeros((0,4),dtype=int)

        #inititate the class    
        self.stretch = self.Stretch(bars, self.positions, self.n_verts, self.m_sqrt, self.seams, self.seams_IJK)
        #self.shear = self.Stretch(diag_bars, self.positions, self.n_verts, self.m_sqrt, self.seams, self.seams_IJK)
        self.shear = self.Shear(shear_neighs, shear_corners, self.positions, self.n_verts, self.m_sqrt, self.seams, self.seams_IJK)

    class Stretch:
        def __init__(self, bars, X, n_verts, m_sqrt, seams, IJKs):
            self.n_verts = n_verts
            self.bars = bars; 
            self.bars1 = self.bars[:,1]
            self.bars0 = self.bars[:,0]
            self.n_conds = bars.shape[0]
            self.I = np.tile(np.arange(self.n_conds), 6)
            v1 = bars[:, 0]; v2 = bars[:, 1]
            self.J = np.concatenate([v1,v1 + n_verts, v1 + 2 * n_verts,
                                     v2,v2 + n_verts, v2 + 2 * n_verts])
            #seams
            self.seams = seams
            self.n_seams = seams.shape[0]
            self.Is = IJKs[0]
            self.Js = IJKs[1]
            self.Ks = IJKs[2]

            #for the control u
            self.II = np.concatenate([self.I,self.Is+self.n_conds])
            self.JJ = np.concatenate([self.J,self.Js])
            self.Ku = []
            #initial condition
            self.val0 = np.zeros((self.n_conds,))
            self.grad = sp.csc_matrix((np.arange(self.II.shape[0]), (self.II, self.JJ)), 
                                       shape=(self.n_conds + 3*self.n_seams, 3*self.n_verts))
            self.gradT = sp.csr_matrix((np.arange(self.II.shape[0]), (self.JJ, self.II)), 
                                       shape=(3*self.n_verts,self.n_conds + 3*self.n_seams))
            self.order = self.grad.data.astype(np.int64)
            self.orderT = self.gradT.data.astype(np.int64)
            self.m_sqrt = m_sqrt
            self.m_sqrt_JJ = self.m_sqrt[self.JJ]
            self.val0 = self.evaluate(X,np.zeros((0,)),[])[:self.n_conds]
            self.abs_val0 = np.abs(self.val0)
            self.factor = None

        def update_u(self,I,J,K):
            self.Ku = K    
            if len(I) > 0:
               self.II = np.concatenate([self.I,self.Is+self.n_conds,I + self.n_conds + 3*self.n_seams])
               self.JJ = np.concatenate([self.J,self.Js,J])  
            else:
               self.II = np.concatenate([self.I,self.Is+self.n_conds])
               self.JJ = np.concatenate([self.J,self.Js])
            self.m_sqrt_JJ = self.m_sqrt[self.JJ]
            self.grad = sp.csc_matrix((np.arange(len(self.II)), (self.II, self.JJ)), 
                                       shape=(self.n_conds+len(I)+3*self.n_seams, 3*self.n_verts))
            self.order = self.grad.data.astype(np.int64)
            self.gradT = sp.csr_matrix((np.arange(len(self.II)), (self.JJ, self.II)), 
                                       shape=(3*self.n_verts,self.n_conds+len(I)+3*self.n_seams))
            self.orderT = self.gradT.data.astype(np.int64)

        def evaluate(self,phi,u,control,grad=True):
            phi_mat = phi.reshape((self.n_verts, 3), order='F')
            vec = phi_mat[self.bars1,:] - phi_mat[self.bars0,:]; 
            longs = np.einsum('ij,ij->i', vec, vec); 
            val_str = longs - self.val0
            if grad:
                grad1 = 2*(vec).flatten(order='F')
                grad0 = - grad1
                K = np.concatenate([grad0,grad1,self.Ks,self.Ku])*self.m_sqrt_JJ + 1e-16
                self.grad.data = K[self.order]
                self.gradT.data = K[self.orderT]
            val_u = phi_mat[control].flatten(order='F') - u
            val_s = (phi_mat[self.seams[:,0]]-phi_mat[self.seams[:,1]]).flatten(order='F')
            val = np.concatenate([val_str,val_s,val_u])
            return val

    class Shear:
        def __init__(self, shear_neighs, shear_corners, X, n_verts, m_sqrt, seams, IJKs):
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

            #seams
            self.seams = seams
            self.n_seams = seams.shape[0]
            self.Is = IJKs[0]
            self.Js = IJKs[1]
            self.Ks = IJKs[2]

            #for the control u
            self.II = np.concatenate([self.I,self.Is+self.n_conds])
            self.JJ = np.concatenate([self.J,self.Js])
            self.Ku = []
            #initial condition
            self.val0 = np.zeros((self.n_conds,))
            self.grad = sp.csc_matrix((np.arange(len(self.II)), (self.II, self.JJ)), 
                                      shape=(self.n_conds+3*self.n_seams, 3*self.n_verts))
            self.order = self.grad.data.astype(np.int64)
            self.gradT = sp.csr_matrix((np.arange(len(self.II)), (self.JJ, self.II)), 
                                      shape=(3*self.n_verts,self.n_conds+3*self.n_seams))
            self.orderT = self.gradT.data.astype(np.int64)
            self.m_sqrt = m_sqrt
            self.m_sqrt_JJ = self.m_sqrt[self.JJ]
            self.val0 = self.evaluate(X,np.zeros((0,)),[])[:self.n_conds]
            self.abs_val0 = np.abs(self.val0)
            self.factor = None

        def update_u(self,I,J,K):
            self.Ku = K    
            if len(I) > 0:
               self.II = np.concatenate([self.I,self.Is+self.n_conds,I + self.n_conds + 3*self.n_seams])
               self.JJ = np.concatenate([self.J,self.Js,J])  
            else:
               self.II = np.concatenate([self.I,self.Is+self.n_conds])
               self.JJ = np.concatenate([self.J,self.Js])
            self.m_sqrt_JJ = self.m_sqrt[self.JJ]
            self.grad = sp.csc_matrix((np.arange(len(self.II)), (self.II, self.JJ)), 
                                       shape=(self.n_conds+len(I)+3*self.n_seams, 3*self.n_verts))
            self.order = self.grad.data.astype(np.int64)
            self.gradT = sp.csr_matrix((np.arange(len(self.II)), (self.JJ, self.II)), 
                                       shape=(3*self.n_verts,self.n_conds+len(I)+3*self.n_seams))
            self.orderT = self.gradT.data.astype(np.int64)

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
                                   _grad0,_grad1,_grad2,self.Ks,self.Ku])*self.m_sqrt_JJ + 1e-16  
                self.grad.data = K[self.order]
                self.gradT.data = K[self.orderT]
            val_u = phi_mat[control,:].flatten(order='F') - u
            val_s = (phi_mat[self.seams[:,0]]-phi_mat[self.seams[:,1]]).flatten(order='F')
            val = np.concatenate([val_shr,val_s,val_u])
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
        ps.register_surface_mesh(self.label, self.positions, self.triangles, smooth_shade=True, transparency=0.9, edge_width = 0)
        #ps.register_surface_mesh(self.label, self.positions, self.faces, smooth_shade=True, transparency=0.9)#, edge_width = 0)
        ps.register_curve_network(self.label,self.positions,self.edges_matrix,enabled=False)
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("tile_reflection")  # set +Z as up direction
        ps.set_ground_plane_height(-0.005) # adjust the plane height

    
    def plotMesh(self):    
        if self.polyscoped is False:
            self.preparePolyscope()
        """Plot the current mesh"""
        ps.get_surface_mesh(self.label).update_vertex_positions(self.positions)
        #ps.get_surface_mesh(self.label).update_vertex_positions(self.positions)
        ps.get_curve_network(self.label).update_node_positions(self.positions)
        if self.rad is not None:
           ps.get_curve_network(self.label).set_radius(rad=self.rad,relative=False)
        ps.show()

    def makeMovie(self, speed = 1, repeat = True, smooth = 0):
        if self.polyscoped is False:
            self.preparePolyscope()
        self.ps_frame = 0
        skip = speed
        ps.get_curve_network(self.label).set_radius(rad=self.rad,relative=False)

        def goThroughHistory():
            # Update Polyscope visualization
            phi_mat = self.history_pos[self.ps_frame]
            for _ in range(smooth):
                phi_mat = self.S@phi_mat
            ps.get_surface_mesh(self.label).update_vertex_positions(phi_mat)
            ps.get_curve_network(self.label).update_node_positions(phi_mat)

            # Advance simulation time by skipping frames accordingly
            self.ps_frame += skip
            if self.ps_frame >= len(self.history_pos):
                if repeat:
                   self.ps_frame = 0  # Loop back to start
                else:
                   #display last frame before stopping
                   phi_mat = self.history_pos[-1]
                   for _ in range(smooth):
                       phi_mat = self.S@phi_mat
                   ps.get_surface_mesh(self.label).update_vertex_positions(phi_mat)
                   ps.get_curve_network(self.label).update_node_positions(phi_mat)
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
        #assert diff_rel <= 50, f"Relative difference between smallest and biggest edge is '{diff_rel}'% more than 50%, please re-define mesh"
        #take into account diagonals
        mid_faces =  (self.positions[self.f0] + self.positions[self.f1] + self.positions[self.f2])/3
        l0 = self.computeNorm(mid_faces-self.positions[self.f0])
        l1 = self.computeNorm(mid_faces-self.positions[self.f1])
        l2 = self.computeNorm(mid_faces-self.positions[self.f2])
        #constant radious of the balls
        #self.rad = self.thck*np.mean(longs)/2.05
        self.max_step = self.max_mov*np.mean(longs)
        self.eps_ee = 1.1*np.max(longs)
        self.eps_nf = 1.1*np.max([l0,l1,l2])

        #matrix of radiouses
        matrix_rads = 2*self.rad*np.ones((self.n_verts,self.n_verts),dtype=float)
        #reduce in case it is too big
        sum_rads = np.minimum(2*self.rad,0.976*longs)
        matrix_rads[e0,e1] = sum_rads; matrix_rads[e1,e0] = sum_rads   
        #save matrix for fast indixing
        self.matrix_rads = matrix_rads
        #edges that share a node
        #S = self.A0 @ self.A0.T
        #ei, ej = S.nonzero()

 
    def setSimulatorParameters(self, dt = 1/60, tol = 0.0075, sub_steps = 10,
                               rho = 0.1, delta = 0.1, alpha = 0.2,
                               kappa = 0.5*1e-4, kappa_bnd = 0.05*1e-4, kappa_flt = 0.5*1e-4,
                               str = 0.01*1e-4, shr = 10*1e-4, slf = 1*1e-4,
                               mu_f = 0.2, mu_s = 0.35, thck = 0.95, max_mov= 0.1):
        #solver parameters
        self.frame_rate = dt #desired frame rate
        self.sub_steps = sub_steps
        self.dt = dt/self.sub_steps #time step
        self.t_int = np.linspace(1/self.sub_steps,1,self.sub_steps) #for interpolating the controls when substepping
        self.tol = tol #tolerance for constraints
        self.implicitEuler = False

        #physical parameters
        self.g = 9.8 #gravity acceleration in m/s**2
        self.rho = rho # density of the cloth
        self.delta = delta # virtual mass 
        self.alpha = alpha # slow damping 
        self.kappa = kappa # bending stiffness
        self.kappa_bnd = kappa_bnd # bending stiffness
        self.kappa_flt = kappa_flt # bending stiffness
        self.beta = 0.02*self.kappa # fast damping: do not change in general
        self.str = str/(self.dt**2) # stretch elasticity
        self.shr = shr/(self.dt**2) # shear elasticity
        self.slf = slf/(self.dt**2) # self-collisions elasticity
        self.mu_floor = mu_f #friction with to the floor
        self.mu_self = mu_s #friction for self-collisions

        #self-collision parameters
        self.thck = thck
        self.mov_tol = 0.02 #when some node moves 2.5% or more than its previous position, run computeClosePairs()
        self.max_mov = max_mov #between 0 and 1 fraction of mean edge length that the control nodes can move in one time step
        self.computeRadiouses()
        #self.eps_sus = 3.5*self.rad #threshold for detecting close balls in computeClosePairs()


        #factorize implicit step matrix E for fast unconstrained step
        D = self.alpha*self.M + self.beta*self.K 
        K = self.kappa*self.K + self.kappa_bnd*self.Kb; 
        M = self.rho*self.M; 
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
        self.dt2_delta_Fg = (self.dt**2)*self.delta*self.g*self.Fg
        self.half_dt2_delta_Fg = 0.5*(self.dt**2)*self.delta*self.g*self.Fg  

        #aerodynamics    
        self.half_dt2_Fg = 0.5*(self.dt**2)*self.g*self.Fg
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

    def addTable(self,center,dimensions,mu):
        self.table = True
        self.mu_table = mu
        self.table_center = np.array(center)
        self.table_half_size = np.array(dimensions)/2
        self.box_min = self.table_center - self.table_half_size
        self.box_max = self.table_center + self.table_half_size
        cx, cy, cz = self.table_center
        hx, hy, hz = 0.95*self.table_half_size

        # Vertices: 8 corners of the rectangular table box
        self.table_vertices = np.array([
            [cx - hx, cy - hy, cz - hz],  # 0
            [cx + hx, cy - hy, cz - hz],  # 1
            [cx + hx, cy + hy, cz - hz],  # 2
            [cx - hx, cy + hy, cz - hz],  # 3

            [cx - hx, cy - hy, cz + hz],  # 4
            [cx + hx, cy - hy, cz + hz],  # 5
            [cx + hx, cy + hy, cz + hz],  # 6
            [cx - hx, cy + hy, cz + hz],  # 7
        ])

        # Quadrilateral faces
        self.table_faces = np.array([
            [0, 3, 2, 1],  # bottom
            [7, 6, 5, 4],  # top

            [0, 1, 5, 4],  # front, y-
            [3, 7, 6, 2],  # back, y+

            [0, 4, 7, 3],  # left, x-
            [1, 2, 6, 5],  # right, x+
        ], dtype=int)
        if self.polyscoped is False:
            self.preparePolyscope()
        ps.register_surface_mesh("Table", self.table_vertices, self.table_faces, smooth_shade=True, edge_width = 1)

    
    def tableCollisions(self,phi):    
        phi_mat = phi.reshape((-1, 3), order="F")
        p = phi_mat.copy()

        # Is particle center inside the box?
        inside = np.all((p >= self.box_min) & (p <= self.box_max), axis=1)

        # --------------------------------------------------
        # Case 1: particle center is outside the box
        # --------------------------------------------------
        closest = np.clip(p, self.box_min, self.box_max)

        direction = p - closest
        dist = self.computeNorm(direction) 

        outside_hit = (~inside) & (dist < self.rad)

        if np.any(outside_hit):
            n = direction[outside_hit] / dist[outside_hit, None]
            p[outside_hit] = closest[outside_hit] + n * self.rad

        # --------------------------------------------------
        # Case 2: particle center is inside the box
        # Push it to the nearest face, plus radius.
        # --------------------------------------------------
        if np.any(inside):
            p_inside = p[inside]

            dist_to_min = p_inside - self.box_min
            dist_to_max = self.box_max - p_inside

            distances = np.concatenate([dist_to_min, dist_to_max], axis=1)
            closest_face = np.argmin(distances, axis=1)

            corrected = p_inside.copy()

            for i, face in enumerate(closest_face):
                if face < 3:
                    axis = face
                    corrected[i, axis] = self.box_min[axis] - self.rad
                else:
                    axis = face - 3
                    corrected[i, axis] = self.box_max[axis] + self.rad

            p[inside] = corrected
        #friction
        dlt_phi = p - phi_mat
        nu_mat = dlt_phi.reshape((self.n_verts, 3), order='F')
        norm_Fn = self.computeNorm(nu_mat)
        nu = nu_mat/(norm_Fn[:,np.newaxis] + 1e-12)
        v = self.positions - p
        vt = v - (self.innerProduct(v,nu)[:,np.newaxis])*nu 
        #compute friction force vector
        F_mu = self.frictionForce(self.mu_table,norm_Fn,vt,cap = True)
        phi += (dlt_phi + F_mu).flatten(order="F") 
        return phi

    def floorCollisions(self,phi):
        phi_mat = phi.reshape((self.n_verts, 3), order='F').copy()
        ind_col = np.nonzero(phi_mat[:,2] < 0)[0]
        self.flr = False #bookeeping if floor collisions occurred
        if ind_col.shape[0] > 0:
            self.flr = True
            #normal forces
            #norm_Fn = self.nodes_faces_count[ind_col]*np.abs(phi_mat[ind_col,2]) #normal force scaled
            norm_Fn = np.abs(phi_mat[ind_col,2]) #normal force 
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
    
    def cullRedundantEdgeConstraints(self, max_per_edge=3):
        """
        Keep only the most penetrating edge-edge constraints, with a cap on how
        many constraints each mesh edge can participate in.
        """

        ind = self.ind_slf_ee

        if ind.shape[0] == 0:
            return

        # Candidate edge ids
        eA = self.near_ee0[ind]
        eB = self.near_ee1[ind]

        # Most penetrating first: vals_ee is C = distance - radius.
        # Negative is penetrating, more negative is more important.
        order = np.argsort(self.vals_ee[ind])

        n_edges = self.e0.shape[0]
        edge_count = np.zeros(n_edges, dtype=int)

        keep = np.zeros(ind.shape[0], dtype=bool)

        for loc in order:
            ea = eA[loc]
            eb = eB[loc]

            if edge_count[ea] < max_per_edge and edge_count[eb] < max_per_edge:
                keep[loc] = True
                edge_count[ea] += 1
                edge_count[eb] += 1

        self.ind_slf_ee = ind[keep]
    
    @profile
    def selfCollisions(self,phi,n_iter,s,max_iters=50):    
        if n_iter == 0:
            #precompute objects for selfcollisions
            self.prepareCollisions(phi)   


        #1) check for possible interior faces selfcollisions
        self.updateCollisionsFaces(phi)

        if self.error_nf < -self.tol: #correct detected self-collisions
            #add new and previous selfcollisions
            ind_s = np.nonzero((self.vals_nf/(2*self.rad)) < -self.tol)[0]
            self.ind_slf_nf = self.unionMask(self.ind_slf_nf,ind_s)
            #print('Face constraints: ',self.ind_slf_nf.shape[0])
            #correction for positions
            dlt_phi = self.solveFacesLCP(max_iters)
            phi += dlt_phi

            #self.checkCollisionsFaces(phi)
            
            #apply friction if needed
            if self.mu_self > 0 and n_iter < 5:
                F_mu = self.computeFrictionCorrection(phi,dlt_phi)
                phi += F_mu
        #self.updateCollisionsFaces(phi)

        #2) check for possible edges selfcollisions
        self.updateCollisionsEdges(phi)

        if self.error_ee < -self.tol: #correct detected self-collisions
            #add new and previous selfcollisions
            ind_s = np.nonzero((self.vals_ee/(2*self.rad)) < -self.tol)[0]
            self.ind_slf_ee = self.unionMask(self.ind_slf_ee,ind_s)
            #self.cullRedundantEdgeConstraints(max_per_edge=3)
            #print('Edge constraints: ',self.ind_slf_ee.shape[0])
            #correction for positions
            dlt_phi = self.solveEdgesLCP(max_iters)
            #dlt_phi = 0*phi
            phi += dlt_phi

            #self.checkCollisionsEdges(phi)
            
            #apply friction if needed
            if self.mu_self > 0 and n_iter < 5:
                F_mu = self.computeFrictionCorrection(phi,dlt_phi)
                phi += F_mu
        #self.updateCollisionsEdges(phi)
            
        return phi
    
    @profile
    def updateCollisionsFaces(self,phi): 
        phi_mat = phi.reshape((self.n_verts, 3), order='F') 
        #assume we already have the baryentric coordinates
        p = phi_mat[self.near_nf0]
        q0 = phi_mat[self.f0[self.near_nf1]]
        q1 = phi_mat[self.f1[self.near_nf1]]
        q2 = phi_mat[self.f2[self.near_nf1]]
        #closest points
        q = self.w0*q0 + self.w1*q1 + self.w2*q2 

        #simplified CCD for the faces
        pq = q - p
        #normal
        norm_pq = self.computeNorm(pq)
        normal_all = pq / norm_pq[:,np.newaxis]
        #orient normal
        res0 = self.innerProduct(self.pq0_nf,normal_all); flip = (res0 < 0); 
        normal_all[flip] = -normal_all[flip]; norm_pq[flip] = -norm_pq[flip]             
        #evaluate the constraints
        self.vals_nf = norm_pq - 2*self.rad
        self.normals_nf = normal_all 
        if self.vals_nf.shape[0] > 0:
           self.error_nf = np.min(self.vals_nf/(2*self.rad))
        else:
           self.error_nf = 1

    def checkCollisionsFaces(self,phi): 
        phi_mat = phi.reshape((self.n_verts, 3), order='F') 
        #assume we already have the baryentric coordinates
        p = phi_mat[self.near_nf0]
        q0 = phi_mat[self.f0[self.near_nf1]]
        q1 = phi_mat[self.f1[self.near_nf1]]
        q2 = phi_mat[self.f2[self.near_nf1]]
        q3 = phi_mat[self.f3[self.near_nf1]]
        #closest points
        q = self.w0*q0 + self.w1*q1 + self.w2*q2 + self.w3*q3

        #simplified CCD for the faces
        pq = q - p
        res = self.innerProduct(pq,self.normals_nf)
        print('Error faces after LCP',np.min((res - 2*self.rad)/(2*self.rad)))
    

    @profile
    def updateCollisionsEdges(self,phi): 
        phi_mat = phi.reshape((self.n_verts, 3), order='F') 
        #assume we already have the baryentric coordinates
        p0 = phi_mat[self.e0[self.near_ee0]]
        p1 = phi_mat[self.e1[self.near_ee0]]
        q0 = phi_mat[self.e0[self.near_ee1]]
        q1 = phi_mat[self.e1[self.near_ee1]]
        #closest points
        p = (1-self.ss)*p0 + self.ss*p1
        q = (1-self.tt)*q0 + self.tt*q1

        #simplified CCD for the edges
        pq = q - p
        #normal
        norm_pq = self.computeNorm(pq)
        normal_all = pq / norm_pq[:,np.newaxis]
        #orient normal
        res0 = self.innerProduct(self.pq0_ee,normal_all); flip = (res0 < 0); 
        normal_all[flip] = -normal_all[flip]; norm_pq[flip] = -norm_pq[flip]             
        #evaluate the constraints
        self.vals_ee = norm_pq - 2*self.rad
        self.normals_ee = normal_all 
        if self.vals_ee.shape[0] > 0:
           self.error_ee = np.min(self.vals_ee/(2*self.rad))
        else:
           self.error_ee = 1

    def checkCollisionsEdges(self,phi): 
        phi_mat = phi.reshape((self.n_verts, 3), order='F') 
        #assume we already have the baryentric coordinates
        p0 = phi_mat[self.e0[self.near_ee0]]
        p1 = phi_mat[self.e1[self.near_ee0]]
        q0 = phi_mat[self.e0[self.near_ee1]]
        q1 = phi_mat[self.e1[self.near_ee1]]
        #closest points
        p = (1-self.ss)*p0 + self.ss*p1
        q = (1-self.tt)*q0 + self.tt*q1

        #simplified CCD for the edges
        pq = q - p
        res = self.innerProduct(pq,self.normals_ee);         
        #evaluate the constraints
        print('Error after LCP',(res - 2*self.rad)/(2*self.rad))




    def scatterEdgesBincount(self, ind_p0, ind_p1, ind_q0, ind_q1,
                            aa, bb, cc, dd, dlt, out):
        n = self.n_verts

        for k in range(3):
            dk = dlt[:, k]

            out[:, k] = (
                np.bincount(ind_p0, weights=-aa * dk, minlength=n)
                + np.bincount(ind_p1, weights=-bb * dk, minlength=n)
                + np.bincount(ind_q0, weights= cc * dk, minlength=n)
                + np.bincount(ind_q1, weights= dd * dk, minlength=n)
            )
    @profile
    def solveEdgesLCP(self, max_iter = 50):
        #take only needed normals
        normals = self.normals_ee[self.ind_slf_ee]
        #and barycentric coordinates
        aa = self.a_e[self.ind_slf_ee]
        bb = self.b_e[self.ind_slf_ee]
        cc = self.c_e[self.ind_slf_ee]
        dd = self.d_e[self.ind_slf_ee]
        #indices of involved nodes
        e0_col = self.near_ee0[self.ind_slf_ee]
        e1_col = self.near_ee1[self.ind_slf_ee]
        ind_p0 = self.e0[e0_col]; ind_p1 = self.e1[e0_col]
        ind_q0 = self.e0[e1_col]; ind_q1 = self.e1[e1_col]
        ind_all = np.concatenate([ind_p0,ind_p1,ind_q0,ind_q1])
    
        #counts to take average impulses 
        #count = np.bincount(ind_all, minlength=self.n_verts)
        
        eps_w = 1e-3
        count = np.zeros(self.n_verts)
        np.add.at(count, ind_p0[aa > eps_w], 1.0)
        np.add.at(count, ind_p1[bb > eps_w], 1.0)
        np.add.at(count, ind_q0[cc > eps_w], 1.0)
        np.add.at(count, ind_q1[dd > eps_w], 1.0)
        
        #averages
        avg = 1/(count + 1e-12); avg[count == 0] = 0; 
        #mass inverses: set controled to zero
        w = self.m_inv.copy(); w[self.control] = 0
        wa  = (avg*w)[:,np.newaxis]      

        #initial impulses
        num = -self.vals_ee[self.ind_slf_ee]; 
        den = (aa**2)*w[ind_p0] + (bb**2)*w[ind_p1] + (cc**2)*w[ind_q0] + (dd**2)*w[ind_q1] + self.slf
        landa = np.maximum(0,num/den)

        #corrections
        dlt = landa[:,np.newaxis]*normals
        dlt_a = -aa[:,np.newaxis]*dlt
        dlt_b = -bb[:,np.newaxis]*dlt
        dlt_c = +cc[:,np.newaxis]*dlt
        dlt_d = +dd[:,np.newaxis]*dlt
        dlt_all = np.concatenate([dlt_a,dlt_b,dlt_c,dlt_d], axis = 0)
        #global correction
        dlt_tot = np.zeros((self.n_verts,3))
        np.add.at(dlt_tot,ind_all,dlt_all); 
        dlt_phi = wa*dlt_tot

        #iterative process
        error_l = -1; ii = 0
        while error_l < -self.tol and ii < max_iter:  
            dlt_pq = - (aa[:,np.newaxis]*dlt_phi[ind_p0]) - (bb[:,np.newaxis]*dlt_phi[ind_p1]) + (cc[:,np.newaxis]*dlt_phi[ind_q0]) + (dd[:,np.newaxis]*dlt_phi[ind_q1])
                      
            dlt_vals = -self.innerProduct(normals,dlt_pq)
            #compute multipliers
            res = num + dlt_vals - self.slf*landa
            error_l = np.min(-res)/(2*self.rad)
            #print('error LCP: ',error_l)
            landa = np.maximum(0, landa + res/den)
            #corrections
            dlt = landa[:,np.newaxis]*normals 
            dlt_a = -aa[:,np.newaxis]*dlt
            dlt_b = -bb[:,np.newaxis]*dlt
            dlt_c = +cc[:,np.newaxis]*dlt
            dlt_d = +dd[:,np.newaxis]*dlt
            dlt_all = np.concatenate([dlt_a,dlt_b,dlt_c,dlt_d], axis = 0)
            #global correction
            dlt_tot.fill(0.0)
            np.add.at(dlt_tot,ind_all,dlt_all);         
            dlt_phi = wa*dlt_tot
            ii += 1
        #print('iterations LCP: ',ii)
        return dlt_phi.flatten(order='F')
    
    @profile
    def solveFacesLCP(self, max_iter = 50):
        #take only needed normals
        normals = self.normals_nf[self.ind_slf_nf]
        #and barycentric coordinates
        w0 = self.w0[self.ind_slf_nf]
        w1 = self.w1[self.ind_slf_nf]
        w2 = self.w2[self.ind_slf_nf]
        #indices of involved nodes
        ind_p = self.near_nf0[self.ind_slf_nf]
        f_col = self.near_nf1[self.ind_slf_nf]
        ind_q0 = self.f0[f_col]; ind_q1 = self.f1[f_col]; ind_q2 = self.f2[f_col]; 
        ind_all = np.concatenate([ind_p,ind_q0,ind_q1,ind_q2])
    
        #counts to take average impulses 
        count = np.bincount(ind_all, minlength=self.n_verts)
        
        #averages
        avg = 1/(count + 1e-12); avg[count == 0] = 0; 
        #mass inverses: set controled to zero
        w = self.m_inv.copy(); w[self.control] = 0
        wa  = (avg*w)[:,np.newaxis]      

        #initial impulses
        num = -self.vals_nf[self.ind_slf_nf]; 
        den = w[ind_p] + (w0[:, 0]**2)*w[ind_q0] + (w1[:, 0]**2)*w[ind_q1] + (w2[:, 0]**2)*w[ind_q2] + self.slf
        landa = np.maximum(0,num/den)

        #corrections
        dlt = landa[:,np.newaxis]*normals
        dlt0 = +w0*dlt
        dlt1 = +w1*dlt
        dlt2 = +w2*dlt
        dlt_all = np.concatenate([-dlt,dlt0,dlt1,dlt2], axis = 0)
        #global correction
        dlt_tot = np.zeros((self.n_verts,3))
        np.add.at(dlt_tot,ind_all,dlt_all); 
        dlt_phi = wa*dlt_tot

        #iterative process
        error_l = -1; ii = 0
        while error_l < -self.tol and ii < max_iter:  
            dlt_pq =  (w0*dlt_phi[ind_q0]) + (w1*dlt_phi[ind_q1]) + (w2*dlt_phi[ind_q2]) - dlt_phi[ind_p]
                      
            dlt_vals = -self.innerProduct(normals,dlt_pq)
            #compute multipliers
            res = num + dlt_vals - self.slf*landa
            error_l = np.min(-res)/(2*self.rad)
            #print('error LCP: ',error_l)
            landa = np.maximum(0, landa + res/den)
            #corrections
            dlt = landa[:,np.newaxis]*normals
            dlt0 = +w0*dlt
            dlt1 = +w1*dlt
            dlt2 = +w2*dlt
            dlt_all = np.concatenate([-dlt,dlt0,dlt1,dlt2], axis = 0)
            #global correction
            dlt_tot.fill(0.0)
            np.add.at(dlt_tot,ind_all,dlt_all); 
            dlt_phi = wa*dlt_tot
            ii += 1
        #print('iterations LCP: ',ii)
        return dlt_phi.flatten(order='F')


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
        den = w[b0_col] + w[b1_col] + self.slf
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
            res = num + dlt_vals - self.slf*landa
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
    
    def buildShareNodeMatrix(self):
        n = self.n_edges
        share_node = np.zeros((n, n), dtype=bool)
        S = self.A0 @ self.A0.T
        ei, ej = S.nonzero()
        share_node[ei,ej] = True
        #S2 = self.A1.T @ self.A1
        #ei, ej = S2.nonzero()
        #share_node[ei,ej] = True
        self.share_node = share_node

        node_in_face = self.A2.toarray().astype(bool)
        self.node_in_face = node_in_face

    
    def buildShareEdgeMatrix(self):
        n = self.n_verts
        # --- Union-Find over seam equivalences ---
        parent = np.arange(n, dtype=int)
        rank = np.zeros(n, dtype=int)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for a, b in self.seams:
            union(a, b)

        reps = np.array([find(i) for i in range(n)], dtype=int)

        # group members by representative
        groups = {}
        for idx, r in enumerate(reps):
            groups.setdefault(r, []).append(idx)

        share_edge = np.zeros((n, n), dtype=bool)

        # --- (1) clique within each equivalence class ---
        for members in groups.values():
            if len(members) > 1:
                m = np.array(members, dtype=int)
                share_edge[np.ix_(m, m)] = True

        # --- (2) lift real edges across equivalence classes ---
        for u, v in self.edges_matrix:
            gu = np.array(groups[reps[u]], dtype=int)
            gv = np.array(groups[reps[v]], dtype=int)
            share_edge[np.ix_(gu, gv)] = True
            share_edge[np.ix_(gv, gu)] = True

        self.share_edge = share_edge
    @profile
    def computeClosePairs2(self, phi_mat):
        # -------------------------
        # edge-edge broad phase
        # -------------------------
        phi_e = 0.5 * (phi_mat[self.e0] + phi_mat[self.e1])
        tree_e = cKDTree(phi_e)

        pairs = tree_e.query_pairs(self.eps_ee, output_type="ndarray")

        if pairs.shape[0] > 0:
            ei = pairs[:, 0]
            ej = pairs[:, 1]

            mask = ~self.share_node[ei, ej]
            ei = ei[mask]
            ej = ej[mask]
        else:
            ei = np.empty(0, dtype=np.int64)
            ej = np.empty(0, dtype=np.int64)

        self.near_ee0 = ei
        self.near_ee1 = ej

        # -------------------------
        # node-face broad phase
        # -------------------------
        phi_f = (phi_mat[self.f0] + phi_mat[self.f1] + phi_mat[self.f2]) / 3.0
        self.mid_faces = phi_f

        tree_n = cKDTree(phi_mat)

        neigh_lists = tree_n.query_ball_point(phi_f, r=self.eps_nf)
        counts = np.fromiter((len(x) for x in neigh_lists), dtype=np.int64)

        if counts.sum() > 0:
            fi = np.repeat(np.arange(phi_f.shape[0]), counts)
            nj = np.concatenate(neigh_lists).astype(np.int64)

            mask = ~self.node_in_face[fi, nj]
            fi = fi[mask]
            nj = nj[mask]
        else:
            fi = np.empty(0, dtype=np.int64)
            nj = np.empty(0, dtype=np.int64)

        self.near_nf0 = nj
        self.near_nf1 = fi

        self.mask_col = np.zeros(
            max(ei.shape[0], nj.shape[0]),
            dtype=bool
        )
    
    @profile
    def computeClosePairs(self,phi_mat):
        #build the tree only for the edges
        phi_e = 0.5*(phi_mat[self.e0]+phi_mat[self.e1])
        tree_e = KDTree(phi_e)

        #node-node close pairs
        dists, neighs = tree_e.query(phi_e, k=self.ke+1) #query it for k nodes neighbors
        #reshape removing the first pair
        dist = dists[:,1:].reshape(-1)
        ej = neighs[:,1:].reshape(-1)
        #remove far away pairs and duplicates
        mask = (dist < self.eps_ee) & (self.ei < ej)
        ei = self.ei[mask]; ej = ej[mask]
        #second mask
        mask2 = ~self.share_node[ei,ej]
        ei = ei[mask2]; ej = ej[mask2]
        #potential colliding nodes-nodes
        self.near_ee0 = ei; self.near_ee1 = ej

        #build the tree only for the faces
        tree_n = KDTree(phi_mat)

        #node-face close pairs
        phi_f = (phi_mat[self.f0] + phi_mat[self.f1] + phi_mat[self.f2])/3
        self.mid_faces = phi_f
        dists, neighs = tree_n.query(phi_f, k=self.kf) #query it for k nodes neighbors
        #reshape 
        dist = dists.reshape(-1)
        nj = neighs.reshape(-1)
        #remove far away pairs and duplicates
        mask = (dist < self.eps_nf) 
        fi = self.fi[mask]; nj = nj[mask]
        #second mask
        mask2 = ~self.node_in_face[fi,nj]
        fi = fi[mask2]; nj = nj[mask2]
        #potential colliding nodes-faces
        self.near_nf0 = nj; self.near_nf1 = fi


        #mask for indices
        self.mask_col = np.zeros(np.maximum(ei.shape[0],nj.shape[0]), dtype=bool)

    @profile
    def updateClosePairs(self,phi_mat):
        updated = False
        #check close pairs
        diff = phi_mat - self.last_check
        mov = np.sqrt(np.max(self.innerProduct(diff, diff)/self.den_last))
        if (mov > self.mov_tol) or (self.total_iters == 0) or self.update_chol: #only check when at least 1 node has moved more than mov_eps
            self.computeClosePairs(phi_mat) #update close pairs
            self.last_check = phi_mat.copy() #update last checked mesh
            self.den_last = self.innerProduct(self.last_check,self.last_check)
            updated = True          
            #print("Close node-face")
            #print(np.vstack([self.near_nf0,self.near_nf1]).T)

    @profile
    def computeBarycentricFaces(self, phi_mat):
        #fancy indexing (precompute interior)
        p = phi_mat[self.near_nf0]
        q0 = phi_mat[self.f0[self.near_nf1]]
        q1 = phi_mat[self.f1[self.near_nf1]]
        q2 = phi_mat[self.f2[self.near_nf1]]


        # ------------------------------------------------------------
        # Interior candidate:
        #
        #     p-q0 = alpha*(q1-q0) + beta*(q2-q0)
        # ------------------------------------------------------------

        alpha, beta, nonsing, _, _ = self.projectVectorInPlane(p - q0, q1-q0, q2-q0)
        gamma = 1 - alpha - beta
        self.w1 = alpha[:,np.newaxis]
        self.w2 = beta[:,np.newaxis]
        self.w0 = gamma[:,np.newaxis]

        q = self.w0*q0 + self.w1*q1 + self.w2*q2

        norm_pq = self.computeNorm(q-p)

        valid = (
            nonsing & (norm_pq < 4.5*self.rad)
            & (alpha > 0) & (alpha < 1)
            & (beta > 0) & (beta < 1)
            & (gamma > 0) & (gamma < 1)
        )


        #update arrays
        self.near_nf0 = self.near_nf0[valid]
        self.near_nf1 = self.near_nf1[valid]
        self.w0 = self.w0[valid]
        self.w1 = self.w1[valid]
        self.w2 = self.w2[valid]

        #ps.register_point_cloud('close node-face',np.concatenate((p[valid],q[valid]),axis=0))
        #ps.get_point_cloud('close node-face').set_radius(rad=self.rad,relative=False)


    @profile
    def computeBarycentricEdges(self, phi_mat):
        #fancy indexing (precompute interior)
        p0 = phi_mat[self.e0[self.near_ee0]]
        p1 = phi_mat[self.e1[self.near_ee0]]
        q0 = phi_mat[self.e0[self.near_ee1]]
        q1 = phi_mat[self.e1[self.near_ee1]]
        #direction vectors
        dp = p1 - p0; dq = q1 - q0

        # ------------------------------------------------------------
        # Interior line-line candidate:
        #
        #     q0 - p0 = u dp - v dq
        # ------------------------------------------------------------

        u_int, v_int, nonsing, dp2, dq2 = self.projectVectorInPlane(q0 - p0, dp, -dq)

        valid_int = (
            nonsing
            & (u_int >= 0.0) & (u_int <= 1.0)
            & (v_int >= 0.0) & (v_int <= 1.0)
        )

        u = u_int.copy()
        v = v_int.copy()

        # ------------------------------------------------------------
        # Only non-interior / singular cases need boundary tests.
        # ------------------------------------------------------------

        bad = ~valid_int

        if np.any(bad):
            #only do all these computations for non-interior ones
            p0b = p0[bad]; p1b = p1[bad]
            q0b = q0[bad]; q1b = q1[bad]

            dpb = dp[bad]; dqb = dq[bad]
            dp2b = dp2[bad]; dq2b = dq2[bad]

            nb = p0b.shape[0]

            # p0 against q0-q1
            v_p0, d2_p0 = self.closestPointNodeEdge(
                p0b, q0b, dqb, dq2b
            )
            u_p0 = np.zeros(nb)

            # p1 against q0-q1
            v_p1, d2_p1 = self.closestPointNodeEdge(
                p1b, q0b, dqb, dq2b
            )
            u_p1 = np.ones(nb)

            # q0 against p0-p1
            u_q0, d2_q0 = self.closestPointNodeEdge(
                q0b, p0b, dpb, dp2b
            )
            v_q0 = np.zeros(nb)

            # q1 against p0-p1
            u_q1, d2_q1 = self.closestPointNodeEdge(
                q1b, p0b, dpb, dp2b
            )
            v_q1 = np.ones(nb)

            u_all = np.stack([u_p0, u_p1, u_q0, u_q1], axis=1)
            v_all = np.stack([v_p0, v_p1, v_q0, v_q1], axis=1)

            d2_all = np.stack([d2_p0, d2_p1, d2_q0, d2_q1], axis=1)

            ind = np.argmin(d2_all, axis=1)
            rows = np.arange(nb)

            u[bad] = u_all[rows, ind]
            v[bad] = v_all[rows, ind]

        self.ss = u[:,np.newaxis]
        self.tt = v[:,np.newaxis]
        self.a_e = 1-u
        self.b_e = u
        self.c_e = 1-v
        self.d_e = v 
        #TODO: only do this for interior ones and reuse computed distances
        p = p0 + self.ss * dp
        q = q0 + self.tt * dq
        norm_pq = self.computeNorm(q-p)
        inds_cls = (norm_pq < 4.5*self.rad)

        #update arrays
        self.near_ee0 = self.near_ee0[inds_cls]
        self.near_ee1 = self.near_ee1[inds_cls]
        self.ss = self.ss[inds_cls]
        self.tt = self.tt[inds_cls]
        self.a_e = self.a_e[inds_cls]
        self.b_e = self.b_e[inds_cls]
        self.c_e = self.c_e[inds_cls]
        self.d_e = self.d_e[inds_cls]

        #ps.register_point_cloud('close edge-edge',np.concatenate((p[inds_cls],q[inds_cls]),axis=0))
        #ps.get_point_cloud('close edge-edge').set_radius(rad=self.rad,relative=False)


    
    def projectVectorInPlane(self,q,q1,q2):
        b1 = self.innerProduct(q,q1)
        b2 = self.innerProduct(q,q2)
        a11 = self.innerProduct(q1,q1)
        a12 = self.innerProduct(q1,q2)
        a22 = self.innerProduct(q2,q2)
        return self.solve2x2system(b1,b2,a11,a12,a12,a22)
    
    def solve2x2system(self, b1, b2, a11, a12, a21, a22, eps=1e-10):
        """
        Vectorized 2x2 solve.

        Singular / near-singular systems return x = y = 0, but should be ignored
        through the nonsing mask.
        """
        deter = a11 * a22 - a12 * a21

        # Relative singularity test. For the Gram matrix this is more meaningful
        # than comparing deter to an absolute number.
        scale = np.abs(a11 * a22) + eps
        nonsing = np.abs(deter) > eps * scale

        x = np.zeros_like(b1)
        y = np.zeros_like(b2)

        x[nonsing] = (
            b1[nonsing] * a22[nonsing]
            - a12[nonsing] * b2[nonsing]
        ) / deter[nonsing]

        y[nonsing] = (
            b2[nonsing] * a11[nonsing]
            - a21[nonsing] * b1[nonsing]
        ) / deter[nonsing]

        return x, y, nonsing, a11, a22
    
    def clampVector(self,u):
        return np.maximum(0.0, np.minimum(1.0, u))
    
    def closestPointNodeEdge(self, x, e0, de, de2, eps=1e-12):
        """
        Closest point from nodes x to segments e0 + u de.

        Parameters
        ----------
        x : (n, 3)
            Query points.
        e0 : (n, 3)
            Segment start points.
        de : (n, 3)
            Precomputed edge directions.
        de2 : (n,)
            Precomputed squared edge lengths.

        Returns
        -------
        u : (n,)
            Clamped coordinate on the edge.
        d2 : (n,)
            Squared distance.
        """
        u = self.innerProduct(x - e0, de) / (de2 + eps)
        u = self.clampVector(u)

        p = e0 + u[:, np.newaxis] * de
        d = x - p
        d2 = self.innerProduct(d, d)

        return u, d2

    
    @profile
    def prepareCollisions(self,phi):
        phi_mat = phi.reshape((self.n_verts, 3), order='F') 
        self.updateClosePairs(phi_mat)
        self.computeBarycentricEdges(phi_mat)
        self.computeBarycentricFaces(phi_mat)
        #do costly indexing operations only once
        p0 = self.positions[self.e0[self.near_ee0]]
        p1 = self.positions[self.e1[self.near_ee0]]
        q0 = self.positions[self.e0[self.near_ee1]]
        q1 = self.positions[self.e1[self.near_ee1]]
        #closest points
        p = (1-self.ss)*p0 + self.ss*p1
        q = (1-self.tt)*q0 + self.tt*q1
        self.pq0_ee = q - p
        #now for the other case
        p = self.positions[self.near_nf0]
        q0 = self.positions[self.f0[self.near_nf1]]
        q1 = self.positions[self.f1[self.near_nf1]]
        q2 = self.positions[self.f2[self.near_nf1]]
        #closest points
        q = self.w0*q0 + self.w1*q1 + self.w2*q2 
        self.pq0_nf = q - p
        #store past collisions
        self.ind_slf_ee = self.empty
        self.ind_slf_nf = self.empty
        #store if floor collisions have happened
        self.flr = True
    
    def projectControl(self,phi,u_mat,control,n_ctr):
        if n_ctr > 0:
            phi_mat = phi.reshape((self.n_verts, 3), order='F')
            phi_mat[control] = u_mat
            phi = phi_mat.reshape((self.n_verts*3, ), order='F')
        return phi 

    def projectConstraints(self,constraints,phi,u,control,landa,par,den_error,n):
        #evaluate constraints
        if n == 0:
            val = constraints.evaluate(phi,u,control,grad=True)
            if self.update_chol or constraints.factor is None:
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
        aux_error = (val[:constraints.n_conds] + par*landa[:constraints.n_conds])/(constraints.abs_val0 + den_error)
        error = np.linalg.norm(aux_error,ord=np.inf) 

        return phi, landa, error
    
    def ImplicitEuler(self):
        q = self.dt2_delta_Fg + (self.dt_rho_M * self.velocities) + (self.rho_M_plus_dt_D @ self.positions)
        #solve the sistem with the cholesky factor
        x = self.factor_E(q)
        return x.reshape((3*self.n_verts,),order='F')

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
           pos0 = self.positions[control].flatten(order='F')
           U = []
           for s in range(self.sub_steps):
               U.append(pos0 + self.t_int[s]*(u - pos0))
        else:
           u = np.zeros((0,))
           U = [u]*self.sub_steps
        #check if we need to update cholesky decomp. of constraints
        self.update_chol = False
        if self.control != control:
            #update internal variables
            self.control = control
            self.update_chol = True
            self.share_control[:] = False
            self.share_control[np.ix_(control, control)] = True
            if n_ctr > 0:
                Iu = np.arange(3*n_ctr)
                Ju = np.concatenate((control, [x+self.n_verts for x in control], [x+2*self.n_verts for x in control]))
                Ku = np.ones_like(Iu)            
            else:
                Iu = self.empty; Ju = self.empty; Ku = self.empty
            self.shear.update_u(Iu,Ju,Ku)
            self.stretch.update_u(Iu,Ju,Ku)
        return U
    
    def limitControlVelocity(self, u_raw):
        u_raw_mat = u_raw.reshape((len(self.control), 3), order="F")

        u_used = self.positions[self.control]

        du = u_raw_mat - u_used
        dist = self.computeNorm(du)
        scale = np.minimum(1.0, self.max_step / (dist + 1e-12))

        u_clmp = u_used + scale[:, None] * du

        return u_clmp.flatten(order="F")


    @profile
    def simulate(self, u, control):

        #process the control inputs
        U = self.processControlInputs(u,control)

        #substepping
        n_iter_sub = 0
        for s in range(self.sub_steps):

            #current position of the cloth 
            phi0 = self.positions.reshape((3*self.n_verts,),order = 'F')

            #interpolated control
            u_raw = U[s]; #u_mat = u.reshape((n_ctr,3),order='F')
            u = self.limitControlVelocity(u_raw)

            #unconstrained step to correct
            phi = self.unconstrainedStep(self.implicitEuler)

            #lagrange multipliers for the shear and stretch constraints
            lambda_shr = np.zeros((self.shear.n_conds + u.shape[0] + 3*self.n_seams,)); 
            lambda_str = np.zeros((self.stretch.n_conds + u.shape[0] + 3*self.n_seams,)); 

            #solver variables for inextensiblity 
            n_iter = 0; error_str = np.inf; error_shr = np.inf; self.error_ee = 0; self.error_nf = 0

            while (error_str > self.tol or error_shr > self.tol or self.error_nf < -np.inf or self.error_ee < -np.inf) and n_iter < 100: 

                #shearing
                phi, lambda_shr, error_shr = self.projectConstraints(self.shear,phi,u,control,
                                                                    lambda_shr,self.shr,0.005,s%5)

                #stretching
                phi, lambda_str, error_str = self.projectConstraints(self.stretch,phi,u,control,
                                                                    lambda_str,self.str,0,0)   
                
                
                #self-collisions
                phi = self.selfCollisions(phi,n_iter,s); 

                #iteration count 
                n_iter += 1

                #print('global edges error: ',self.error_ee)

            #print("global iters:",n_iter)

            if self.table is True:
                phi = self.tableCollisions(phi)

            """    

            inds_ee = np.nonzero(self.vals_ee < np.inf)[0]
            if inds_ee.shape[0] < 0:
                print("Close edge-edge")
                print(np.vstack([self.near_ee0[inds_ee],self.near_ee1[inds_ee]]).T)
                print("Barycentric")
                print(np.hstack([self.ss[inds_ee],self.tt[inds_ee]]))
                print("error")
                print(self.vals_ee[inds_ee])
            
            inds_nf = np.nonzero(self.vals_nf < np.inf)[0]
            if inds_nf.shape[0] < 0:
                print("Close node-face")
                print(np.vstack([self.near_nf0[inds_nf],self.near_nf1[inds_nf]]).T)
                print("Barycentric")
                print(np.hstack([self.w0[inds_nf],self.w1[inds_nf],self.w2[inds_nf]]))
                print("error")
                print(self.vals_nf[inds_nf])
            """
            
                


            #floor collisions
            phi = self.floorCollisions(phi)

            #update internal cloth variables
            dphi = (phi-phi0)/self.dt
            self.positions = phi.reshape((self.n_verts, 3), order='F')
            self.velocities = dphi.reshape((self.n_verts, 3), order='F')
            n_iter_sub += n_iter

        #save final positions and velocities
        self.history_pos.append(self.positions)
        self.history_vel.append(self.velocities)
        self.total_iters += n_iter_sub/self.sub_steps

        #warnings
        if self.total_iters/(len(self.history_pos)-1) > 4 and self.warning == False:
           print("WARNING: average of more than 4 iterations taken, for better performance reduce dt or increase thck")
           self.warning = True

