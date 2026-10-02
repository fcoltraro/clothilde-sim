import numpy as np
from scipy.spatial import cKDTree

def createMesh(interval, npx, npy, f1, f2, f3):
    #se crea a partir de una parametrizacion de la forma (f1(x,y),f2(x,y),f3(x,y))
    #donde (x,y) estan en el rectangulo [ax,bx]x[ay,by] and interval = [ax bx ay by]
    # Allocate space for the nodal coordinates matrix
    X = np.zeros((npx*npy, 3))
    xs = np.linspace(interval[0], interval[1], npx).reshape(-1, 1)
    unos = np.ones((npx, 1))
    # Nodes' coordinates
    yys = np.linspace(interval[2], interval[3], npy)
    for i in range(npy):
        ys = yys[i] * unos
        posi = np.arange((i * npx), ((i + 1) * npx))
        X[posi, :] = np.column_stack((f1(xs, ys), f2(xs, ys), f3(xs, ys)))
    # Elements (quadrilaterals)
    nx = npx - 1
    ny = npy - 1
    T = np.zeros((nx * ny, 4), dtype=int)
    for a in range(1, ny + 1):
        for b in range(1, nx + 1):
            ielem = (a - 1) * nx + b - 1
            inode = (a - 1) * npx + b - 1
            T[ielem, :] = [inode, inode + 1, inode + npx + 1, inode + npx]
    return X, T

def createRectangularMesh(a,b,na,nb,h = 0.5):
    #coordinate function for a flat cloth
    def f1(x, y):
        return x
    def f2(x, y):
        return y
    def f3(x, y):
        return h*(0.25-x**2) #avoid singular case
    #rectangle; a and b are the sides of the rectangle and na nb the number of nodes
    rect = [-a/2, a/2, -b/2, b/2]
    #create the mesh
    X, T = createMesh(rect, na, nb, f1, f2, f3)   
    return X, T  

def quad_cylinder_mesh(R, H, h, f=1.0):
    """
    Quad mesh of a (possibly flattened) cylinder.

    Parameters
    ----------
    R : float
        Base radius
    H : float
        Height
    h : float
        Target quad edge length
    alpha : float, optional
        Flattening factor in x-direction (alpha=1 -> circular cylinder,
        alpha<1 -> flattened / elliptical cylinder)

    Returns
    -------
    V : (N, 3) ndarray
        Vertex positions
    F : (M, 4) ndarray
        Quad faces
    """

    n_theta = max(3, int(round(2 * np.pi * R / h)))
    n_z     = max(1, int(round(H / h)))

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z = np.linspace(0.0, H, n_z + 1)

    Theta, Z = np.meshgrid(theta, z, indexing="ij")

    X = f * R * np.cos(Theta)
    Y = R * np.sin(Theta)

    V = np.column_stack((X.ravel(), Z.ravel(), Y.ravel()))

    F = []
    for i in range(n_theta):
        ip = (i + 1) % n_theta
        for j in range(n_z):
            v0 = i  * (n_z + 1) + j
            v1 = ip * (n_z + 1) + j
            v2 = ip * (n_z + 1) + (j + 1)
            v3 = i  * (n_z + 1) + (j + 1)
            F.append([v0, v1, v2, v3])

    return V, np.asarray(F, dtype=np.int64)


def duplicate_node_pairs(X, tol=1e-9):
    """
    Parameters
    ----------
    X : (n, 3) array
        Node positions
    tol : float
        Distance tolerance for considering two nodes identical

    Returns
    -------
    pairs : (r, 2) array of int
        Each row [i, j] means X[i] and X[j] are the same (within tol), with i < j
    """
    X = np.asarray(X)
    tree = cKDTree(X)

    # Get all unordered pairs within tolerance
    pairs = tree.query_pairs(r=tol)

    if not pairs:
        return np.empty((0, 2), dtype=int)

    # Convert set of tuples to sorted array
    pairs = np.array(list(pairs), dtype=int)
    pairs.sort(axis=1)  # ensure (i, j) with i < j
    return pairs

import numpy as np


def weld_quad_mesh(X, T, tol=1e-10, remove_degenerate=True):
    """
    Merge coincident vertices of a quadrilateral mesh.

    Parameters
    ----------
    X : (n, d) ndarray
        Vertex coordinates.
    T : (m, 4) ndarray of int
        Quadrilateral connectivity.
    tol : float
        Two vertices are merged when their coordinates agree up to this
        spatial tolerance.
    remove_degenerate : bool
        Remove quads that contain repeated vertices after welding.

    Returns
    -------
    X_clean : (n_clean, d) ndarray
        Welded vertex coordinates.
    T_clean : (m_clean, 4) ndarray
        Remapped quadrilateral connectivity.
    old_to_new : (n,) ndarray
        old_to_new[i] is the new index corresponding to old vertex i.
    groups : list[list[int]]
        Original vertices merged into each cleaned vertex.
    """

    X = np.asarray(X)
    T = np.asarray(T, dtype=int)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_vertices, dimension).")

    if T.ndim != 2 or T.shape[1] != 4:
        raise ValueError("T must have shape (n_quads, 4).")

    if np.any(T < 0) or np.any(T >= len(X)):
        raise ValueError("T contains invalid vertex indices.")

    if tol <= 0:
        raise ValueError("tol must be positive.")

    # Quantize coordinates so nearby points receive the same key.
    keys = np.round(X / tol).astype(np.int64)

    _, first_indices, inverse = np.unique(
        keys,
        axis=0,
        return_index=True,
        return_inverse=True,
    )

    # np.unique sorts the keys, so the resulting order is not necessarily
    # the order of first appearance. Reorder cleaned vertices by first use.
    order = np.argsort(first_indices)
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))

    old_to_new = inverse_order[inverse]

    representative_indices = first_indices[order]
    X_clean = X[representative_indices].copy()
    T_clean = old_to_new[T]

    groups = [[] for _ in range(len(X_clean))]
    for old_index, new_index in enumerate(old_to_new):
        groups[new_index].append(old_index)

    if remove_degenerate:
        # A valid quadrilateral must still have four distinct vertices.
        valid = np.array(
            [len(np.unique(quad)) == 4 for quad in T_clean],
            dtype=bool,
        )
        T_clean = T_clean[valid]

    return X_clean, T_clean, old_to_new, groups

from collections import deque, defaultdict
from scipy.sparse import lil_matrix, csr_matrix

def refine_rect_quad_mesh(T, n, m, num_vertices=None):
    """
    Refine a structured rectangular quadrilateral mesh.

    Parameters
    ----------
    T : (F,4) int array
        Coarse quad connectivity. Each quad must be cyclically ordered.
    n : int
        Number of subdivisions per coarse cell in the vertical direction.
    m : int
        Number of subdivisions per coarse cell in the horizontal direction.
    num_vertices : int or None
        Number of coarse vertices. If None, inferred as T.max()+1.

    Returns
    -------
    Tf : (Ff,4) int array
        Refined quad connectivity.
    S : scipy.sparse.csr_matrix, shape (Nf, N)
        Prolongation/interpolation matrix. If X is (N,d), then
        Xf = S @ X is (Nf,d).
        The first N rows of S form the identity, so original vertices are kept
        at the top of Xf.
    """

    T = np.asarray(T, dtype=int)
    if T.ndim != 2 or T.shape[1] != 4:
        raise ValueError("T must have shape (F,4)")
    if n < 1 or m < 1:
        raise ValueError("n and m must be positive integers")

    N = int(T.max()) + 1 if num_vertices is None else int(num_vertices)
    F = T.shape[0]

    # ------------------------------------------------------------------
    # 1) Recover the logical coarse grid (rows x cols) from the quad mesh
    # ------------------------------------------------------------------

    # Build face adjacency through undirected edges
    edge_to_faces = defaultdict(list)
    for f, q in enumerate(T):
        for k in range(4):
            a = q[k]
            b = q[(k + 1) % 4]
            e = tuple(sorted((a, b)))
            edge_to_faces[e].append((f, k))

    face_nbrs = [[None] * 4 for _ in range(F)]
    for e, lst in edge_to_faces.items():
        if len(lst) == 2:
            (f0, k0), (f1, k1) = lst
            face_nbrs[f0][k0] = (f1, k1)
            face_nbrs[f1][k1] = (f0, k0)
        elif len(lst) != 1:
            raise ValueError("Non-manifold edge detected")

    # Assign integer cell coordinates to faces by BFS
    # Local edge convention for a face q=[v0,v1,v2,v3]:
    # edge 0: v0-v1  -> neighbor at (-1, 0)
    # edge 1: v1-v2  -> neighbor at ( 0,+1)
    # edge 2: v2-v3  -> neighbor at (+1, 0)
    # edge 3: v3-v0  -> neighbor at ( 0,-1)
    edge_dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    face_rc = {0: (0, 0)}
    q = deque([0])

    while q:
        f = q.popleft()
        r, c = face_rc[f]
        for k in range(4):
            nbr = face_nbrs[f][k]
            if nbr is None:
                continue
            g, _ = nbr
            rr = r + edge_dirs[k][0]
            cc = c + edge_dirs[k][1]
            if g not in face_rc:
                face_rc[g] = (rr, cc)
                q.append(g)
            else:
                if face_rc[g] != (rr, cc):
                    raise ValueError("Mesh is not a consistent structured rectangular quad mesh")

    # Shift to start at (0,0)
    min_r = min(r for r, c in face_rc.values())
    min_c = min(c for r, c in face_rc.values())
    face_rc = {f: (r - min_r, c - min_c) for f, (r, c) in face_rc.items()}

    H = max(r for r, c in face_rc.values()) + 1   # number of coarse cells vertically
    W = max(c for r, c in face_rc.values()) + 1   # number of coarse cells horizontally

    if H * W != F:
        raise ValueError("The quad mesh is not a full rectangular grid")

    # Assign coarse grid coordinates to vertices
    # For a face q=[v0,v1,v2,v3] at cell (r,c), assign:
    # v0 -> (r,c), v1 -> (r,c+1), v2 -> (r+1,c+1), v3 -> (r+1,c)
    vertex_rc = {}
    for f, quad in enumerate(T):
        r, c = face_rc[f]
        corners = [
            (r, c),
            (r, c + 1),
            (r + 1, c + 1),
            (r + 1, c),
        ]
        for v, rc in zip(quad, corners):
            if v in vertex_rc:
                if vertex_rc[v] != rc:
                    raise ValueError("Inconsistent vertex placement; check quad orientations/orderings")
            else:
                vertex_rc[v] = rc

    if len(vertex_rc) != N:
        raise ValueError("Some vertices were not assigned logical grid coordinates")

    coarse_grid = -np.ones((H + 1, W + 1), dtype=int)
    for v, (r, c) in vertex_rc.items():
        if coarse_grid[r, c] != -1 and coarse_grid[r, c] != v:
            raise ValueError("Duplicate coarse-grid placement detected")
        coarse_grid[r, c] = v

    if np.any(coarse_grid < 0):
        raise ValueError("The mesh does not define a complete rectangular coarse grid")

    # ------------------------------------------------------------------
    # 2) Create fine-grid vertex indexing, keeping original vertices first
    # ------------------------------------------------------------------

    Hf = H * n
    Wf = W * m

    fine_idx = -np.ones((Hf + 1, Wf + 1), dtype=int)

    # Original coarse vertices stay at their original indices 0..N-1
    for r in range(H + 1):
        for c in range(W + 1):
            fine_idx[r * n, c * m] = coarse_grid[r, c]

    next_idx = N
    for r in range(Hf + 1):
        for c in range(Wf + 1):
            if fine_idx[r, c] == -1:
                fine_idx[r, c] = next_idx
                next_idx += 1

    Nf = next_idx

    # ------------------------------------------------------------------
    # 3) Build prolongation matrix S, so Xf = S @ X
    # ------------------------------------------------------------------

    S = lil_matrix((Nf, N), dtype=float)

    # Original vertices: identity rows
    for i in range(N):
        S[i, i] = 1.0

    def coarse_cell_and_local_coords(rf, cf):
        """
        For fine-grid logical coordinates (rf, cf), return
        coarse cell (r0, c0) and local bilinear coordinates (v,u) in [0,1].
        v = vertical local coordinate
        u = horizontal local coordinate
        """
        if rf == Hf:
            r0 = H - 1
            v = 1.0
        else:
            r0 = rf // n
            v = (rf - r0 * n) / n

        if cf == Wf:
            c0 = W - 1
            u = 1.0
        else:
            c0 = cf // m
            u = (cf - c0 * m) / m

        return r0, c0, v, u

    for rf in range(Hf + 1):
        for cf in range(Wf + 1):
            idx = fine_idx[rf, cf]

            # original vertices already set
            if idx < N:
                continue

            r0, c0, v, u = coarse_cell_and_local_coords(rf, cf)

            v00 = coarse_grid[r0,     c0]
            v10 = coarse_grid[r0,     c0 + 1]
            v11 = coarse_grid[r0 + 1, c0 + 1]
            v01 = coarse_grid[r0 + 1, c0]

            w00 = (1.0 - u) * (1.0 - v)
            w10 = u * (1.0 - v)
            w11 = u * v
            w01 = (1.0 - u) * v

            S[idx, v00] = w00
            S[idx, v10] = w10
            S[idx, v11] = w11
            S[idx, v01] = w01

    S = S.tocsr()

    # ------------------------------------------------------------------
    # 4) Build refined quad connectivity
    # ------------------------------------------------------------------

    Tf = []
    for r in range(Hf):
        for c in range(Wf):
            q = [
                fine_idx[r,     c],
                fine_idx[r,     c + 1],
                fine_idx[r + 1, c + 1],
                fine_idx[r + 1, c],
            ]
            Tf.append(q)

    Tf = np.asarray(Tf, dtype=int)
    return Tf, S

