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

