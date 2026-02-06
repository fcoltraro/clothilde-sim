import numpy as np

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
        return h*(1-x**2) #avoid singular case
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

