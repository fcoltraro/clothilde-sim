import polyscope as ps
import numpy as np

ps.init()
pts = np.array([[0,0,0],[1,0,0],[0,1,0]])
ps.register_point_cloud("p", pts)
ps.show()