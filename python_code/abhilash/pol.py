import numpy as np
import polyscope as ps

ps.init()

unit_verts = np.array([
    [0, 0, 0], [1/2, 0, 0], [1/2, 1/2, 0], [0, 1/2, 0],
    [0, 0, 1], [1/2, 0, 1], [1/2, 1/2, 1], [0, 1/2, 1]
])

# Connectivity for ONE cube
cube_cell = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])

# Create the Left and Right versions using vector broadcasting
verts_left  = unit_verts + np.array([-2, 0, 0])  # Shift left by 2 units
verts_right = unit_verts + np.array([ 1, 0, 0])  # Shift right by 1 unit

all_verts = np.vstack([verts_left, verts_right])
all_cells = np.vstack([cube_cell, cube_cell + 8])

ps_vol = ps.register_volume_mesh("Dual Cubes", all_verts, mixed_cells=all_cells)

# ps_vol.set_enabled(False) # disable
# ps_vol.set_enabled() # default is true

# ps_vol.set_color((0.3, 0.6, 0.8)) # rgb triple on [0,1]
# ps_vol.set_interior_color((0.4, 0.7, 0.9))
# ps_vol.set_edge_color((0.8, 0.8, 0.8)) 
# ps_vol.set_edge_width(1.0)
# ps_vol.set_material("wax")
# ps_vol.set_transparency(0.5)

# # alternately:
# ps.register_volume_mesh("test volume mesh 2", verts, mixed_cells=cells, enabled=False, 
#                          color=(1., 0., 0.), interior_color=(0., 1., 0.),
#                          edge_color=((0.8, 0.8, 0.8)), edge_width=1.0, 
#                          material='candy', transparency=0.5)

# ps.show()

ps_vol.set_color((0.3, 0.6, 0.8))
ps_vol.set_edge_width(1.0)
ps_vol.set_material("wax")

ps.show()