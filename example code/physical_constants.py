import numpy as np

# physical parameters
NA_objective = 0.95
Mag = 40
NA_tube_lens = NA_objective/Mag
wavelength = 500e-9
k0 = 2 * np.pi / wavelength

#Dipole polarization
p_phi, p_theta = 0, 0
p = np.array((np.sin(p_theta)*np.cos(p_phi), np.sin(p_theta)*np.sin(p_phi), np.cos(p_theta)))


M = 63
MM = 2 * M + 1
num_slices = MM
m = np.linspace(-M,M,num_slices)

xx, yy = np.meshgrid(m,m)
delta_k = k0 * NA_tube_lens / M
R = np.sqrt(xx**2 + yy**2)
THETAi = (np.arcsin(delta_k*R / k0 ))
THETAi = (THETAi > np.arcsin(NA_tube_lens))*np.arcsin(NA_tube_lens) + (THETAi < np.arcsin(NA_tube_lens))*THETAi
PHI = np.real(np.arctan2(xx, yy))


k_xy = delta_k * R
k_z = np.sqrt(k0**2 - k_xy**2)
voxel_size = 1e-6 # um
z_max = (num_slices - 2) * voxel_size * Mag
z_max = z_max / 2
z_min = -z_max
z_val = np.linspace(z_min,z_max,num_slices)
