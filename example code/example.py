import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from time import time
import sys
from multiprocessing import Pool
from functions import *
from physical_constants import *

def apodization_0(theta):
    return 1/(np.sqrt(np.cos(theta)))

def apodization_1(theta):
    return np.sqrt(np.cos(theta))

def theta_0(NA_0,NA_1,theta_1):
    return np.arcsin((NA_0/NA_1)*np.sin(theta_1))

def k_0(phi, NA_0, NA_1, theta_1):
    theta0 = theta_0(NA_0, NA_1, theta_1)
    return np.array((np.sin(theta0)*np.cos(phi), np.sin(theta0)*np.sin(phi), np.cos(theta0)))

def E_0(p, phi, NA_0, NA_1, theta_1):
    k0 = np.transpose(k_0(phi, NA_0, NA_1, theta_1),(1,2,0))
    return np.cross(np.cross(k0,p),k0)

def R_z(phi):
    return np.array(((np.cos(phi),np.sin(phi),np.zeros_like(phi)),
                     (-np.sin(phi),np.cos(phi),np.zeros_like(phi)),
                     (np.zeros_like(phi),np.zeros_like(phi),np.ones_like(phi))))

def L_0(theta):
    return np.array(((np.cos(-theta),np.zeros_like(theta),np.sin(-theta)),
                     (np.zeros_like(theta),np.ones_like(theta),np.zeros_like(theta)),
                     (-np.sin(-theta),np.zeros_like(theta),np.cos(-theta))))

def E_transform(phi, theta, apod, E0):
    a,b = np.shape(apod)
    transform = apod.reshape(a,b,1,1)*np.linalg.inv(np.transpose(R_z(phi),(2,3,0,1)))@np.transpose(L_0(theta),(2,3,0,1))@np.transpose(R_z(phi),(2,3,0,1))
    return np.einsum('abji,abi->abj', transform, E0)

def find_intensity(E_mat,MM,num_slices,N,M,back_aperture_obliqueness,z_val,k_z):
    E_field = np.zeros((MM,MM,3,num_slices),dtype=np.complex128)

    for k_index in range(num_slices):
        if k_index%(num_slices//100)==0:
            done = (k_index*100)//num_slices+1
            sys.stdout.write('\r')
            sys.stdout.write("[%-100s] %d%%" % ('='*done, done))
            sys.stdout.flush()

        E_field[:, :, :, k_index] = propagate(M, N, k_z, z_val[k_index], back_aperture_obliqueness, E_mat)

    intensity = np.sum(np.abs(E_field)**2,axis=2)

    sys.stdout.write('\n')

    return intensity

# def find_field(MM, num_slices, m, M, k0, NA_tube_lens, z_max):
if __name__ == '__main__':
    pupil = (R<M)*1
    circshift_pupil = np.roll(pupil,M//2,axis=1)

    E_mat = np.zeros((MM,MM,3))

    E0 = E_0(p, PHI, NA_objective, NA_tube_lens, THETAi)

    theta0 = theta_0(NA_objective, NA_tube_lens, THETAi)
    apod_0 = apodization_0(theta0)
    Ef = E_transform(PHI, theta0, apod_0, E0)

    apod_1 = apodization_1(THETAi)
    Ei = E_transform(PHI, THETAi, apod_1, Ef)

    E_mat = Ei*pupil.reshape(MM,MM,1)*circshift_pupil.reshape(MM,MM,1)

    back_aperture_obliqueness = 1 / np.cos(THETAi)
    plt.imshow(back_aperture_obliqueness)
    plt.show()
    exit()

    N = int(3 * (( np.floor((( wavelength * M / (NA_tube_lens * voxel_size)) / 3) / 2)) * 2 + 1))
    intensity = find_intensity(E_mat,MM,num_slices,N,M,back_aperture_obliqueness,z_val,k_z)

    intensity /= np.amax(intensity)
    img_16 = ((2**16-1)*intensity).astype(np.uint16)

    save_stack(img_16,'image/')
