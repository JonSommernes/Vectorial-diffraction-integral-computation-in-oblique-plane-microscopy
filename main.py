import numpy as np
import matplotlib.pyplot as plt
from time import time
import sys
from functions import *
from scipy.io import savemat

def R_x(alpha):
    return np.array(((np.cos(alpha), np.zeros_like(alpha), np.sin(alpha)),
                    (np.zeros_like(alpha), np.ones_like(alpha), np.zeros_like(alpha)),
                    (-np.sin(alpha), np.zeros_like(alpha), np.cos(alpha))))

def R_y(theta):
    return np.array(((np.cos(theta), np.zeros_like(theta), -np.sin(theta)),
                     (np.zeros_like(theta), np.ones_like(theta), np.zeros_like(theta)),
                     (np.sin(theta), np.zeros_like(theta), np.cos(theta))))

def R_z(phi):
    return np.array(((np.cos(phi), np.sin(phi), np.zeros_like(phi)),
                     (-np.sin(phi), np.cos(phi), np.zeros_like(phi)),
                     (np.zeros_like(phi), np.zeros_like(phi), np.ones_like(phi))))

def Fresnel(tp,ts):
    return np.array(((tp, np.zeros_like(tp), np.zeros_like(tp)),
                     (np.zeros_like(tp), ts, np.zeros_like(tp)),
                     (np.zeros_like(tp), np.zeros_like(tp), np.ones_like(tp))))

def L_refraction(theta):
    return np.array(((np.cos(-theta),np.zeros_like(theta),np.sin(-theta)),
                     (np.zeros_like(theta),np.ones_like(theta),np.zeros_like(theta)),
                     (-np.sin(-theta),np.zeros_like(theta),np.cos(-theta))))

def k_0(phi, theta):
    return np.array((np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(theta)))

def E_0(p, phi, theta, Ae):
    k0 = np.transpose(k_0(phi,theta),(1,2,0))
    return Ae*np.cross(np.cross(k0,p),k0)

def find_angles(NA_6, NA_5, NA_4, n_4, n_5, alpha_s, k0, num_slices):
    M = (num_slices-1)/2
    m = np.linspace(-M,M,num_slices)
    xx, yy = np.meshgrid(m,m)
    R = np.sqrt(xx**2 + yy**2)
    delta_k = k0 * NA_6 / M

    phi_6 = np.real(np.arctan2(xx, yy))
    theta_6 = (np.arcsin(delta_k*R / k0 ))
    theta_6 = (theta_6 > np.arcsin(NA_6))*np.arcsin(NA_6) + (theta_6 < np.arcsin(NA_6))*theta_6


    phi_5s = phi_6
    theta_5s = np.arcsin((NA_5/(n_5*NA_6))*np.sin(theta_6))

    phi_4s = phi_5s
    theta_4s = np.arcsin(n_5*np.sin(theta_5s)/n_4)


    R_x_as = R_x(alpha_s)

    k4_s = np.array((np.sin(theta_4s) * np.cos(phi_4s),
                     np.sin(theta_4s) * np.sin(phi_4s),
                     np.cos(theta_4s)))


    k4 = np.zeros_like(k4_s)
    for i in range(num_slices):
        k4[:,:,i] = R_x_as@k4_s[:,:,i]

    phi_4 = (np.arctan2(k4[1], k4[0]))
    theta_4 = np.arctan2(np.sqrt(k4[0]**2+k4[1]**2),k4[2])
    # theta_4 = np.arccos(k4[2]/np.sqrt(k4[0]**2+k4[1]**2+k4[2]**2))
    # theta_4 = (np.arcsin(np.sqrt(k4[0]**2+k4[1]**2)))
    # theta_4 = (theta_4 > np.arcsin(NA_4))*np.arcsin(NA_4) + (theta_4 < np.arcsin(NA_4))*theta_4


    # plt.subplot(211)
    # plt.imshow(theta_4)
    # plt.subplot(212)
    # plt.imshow(theta_4s)
    plt.subplot(221)
    plt.imshow(k4[0])
    plt.subplot(222)
    plt.imshow(k4[1])
    plt.subplot(223)
    plt.imshow(k4[2])
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()
    plt.imshow(phi_4)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    angles = [phi_4, phi_4s, phi_5s, phi_6,
              theta_4, theta_4s, theta_5s, theta_6]

    return angles

def find_angles_test(NA_6, NA_5, NA_4, n_4, n_5, alpha_s, k0, num_slices):
    M = (num_slices-1)/2
    m = np.linspace(-M,M,num_slices)
    xx, yy = np.meshgrid(m,m)
    R = np.sqrt(xx**2 + yy**2)
    delta_k = k0 * NA_4 / M

    phi_4 = np.real(np.arctan2(xx, yy))
    theta_4 = (np.arcsin(delta_k*R / k0 ))
    theta_4 = np.nan_to_num(theta_4, nan=np.nanmax(theta_4))

    print(np.amax(theta_4))

    # k4_s = np.array((np.sin(theta_4s) * np.cos(phi_4s),
    #                  np.sin(theta_4s) * np.sin(phi_4s),
    #                  np.cos(theta_4s)))
    #
    # k4 = np.zeros_like(k4_s)
    # for i in range(num_slices):
    #     k4[:,:,i] = R_x_as@k4_s[:,:,i]
    #
    # R_x_as = R_x(alpha_s)
    #
    # phi_4s = phi_5s
    # theta_4s = np.arcsin(n_5*np.sin(theta_5s)/n_4)




    plt.imshow(theta_4)
    plt.show()
    exit()

def find_E_fields(angles, Ae, p, alpha_s):
    phi_4, phi_4s, phi_5s, phi_6, theta_4, theta_4s, theta_5s, theta_6 = angles

    E4 = E_0(p, phi_4, theta_4, Ae)

    R_x_as = R_x(alpha_s)
    E4_s = np.einsum('ji,abi->abj', R_x_as, E4)

    Rz_4 = R_z(phi_4s)

    Rz_5 = R_z(phi_5s)

    t_p = 2 * np.cos(theta_4s) / (np.cos(theta_5s) + n_5 * np.cos(theta_4s))
    t_s = 2 * np.cos(theta_4s) / (np.cos(theta_4s) + n_5 * np.cos(theta_5s))

    # Fresnel matrix
    Ft = Fresnel(t_p,t_s)

    R_ys_4 = R_y(theta_4s)

    # Rotation matrix from meridional/saggital to S/P
    R_ys_5 = R_y(theta_5s)

    # Jones matrix calculus of Fresnel-refraction
    Rz_5_inv = np.linalg.inv(np.transpose(Rz_5,(2,3,0,1))).transpose((2,3,0,1))
    R_ys_5_inv = np.linalg.inv(np.transpose(R_ys_5,(2,3,0,1))).transpose((2,3,0,1))
    transform_5s = (Rz_5_inv * R_ys_5_inv * Ft * R_ys_4 * Rz_4).transpose((2,3,0,1))
    E5_s = np.einsum('abji,abi->abj', transform_5s, E4_s)

    L5 = L_refraction(theta_5s)

    # Lens apodisation to obey the Abbe sine condition in focussing
    A5 = np.sqrt(n_5) / np.sqrt(np.cos(theta_5s))

    transform_5 = (A5 * Rz_5_inv * L5 * Rz_5).transpose((2,3,0,1))
    E5 = np.einsum('abji,abi->abj', transform_5, E5_s)

    Rz_6 = R_z(phi_6)
    Rz_6_inv = np.linalg.inv(np.transpose(Rz_6,(2,3,0,1))).transpose((2,3,0,1))

    # Lens refraction - the tube lens refracts clock-wise, so theta needs to be
    # with negative sign
    L6 = L_refraction(theta_6)

    # Lens apodisation to obey the Abbe sine condition in focussing
    A6 = np.sqrt(np.cos(theta_6))

    transform_6 = (A6 * Rz_6_inv * L6 * Rz_6).transpose((2,3,0,1))
    E6 = np.einsum('abji,abi->abj', transform_6, E5)

    return E6

def find_intensity(E_mat,num_slices,N,back_aperture_obliqueness,z_val,k_z):
    M = (num_slices-1)//2

    E_field = np.zeros((num_slices,num_slices,3,num_slices),dtype=np.complex128)

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

if __name__ == '__main__':
    p_phi, p_theta = 0, 0
    p = np.array((np.sin(p_theta)*np.cos(p_phi), np.sin(p_theta)*np.sin(p_phi), np.cos(p_theta)))

    anisotropy = 0 # [0  / 0.4]
    if anisotropy == 0:
        Ae = 1
    elif anisotropy == 0.4:
        Ae = p@P
    tilt = 30 # [ 0 / 20 / 30]
    lightsheet_polarization = 'p' # ['p' / 's']

    #Optical axis unit vectors
    alpha_s = 90-tilt
    O_1 = np.array((0,0,1))
    O_2 = np.array((0,np.sin(alpha_s),np.cos(alpha_s)))

    wavelength = 500e-9
    k0 = 2 * np.pi / wavelength

    NA_1 = 1.27
    NA_4 = 1
    NA_5 = 1
    Mag = 40
    NA_6 = NA_5/Mag

    n_1 = 1.33
    n_4 = 1
    n_5 = 1.7

    num_slices = 127
    M = (num_slices-1)/2
    angles = find_angles(NA_6, NA_5, NA_4, n_4, n_5, alpha_s, k0, num_slices)
    phi_4, phi_4s, phi_5s, phi_6, theta_4, theta_4s, theta_5s, theta_6 = angles

    E6 = find_E_fields(angles, Ae, p, alpha_s)


    back_aperture_obliqueness = 1 / np.cos(theta_6)

    m = np.linspace(-M,M,num_slices)
    xx, yy = np.meshgrid(m,m)
    delta_k = k0 * NA_6 / M
    R = np.sqrt(xx**2 + yy**2)
    k_xy = delta_k * R
    k_z = np.sqrt(k0**2 - k_xy**2)


    pupil = (theta_5s<np.arcsin(NA_5/n_5))*1

    shift = int(tilt/90*M)

    circshift_pupil = np.roll(pupil,shift,axis=1)
    pupil_prod = pupil*circshift_pupil

    #Do struff after this correct
    # E_mat = E6
    E_mat = E6*pupil_prod.reshape(num_slices,num_slices,1)

    voxel_size = 1e-6 # um
    N = int(3 * (( np.floor((( wavelength * M / (NA_6 * voxel_size)) / 3) / 2)) * 2 + 1))

    z_max = ((num_slices - 2) * voxel_size * Mag)/2
    z_min = -z_max
    z_val = np.linspace(z_min,z_max,num_slices)

    intensity = find_intensity(E_mat,num_slices,N,back_aperture_obliqueness,z_val,k_z)

    intensity /= np.amax(intensity)
    img_16 = ((2**16-1)*intensity).astype(np.uint16)

    XZ = np.log(img_16[num_slices//2,:,:])

    cont_im = XZ/XZ.max()
    gamm_im = cont_im**4

    plt.imshow(gamm_im)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    YZ = np.log(img_16[:,num_slices//2,:])

    cont_im = YZ/YZ.max()
    gamm_im = cont_im**4

    plt.imshow(gamm_im)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()


    save_stack(img_16,'image/')
