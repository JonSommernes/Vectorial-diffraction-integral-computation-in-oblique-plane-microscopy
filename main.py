import numpy as np
import matplotlib.pyplot as plt
from time import time
import sys
from functions import *
from scipy.io import savemat

def R_x(alpha):
    """Making the coordinate rotation matrix for clockwise x-rotation

    Parameters
    ----------
    alpha : floating point array
        Rotation angle in radians

    Returns
    -------
    floating point array
        Complete rotation matrix
    """
    zero = np.zeros_like(alpha)
    one = np.ones_like(alpha)
    return np.array(((np.cos(alpha), zero, np.sin(alpha)),
                    (zero, one, zero),
                    (-np.sin(alpha), zero, np.cos(alpha))))

def R_y(alpha):
    """Making the coordinate rotation matrix for clockwise y-rotation

    Parameters
    ----------
    alpha : floating point array
        Rotation angle in radians

    Returns
    -------
    floating point array
        Complete rotation matrix
    """
    zero = np.zeros_like(alpha)
    one = np.ones_like(alpha)
    return np.array(((np.cos(alpha), zero, -np.sin(alpha)),
                     (zero, one, zero),
                     (np.sin(alpha), zero, np.cos(alpha))))

def R_z(alpha):
    """Making the coordinate rotation matrix for clockwise z-rotation

    Parameters
    ----------
    alpha : floating point array
        Rotation angle in radians

    Returns
    -------
    floating point array
        Complete rotation matrix
    """
    zero = np.zeros_like(alpha)
    one = np.ones_like(alpha)
    return np.array(((np.cos(alpha), np.sin(alpha), zero),
                     (-np.sin(alpha), np.cos(alpha), zero),
                     (zero, zero, one)))

def Fresnel(tp,ts):
    """Making the Fresnel transmission matrix

    Parameters
    ----------
    tp : type
        Parallel transmission coefficient
    ts : float
        Sagittal transmission coefficient

    Returns
    -------
    floating point array
        Complete Fresnel matrix
    """
    zero = np.zeros_like(tp)
    one = np.ones_like(tp)
    return np.array(((tp, zero, zero),
                     (zero, ts, zero),
                     (zero, zero, one)))

def L_refraction(theta):
    """Making the ray refraction matrix for the meridional plane

    Parameters
    ----------
    theta : floating point array
        Refraction angle in radians

    Returns
    -------
    floating point array
        Complete refraction matrix
    """
    zero = np.zeros_like(theta)
    one = np.ones_like(theta)
    return np.array(((np.cos(-theta),zero,np.sin(-theta)),
                     (zero,one,zero),
                     (-np.sin(-theta),zero,np.cos(-theta))))

def k_0(phi, theta):
    """Generating x-, y-, and z-component of k0 based on lens position

    Parameters
    ----------
    phi : floating point array
        Azimuthal angle on lens
    theta : floating point array
        Polar angle on lens

    Returns
    -------
    floating point array
        k0 in Cartesian coordinates
    """
    return np.array((np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(theta)))

def E_0(p, phi, theta, Ae):
    """Generating the initial electric field based
       on dipole orientation and anisotropy

    Parameters
    ----------
    p : floating point array
        Polarization of dipole in Cartesian coordinates
    phi : floating point array
        Lens azimuth angle
    theta : floating point array
        Lens polar angle
    Ae : float
        Anisotropy coefficient

    Returns
    -------
    floating point array
        Initial electric field
    """
    k0 = np.transpose(k_0(phi,theta),(1,2,0))
    return Ae*np.cross(np.cross(k0,p),k0)

def find_angles(NA_6, NA_5, NA_4, n_4, n_5, alpha_s, k0, num_slices, xx, yy, R, delta_k):
    phi_6 = np.real(np.arctan2(xx, yy))
    theta_6 = (np.arcsin(delta_k*R / k0 ))
    # theta_6 = (theta_6 > np.arcsin(NA_6))*np.arcsin(NA_6) + (theta_6 < np.arcsin(NA_6))*theta_6


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

    k4[2] = -k4[2]

    phi_4 = (np.arctan2(k4[1], k4[0]))
    theta_4 = np.arctan2(np.sqrt(k4[0]**2+k4[1]**2),k4[2])
    # theta_4 = np.arccos(k4[2]/np.sqrt(k4[0]**2+k4[1]**2+k4[2]**2))
    # theta_4 = (np.arcsin(np.sqrt(k4[0]**2+k4[1]**2)))
    # theta_4 = (theta_4 > np.arcsin(NA_4))*np.arcsin(NA_4) + (theta_4 < np.arcsin(NA_4))*theta_4

    # plt.subplot(131)
    # plt.imshow(np.abs(k4_s[0]))
    # plt.subplot(132)
    # plt.imshow(np.abs(k4_s[1]))
    # plt.subplot(133)
    # plt.imshow(k4_s[2])
    # plt.show()
    #
    # plt.subplot(131)
    # plt.imshow(np.abs(k4[0]))
    # plt.subplot(132)
    # plt.imshow(np.abs(k4[1]))
    # plt.subplot(133)
    # plt.imshow(k4[2])
    # plt.show()
    #
    min = np.nanmin(theta_4)
    tmp = np.where(theta_4==min)
    diff = (tmp[1]-tmp[0])[0]
    plt.subplot(141)
    plt.imshow(theta_6,cmap='inferno')
    plt.colorbar()
    plt.subplot(142)
    plt.imshow(theta_5s,cmap='inferno')
    plt.colorbar()
    plt.subplot(143)
    plt.imshow(theta_4s,cmap='inferno')
    plt.colorbar()
    plt.subplot(144)
    plt.imshow(theta_4,cmap='inferno')
    plt.colorbar()
    plt.show()
    #
    # plt.subplot(141)
    # plt.imshow(phi_6,cmap='inferno')
    # plt.colorbar()
    # plt.subplot(142)
    # plt.imshow(phi_5s,cmap='inferno')
    # plt.colorbar()
    # plt.subplot(143)
    # plt.imshow(phi_4s,cmap='inferno')
    # plt.colorbar()
    # plt.subplot(144)
    # plt.imshow(phi_4,cmap='inferno')
    # plt.colorbar()
    # plt.show()

    angles = [phi_4, phi_4s, phi_5s, phi_6,
              theta_4, theta_4s, theta_5s, theta_6, diff]

    return angles

def find_E_fields(angles, Ae, p, alpha_s):
    phi_4, phi_4s, phi_5s, phi_6, theta_4, theta_4s, theta_5s, theta_6, diff = angles

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
    E6 = np.nan_to_num(E6)

    # plt.subplot(131)
    # plt.imshow(E4[:,:,0],cmap='inferno')
    # plt.colorbar()
    # plt.subplot(132)
    # plt.imshow(E4[:,:,1],cmap='inferno')
    # plt.colorbar()
    # plt.subplot(133)
    # plt.imshow(E4[:,:,2],cmap='inferno')
    # plt.colorbar()
    # plt.show()
    #
    # plt.subplot(131)
    # plt.imshow(E5[:,:,0],cmap='inferno')
    # plt.colorbar()
    # plt.subplot(132)
    # plt.imshow(E5[:,:,1],cmap='inferno')
    # plt.colorbar()
    # plt.subplot(133)
    # plt.imshow(E5[:,:,2],cmap='inferno')
    # plt.colorbar()
    # plt.show()
    #
    # plt.subplot(131)
    # plt.imshow(E6[:,:,0],cmap='inferno')
    # plt.colorbar()
    # plt.subplot(132)
    # plt.imshow(E6[:,:,1],cmap='inferno')
    # plt.colorbar()
    # plt.subplot(133)
    # plt.imshow(E6[:,:,2],cmap='inferno')
    # plt.colorbar()
    # plt.show()

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
    #Defining the polarization of the dipole
    p_phi, p_theta = 0, 0
    p = np.array((np.sin(p_theta)*np.cos(p_phi),
                  np.sin(p_theta)*np.sin(p_phi),
                  np.cos(p_theta)))

    #Defining the light sheet parameters
    tilt = 30 # [ 0 / 20 / 30]
    anisotropy = 0 # [0  / 0.4]
    lightsheet_polarization = 'p' # ['p' / 's']
    if anisotropy == 0:
        Ae = 1
    elif anisotropy == 0.4:
        Ae = p@P

    #Optical axis unit vectors
    alpha_s = 90-tilt
    O_1 = np.array((0,0,1))
    O_2 = np.array((0,np.sin(alpha_s),np.cos(alpha_s)))

    #Fidning the wavenumber of the light
    wavelength = 500e-9
    k0 = 2 * np.pi / wavelength

    #Defining voxel size of the camera
    voxel_size = 5e-6 # um

    #Defining the lens apertures
    NA_1 = 1.27
    NA_4 = 0.95
    NA_5 = 1
    Mag = 40
    NA_6 = NA_5/Mag

    #Defining refractive indexec
    n_1 = 1.33
    n_4 = 1
    n_5 = 1.7

    #Defining the resolution and half pixel length
    num_slices = 127
    M = (num_slices-1)/2

    #Defining a meshgrid for x, y, and R
    m = np.linspace(-M,M,num_slices)
    xx, yy = np.meshgrid(m,m)
    R = np.sqrt(xx**2 + yy**2)

    #Defining k for each position of lens
    delta_k = k0 * NA_6 / M
    k_xy = delta_k * R
    k_z = np.sqrt(k0**2 - k_xy**2)

    #Finding the angles corresponding to position at objective lens
    angles = find_angles(NA_6, NA_5, NA_4, n_4, n_5, alpha_s, k0, num_slices, xx, yy, R, delta_k)
    phi_4, phi_4s, phi_5s, phi_6, theta_4, theta_4s, theta_5s, theta_6, diff = angles

    #Finding the electrical field after lens 6
    E6 = find_E_fields(angles, Ae, p, alpha_s)

    #???
    back_aperture_obliqueness = 1 / np.cos(theta_6)



    #Defining a pupil corresponding to lens 4
    pupil = (theta_4s<np.arcsin(NA_4/n_4))*1

    #Finding a shift corresponding to lens tilt
    shift = int(tilt/90*M)

    #Finding the pupil corresponding to lens 5
    circshift_pupil = np.roll(pupil,-diff,axis=0)
    zero_strip = np.ones_like(circshift_pupil)
    zero_strip[-diff:] = 0

    circshift_pupil *= zero_strip

    pupil_2 = (theta_5s<np.arcsin(NA_5/n_5))*1


    effective_aperture = (pupil_2*circshift_pupil).reshape(num_slices,num_slices,1)

    #Propogating through pupils
    E_mat = E6*effective_aperture

    plt.imshow(E_mat[:,:,0])
    plt.show()

    #Finding the padding size for fourier transforms
    N = int(3*((np.floor(((wavelength*M/(NA_6*voxel_size))/3)/2))*2+1))

    #Defining minimum and maximum value of z
    z_max = ((num_slices-2)*voxel_size*Mag)/2
    z_min = -z_max

    #Finding z-layers of the image
    z_val = np.linspace(z_min,z_max,num_slices)

    #Finding the image of dipole based on electric field
    intensity = find_intensity(E_mat,num_slices,N,back_aperture_obliqueness,z_val,k_z)

    #Scaling the image to fit in 16-bit image
    intensity /= np.amax(intensity)
    img_16 = ((2**16-1)*intensity).astype(np.uint16)

    #Finding the XZ-cross section of the image and gamma transforming it
    XZ = np.log(img_16[num_slices//2,:,:])
    cont_XZ = XZ/XZ.max()
    gamm_XZ = cont_XZ**4

    #Plotting the cross section
    plt.imshow(gamm_XZ)
    plt.colorbar()
    plt.show()

    #Finding the YZ-cross section of the image and gamma transforming it
    YZ = np.log(img_16[:,num_slices//2,:])
    cont_YZ = YZ/YZ.max()
    gamm_YZ = cont_YZ**4

    #Plotting the cross section
    plt.imshow(gamm_YZ)
    plt.colorbar()
    plt.show()

    #Saving the image stack
    save_stack(img_16,'image/')
