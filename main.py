import numpy as np
import matplotlib.pyplot as plt
import sys, os
from functions import *
import json
import warnings

def loadbar(counter,len):
    counter +=1
    done = (counter*100)//len
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % ('='*done, done))
    sys.stdout.flush()
    if counter == len:
        print('\n')

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

def find_angles(NAs, n_i, alpha_s, k0, num_slices, meshgrids, delta_k):
    """Calculating all angle based on NA, refractive indexec,
    image resolution, and wavenumber.

    Parameters
    ----------
    NAs : List
        List of lens NAs for lens 6, 5, and 4 respectively
    n_i : List
        List of refractive index 5, and 4 respectively
    alpha_s : float
        Angle between optical axes of the system
    k0 : float
        Wavenumber of the emission light
    num_slices : int
        Image resolution
    meshgrids : List
        meshgrid xx, yy, and R respectively
    delta_k : float
        k*NA/Magnification
    plot : boolean
        If true debugging plots will show

    Returns
    -------
    List
        Angles for position 4, 4s, 5s, and 6 in azimuth
        and polar respectively, and shift of angle.
    """
    NA_6, NA_5, NA_4 = NAs
    n_5, n_4 = n_i
    xx, yy, R = meshgrids

    phi_6 = np.real(np.arctan2(xx, yy))
    theta_6 = (np.arcsin(delta_k*R / k0 ))

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

    min = np.nanmin(theta_4)
    tmp = np.where(theta_4==min)
    diff = (tmp[1]-tmp[0])[0]

    phi = [phi_4, phi_4s, phi_5s, phi_6]
    theta = [theta_4, theta_4s, theta_5s, theta_6]

    angles = [phi, theta, diff]

    return angles

def find_E_fields(angles, Ae, p, alpha_s):
    """Calculating electric field in image space

    Parameters
    ----------
    angles : list
        Output of find_angles function
    Ae : float
        Anisotropy coeficient
    p : floating point vector
        Polarization of dipole
    alpha_s : float
        Angle between optical axes of the system
    plot : boolean
        If true debugging plots witt show

    Returns
    -------
    floating point array
        Full electric field in image space (not phase information)
    """
    phi, theta, diff = angles
    phi_4, phi_4s, phi_5s, phi_6 = phi
    theta_4, theta_4s, theta_5s, theta_6 = theta

    #Finding the initial electric field
    E4 = E_0(p, phi_4, theta_4, Ae)

    #Rotating the coordinate system to the second optical axis
    R_x_as = R_x(alpha_s)
    E4_s = np.einsum('ji,abi->abj', R_x_as, E4)

    #Defines the Rotation matrices before and after a refractive index change
    Rz_4 = R_z(phi_4s)
    Rz_5 = R_z(phi_5s)

    #Defining the Fresnel coefficients for the refracive index change to snouty
    t_p = 2 * np.cos(theta_4s)/(np.cos(theta_5s)+n_5*np.cos(theta_4s))
    t_s = 2 * np.cos(theta_4s)/(np.cos(theta_4s)+n_5*np.cos(theta_5s))
    Ft = Fresnel(t_p,t_s)

    #Rotation matrix in y before and after refractive index change
    R_ys_4 = R_y(theta_4s)
    R_ys_5 = R_y(theta_5s)

    #Inversing the rotation matrices in y and z after refractive index change
    Rz_5_inv = np.linalg.inv(np.transpose(Rz_5,(2,3,0,1))).transpose((2,3,0,1))
    R_ys_5_inv = np.linalg.inv(np.transpose(R_ys_5,(2,3,0,1))).transpose((2,3,0,1))

    #Makes the transformation matrix and dots with field 4s
    transform_5s = (Rz_5_inv * R_ys_5_inv * Ft * R_ys_4 * Rz_4).transpose((2,3,0,1))
    E5_s = np.einsum('abji,abi->abj', transform_5s, E4_s)

    #Finds the lens refraction of snouty lens
    L5 = L_refraction(theta_5s)

    #Lens apodisation for snouty
    A5 = np.sqrt(n_5) / np.sqrt(np.cos(theta_5s))

    #Makes the transformation matrix and dots with field 5s
    transform_5 = (A5 * Rz_5_inv * L5 * Rz_5).transpose((2,3,0,1))
    E5 = np.einsum('abji,abi->abj', transform_5, E5_s)

    #Defines rotation matrix in z and its inverse for lens 6
    Rz_6 = R_z(phi_6)
    Rz_6_inv = np.linalg.inv(np.transpose(Rz_6,(2,3,0,1))).transpose((2,3,0,1))

    #Finds the lens refraction of tube lens
    L6 = L_refraction(theta_6)

    #Lens apodization of tube lens
    A6 = np.sqrt(np.cos(theta_6))

    #Makes the transformation matrix and dots with field 5
    transform_6 = (A6 * Rz_6_inv * L6 * Rz_6).transpose((2,3,0,1))
    E6 = np.einsum('abji,abi->abj', transform_6, E5)

    #Replaces all NaN values with 0 in the final E_field
    E6 = np.nan_to_num(E6)

    return E6

def find_intensity(E_mat,num_slices,N,back_aperture_obliqueness,z_val,k_z):
    """Find intensity image from electric field

    Parameters
    ----------
    E_mat : floating point array
        Electric field in image space
    num_slices : int
        Image resolution
    N : int
        Padding size for fourier transform
    back_aperture_obliqueness : floating point array
        ???????????????????????????????????????????
    z_val : floating point vector
        z-values for all image planes
    k_z : floating point array
        ???????????????????????????????????????????

    Returns
    -------
    floating point array
        Image stack in floating point values
    """
    M = (num_slices-1)//2

    E_field = np.zeros((num_slices,num_slices,3,num_slices),dtype=np.complex128)

    for k_index in range(num_slices):
        E_field[:, :, :, k_index] = propagate(M, N, k_z, z_val[k_index], back_aperture_obliqueness, E_mat)

    intensity = np.sum(np.abs(E_field)**2,axis=2)

    return intensity

def calculate_image(k0,voxel_size,p,alpha_s,Ae,NAs,n_i,num_slices,wavelength):
    """Uses the microscope and dipole variables to calculate image stack

    Parameters
    ----------
    k0 : float
        Wavenumber of emission light
    voxel_size : float
        Size of pixels in camera (sampling size)
    p : floating point array
        Dipole polarization
    alpha_s : float
        Tilt of snouty microscope
    Ae : float
        Anisotropy
    NAs : list
        List of lens NAs in the microscope
    n_i : list
        List of refractive indexes in system
    num_slices : int
        Resolution of image stack

    Returns
    -------
    integer array
        3D image stack
    """
    M = (num_slices-1)/2

    #Defining a meshgrid for x, y, and R
    m = np.linspace(-M,M,num_slices)
    xx, yy = np.meshgrid(m,m)
    R = np.sqrt(xx**2 + yy**2)

    #Defining k for each position of lens
    delta_k = k0 * NA_6 / M
    k_xy = delta_k * R
    k_z = np.sqrt(k0**2 - k_xy**2)

    meshgrids = [xx, yy, R]

    #Finding the angles corresponding to position at objective lens
    warnings.filterwarnings('ignore')
    angles = find_angles(NAs, n_i, alpha_s, k0, num_slices, meshgrids, delta_k)
    warnings.resetwarnings()

    #Unpacking the angles
    phi, theta, diff = angles
    phi_4, phi_4s, phi_5s, phi_6 = phi
    theta_4, theta_4s, theta_5s, theta_6 = theta

    #Finding the electrical field after lens 6
    E6 = find_E_fields(angles, Ae, p, alpha_s)

    #???
    back_aperture_obliqueness = 1 / np.cos(theta_6)

    #Defining a pupil corresponding to lens 4
    pupil = (theta_4s<np.arcsin(NA_4/n_4))*1

    #Finding the shifted pupil
    circshift_pupil = np.roll(pupil,-diff,axis=0)
    zero_strip = np.ones_like(circshift_pupil)
    zero_strip[-diff:] = 0
    circshift_pupil *= zero_strip

    #Finding the snouty pupil
    pupil_s = (theta_5s<np.arcsin(NA_5/n_5))*1

    #Calculating the effective pupil area
    effective_aperture = (pupil_s*circshift_pupil).reshape(num_slices,num_slices,1)

    #Propogating through pupils
    E_mat = E6*effective_aperture

    #Finding the padding size for fourier transforms
    N = int(3*((np.floor(((wavelength*M/(NA_6*voxel_size))/3)/2))*2+1))

    #Defining minimum and maximum value of z
    z_max = ((num_slices-2)*voxel_size*Mag)/2
    z_min = -z_max

    #Finding z-layers of the image
    z_val = np.linspace(z_min,z_max,num_slices)

    #Finding the image of dipole based on electric field
    intensity = find_intensity(E_mat,num_slices,N,back_aperture_obliqueness,z_val,k_z)

    return intensity

def imaging(voxel_size,NAs,n_i,num_slices,center_wl,wl_bandwidth,l_p,tilt,anisotropy):
    """Function that randomizes variables for every iteration of imaging

    Parameters
    ----------
    voxel_size : float
        Pixel size of camera
    NAs : list of floats
        list of NA 6, 5, and 4 respectively
    n_i : list of floats
        list of n_5, and n_4
    num_slices : int
        Reconstruction size in pixels
    center_wl : float
        Average wavelength of light emission
    wl_bandwidth : float
        Variance of light emission
    l_p : floating point numpy array
        Lightsheet polarization in x, y, and z
    tilt : int
        Tilt of lightsheet in degrees
    anisotropy : float
        Anisotropy of the dipole

    Returns
    -------
    integer array
        3D image stack
    """
    #Fidning the wavenumber of the light
    wavelength = np.random.normal(center_wl,wl_bandwidth)*1e-9
    # wavelength = 500e-9 #nm
    k0 = 2 * np.pi / wavelength

    #Defining the polarization of the dipole
    # p_phi, p_theta = 0, 0
    p_phi = np.random.uniform(0,2*np.pi)
    p_theta = np.random.uniform(-np.pi/2,np.pi/2)
    p = np.array((np.sin(p_theta)*np.cos(p_phi),
                  np.sin(p_theta)*np.sin(p_phi),
                  np.cos(p_theta)))

    alpha_s = tilt

    if anisotropy == 0:
        Ae = 1
    elif anisotropy == 0.4:
        Ae = p@l_p

    #Calculating the image stack
    intensity = calculate_image(k0,voxel_size,p,alpha_s,Ae,NAs,n_i,num_slices,wavelength)

    return intensity

if __name__ == '__main__':
    center_wl = 500
    wl_bandwidth = 25

    #Defining voxel size of the camera
    voxel_size = 5e-6 #um

    #Defining the lens apertures
    NA_4 = 0.95
    NA_5 = 1
    Mag = 40
    NA_6 = NA_5/Mag
    NAs = [NA_6, NA_5, NA_4]

    #Defining refractive indexec
    n_4 = 1
    n_5 = 1.7
    n_i = [n_5, n_4]

    #Defining the resolution
    num_slices = 127

    timepoints = 100

    #Defining the light sheet parameters (tilt in x-axis)
    tilt = 30 # [ 0 / 20 / 30]
    #Defining anisotropy and exitation level
    anisotropy = 0 # [0  / 0.4]
    #Defining lightsheet polarization in x, y, and z
    lightsheet_polarization = 's' # ['p' / 's']

    for tilt in [0,20,30]:
        for anisotropy in [0,0.4]:
            for lightsheet_polarization in ['p','s']:

                dir_1 = 'tilt_{}_degrees'.format(tilt)+'__'
                dir_2 = 'anisotropy_{}'.format(anisotropy)+'__'
                dir_3 = 'lightsheet_polarization_'+lightsheet_polarization
                dir = dir_1+dir_2+dir_3

                inp = 'n'
                if dir in os.listdir('images/'):
                    print(dir+' already exists. Do you wish to replace the image? [y/n]')
                    while 1:
                        inp = input()
                        if inp == 'y':
                            print('Replacing existing image.\n')
                            break
                        elif inp == 'n':
                            print('Continuing to next image.\n')
                            break
                        else:
                            print('Command not recognized, try again.')

                if dir not in os.listdir('images/') or inp == 'y':

                    if lightsheet_polarization == 'p':
                        l_x = 0
                        l_y = 1
                        l_z = 0
                    elif lightsheet_polarization == 's':
                        l_x = np.sin(tilt)
                        l_y = 0
                        l_z = np.cos(tilt)
                    l_p = np.array((l_x,l_y,l_z))

                    image = np.zeros((num_slices,num_slices,num_slices))

                    for i in range(timepoints):
                        if i%(timepoints//100)==0:
                            loadbar(i,timepoints)
                        image += imaging(voxel_size,NAs,n_i,num_slices,center_wl,wl_bandwidth,l_p,tilt,anisotropy)

                    #Scaling the image to fit in 16-bit image
                    image /= np.amax(image)
                    img_16 = ((2**16-1)*image).astype(np.uint16)


                    if lightsheet_polarization == 'p':
                        lig_pol = 'Paralell'
                    elif lightsheet_polarization == 's':
                        lig_pol = 'Senkrechte'

                    data = {'Emission wavelength [nm]' : center_wl*1e9,
                            'Light sheet angle [degrees]' : tilt,
                            'Light sheet polarization' : lig_pol,
                            'Voxel size [microns]' : voxel_size*1e6,
                            'Anisotropy value' : anisotropy,
                            'Time points' : timepoints,
                            'Magnification' : Mag}

                    if dir not in os.listdir('images/'):
                        os.mkdir('images/'+dir)
                        os.mkdir('images/'+dir+'/'+'image')
                    with open('images/'+dir+'/'+"data.json", 'w') as output:
                        json.dump(data, output, indent=4)

                    #Saving the image stack
                    save_stack(img_16,'images/'+dir+'/'+'image/')
