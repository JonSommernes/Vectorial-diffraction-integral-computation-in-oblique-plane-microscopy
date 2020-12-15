import numpy as np
from PIL import Image

def paddedfft2(M, N, matrix):
    tmp = int((N-2*M)//2)

    matrix_fft = np.pad(matrix,((tmp,tmp),(0,0)))
    matrix_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(matrix_fft),axis=0))
    matrix_fft = matrix_fft[tmp :tmp + 1 + 2*M,:]

    matrix_fft2 = np.pad(matrix_fft,((0,0),(tmp,tmp)))
    matrix_fft2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(matrix_fft2),axis=1))
    matrix_fft2 = matrix_fft2[:,tmp :tmp + 1 + 2*M]
    return matrix_fft2

def propagate(M, N, kz, z, back_aperture_obliqueness, back_aperture_field):
    field_x = np.exp(1j * kz * z) * back_aperture_obliqueness * back_aperture_field[:, :, 0]
    field_y = np.exp(1j * kz * z) * back_aperture_obliqueness * back_aperture_field[:, :, 1]
    field_z = np.exp(1j * kz * z) * back_aperture_obliqueness * back_aperture_field[:, :, 2]

    field_x = paddedfft2(M, N, field_x)
    field_y = paddedfft2(M, N, field_y)
    field_z = paddedfft2(M, N, field_z)

    electric_field = np.zeros((field_x.shape[0], field_x.shape[1], 3),dtype=np.complex128)
    electric_field[:, :, 0] = field_x
    electric_field[:, :, 1] = field_y
    electric_field[:, :, 2] = field_z

    return electric_field

def save_stack(stack,path):
    num_slices = stack.shape[2]
    for i in range(num_slices):
          im = Image.fromarray(stack[:,:,i].astype(stack.dtype))
          im.save(path+'{}.tiff'.format(i))
