# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_blurring.ipynb (unless otherwise specified).

__all__ = ['hanser_defocus', 'hanserDefocus']

# Cell
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
import albumentations as albu
import numpy as np
import random

# Cell
# TODO: accept non-square images as input
# TODO: check all this, it has been ages. especially with different input,
# sizes, that sampling frequency stuff changes a lot and I had just hacked it
# TODO: do these in pytorch. at the time they didn't have imaginary support
# but now they do (I think?)

# TODO: get rid of the z here, it is just for debugging
def hanser_defocus(target, scale=None, z = None, p_identity = 0.01, crop = 'random'):
    # TODO: remove the cropping, it is to ensure hanser defocus gets
    # square images as input but the _blur_function() should be able
    # to handle non-square anyway
    min_side = min(target.shape[:2])
    if crop == 'random':
        random_crop = albu.RandomCrop(min_side, min_side)
        target = random_crop(image=target)['image']
    elif crop == 'center':
        center_crop = albu.CenterCrop(min_side, min_side)
        target = center_crop(image=target)['image']
#     scale = 1#480.0/min_side
    if random.random() > p_identity:
        input_, z = _blur_function(np.float64(target)/255.0, scale=scale, z=z)
        # why omit the border pixels? because if you do augmentations later,
        # the fft artifacts at the borders may happen to be affine transformed to the center
        # of the image.

        return np.uint8(np.ascontiguousarray(input_*255.0))[2:-2,2:-2], np.ascontiguousarray(target)[2:-2,2:-2]
    else:
        return target[1:-1,1:-1], target[2:-2,2:-2]

def _blur_function(image, scale, z=None):
    if z is None:
        z = random.random()*scale
    imageHanserDefocus = hanserDefocus(image, z, numPixels=image.shape[0])
    return imageHanserDefocus, z

# TODO: rename this, make it clear that we are not to import this but the hanser_defocus function
def hanserDefocus(image, z=0, numPixels = 96):

    lambda_     = 1e-6              # wavelength
    k           = 2*np.pi/lambda_
    L           = 0.1               # Size of the calculation plane
    pixelSize   = L/numPixels

    # Forget the y coordinate, just work with square images

    # Create the spatial domain coordinates
    x = np.arange(-(numPixels/2), (numPixels/2))*pixelSize

    # Sampling period, i.e., distance between two sample points in the spatial domain
    dx = x[1] - x[0]
    # Sampling period BUT in the frequency domain, ~ equal to 1/L
    df = 1/(numPixels*dx)

    # Create the frequency domain coordinates
    fx = np.arange(-(numPixels/2), (numPixels/2))*df

    def ft2(g, dx):
        G = fftshift(fft2(ifftshift(g))) * dx**2
        return G

    def ift2(G, df):
        # Instead of passing numPixels as a parameter, just read it from the input size
        numPixels = G.shape[0]
        # Note that (df*numPixels) is equal to 1/dx
        g = fftshift(ifft2(ifftshift(G))) * (df*numPixels)**2
        return g

    # While we are at it, let's also implement the convolution theorem
    def conv2(g1, g2, dx):
        # Switch to frequency domain and multiply, a.k.a. convolution theorem
        G1 = ft2(g1, dx)
        G2 = ft2(g2, dx)
        G_out = G1*G2
        # Instead of passing numPixels as a parameter, just read it from the input size
        numPixels = g1.shape[0]
        # Switch back to the spatial domain
        # Note that 1/(numPixels*dx) is equal to df (or 1/L)
        g_out = ift2(G_out, 1/(numPixels*dx))
        return g_out

    def createPupil(L, numPixels, pupilRadius):
        # Create a mask, where we have 1s inside a circular aperture and 0s outside
        W, H            = np.meshgrid(np.linspace(-L/2, L/2, num=numPixels), np.linspace(-L/2, L/2, num=numPixels))
        pupilMask       = np.sqrt(W**2 + H**2) <= pupilRadius

        # Not necessary but for completeness: Our aperture is just a hole, it does not change the phase of the wavefront
        pupil = pupilMask + 0j
        # Calculate the intensity
        I_spatial = (np.abs(pupil)**2).sum()*dx*dx
        # normalize it so that its total power is 1
        pupil = pupil * np.sqrt(1 / I_spatial)

        return pupil

    pupilRadius = 0.1
    FX,FY = np.meshgrid(fx,fx)

    phaseAngle   = 1j * z * 2*np.pi * np.sqrt((1 / lambda_)**2 - FX**2 - FY**2)
    defocusTerm  = np.exp(phaseAngle)

    pupil = createPupil(L, numPixels, pupilRadius)
    h = ft2(pupil* defocusTerm, dx)
    psf = np.abs(h)**2

    imageHanserDefocus = np.zeros((numPixels,numPixels,3))
    imageHanserDefocus[...,0] = np.abs(conv2(image[...,0], psf, df))
    imageHanserDefocus[...,1] = np.abs(conv2(image[...,1], psf, df))
    imageHanserDefocus[...,2] = np.abs(conv2(image[...,2], psf, df))
    return imageHanserDefocus.astype(np.float32)