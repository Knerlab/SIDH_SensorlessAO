#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:57:56 2023

@author: shaohli
"""


import os, time
import numpy as np
import tifffile as tf
import numpy.random as rd
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import pandas as pd
from scipy.optimize import fmin
import Zernike36 as Z
from operator import add
from scipy import signal


pi = np.pi
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
cm = ndimage.measurements.center_of_mass

class hologram_simulation(object):
    
    def __init__(self):
        # self.img = img_stack
        # self.nx, self.ny, self.nz = img_stack.shape
        # self.nx = self.nx/2
        # self.ny =  self.ny/2 
        self.wl = 0.00067  #Wavelenth of the light
        self.na = 1.42   # Numerical aperture of objective
        self.dx = 0.016 #
        self.nx = 256
        self.ny = 256
        self.dp = 1 / (self.nx * 2 * self.dx)
        self.k = 2 * np.pi / self.wl
        
        self.dz = 0.0002                        # z stepsize 500nm
        
        self.f_o = 3       # Focal length of objective (mm)
        # self.z_s = 3      # Distance between sample and objective (mm)
        self.f_slm = 300
          
        self.f_TL = 180.         #focal length of tube lens
        self.f_2 = 120.          #focal length of second lens
        self.d1 = 183.           #distance between objective and tube lens
        self.d2 = self.f_TL + self.f_2          #distance between tube lens and second lens
        self.f_3 = 120.          #focal length of third lens
        self.f_4 = 100.          #focal length of fourth lens
        self.d3 = self.f_2 + self.f_3         #distance between second lens and third lens
        self.d4 = self.f_4 + self.f_3          #distance between third lens and fourth lens
        self.d5 = self.f_4          #distance between fourth lens and interferometer
        
        self.xr = np.arange(-self.nx, self.nx)
        self.yr = np.arange(-self.ny, self.ny)
        self.xv, self.yv = np.meshgrid(self.xr, self.yr, indexing='ij', sparse=True)
        self.xv1 = self.dx * self.xv
        self.yv1 = self.dx * self.yv
        self.xv2 = self.dx * self.xv
        self.yv2 = self.dx * self.yv

        self.Nph = 6000
        self.bg = 5
        self.zarr = [0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,
                     0,   0,   0,   0,   0,
                     0,   0,   0,   0]
        
        t=time.localtime() 
        x = np.array([1e4,1e2,1])    
        t1 = int((t[0:3]*x).sum())
        t2 = int((t[3:6]*x).sum())
        fnt = "%s%s" %(t1,t2)
        parent_dir = '/Users/shaohli/Desktop/Holo_Aberation/AO'
        newfold = fnt + '_ao_iteration' + '_STD_noABER' + '/'
        self.newfold = os.path.join(parent_dir, newfold) 
        try:
            os.mkdir(self.newfold)
        except:
            print('Directory already exists')
        
        # self.path = time.strftime("%Y%m%d_%H%M%S" + 'bg' + str(self.bg)) 

        
    def __del__(self):
        pass
    
    def FT(self,x):
        # this only defines the correct fwd fourier transform including proper shift of the frequencies
        return fft2(fftshift(x)) # Fourier transform and shift
 
    def iFT(self,x):
        # this only defines the correct inverse fourier transform including proper shift of the frequencies
        return ifftshift(ifft2(x)) # inverse Fourier transform and shift
        
    
    def recon_dist_calc_realsetting(self, z_s, z_h):
        f_4 = self.f_4
        d4 = self.f_4 + self.f_3          #distance between third lens and fourth lens
        d5 = self.f_4          #distance between fourth lens and interferometer
        f_obj = self.f_o
        f_TL = self.f_TL
        d1 = self.d1
        d2 = self.d2
        d3 = self.d3
        f_2 = self.f_2
        f_3 = self.f_3
        f_mirr = self.f_slm
        if(np.abs(z_s)) == f_obj:
            z_r = (z_h-f_mirr)
        else:
            f_e = (z_s*f_obj)/(f_obj-z_s)
            f_bar1 = (f_TL*(f_e+d1))/(f_TL-(f_e+d1))
            f_bar2 = (f_2*(f_bar1+d2))/(f_2-(f_bar1+d2))
            f_bar3 = (f_3*(f_bar2+d3))/(f_3-(f_bar2+d3))
            f_bar4 = (f_4*(f_bar3+d4))/(f_4-(f_bar3+d4))
            f_bar5 = (f_mirr*(f_bar4+d5)/(f_mirr-(f_bar4+d5)))
            z_r = (((f_bar5+z_h)*(f_bar4+d5+z_h))/(f_bar5-f_bar4-d5))
        return np.abs(z_r)
    
    def Transverse_Magnification_realsetting(self, z_s, z_h):
        f_4 = self.f_4
        d4 = f_4 + self.f_3          #distance between third lens and fourth lens
        d5 = f_4            #distance between fourth lens and interferometer
        f_obj = self.f_o
        f_TL = self.f_TL
        d1 = self.d1
        d2 = self.d2
        d3 = self.d3
        f_2 = self.f_2
        f_3 = self.f_3
        if z_s == self.f_o:
            trans_mag = z_h/f_obj
        else:
            f_e = (z_s*f_obj)/(f_obj-z_s)
            f_bar1 = (f_TL*(f_e+d1))/(f_TL-(f_e+d1))
            f_bar2 = (f_2*(f_bar1+d2))/(f_2-(f_bar1+d2))
            f_bar3 = (f_3*(f_bar2+d3))/(f_3-(f_bar2+d3))
            f_bar4 = (f_4*(f_bar3+d4))/(f_4-(f_bar3+d4))
            trans_mag = (z_h*f_e*f_bar1*f_bar2*f_bar3*f_bar4)/(z_s*(f_e+d1)*(d2+f_bar1)*(d3+f_bar2)*(d4+f_bar3)*(d5+f_bar4))
        return np.abs(trans_mag)

    def Hologram_radius_realsetting(self, z_s, z_h, f_slm, f_o , na):
        f_4 = self.f_4
        d5 = f_4 
        d4 = f_4 + self.f_3   
        M1=np.array([[1,z_h],[0,1]])
        M2=np.array([[1,0],[(-1/f_slm),1]])
        M3=np.array([[1,d5],[0,1]])
        M4=np.array([[1,0],[(-1/f_4),1]])
        M5=np.array([[1,d4],[0,1]])
        M6=np.array([[1,0],[(-1/self.f_3),1]])
        M7=np.array([[1,self.d3],[0,1]])
        M8=np.array([[1,0],[(-1/self.f_2),1]])
        M9=np.array([[1,self.d2],[0,1]])
        M10=np.array([[1,0],[(-1/self.f_TL),1]])
        M11=np.array([[1,self.d1],[0,1]])
        M12=np.array([[1,0],[(-1/f_o),1]])
        M13=np.array([[1,z_s],[0,1]])
        
        # M_1 = M11.dot(M12).dot(M13)
        # Holo_R_1 = M_1[0,1] * na
        # print('Holo_R_obj pupil:',Holo_R_1)
        
        # M_2 = M7.dot(M8).dot(M9).dot(M10).dot(M11).dot(M12).dot(M13)
        # Holo_R_2 = M_2[0,1] * na
        # print('Holo_R_120 pupil:',Holo_R_2)
        
        # M_3 = M3.dot(M4).dot(M5).dot(M6).dot(M7).dot(M8).dot(M9).dot(M10).dot(M11).dot(M12).dot(M13)
        # Holo_R_3 = M_3[0,1] * na
        # print('Holo_R_entry pupil:',Holo_R_3)
        
        M = M1.dot(M2).dot(M3).dot(M4).dot(M5).dot(M6).dot(M7).dot(M8).dot(M9).dot(M10).dot(M11).dot(M12).dot(M13)
        Holo_R_real = M[0,1] * na
        return Holo_R_real
    
    
    def Linear_phase_function(self, x, y, xs, ys, wl):
        L = np.exp((1j * 2 * np.pi ) * (wl ** (-1)) * (sx * x + sy * y))
        return L
    
    def Quadratic_phase_function(self, b, wl, x, y):
        Q = np.exp(((1j * b * np.pi ) / wl) * (x ** 2 + y ** 2))
        return Q
    
    def Pupil_Radius(self,z_s):
        f_4 = self.f_4
        d4 = f_4 + self.f_3          #distance between third lens and fourth lens
        d5 = f_4          #distance between fourth lens and interferometer
        f_o = self.f_o
        f_TL = self.f_TL
        d1 = self.d1
        d2 = self.d2
        d3 = self.d3
        f_2 = self.f_2
        f_3 = self.f_3
        M1=np.array([[1,d5],[0,1]])
        M2=np.array([[1,0],[(-1/f_4),1]])
        M3=np.array([[1,d4],[0,1]])
        M4=np.array([[1,0],[(-1/f_3),1]])
        M5=np.array([[1,d3],[0,1]])
        M6=np.array([[1,0],[(-1/f_2),1]])
        M7=np.array([[1,d2],[0,1]])
        M8=np.array([[1,0],[(-1/f_TL),1]])
        M9=np.array([[1,d1],[0,1]])
        M10=np.array([[1,0],[(-1/f_o),1]])
        M11=np.array([[1,z_s],[0,1]])
        M = M1.dot(M2).dot(M3).dot(M4).dot(M5).dot(M6).dot(M7).dot(M8).dot(M9).dot(M10).dot(M11)
        pupil_Radius = M[0,1] * self.na
        return pupil_Radius
    
    def Pupil_Radius_2(self,z_s):
        f_4 = self.f_4
        d4 = f_4 + self.f_3          #distance between third lens and fourth lens
        d5 = f_4          #distance between fourth lens and interferometer
        f_o = self.f_o
        f_TL = self.f_TL
        d1 = self.d1
        d2 = self.d2
        d3 = self.d3
        f_2 = self.f_2
        f_3 = self.f_3
        M5=np.array([[1,f_2],[0,1]])
        M6=np.array([[1,0],[(-1/f_2),1]])
        M7=np.array([[1,d2],[0,1]])
        M8=np.array([[1,0],[(-1/f_TL),1]])
        M9=np.array([[1,d1],[0,1]])
        M10=np.array([[1,0],[(-1/f_o),1]])
        M11=np.array([[1,z_s],[0,1]])
        M = M5.dot(M6).dot(M7).dot(M8).dot(M9).dot(M10).dot(M11)
        pupil_Radius = M[0,1] * self.na
        return pupil_Radius
    
    
    def mask_off_center_beads(self, z_s, posx, posy):   
        xs1 = -posx * 3750
        ys1 = -posy * 3750
        HF_R =  1.1 * self.Hologram_radius_realsetting(z_s, z_h, self.f_slm, self.f_o, self.na)
        maskRadius = HF_R / self.dx
        mask = ((self.xv - xs1) ** 2 + (self.yv - ys1) ** 2) <= maskRadius**2
        return mask
     
    
    def Pupil_Mask(self,z_s):
        maskRadius = self.Pupil_Radius(z_s) # Radius of amplitude mask for defining the pupil
        # print('pupil')
        # print(maskRadius)
        maskRadius = maskRadius / self.dx
        maskCenter = np.floor(0)
        xv, yv     = np.meshgrid(np.arange(-self.nx, self.nx), np.arange(-self.ny, self.ny))
        mask       = np.sqrt((xv - maskCenter)**2 + (yv- maskCenter)**2) < maskRadius
        return maskRadius, mask

    def getZArrWF(self, zarr):
        ph = np.zeros((self.nx*2, self.ny*2),dtype=np.float32)
        maskRadius, mask = self.Pupil_Mask(self.f_o)
        for j,m in enumerate(zarr):
            ph = ph + m*Z.Zm(j,rad=maskRadius,orig=None,Nx=self.nx*2)
        bpp = mask*np.exp(1j*ph)
        return bpp
    
    def getZArr_out_of_focus(self, zarr,z_s,z_h):
        ph = np.zeros((self.nx*2, self.ny*2),dtype=np.float32)
        HF_R = 1.5 * self.Hologram_radius_realsetting(z_s, z_h, self.f_slm, self.f_o, self.na)
        maskRadius = HF_R / self.dx
        mask = (self.xv ** 2 + self.yv ** 2) <= maskRadius**2
        for j,m in enumerate(zarr):
            ph = ph + m*Z.Zm(j,rad=maskRadius,orig=None,Nx=self.nx*2)
        bpp = mask*np.exp(1j*ph) 
        # tf.imshow(bpp)
        return bpp, mask
    
      
    def getZArr_HF(self, zarr, z_s, z_h):     
        ''' get the aberration at the complex hologram (H_F) after back propogate f_d'''
        ph = np.zeros((self.nx*2, self.ny*2),dtype=np.float32)
        HF_R = 0.55 *  self.Pupil_Radius(z_s)
        maskRadius = HF_R / self.dx
        mask = (self.xv ** 2 + self.yv ** 2) <= maskRadius**2
        for j,m in enumerate(zarr):
            ph = ph + m*Z.Zm(j,rad=maskRadius,orig=None,Nx=self.nx*2)
        bpp = mask*np.exp(1j*ph) 
        # tf.imshow(np.angle(bpp))
        return bpp, mask
    
    
    def Pupil(self,z_s):
        maskRadius, mask = self.Pupil_Mask(z_s)
        # print(maskRadius)
        amp   = np.ones((self.nx*2, self.ny*2)) * mask
        phase = 2j * np.pi * np.ones((self.nx*2, self.ny*2))
        pupil = amp * np.exp(phase)                                      #the pupil
        return pupil
    
    
    def I_function(self, z_s, x, y, wl, theta, z_h, xs, ys):
        ''' use qudratic phase function to get the hologram, aberration apply at the pupil'''
        if (np.abs(z_s)) == self.f_o:
            bpp = self.getZArrWF(self.zarr)
            pupil = self.Pupil(z_s) 
            maskRadius_p, mask_p = self.Pupil_Mask(z_s)
            I_1 = pupil *((1/2)+((1/2) * np.exp(1j*theta)*self.Quadratic_phase_function(-1/self.f_slm, wl, x, y))) * bpp * mask_p
            I_2 = self.Quadratic_phase_function(1/z_h, wl, x, y) * self.msk
            I_h = ifftshift(ifft2(fft2(I_1)*fft2(I_2))) 
        else:
            bpp2, mask_p = self.getZArr_out_of_focus(self.zarr,z_s,z_h)
            mask_holo = self.mask_off_center_beads(z_s, xs, ys)
            bpp = self.getZArrWF(self.zarr)
            f_o = self.f_o
            f_e = (z_s*f_o)/(f_o-z_s)
            f_bar1 = (self.f_TL*(f_e+self.d1))/(self.f_TL-(f_e+self.d1))
            f_bar2 = (self.f_2*(f_bar1+self.d2))/(self.f_2-(f_bar1+self.d2))
            f_bar3 = (self.f_3*(f_bar2+self.d3))/(self.f_3-(f_bar2+self.d3))
            f_bar4 = (self.f_4*(f_bar3+self.d4))/(self.f_4-(f_bar3+self.d4))
        
            Lx1 =  (-xs * f_e * f_bar1 * f_bar2 * f_bar3 * f_bar4 ) / (z_s * (f_e +self.d1) * (f_bar1 +self.d2)  * (f_bar2 +self.d3)  * (f_bar3 +self.d4)  * (f_bar4 +self.d5))
            Ly1 =  (-ys * f_e * f_bar1 * f_bar2 * f_bar3 * f_bar4 ) / (z_s * (f_e +self.d1) * (f_bar1 +self.d2)  * (f_bar2 +self.d3)  * (f_bar3 +self.d4)  * (f_bar4 +self.d5))
        
            I_1 = 1/2 * self.Linear_phase_function(x, y, Lx1, Ly1, wl) * self.Quadratic_phase_function(1/(f_bar4 + self.d5), wl, x, y) * bpp * mask_p
            I_2 = I_1 * np.exp(1j*theta) * self.Quadratic_phase_function(-1/self.f_slm, wl, x, y) * mask_p
            I_3 = self.Quadratic_phase_function(1/z_h, wl, x, y)  
            
            I_hp = I_1 + I_2
        
            I_h = self.iFT(self.FT(I_hp)*self.FT(I_3)) * mask_holo
            
            # tf.imshow(np.abs((I_h*np.conj(I_h))))
            
        return np.abs((I_h*np.conj(I_h)))
    
    
    def Inverse_I_function_HF(self, imgstack, x, y, wl, z_h, zarr, recon_dist):               #correction applied at virtual pupil
        '''apply AO to complex hologram at BS'''       
        z_s = 3.0001                         
        phiin, HF_mask = self.getZArr_HF(zarr, z_s, z_h)
        theta1 = 0 * np.pi / 3                               #Theta timed 2 cause there are two exp(i*theta)
        theta2 = 2 * np.pi / 3
        theta3 = 4 * np.pi / 3
        final_intensity = (imgstack[0] * (np.exp(-1j*theta3)-np.exp(-1j*theta2)) +
                            imgstack[1] * (np.exp(-1j*theta1)-np.exp(-1j*theta3)) +
                            imgstack[2] * (np.exp(-1j*theta2)-np.exp(-1j*theta1)))
        
        f_vir = 160
        
        HF_R_2 = 1.7 *  self.Pupil_Radius(z_s)
        maskRadius_2 = HF_R_2 / self.dx
        mask_2 = (self.xv ** 2 + self.yv ** 2) <= maskRadius_2**2
        
        I_2 = self.Quadratic_phase_function(1/(self.f_slm+f_vir-z_h), wl, x, y) * mask_2
        I_3 = self.Quadratic_phase_function(-1/f_vir, wl, x, y) 
        I_4 = self.Quadratic_phase_function(1/(f_vir), wl, x, y)  * mask_2
        
        
        I_f =  self.iFT(self.FT(final_intensity) * self.FT(I_2))       # before the virtual lens

        I_f = I_f * I_3                                           
        I_f = self.iFT(self.FT(I_f) * self.FT(I_4)) * phiin         # at virtual pupil plane
        
   
        I_b = self.Quadratic_phase_function(1/(self.f_slm+f_vir-z_h), wl, x, y) 
    
    
        I_f = self.iFT(self.FT(I_f) * np.conj( self.FT(I_4)))            # after the virtual lens
    
        
        I_f = I_f / I_3
        I_f = self.iFT(self.FT(I_f) * np.conj( self.FT(I_b)))       # at camera plane
        
        ######################################
        
        recon_temp = np.exp((1j*pi/self.wl/recon_dist)*((x)**2+(y)**2))
        g = ifftshift(ifft2(fft2(I_f)*fft2(recon_temp)))
        
        # # g  = np.roll(g, 2, axis=1) # right
        # # g  = np.roll(g, -2, axis=0) # right
        # tf.imshow(np.abs(g))
        # tf.imshow(np.abs(g_ori))

        return np.abs(g), I_f


   
    
    def recon(self, recon_dist, imgstack, xv, yv, msk):
        theta1 = 0 * np.pi / 3                               #Theta timed 2 cause there are two exp(i*theta)
        theta2 = 2 * np.pi / 3
        theta3 = 4 * np.pi / 3
        final_intensity = (imgstack[0] * (np.exp(-1j*theta3)-np.exp(-1j*theta2)) +
                           imgstack[1] * (np.exp(-1j*theta1)-np.exp(-1j*theta3)) +
                           imgstack[2] * (np.exp(-1j*theta2)-np.exp(-1j*theta1)))
        recon_temp = np.exp((1j*pi/self.wl/recon_dist)*((xv)**2+(yv)**2)) * msk
        g = ifftshift(ifft2(fft2(final_intensity)*fft2(recon_temp)))
        return  final_intensity, np.abs(g)
    
    
    
    def generate_holoImgs(self, z_s, z_h, recon_dist, sx, sy, verbose=False):
        self.imgstack = np.zeros((3, self.nx*2, self.ny*2))
        theta_stack = np.array([ 0 * np.pi / 3 , 2 * np.pi / 3,  4 * np.pi / 3  ])
        self.Holo_R = self.Hologram_radius_realsetting(z_s, z_h, self.f_slm, self.f_o, self.na)
        # print( self.Holo_R)
        self.radius = self.Holo_R / self.dx
        self.msk = (self.xv ** 2 + self.yv ** 2) <= self.radius**2
        Mt = self.Transverse_Magnification_realsetting(z_s, z_h)
        for i in range(3):
            h = np.zeros((self.nx*2, self.ny*2))
            bg_metrix = rd.poisson ( self.bg * np.ones((self.nx*2, self.ny*2)) )
            theta = theta_stack[i]
            # h = self.I_function(sx, sy, self.xv2, self.yv2, self.wl, -recon_dist, theta ,Mt)
            h = self.I_function(z_s, self.xv2, self.yv2, self.wl, theta, z_h, sx, sy)
            # h = self.I_function(z_s, self.xv2, self.yv2, self.wl, theta, z_h)
            # h = self.I_function(z_s, self.xv2, self.yv2, self.wl, theta, z_h, z_r)
            # h = self.I_function_seperated(z_s, self.xv2, self.yv2, self.wl, theta, z_h)
            # h = self.I_function(sx, sy, self.xv2, self.yv2, self.wl, recon_dist, theta ,Mt)
            # h = h * self.msk
            h = h / h.sum()                         #normalization
            h = h * self.Nph
            h = rd.poisson(h) + bg_metrix
            self.imgstack[i] = h
        fn = os.path.join(self.newfold,'Holo_stack.tif')
        # tf.imsave(fn,self.imgstack)
        if verbose:
            tf.imshow(np.abs(self.imgstack[0]))
            tf.imshow(np.abs(self.imgstack[1]))
            tf.imshow(np.abs(self.imgstack[2]))
        
            
    def finch_recon(self,recon_dist, verbose=False):
        self.final_intensity, self.holo_recon = self.recon(recon_dist, self.imgstack, self.xv1, self.yv1, self.msk)
        if verbose:
            # tf.imshow(np.abs(self.final_intensity))
            # tf.imshow(np.angle(self.final_intensity))
            # tf.imshow(np.abs(self.holo_recon))
            fn = os.path.join(self.newfold,'Befroe_sensorlessAO.tif')
            tf.imsave(fn, np.abs(self.holo_recon))
        
            
    def finch_recon3D(self, HF, z_s, z_h, sx, sy, k, verbose=False):                                                         # 3D reconstruction of FINCH
        z_depth = np.arange((z_s-0.01), (z_s+0.01), self.dz)                           #z_depth(-10um to 10um, step size:1 um)
        zr_depth = np.zeros(len(z_depth))
        for i in range(len(z_depth)):
            zr_depth[i] = self.recon_dist_calc_realsetting(z_depth[i], z_h)                          #z_depth : 20um
        self.Recon_3d_stack = np.zeros((len(zr_depth), self.nx*2, self.ny*2))
        self.intensity_stack = np.zeros((len(zr_depth), self.nx*2, self.ny*2))
        for m in range(len(zr_depth)):
            recon_temp = np.exp((1j*pi/self.wl/zr_depth[m])*((self.xv1)**2+(self.yv1)**2)) * self.msk
            g = ifftshift(ifft2(fft2(HF)*fft2(recon_temp)))
            # g  = np.roll(g, 2, axis=1) # right
            # g  = np.roll(g, -2, axis=0) # right
            self.Recon_3d_stack[m,:,:] = np.abs(g)
        if verbose:
            if k == 1:
                # tf.imwrite('Recon_3d_stack_afterAO.tif',self.Recon_3d_stack)
                fn = os.path.join(self.newfold, 'Recon_3d_stack_afterAO.tif')
                tf.imsave(fn,self.Recon_3d_stack) 
            elif k == 0:
                # tf.imwrite('Recon_3d_stack_beforeAO.tif',self.Recon_3d_stack)
                fn = os.path.join(self.newfold, 'Recon_3d_stack_beforeAO.tif')
                tf.imsave(fn,self.Recon_3d_stack) 
        
           
    def zernikemodes(self):
        self.zn = 24
        zarr_i = np.zeros((self.zn,self.zn))
        row, col = np.diag_indices_from(zarr_i)
        zarr_i[row, col] = 1
        return zarr_i

    def gaussianArr(self, shape, sigma, peakVal, orig=None, dtype=np.float32):
        nx,ny = shape
        if orig == None:
            ux = nx/2.
            uy = ny/2.
        else:
            ux,uy = orig
        g = np.fromfunction(lambda i,j: np.exp(-((i-ux)**2.+(j-uy)**2.)/(2.*sigma**2.)),(nx,ny),dtype=dtype)*peakVal#/(sigma*sqrt(2*np.pi))
        return g
  
    def snr(self, wl,img,dx):
        nx,ny = img.shape
        maskRadius, mask = self.Pupil_Mask(self.f_o)
        radius = int(nx/4)
        msk = (self.xv ** 2 + self.yv ** 2) <= (radius)**2
        img = img * msk    
        # tf.imshow(img)
        siglp = maskRadius * 0.04
        sighp = maskRadius * 0.7      
        msk_1 =  (self.xv ** 2 + self.yv ** 2) <= (maskRadius)**2      
        lp = self.gaussianArr(shape=(nx,nx), sigma=siglp, peakVal=1, orig=None, dtype=np.float32)
        hp = 1-self.gaussianArr(shape=(nx,nx), sigma=sighp, peakVal=1, orig=None, dtype=np.float32)
        aft = fftshift(fft2(img))
        hpC = (np.abs(hp*aft*msk_1)).sum()
        lpC = (np.abs(lp*aft*msk_1)).sum()
        res = hpC/lpC
        return res
    
    
    def peak(self, x, y):
        a,b,c = np.polyfit(x, y, 2)
        
        m = len(x)
        xx = np.arange(x[0],x[m-1],0.01)
        yy = np.polyval([a,b,c],xx )
        plt.plot(xx, yy)
        
        zmax = -1*b/(a*2.0)
        if (a>0):
            print('no maximum')
            return 0.
        elif (zmax>=x.max()) or (zmax<=x.min()):
            print('maximum exceeding range')
            return 0.
        else:
            return zmax
            
        
    def get_STD_noAO(self, z_s, iteration, z_h, sx, sy):       #Calculate the STD of each Z_h(different radius)
        locs_xy = np.zeros((2,iteration))
        FWHM_xy = np.zeros((1,iteration))
        Recon_std_xy = np.zeros((2,1))
        FWHM_mean_xy = np.zeros((1,1))
        Locs_xy = np.zeros((2,1))
        
        
        locs_zy = np.zeros((2,iteration))
        FWHM_zy = np.zeros((2,iteration))
        Recon_std_zy = np.zeros((2,1))
        FWHM_mean_zy = np.zeros((2,1))
        Locs_zy = np.zeros((2,1))
        
        # z_h_perfect, R_prefect = self.Get_ZH_at_perfect_overlap_realsetting(z_s, f_slm2)
        # Mt = self.Transverse_Magnification(z_s,z_h)
        Mt = self.Transverse_Magnification_realsetting(z_s,z_h)
        for i in range(iteration):
            #STD calculate for xy plane
            recon_dist = self.recon_dist_calc_realsetting(z_s, z_h)
            self.generate_holoImgs(z_s, z_h, sx, sy, False)
            
            
            I_f_noAO, recon_noAO = self.recon(recon_dist, self.imgstack, self.xv1, self.yv1, self.msk)
            Recon_xy = recon_noAO
            "p=(amplitude, x_center, y_center, sigma_xy, background)"
            sec = 16              #zise of sub_XY 
            init_params_xy = (1.e4, sec, sec, 2.,  1.e2)    
            Sub_Recon_xy = np.abs(Recon_xy)[self.nx-sec:self.nx+sec, self.ny-sec:self.ny+sec]
            # tf.imshow(Sub_Recon_xy)
            out_xy = self.GaussianFit2D(Sub_Recon_xy , init_params_xy)
            locs_xy[0,i] = out_xy[1]
            locs_xy[1,i] = out_xy[2]
            FWHM_xy[0,i] = 2.35*out_xy[3]
            # if i == 1:
            # self.gauss2Dplot(out_xy,Sub_Recon_xy)
            
            self.holo_recon = []
            
            #STD calculate for zy plane
            I_f_noAO, recon_noAO = self.recon(recon_dist, self.imgstack, self.xv1, self.yv1, self.msk)
            self.finch_recon3D(I_f_noAO, z_s, z_h, sx, sy, 0, verbose=False)         #self.Recon_3d_stack
            secz = 16             #zise of sub_Z
            secy = 8              #zise of sub_Z
            nz, nx, ny = self.Recon_3d_stack.shape
            nz = int(nz/2)
            Recon_zy = self.zslides_trans(self.Recon_3d_stack)
            "p=(amplitude, z_center, y_center, sigma_z, sigma_y, background)"
            init_params_zy = (1.e4, secz, sec, 4., 2., 4.e2)    
            Sub_Recon_zy = np.abs(Recon_zy)[nz-secz:nz+secz,self.ny-secy:self.ny+secy]
            # tf.imshow(Sub_Recon_zy)
            out_zy = self.GaussianFit2DZ(Sub_Recon_zy, init_params_zy)    #3d guassian fit
            
            # if i == 1:
            # self.gauss2DplotZ(out_zy,Sub_Recon_zy)
                
            locs_zy[0,i] = out_zy[1]     #z
            locs_zy[1,i] = out_zy[2]     #y
            FWHM_zy[0,i] = 2.35*out_zy[3]  #FWHM for z
            FWHM_zy[1,i] = 2.35*out_zy[4]  #FWHM for y
            
            # if i == 1:
            #     "f_p=(amplitude, z_center, x_center, y_center, sigma_xy, (sigma_z), background)"
            #     f_p = (out_xy[0],out_zy[1],out_xy[1],out_xy[2],out_xy[3],out_zy[3],out_xy[4])
            #     self.save_fit_data(f_p, Recon_sub)
            #out_imgs[i,:,:] = abs(self.holo_recon)
            
        # print(FWHM_zy)
        # print(FWHM_xy)
        
        M = []
        for i in range(len(locs_zy[0])):
            if FWHM_zy[0,i] > 30 or  FWHM_xy[0,i] > 15 or  FWHM_xy[0,i] < 0 or FWHM_zy[0,i] < 0:
                M.append(i) 
        FWHM_zy = np.delete(FWHM_zy, M, 1)
        locs_zy = np.delete(locs_zy, M, 1)
        FWHM_xy = np.delete(FWHM_xy, M, 1)
        locs_xy = np.delete(locs_xy, M, 1)
        
        
        
        
        Locs_xy[0] = np.mean(locs_xy[0,:])   #x
        Locs_xy[1] = np.mean(locs_xy[1,:])   #y
        Recon_std_xy[0] = np.std(locs_xy[0,:]) *  self.dx / Mt 
        Recon_std_xy[1] = np.std(locs_xy[1,:]) *  self.dx / Mt
        FWHM_mean_xy[0] = np.mean(FWHM_xy[0,:]) *  self.dx / Mt 
        
        Locs_zy[0] = np.mean(locs_zy[0,:])   #z
        Locs_zy[1] = np.mean(locs_zy[1,:])   #y
        Recon_std_zy[0] = np.std(locs_zy[0,:]) *  self.dz 
        Recon_std_zy[1] = np.std(locs_zy[1,:]) *  self.dx / Mt
        FWHM_mean_zy[0] = np.mean(FWHM_zy[0,:]) *  self.dz 
        FWHM_mean_zy[1] = np.mean(FWHM_zy[1,:]) *  self.dx / Mt
        # print('FWHM_xy(nm)')
        # print(FWHM_mean_xy)
        # print('FWHM_zy(nm)')
        # print( FWHM_mean_zy)
        
        
        # print('iteration')
        # print(i)
        #tf.imsave('test.tif', out_imgs, photometric='minisblack')
        return Locs_xy, Locs_zy, Recon_std_xy, Recon_std_zy, FWHM_mean_xy, FWHM_mean_zy
    
    def get_STD_withAO(self, z_s, iteration, z_h, zarr_final, sx, sy):       #Calculate the STD of each Z_h(different radius)
        locs_xy = np.zeros((2,iteration))
        FWHM_xy = np.zeros((1,iteration))
        Recon_std_xy = np.zeros((2,1))
        FWHM_mean_xy = np.zeros((1,1))
        Locs_xy = np.zeros((2,1))
        
        
        locs_zy = np.zeros((2,iteration))
        FWHM_zy = np.zeros((2,iteration))
        Recon_std_zy = np.zeros((2,1))
        FWHM_mean_zy = np.zeros((2,1))
        Locs_zy = np.zeros((2,1))
        
        Mt = self.Transverse_Magnification_realsetting(z_s,z_h)
        self.Holo_R = self.Hologram_radius_realsetting(z_s, z_h, self.f_slm, self.f_o, self.na)
        self.radius = self.Holo_R / self.dx
        msk = (self.xv ** 2 + self.yv ** 2) <= self.radius**2
        for i in range(iteration):
            #STD calculate for xy plane
            recon_dist = self.recon_dist_calc_realsetting(z_s, z_h)
            self.generate_holoImgs(z_s, z_h, sx, sy, False)
            recon, I_f_AO = self.Inverse_I_function_HF(self.imgstack, self.xv1, self.yv1, self.wl, z_h, zarr_final, recon_dist) 
            Recon_xy = recon
            "p=(amplitude, x_center, y_center, sigma_xy, background)"
            sec = 14             #zise of sub_XY 
            init_params_xy = (2.e15, sec+1, sec-1, 2.,  3.e14)    
            Sub_Recon_xy = np.abs(Recon_xy)[self.nx-sec:self.nx+sec, self.ny-sec:self.ny+sec]
            # tf.imshow(Sub_Recon_xy)
            out_xy = self.GaussianFit2D(Sub_Recon_xy , init_params_xy)
            # print(out_xy)
            locs_xy[0,i] = out_xy[1]
            locs_xy[1,i] = out_xy[2]
            FWHM_xy[0,i] = 2.35*out_xy[3]
       
            # self.gauss2Dplot(out_xy,Sub_Recon_xy)
            
            self.holo_recon = []
            
            #STD calculate for zy plane
            self.finch_recon3D(I_f_AO, z_s, z_h, sx, sy, 1, verbose=False)          #self.Recon_3d_stack
            
            secz = 14             #zise of sub_Z
            secy = 6
            nz, nx, ny = self.Recon_3d_stack.shape
            nz = int(nz/2)
            Recon_zy = self.zslides_trans(self.Recon_3d_stack)
            "p=(amplitude, z_center, y_center, sigma_z, sigma_y, background)"
            init_params_zy = (2.e15, secz, 5, 4., 2., 5.e14)    
            Sub_Recon_zy = np.abs(Recon_zy)[nz-secz:nz+secz,self.ny-secy:self.ny+3]
            # tf.imshow(Sub_Recon_zy)
            out_zy = self.GaussianFit2DZ(Sub_Recon_zy, init_params_zy)    #3d guassian fit
            # print(out_zy)
            # self.gauss2DplotZ(out_zy,Sub_Recon_zy)
                
            locs_zy[0,i] = out_zy[1]     #z
            locs_zy[1,i] = out_zy[2]     #y
            FWHM_zy[0,i] = 2.35*out_zy[3]  #FWHM for z
            FWHM_zy[1,i] = 2.35*out_zy[4]  #FWHM for y
            
        # print(FWHM_zy)
        # print(FWHM_xy)
            
        M = []
        for i in range(len(locs_zy[0,:])):
            if FWHM_zy[0,i] > 30 or  FWHM_xy[0,i] > 15 or  FWHM_xy[0,i] < 0 or FWHM_zy[0,i] < 0:
                M.append(i) 
        # print (M)
        FWHM_zy = np.delete(FWHM_zy, M, 1)
        locs_zy = np.delete(locs_zy, M, 1)
        FWHM_xy = np.delete(FWHM_xy, M, 1)
        locs_xy = np.delete(locs_xy, M, 1)
        # print(FWHM_zy)
        # print(FWHM_xy)
            
        
        Locs_xy[0] = np.mean(locs_xy[0,:])   #x
        Locs_xy[1] = np.mean(locs_xy[1,:])   #y
        Recon_std_xy[0] = np.std(locs_xy[0,:]) *  self.dx / Mt 
        Recon_std_xy[1] = np.std(locs_xy[1,:]) *  self.dx / Mt
        FWHM_mean_xy[0] = np.mean(FWHM_xy[0,:]) *  self.dx / Mt 
        
        Locs_zy[0] = np.mean(locs_zy[0,:])   #z
        Locs_zy[1] = np.mean(locs_zy[1,:])   #y
        Recon_std_zy[0] = np.std(locs_zy[0,:]) *  self.dz           #z
        Recon_std_zy[1] = np.std(locs_zy[1,:]) *  self.dx / Mt      #y
        FWHM_mean_zy[0] = np.mean(FWHM_zy[0,:]) *  self.dz         #z
        FWHM_mean_zy[1] = np.mean(FWHM_zy[1,:]) *  self.dx / Mt     #y
        # print('FWHM_xy(nm)')
        # print(FWHM_mean_xy)
        # print('FWHM_zy(nm)')
        # print( FWHM_mean_zy)
        
        
        # print('iteration')
        # print(i)
        #tf.imsave('test.tif', out_imgs, photometric='minisblack')
        return Locs_xy, Locs_zy, Recon_std_xy, Recon_std_zy, FWHM_mean_xy, FWHM_mean_zy


    def Get_STD_vs_Z_s_noAO(self, z_s, iteration, z_h, sx, sy): #Calculate the STD of the hologram with different Z_h(different radius)
        try:
            os.mkdir(self.path)
        except:
            print('Directory already exists')
        Recon_std_XY = np.zeros((2,len(z_s)))
        Recon_std_ZY = np.zeros((2,len(z_s)))
        
        Locs_mean_XY = np.zeros((2,len(z_s)))
        Locs_mean_ZY = np.zeros((2,len(z_s)))
        
        FWHM_XY = np.zeros((1,len(z_s)))
        FWHM_ZY = np.zeros((2,len(z_s)))
        
        Holo_R_stack = np.zeros((len(z_s)))
        for i in range(len(z_s)):
            self.fnd = 'z_s_%.2f' %z_s[i]
            
            Locs_xy, Locs_zy, Recon_std_xy, Recon_std_zy, FWHM_mean_xy, FWHM_mean_zy = self.get_STD_noAO(z_s[i], iteration, z_h, sx, sy)
            Holo_R_stack[i] = self.Holo_R       #hologram radius
             
            Recon_std_XY[0,i] = Recon_std_xy[0]*1e6               #x
            Recon_std_XY[1,i] = Recon_std_xy[1]*1e6               #y
            
            
            Recon_std_ZY[0,i] = Recon_std_zy[0]*1e6               #z
            Recon_std_ZY[1,i] = Recon_std_zy[1]*1e6               #y from zslides
            
            Locs_mean_XY[0,i] = Locs_xy[0]            # Mean of Locs position of x for n iteration
            Locs_mean_XY[1,i] = Locs_xy[1]            # Mean of Locs position of y for n iteration
           
            Locs_mean_ZY[0,i] = Locs_zy[0]            #z
            Locs_mean_ZY[1,i] = Locs_zy[1]            #z
            
            FWHM_XY[0,i] = FWHM_mean_xy[0]*1e6           #xy
            
            FWHM_ZY[0,i] = FWHM_mean_zy[0]*1e6           #z
            FWHM_ZY[1,i] = FWHM_mean_zy[1]*1e6           #y
            
            
            # print('std_xy(nm)')
            # print(Recon_std_XY)
            # print('std_zy(nm)')
            # print(Recon_std_ZY)
            # print('locs_xy')
            # print(Locs_mean_XY)
            # print('locs_zy')
            # print(Locs_mean_ZY)
            # print('FWHM_xy(nm)')
            # print(FWHM_XY)
            # print('FWHM_zy(nm)')
            # print(FWHM_ZY)
            # tf.imwrite(self.path + '/' + self.fnd + '_Holoimages.tif', np.abs(self.imgstack))
            # tf.imwrite(path + '/' + fnd + '_reconimage.tif', np.abs(self.holo_recon))
            # tf.imwrite(self.path + '/' + self.fnd + '_3dreconstack.tif', np.abs(self.Recon_3d_stack))
        save_excel = np.zeros((10,len(z_s)))
        save_excel[0,:] = z_s
        save_excel[1,:] = Holo_R_stack
        
        save_excel[2,:] = Recon_std_XY[0,:]
        save_excel[3,:] = Recon_std_XY[1,:]
        save_excel[4,:] = Recon_std_ZY[0,:]
        
        save_excel[5,:] = Locs_mean_XY[0,:]
        save_excel[6,:] = Locs_mean_XY[1,:]
        save_excel[7,:] = Locs_mean_ZY[0,:]
        
        save_excel[8,:] = FWHM_XY[0,:]
        save_excel[9,:] = FWHM_ZY[0,:]
        
        df = pd.DataFrame(save_excel)
        df.index = ['z_s', 'R_H', 'x_STD', 'y_STD','z_STD','x_Locs','y_Locs','z_Locs','xy_FWHM','z_FWHM']
        df.to_excel(self.newfold + '/'+'ZSchange_noAO_' + str(self.bg) +'bg' + 'STD.xlsx', index=True)
        
        # '''plot all the results'''
        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s, Holo_R_stack)
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("R_h(mm)")
        # plt.show()


        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,Locs_mean_ZY[0,:])
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("z_locs")
        # plt.show()

        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,Locs_mean_XY[0,:])
        # plt.plot(z_s,Locs_mean_XY[1,:])
        # plt.legend(['x','y'])
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("xy_locs")
        # plt.show()

        
        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,Recon_std_ZY[0,:])
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("z_STD(nm)")
        # plt.show()

        
        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,Recon_std_XY[0,:])
        # plt.plot(z_s,Recon_std_XY[1,:])
        # plt.legend(['x','y'])
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("xy_STD(nm)")
        # plt.show()

        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,FWHM_ZY[0,:])          #z
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("FWHM_z(nm)")
        # plt.show()

        
        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,FWHM_XY[0,:])          #xy
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("FWHM_xy(nm)")
        # plt.show()
        
    def Get_STD_vs_Z_s_withAO(self, z_s, iteration, z_h, zarr_final, sx, sy): #Calculate the STD of the hologram with different Z_h(different radius)
        try:
            os.mkdir(self.path)
        except:
            print('Directory already exists')
        Recon_std_XY = np.zeros((2,len(z_s)))
        Recon_std_ZY = np.zeros((2,len(z_s)))
        
        Locs_mean_XY = np.zeros((2,len(z_s)))
        Locs_mean_ZY = np.zeros((2,len(z_s)))
        
        FWHM_XY = np.zeros((1,len(z_s)))
        FWHM_ZY = np.zeros((2,len(z_s)))
        
        Holo_R_stack = np.zeros((len(z_s)))
        for i in range(len(z_s)):
            self.fnd = 'z_s_%.2f' %z_s[i]
            
            Locs_xy, Locs_zy, Recon_std_xy, Recon_std_zy, FWHM_mean_xy, FWHM_mean_zy = self.get_STD_withAO(z_s[i], iteration, z_h, zarr_final, sx, sy)
            Holo_R_stack[i] = self.Holo_R       #hologram radius
             
            Recon_std_XY[0,i] = Recon_std_xy[0]*1e6               #x
            Recon_std_XY[1,i] = Recon_std_xy[1]*1e6               #y
            
            
            Recon_std_ZY[0,i] = Recon_std_zy[0]*1e6               #z
            Recon_std_ZY[1,i] = Recon_std_zy[1]*1e6               #y from zslides
            
            Locs_mean_XY[0,i] = Locs_xy[0]            # Mean of Locs position of x for n iteration
            Locs_mean_XY[1,i] = Locs_xy[1]            # Mean of Locs position of y for n iteration
           
            Locs_mean_ZY[0,i] = Locs_zy[0]            #z
            Locs_mean_ZY[1,i] = Locs_zy[1]            #z
            
            FWHM_XY[0,i] = FWHM_mean_xy[0]*1e6           #xy
            
            FWHM_ZY[0,i] = FWHM_mean_zy[0]*1e6           #z
            FWHM_ZY[1,i] = FWHM_mean_zy[1]*1e6           #y
            
            
            
            
            # print('std_xy(nm)')
            # print(Recon_std_XY)
            # print('std_zy(nm)')
            # print(Recon_std_ZY)
            # print('locs_xy')
            # print(Locs_mean_XY)
            # print('locs_zy')
            # print(Locs_mean_ZY)
            # print('FWHM_xy(nm)')
            # print(FWHM_XY)
            # print('FWHM_zy(nm)')
            # print(FWHM_ZY)
            # tf.imwrite(self.path + '/' + self.fnd + '_Holoimages.tif', np.abs(self.imgstack))
            # tf.imwrite(path + '/' + fnd + '_reconimage.tif', np.abs(self.holo_recon))
            # tf.imwrite(self.path + '/' + self.fnd + '_3dreconstack.tif', np.abs(self.Recon_3d_stack))
        save_excel = np.zeros((10,len(z_s)))
        save_excel[0,:] = z_s
        save_excel[1,:] = Holo_R_stack
        
        save_excel[2,:] = Recon_std_XY[0,:]
        save_excel[3,:] = Recon_std_XY[1,:]
        save_excel[4,:] = Recon_std_ZY[0,:]
        
        save_excel[5,:] = Locs_mean_XY[0,:]
        save_excel[6,:] = Locs_mean_XY[1,:]
        save_excel[7,:] = Locs_mean_ZY[0,:]
        
        save_excel[8,:] = FWHM_XY[0,:]
        save_excel[9,:] = FWHM_ZY[0,:]
        
        df = pd.DataFrame(save_excel)
        df.index = ['z_s', 'R_H', 'x_STD', 'y_STD','z_STD','x_Locs','y_Locs','z_Locs','xy_FWHM','z_FWHM']
        df.to_excel(self.newfold + '/'+'ZSchange_withAO_iter3' + str(self.bg) +'bg' + 'STD.xlsx', index=True)
        
        # '''plot all the results'''
        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s, Holo_R_stack)
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("R_h(mm)")
        # plt.show()
   
        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,Locs_mean_ZY[0,:])
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("z_locs")
        # plt.show()
        
        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,Locs_mean_XY[0,:])
        # plt.plot(z_s,Locs_mean_XY[1,:])
        # plt.legend(['x','y'])
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("xy_locs")
        # plt.show()
        
        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,Recon_std_ZY[0,:])
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("z_STD(nm)")
        # plt.show()

        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,Recon_std_XY[0,:])
        # plt.plot(z_s,Recon_std_XY[1,:])
        # plt.legend(['x','y'])
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("xy_STD(nm)")
        # plt.show()
        
        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,FWHM_ZY[0,:])          #z
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("FWHM_z(nm)")
        # plt.show()

        # fig = plt.figure(figsize=(10,5))
        # plt.plot(z_s,FWHM_XY[0,:])          #xy
        # plt.xlabel("z_s(mm)")
        # plt.ylabel("FWHM_xy(nm)")
        # plt.show()



    def rms(self, arr):
        n = arr.shape
        square = (arr**2).sum()
        mean = square / n
        root = math.sqrt(mean)
        return root

    def gauss2Dplot(self, p, img):
        nx, ny = img.shape
        x = np.arange(nx)
        y = np.arange(ny)
        [xx, yy] = np.meshgrid(x,y,indexing = 'ij')
        imgfit = p[0]*np.exp(-0.5*((xx-p[1])**2 + (yy-p[2])**2)/p[3]**2)+p[4]
        fig = plt.figure(figsize=(7,7))
        plt.imshow(imgfit)
        # fig.savefig(self.path + '/' + self.fnd + 'Fit_ZY.jpg',bbox_inches='tight',dpi=150)
        # fig = plt.figure(figsize=(7,7))
        # plt.imshow(img)
        # fig.savefig(self.path + '/' + self.fnd + 'Ori_ZY.jpg',bbox_inches='tight',dpi=150)
        return imgfit
    
    def GaussianFit2D(self, img, init_params):
        out, fopt, iter1, iter2, warnflag = fmin(self._gausserr2D, init_params, args=(img,), full_output=True)
        # print('out_xy')
        # print(out)
        return out
       
    def _gausserr2D(self, p, img):
        '''
        (p, zshape)
        p=(amplitude, x_center, y_center, sigma, background)
        '''
        nx, ny = img.shape
        x = np.arange(nx)
        y = np.arange(ny)
        [xx, yy] = np.meshgrid(x,y, indexing = 'ij')
        imgfit = p[0]*np.exp(-0.5*(((xx-p[1])**2 + (yy-p[2])**2)/p[3]**2))+p[4]
        err = ((img-imgfit)**2).sum()/(img**2).sum()
        return err
    
       
    def _gausserr2DZ(self, p, img):
        '''
        (p, zshape)
        p=(amplitude, z_center, y_center, sigmaz, sigmay, background)
        '''
        nz, ny = img.shape
        z = np.arange(nz)
        y = np.arange(ny)
        [zz, yy] = np.meshgrid(z,y, indexing = 'ij')
        imgfit = p[0]*np.exp(-0.5*((((zz-p[1])**2)/p[3]**2) + (((yy-p[2])**2)/p[4]**2)))+p[5]
        err = ((img-imgfit)**2).sum()/(img**2).sum()
        return err
    
    def GaussianFit2DZ(self, img, init_params):
        out, fopt, iter1, iter2, warnflag = fmin(self._gausserr2DZ, init_params, args=(img,), full_output=True)
        # print('out_zy')
        # print(out)
        return out
    
    def gauss2DplotZ(self, p, img):
        '''
        (p, zshape)
        p=(amplitude, z_center, y_center, sigmaz, sigmay, background)'''
        nz, ny = img.shape
        z = np.arange(nz)
        y = np.arange(ny)
        [zz, yy] = np.meshgrid(z,y,indexing = 'ij')
        imgfit = p[0]*np.exp(-0.5*((((zz-p[1])**2)/p[3]**2) + (((yy-p[2])**2)/p[4]**2)))+p[5]
        fig = plt.figure(figsize=(7,7))
        plt.imshow(imgfit)
        # fig.savefig(self.path + '/' + self.fnd + 'Fit_ZY.jpg',bbox_inches='tight',dpi=150)
        # fig = plt.figure(figsize=(7,7))
        # plt.imshow(img)
        # fig.savefig(self.path + '/' + self.fnd + 'Ori_ZY.jpg',bbox_inches='tight',dpi=150) 
        return imgfit
    
    
    def zslides_trans(self, img):
        nz, nx, ny = img.shape
        z_y = np.zeros((nz,nx))
        for i in range(nz):
            for j in range(ny):
                z_y[i,j] = max(img[i,:,j])
        # plt.imshow(z_y)
        return z_y
    
    def save_fit_data(self, p, img):
        '''
        (p, zshape)
        p=(amplitude, z_center, x_center, y_center, sigma_xy, sigma_z, background)
        '''
        # print('f_p')
        tf.imwrite(self.path + '/' + self.fnd + 'ori_holoimages.tif', np.abs(img))
        nz, nx, ny = img.shape
        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)
        [zz, xx, yy] = np.meshgrid(z, x, y, indexing = 'ij')
        imgfit = p[0]*np.exp(-0.5*((((((xx-p[2])**2)/p[4]**2) + (((yy-p[3])**2)/p[4]**2) + (((zz-p[1])**2)/p[5]**2) ))))+p[6]
        tf.imwrite(self.path + '/' + self.fnd + 'fit_holoimages.tif', np.abs(imgfit))
        
        ''''plot fit'''
        # #plot fit data
        # plt.figure(figsize = (8,8))
        # ax = plt.axes(projection ="3d")
        # [z1, x1, y1] = np.meshgrid(z, x, y, indexing = 'ij')
        # for i in range(nz):
        #     for j in range(nx):
        #         for k in range(ny):
        #             if imgfit[i,j,k] < 639:
        #                imgfit[i,j,k] = np.nan
        # imgfit2 = np.zeros((nx,ny,nz))
        # for i in range(nz):
        #     imgfit2[:,:,i] = imgfit[i,:,:] 
        # [x1, y1, z1] = np.meshgrid(x, y, z, indexing = 'ij')
        # ax.scatter3D(x1, y1, z1, c=imgfit2, marker='.')
        # plt.show()
        
        # #plot original data
        # [z2, x2, y2] = np.meshgrid(z, x, y, indexing = 'ij')
        # for i in range(nz):
        #     for j in range(nx):
        #         for k in range(ny):
        #             if img[i,j,k] < 639:
        #               img[i,j,k] = np.nan
        # img2 = np.zeros((nx,ny,nz))
        # for i in range(nz):
        #     img2[:,:,i] = img[i,:,:] 
        # [x2, y2, z2] = np.meshgrid(x, y, z, indexing = 'ij')
        # plt.figure(figsize = (8,8))
        # ax = plt.axes(projection ="3d")
        # ax.scatter3D(x2, y2, z2, c=img2, marker='.')
        # plt.show()

        




if __name__ == '__main__':
    sx = 0
    sy = 0
    t = hologram_simulation()
    z_s = np.arange(2.995, 3.005, 0.0002) 
    z_h = 100
    iteration = 100
 
    t.zarr = [0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0]               # Input Aberration Mask
 
    
    
    amp_final = [0,   0,   0,   0,  -0.78900146,
               -1.296192,   -1.459577,   -1.631486,   0,   0,
               0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,
               0,   0,   0,   0]                 # Correction Phsae Mask
    
    



##########################################Calculating STD for No AO#########################################
    
    t.Get_STD_vs_Z_s_noAO(z_s, iteration, z_h, sx,sy)
    
    
##########################################Calculating STD After AO#########################################

    t.Get_STD_vs_Z_s_withAO(z_s, iteration, z_h, amp_final, sx, sy)
    

    
  