#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 01:28:39 2024

@author: shaohli
"""


import os, time
import numpy as np
import tifffile as tf
import numpy.random as rd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import correlate2d
import math
import pandas as pd
from scipy.optimize import fmin
import Zernike36 as Z
from operator import add
import time
# import cv2
# import Image



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
        self.na = 1.42    # Numerical aperture of objective
        self.dx = 0.016 #
        self.nx = 256
        self.ny = 256
        self.dp = 1 / (self.nx * 2 * self.dx)
        self.k = 2 * np.pi / self.wl

        
        kMax = 2 * pi / self.dx     # Value of k at the maximum extent of the pupil function
        self.dk   = 2 * pi / (self.nx * 2 * self.dx)  
        # kx = np.arange((-kMax + dk) / 2, (kMax + dk) / 2, dk)
        # ky = np.arange((-kMax + dk) / 2, (kMax + dk) / 2, dk)
        # self.KX, self.KY = np.meshgrid(kx, ky) # The coordinate system for the pupil function

        
        self.dz = 0.0002                        # z stepsize 200nm
        
        self.f_o = 3       # Focal length of objective (mm)
        # self.z_s = 3      # Distance between sample and objective (mm)
        self.f_slm = 300
          
        self.f_TL = 180.         #focal length of tube lens
        self.f_2 = 120.          #focal length of second lens
        self.d1 = self.f_TL + self.f_o           #distance between objective and tube lens
        self.d2 = self.f_TL + self.f_2          #distance between tube lens and second lens
        self.f_3 = 120.          #focal length of third lens
        self.f_4 = 100.          #focal length of fourth lens
        self.d3 = self.f_2 + self.f_3         #distance between second lens and third lens
        self.d4 = self.f_4 + self.f_3          #distance between third lens and fourth lens
        self.d5 = self.f_4  #distance between fourth lens and interferometer
        
        self.xr = np.arange(-self.nx, self.nx)
        self.yr = np.arange(-self.ny, self.ny)
        self.xv, self.yv = np.meshgrid(self.xr, self.yr, indexing='ij', sparse=True)
        self.xv1 = self.dx * self.xv
        self.yv1 = self.dx * self.yv
        self.xv2 = self.dx * self.xv
        self.yv2 = self.dx * self.yv

        self.Nph = 6000
        self.bg = 0
        self.zarr = [0,   0,   0,   0,   0,
                      0,   0,   0,   0,   0,
                      0,   0,   0,   0,   0,
                      0,   0,   0,   0,   0,
                      0,   0,   0,   0]                #introduced aberration 
        
        t=time.localtime() 
        x = np.array([1e4,1e2,1])    
        t1 = int((t[0:3]*x).sum())
        t2 = int((t[3:6]*x).sum())
        fnt = "%s%s" %(t1,t2)
        parent_dir = '/Users/shaohli/Desktop/Holo_Aberation/AO'
        newfold = fnt + '_ao_iteration' + '_snr' + '/'
        self.newfold = os.path.join(parent_dir, newfold) 
        try:
            os.mkdir(self.newfold)
        except:
            print('Directory already exists')
    

        
    def __del__(self):
        pass
    
    
    def FT(self,x):
        # this only defines the correct fwd fourier transform including proper shift of the frequencies
        return fft2(fftshift(x)) # Fourier transform and shift
 
    def iFT(self,x):
        # this only defines the correct inverse fourier transform including proper shift of the frequencies
        return ifftshift(ifft2(x)) # inverse Fourier transform and shift
        
        
    def Linear_phase_function(self, x, y, xs, ys, wl):
        L = np.exp((1j * 2 * np.pi ) * (wl ** (-1)) * (xs * x + ys * y))
        return L
    
    def Quadratic_phase_function(self, b, wl, x, y):
        Q = np.exp(((1j * b * np.pi ) / wl) * (x ** 2 + y ** 2))
        return Q
        
    def recon_dist_calc_realsetting(self, z_s, z_h):
        ''' calculate Z_r '''
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
        ''' calculate M_t '''
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
        # print(trans_mag)
        return np.abs(trans_mag)

    def Hologram_radius_realsetting(self, z_s, z_h):
        ''' calculate H_R '''
        f_4 = self.f_4
        d5 = f_4 
        d4 = f_4 + self.f_3   
        f_slm = self.f_slm
        f_o =self.f_o
        na = self.na
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
    
        M = M1.dot(M2).dot(M3).dot(M4).dot(M5).dot(M6).dot(M7).dot(M8).dot(M9).dot(M10).dot(M11).dot(M12).dot(M13)
        Holo_R_real = M[0,1] * na
        return Holo_R_real
    
    
    def Pupil_Radius(self,z_s):
        ''' calculate the radius of the pupil at the interferometer plane '''
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
    
    
    
    def Virtual_Pupil_Radius(self,z_s,f_v):
        ''' calculate the radius of the pupil at the interferometer plane '''
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
        f_slm = self.f_slm

        M1=np.array([[1,f_v],[0,1]])
        M2=np.array([[1,0],[(-1/f_v),1]])
        M3=np.array([[1,(f_slm+f_v)],[0,1]])
        M4=np.array([[1,0],[(-1/f_slm),1]])
        M5=np.array([[1,d5],[0,1]])
        M6=np.array([[1,0],[(-1/f_4),1]])
        M7=np.array([[1,d4],[0,1]])
        M8=np.array([[1,0],[(-1/f_3),1]])
        M9=np.array([[1,d3],[0,1]])
        M10=np.array([[1,0],[(-1/f_2),1]])
        M11=np.array([[1,d2],[0,1]])
        M12=np.array([[1,0],[(-1/f_TL),1]])
        M13=np.array([[1,d1],[0,1]])
        M14=np.array([[1,0],[(-1/f_o),1]])
        M15=np.array([[1,z_s],[0,1]])
        M = M1.dot(M2).dot(M3).dot(M4).dot(M5).dot(M6).dot(M7).dot(M8).dot(M9).dot(M10).dot(M11).dot(M12).dot(M13).dot(M14).dot(M15)
        pupil_Radius = M[0,1] * self.na
        return pupil_Radius
     
    
    
    def mask_off_center_beads(self, z_s, posx, posy):   
        xs1 = -posx * 3750
        ys1 = -posy * 3750
        # xs1 = -posx * 6400
        # ys1 = -posy * 6400
        HF_R =  1.2 * self.Hologram_radius_realsetting(z_s, z_h)
        maskRadius = HF_R / self.dx
        mask = ((self.xv - xs1) ** 2 + (self.yv - ys1) ** 2) <= maskRadius**2
        # tf.imshow(mask)
        return mask
     
    
    def Pupil_Mask(self,z_s):
        ''' get the mask of the pupil at the interferometer plane '''
        maskRadius = self.Pupil_Radius(z_s) # Radius of amplitude mask for defining the pupil
        maskRadius = maskRadius / self.dx
        maskCenter = np.floor(0)
        xv, yv     = np.meshgrid(np.arange(-self.nx, self.nx), np.arange(-self.ny, self.ny))
        mask       = np.sqrt((xv - maskCenter)**2 + (yv- maskCenter)**2) < maskRadius
        return maskRadius, mask
    
    def Pupil(self,z_s):
        ''' get the pupil at the interferometer plane '''
        maskRadius, mask = self.Pupil_Mask(z_s)
        amp   = np.ones((self.nx*2, self.ny*2)) * mask
        phase = 2j * np.pi * np.ones((self.nx*2, self.ny*2))
        pupil = amp * np.exp(phase)                                      #the pupil
        return pupil
    
    def zernikemodes(self):
        self.zn = 24
        zarr_i = np.zeros((self.zn,self.zn))
        row, col = np.diag_indices_from(zarr_i)
        zarr_i[row, col] = 1
        return zarr_i
    
    def getZArrWF(self, zarr):
        ''' introduce the aberration at the interferometer pupil plane '''
        ph = np.zeros((self.nx*2, self.ny*2),dtype=np.float32)
        maskRadius, mask = self.Pupil_Mask(self.f_o)
        for j,m in enumerate(zarr):
            ph = ph + m*Z.Zm(j,rad=maskRadius,orig=None,Nx=self.nx*2)
        bpp = mask*np.exp(1j*ph)
        # tf.imshow(np.angle(bpp))
        return bpp
    
    def DM_mask(self, z_s,z_h):     
        ''' get the mask at CCD, the radius of the  H_F is 2 times of H_R'''
        HF_R = 1.1 * self.Hologram_radius_realsetting(z_s, z_h)
        maskRadius = HF_R / self.dx
        mask = (self.xv ** 2 + self.yv ** 2) <= maskRadius**2
        # tf.imshow(bpp)
        return mask
    
    def getZArr_HF(self, zarr, z_s, z_h):     
        ''' get the aberration at the complex hologram (H_F) after back propogate f_d'''
        ph = np.zeros((self.nx*2, self.ny*2),dtype=np.float32)
        HF_R = 0.58 *  self.Pupil_Radius(z_s)
        # HF_R = 1 *  self.Pupil_Radius(z_s)
        maskRadius = HF_R / self.dx
        mask = (self.xv ** 2 + self.yv ** 2) <= maskRadius**2
        for j,m in enumerate(zarr):
            ph = ph + m*Z.Zm(j,rad=maskRadius,orig=None,Nx=self.nx*2)
        bpp = mask*np.exp(1j*ph) 
        # tf.imshow(np.angle(bpp))
        return bpp, mask
    
    def getZArr_out_of_focus(self, zarr,z_s,z_h):
        ph = np.zeros((self.nx*2, self.ny*2),dtype=np.float32)
        HF_R = 1 *  self.Pupil_Radius(z_s)
        # HF_R = 1.5 * self.Hologram_radius_realsetting(z_s, z_h)
        maskRadius = HF_R / self.dx
        mask = (self.xv ** 2 + self.yv ** 2) <= maskRadius**2
        for j,m in enumerate(zarr):
            ph = ph + m*Z.Zm(j,rad=maskRadius,orig=None,Nx=self.nx*2)
        bpp = mask*np.exp(1j*ph) 
        # tf.imshow(np.angle(bpp))
        return bpp, mask

    def gaussianArr(self, shape, sigma, peakVal, orig=None, dtype=np.float32):
        nx,ny = shape
        if orig == None:
            ux = nx/2.
            uy = ny/2.
        else:
            ux,uy = orig
        g = np.fromfunction(lambda i,j: np.exp(-((i-ux)**2.+(j-uy)**2.)/(2.*sigma**2.)),(nx,ny),dtype=dtype)*peakVal#/(sigma*sqrt(2*np.pi))
        return g
  
    def snr(self, wl,img,dx):  #frequency filter
        nx,ny = img.shape
        self.Mt = self.Transverse_Magnification_realsetting(z_s, z_h)
        # print(self.Mt)
        dk   = 1 / (nx * (self.dx/self.Mt) ) 
        maskRadius =  self.na / self.wl / dk
        siglp = maskRadius * 0.012
        sighp = maskRadius * 0.28
        msk =  (self.xv ** 2 + self.yv ** 2) <= (maskRadius*0.3)**2 
        msk_1 =  (self.xv ** 2 + self.yv ** 2) <= (maskRadius*0.3)**2  
        img = img * msk
        tf.imshow(img)
        lp = self.gaussianArr(shape=(nx,nx), sigma=siglp, peakVal=1, orig=None, dtype=np.float32)
        hp = 1-self.gaussianArr(shape=(nx,nx), sigma=sighp, peakVal=1, orig=None, dtype=np.float32)
        aft = fftshift(fft2(img))
        # tf.imshow(aft)
        hpC = (np.abs(hp*aft*msk_1)).sum()
        # tf.imshow(np.abs(hp*aft*msk_1))
        lpC = (np.abs(lp*aft*msk_1)).sum()
        # tf.imshow(np.abs(lp*aft*msk_1))
        res = hpC/lpC
        return res
    

    
    # def snr(self, wl,img,dx):       #peak intensity metric function
    #     nx,ny = img.shape
    #     radius = int(nx/2)
    #     msk = (self.xv ** 2 + self.yv ** 2) <= (radius*0.25)**2
    #     img = img*msk
    #     # tf.imshow(img)
    #     maxv = img.max()
    #     return maxv
    
    
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
    
   
    def I_function(self, z_s, x, y, wl, theta, z_h, xs, ys):
        ''' use qudratic phase function to get the hologram, aberration apply at the pupil'''
        if (np.abs(z_s)) == self.f_o:
            bpp, mask_p = self.getZArrWF(self.zarr)
            pupil = self.Pupil(z_s) 
            I_1 = pupil *((1/2)+((1/2) * np.exp(1j*theta)*self.Quadratic_phase_function(-1/self.f_slm, wl, x, y))) * bpp * mask_p
            I_2 = self.Quadratic_phase_function(1/z_h, wl, x, y) * self.msk
            I_h = ifftshift(ifft2(fft2(I_1)*fft2(I_2))) 
            # tf.imshow(np.angle(bpp))
        else:
            bpp2, mask_p = self.getZArr_out_of_focus(self.zarr,z_s,z_h)
            mask_holo = self.mask_off_center_beads(z_s,xs,ys)
            # bpp, mask_p = self.getZArrWF(self.zarr)
            # tf.imshow(np.abs(mask_holo))
            f_o = self.f_o
            f_e = (z_s*f_o)/(f_o-z_s)
            f_bar1 = (self.f_TL*(f_e+self.d1))/(self.f_TL-(f_e+self.d1))
            f_bar2 = (self.f_2*(f_bar1+self.d2))/(self.f_2-(f_bar1+self.d2))
            f_bar3 = (self.f_3*(f_bar2+self.d3))/(self.f_3-(f_bar2+self.d3))
            f_bar4 = (self.f_4*(f_bar3+self.d4))/(self.f_4-(f_bar3+self.d4))
        
            Lx1 =  (-xs * f_e * f_bar1 * f_bar2 * f_bar3 * f_bar4 ) / (z_s * (f_e +self.d1) * (f_bar1 +self.d2)  * (f_bar2 +self.d3)  * (f_bar3 +self.d4)  * (f_bar4 +self.d5))
            Ly1 =  (-ys * f_e * f_bar1 * f_bar2 * f_bar3 * f_bar4 ) / (z_s * (f_e +self.d1) * (f_bar1 +self.d2)  * (f_bar2 +self.d3)  * (f_bar3 +self.d4)  * (f_bar4 +self.d5))
        
            I_1 = 1/2 * self.Linear_phase_function(x, y, Lx1, Ly1, wl) * self.Quadratic_phase_function(1/(f_bar4 + self.d5), wl, x, y) * bpp2 * mask_p
            I_2 = I_1 * np.exp(1j*theta) * self.Quadratic_phase_function(-1/self.f_slm, wl, x, y) * mask_p
            I_3 = self.Quadratic_phase_function(1/z_h, wl, x, y) * mask_holo
            
            I_hp = I_1 + I_2
        
            I_h = self.iFT(self.FT(I_hp)*self.FT(I_3)) * mask_holo
            
            # tf.imshow(np.abs((I_h*np.conj(I_h))))
            
        return np.abs((I_h*np.conj(I_h)))
    
    def I_function_multiZ(self, x, y, wl, theta, z_h):
        z_s_all = np.arange(2.9951, 3.0051, 0.001) 
        n_emitter_z = len(z_s_all)
        print(n_emitter_z)
        pos = np.zeros((n_emitter_z, 1, 2))
        pos1 = 0.025
        bg_metrix = rd.poisson (self.bg * np.ones((self.nx*2, self.ny*2)) )
        for i in range(n_emitter_z):
            pos[i] = (pos1,pos1)
            pos1 = pos1 - 0.006
            
        h = np.zeros((n_emitter_z, self.nx*2, self.ny*2))
        
        for i in range(n_emitter_z):
            z_s = z_s_all[i]
            msk_holo = self. mask_off_center_beads(z_s, pos[i,0,0], pos[i,0,1])
            # tf.imshow(np.angle(msk_holo ))
            ''' use qudratic phase function to get the hologram, aberration apply at the pupil'''
            if (np.abs(z_s)) == self.f_o:
                bpp = self.getZArrWF(self.zarr)
                pupil = self.Pupil(z_s) 
                maskRadius_p, mask_p = self.Pupil_Mask(z_s)
                I_1 = pupil *((1/2)+((1/2) * np.exp(1j*theta)*self.Quadratic_phase_function(-1/self.f_slm, wl, x, y))) * bpp * mask_p
                I_2 = self.Quadratic_phase_function(1/z_h, wl, x, y) * msk_holo
                I_h = self.iFT(self.FT(I_1)*self.FT(I_2))
            else:
                bpp2, mask_p = self.getZArr_out_of_focus(self.zarr,z_s,z_h)
                bpp = self.getZArrWF(self.zarr)
                # tf.imshow(np.angle(mask_p))
                # tf.imshow(np.angle(bpp))
                
                f_o = self.f_o
                f_e = (z_s*f_o)/(f_o-z_s)
                f_bar1 = (self.f_TL*(f_e+self.d1))/(self.f_TL-(f_e+self.d1))
                f_bar2 = (self.f_2*(f_bar1+self.d2))/(self.f_2-(f_bar1+self.d2))
                f_bar3 = (self.f_3*(f_bar2+self.d3))/(self.f_3-(f_bar2+self.d3))
                f_bar4 = (self.f_4*(f_bar3+self.d4))/(self.f_4-(f_bar3+self.d4))
            
                Lx =  (-pos[i,0,0] * f_e * f_bar1 * f_bar2 * f_bar3 * f_bar4 ) / (z_s * (f_e +self.d1) * (f_bar1 +self.d2)  * (f_bar2 +self.d3)  * (f_bar3 +self.d4)  * (f_bar4 +self.d5))
                Ly =  (-pos[i,0,1] * f_e * f_bar1 * f_bar2 * f_bar3 * f_bar4 ) / (z_s * (f_e +self.d1) * (f_bar1 +self.d2)  * (f_bar2 +self.d3)  * (f_bar3 +self.d4)  * (f_bar4 +self.d5))
        
                I_1 = 1/2 * self.Linear_phase_function(x, y, Lx, Ly, wl) * self.Quadratic_phase_function(1/(f_bar4 + self.d5), wl, x, y) * bpp * mask_p
                I_2 = I_1 * np.exp(1j*theta) * self.Quadratic_phase_function(-1/self.f_slm, wl, x, y) * mask_p
                I_3 = self.Quadratic_phase_function(1/z_h, wl, x, y)  * msk_holo
                
                I_hp = (I_1 + I_2) 
                # I_hp = (I_1 + I_2) * bpp * mask_p
                # tf.imshow(np.angle(I_hp))    
                
                I_h = self.iFT(self.FT(I_hp)*self.FT(I_3))
                I_H = np.abs(I_h * np.conj(I_h)) * msk_holo
                
                h[i] = I_H / I_H.sum()                         #normalization
                h[i] = h[i] * self.Nph
                
                # tf.imshow(np.abs(h[i]))    
                    
        I_e = np.zeros((self.nx*2, self.ny*2))
            
        for i in range(n_emitter_z):
            I_e = I_e + h[i]
        
        I_e = rd.poisson(I_e) + bg_metrix

            
        tf.imshow(np.abs(I_e))    
        return I_e

    
    def generate_holoImgs_multiZ(self, z_h, verbose=False):
        self.imgstack = np.zeros((3, self.nx*2, self.ny*2))
        theta_stack = np.array([ 0 * np.pi / 3 , 2 * np.pi / 3,  4 * np.pi / 3  ])
        # print(self.Mt)
        for i in range(3):
            h = np.zeros((self.nx*2, self.ny*2))
            bg_metrix = rd.poisson (self.bg * np.ones((self.nx*2, self.ny*2)) )
            theta = theta_stack[i]
            h = self.I_function_multiZ( self.xv2, self.yv2, self.wl, theta, z_h)
            # h = h * self.msk
            # h = h / h.sum()                         #normalization
            # h = h * self.Nph
            # h = rd.poisson(h) + bg_metrix
            self.imgstack[i] = h
        fn = os.path.join(self.newfold,'Holo_stack_4beads_pos1.tif')
        tf.imsave(fn,self.imgstack)
        if verbose:
            tf.imshow(np.abs(self.imgstack[0]))
            tf.imshow(np.abs(self.imgstack[1]))
            tf.imshow(np.abs(self.imgstack[2]))
    
    
    
    def I_function_4beads(self, z_s, x, y, wl, theta, z_h):
        ''' use qudratic phase function to get the hologram, aberration apply at the pupil'''
        if (np.abs(z_s)) == self.f_o:
            bpp = self.getZArrWF(self.zarr)
            pupil = self.Pupil(z_s) 
            maskRadius_p, mask_p = self.Pupil_Mask(z_s)
            I_1 = pupil *((1/2)+((1/2) * np.exp(1j*theta)*self.Quadratic_phase_function(-1/self.f_slm, wl, x, y))) * bpp * mask_p
            I_2 = self.Quadratic_phase_function(1/z_h, wl, x, y) * self.msk
            I_h = self.iFT(self.FT(I_1)*self.FT(I_2))
        else:
            bpp2, mask_p = self.getZArr_out_of_focus(self.zarr,z_s,z_h)
            bpp = self.getZArrWF(self.zarr)
            # tf.imshow(np.angle(bpp))
            
            f_o = self.f_o
            f_e = (z_s*f_o)/(f_o-z_s)
            f_bar1 = (self.f_TL*(f_e+self.d1))/(self.f_TL-(f_e+self.d1))
            f_bar2 = (self.f_2*(f_bar1+self.d2))/(self.f_2-(f_bar1+self.d2))
            f_bar3 = (self.f_3*(f_bar2+self.d3))/(self.f_3-(f_bar2+self.d3))
            f_bar4 = (self.f_4*(f_bar3+self.d4))/(self.f_4-(f_bar3+self.d4))
            
            
            pos1 = 0.03
            n_emitter = 4
            pos = np.zeros((n_emitter, 1, 2))
            I_H = np.zeros((n_emitter, self.nx*2, self.ny*2))
            
            pos[0] = (pos1,pos1)
            pos[1] = (-pos1,pos1)
            pos[2] = (pos1,-pos1)
            pos[3] = (-pos1,-pos1)
           
            for i in range(n_emitter):
        
                Lx =  (-pos[i,0,0] * f_e * f_bar1 * f_bar2 * f_bar3 * f_bar4 ) / (z_s * (f_e +self.d1) * (f_bar1 +self.d2)  * (f_bar2 +self.d3)  * (f_bar3 +self.d4)  * (f_bar4 +self.d5))
                Ly =  (-pos[i,0,1] * f_e * f_bar1 * f_bar2 * f_bar3 * f_bar4 ) / (z_s * (f_e +self.d1) * (f_bar1 +self.d2)  * (f_bar2 +self.d3)  * (f_bar3 +self.d4)  * (f_bar4 +self.d5))
        
                I_1 = 1/2 * self.Linear_phase_function(x, y, Lx, Ly, wl) * self.Quadratic_phase_function(1/(f_bar4 + self.d5), wl, x, y) * bpp * mask_p
                I_2 = I_1 * np.exp(1j*theta) * self.Quadratic_phase_function(-1/self.f_slm, wl, x, y) 
                msk = self. mask_off_center_beads(pos[i,0,0], pos[i,0,1])
                I_3 = self.Quadratic_phase_function(1/z_h, wl, x, y)  
                
                I_hp = I_1 + I_2
                
                I_h = self.iFT(self.FT(I_hp)*self.FT(I_3))
                I_H[i] = np.abs(I_h * np.conj(I_h)) * msk
                
            I_e = np.zeros((self.nx*2, self.ny*2))
            for i in range(n_emitter):
                I_e = I_e + I_H[i] 
          
        return I_e

    
    def generate_holoImgs_4beads(self, z_s, z_h, recon_dist, verbose=False):
        self.imgstack = np.zeros((3, self.nx*2, self.ny*2))
        theta_stack = np.array([ 0 * np.pi / 3 , 2 * np.pi / 3,  4 * np.pi / 3  ])
        self.Holo_R = self.Hologram_radius_realsetting(z_s, z_h)
        # print( self.Holo_R)
        self.radius = self.Holo_R / self.dx
        self.msk = (self.xv ** 2 + self.yv ** 2) <= self.radius**2
        self.Mt = self.Transverse_Magnification_realsetting(z_s, z_h)
        # print(self.Mt)
        for i in range(3):
            h = np.zeros((self.nx*2, self.ny*2))
            bg_metrix = rd.poisson (self.bg * np.ones((self.nx*2, self.ny*2)) )
            theta = theta_stack[i]
            h = self.I_function_4beads(z_s, self.xv2, self.yv2, self.wl, theta, z_h)
            # h = h * self.msk
            h = h / h.sum()                         #normalization
            h = h * self.Nph
            # h = rd.poisson(h) + bg_metrix
            self.imgstack[i] = h
        fn = os.path.join(self.newfold,'Holo_stack_4beads.tif')
        tf.imsave(fn,self.imgstack)
        if verbose:
            tf.imshow(np.abs(self.imgstack[0]))
            tf.imshow(np.abs(self.imgstack[1]))
            tf.imshow(np.abs(self.imgstack[2]))
    
    def generate_holoImgs(self, z_s, z_h, recon_dist, sx, sy, verbose=False):
        self.imgstack = np.zeros((3, self.nx*2, self.ny*2))
        theta_stack = np.array([ 0 * np.pi / 3 , 2 * np.pi / 3,  4 * np.pi / 3  ])
        self.Holo_R = self.Hologram_radius_realsetting(z_s, z_h)
        # print( self.Holo_R)
        self.radius = self.Holo_R / self.dx
        self.msk = (self.xv ** 2 + self.yv ** 2) <= self.radius**2
        self.Mt = self.Transverse_Magnification_realsetting(z_s, z_h)
        # print(self.Mt)
        for i in range(3):
            h = np.zeros((self.nx*2, self.ny*2))
            bg_metrix = rd.poisson (self.bg * np.ones((self.nx*2, self.ny*2)) )
            theta = theta_stack[i]
            h = self.I_function(z_s, self.xv2, self.yv2, self.wl, theta, z_h, sx, sy)
            # h = h * self.msk
            h = h / h.sum()                         #normalization
            h = h * self.Nph
            h = rd.poisson(h) + bg_metrix
            self.imgstack[i] = h
        fn = os.path.join(self.newfold,'Holo_stack.tif')
        tf.imsave(fn,self.imgstack)
        if verbose:
            tf.imshow(np.abs(self.imgstack[0]))
            tf.imshow(np.abs(self.imgstack[1]))
            tf.imshow(np.abs(self.imgstack[2]))
            
    def recon(self, recon_dist, imgstack, xv, yv):
        ''' 2d recon '''
        theta1 = 0 * np.pi / 3                               #Theta timed 2 cause there are two exp(i*theta)
        theta2 = 2 * np.pi / 3
        theta3 = 4 * np.pi / 3
        final_intensity = (imgstack[0] * (np.exp(-1j*theta3)-np.exp(-1j*theta2)) +
                           imgstack[1] * (np.exp(-1j*theta1)-np.exp(-1j*theta3)) +
                           imgstack[2] * (np.exp(-1j*theta2)-np.exp(-1j*theta1)))
        recon_temp = np.exp((1j*pi/self.wl/recon_dist)*((xv)**2+(yv)**2)) 
        g = ifftshift(ifft2(fft2(final_intensity)*fft2(recon_temp)))
        return  final_intensity, np.abs(g)
        
            
    def finch_recon(self,recon_dist, verbose=False):
        ''' 2d recon '''
        self.final_intensity, self.holo_recon = self.recon(recon_dist, self.imgstack, self.xv1, self.yv1)
        if verbose:
            # tf.imshow(np.abs(self.final_intensity))
            # tf.imshow(np.angle(self.final_intensity))
            # tf.imshow(np.abs(self.holo_recon))
            fn = os.path.join(self.newfold,'Befroe_sensorlessAO.tif')
            tf.imsave(fn, np.abs(self.holo_recon))
        
            
    def finch_recon3D(self, HF, z_s, z_h, k):   
        ''' 3d recon '''                                                      # 3D reconstruction 
        z_depth = np.arange((z_s-0.01), (z_s+0.01), self.dz)                           #z_depth(-10um to 10um, step size:0.2 um)
        zr_depth = np.zeros(len(z_depth))
        for i in range(len(z_depth)):
            zr_depth[i] = self.recon_dist_calc_realsetting(z_depth[i], z_h)                          #z_depth : 20um
        self.Recon_3d_stack = np.zeros((len(zr_depth), self.nx*2, self.ny*2))
        self.intensity_stack = np.zeros((len(zr_depth), self.nx*2, self.ny*2))
        for m in range(len(zr_depth)):
            recon_temp = np.exp((1j*pi/self.wl/zr_depth[m])*((self.xv1)**2+(self.yv1)**2)) 
            g = ifftshift(ifft2(fft2(HF)*fft2(recon_temp)))
            self.Recon_3d_stack[m,:,:] = np.abs(g)
        if k == 1:
            # tf.imwrite('Recon_3d_stack_afterAO.tif',self.Recon_3d_stack)
            fn = os.path.join(self.newfold, 'Recon_3d_stack_afterAO.tif')
            tf.imsave(fn,self.Recon_3d_stack) 
        elif k == 0:
            # tf.imwrite('Recon_3d_stack_beforeAO.tif',self.Recon_3d_stack)
            fn = os.path.join(self.newfold, 'Recon_3d_stack_beforeAO.tif')
            tf.imsave(fn,self.Recon_3d_stack) 
    
    
    def Inverse_I_function_HF(self, imgstack, x, y, wl, z_h, zarr, recon_dist):               #correction applied at virtual pupil
        '''apply AO to complex hologram at BS'''                                
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
       
        return np.abs(g), I_f

    
   
    
    def ao_optimize_snr_HF(self, imgstack, z_s, z_h, amp_ini, amp_start, amp_stop, amp_step, mode_start, mode_stop, recon_dist):
        results = []
        results.append(('Mode','Amp','Metric(snr)'))
        modes = np.arange(mode_start,mode_stop,1)
        zarr_ini = amp_ini
        zarr_final = np.zeros(24)
        zarr_t = zarr_ini
        for mode in modes:
            amprange = np.arange(amp_start[mode],amp_stop[mode],amp_step[mode])
            dt = np.zeros(amprange.shape)
            dt_stack = np.zeros((24, amprange.size))
            print(zarr_t)
            for k, amp in enumerate(amprange):
                zarr_t[mode] = amp
                zarr = zarr_t
                # zarr = list(map(add, zarr, zarr_final))
                recon, I_f = self.Inverse_I_function_HF(imgstack, self.xv2, self.yv2, self.wl, z_h, zarr, recon_dist)
                temp = recon
                
                fn = "zm%0.2d_amp%.4f" %(mode,amp)
                fn1 = os.path.join(self.newfold,fn+'.tif')
                tf.imsave(fn1, temp)
                
                dt[k]= self.snr(self.wl,temp,self.dx) # metric is snr
                
                results.append((mode,amp,dt[k]))
                print(k, amp, dt[k])
            pmax = self.peak(amprange, dt)
            dt_stack[mode] = dt
            if (pmax!=0.0):
                print('----------------setting mode %d at value of %f' % (mode, pmax))
                zarr_final[mode] = pmax
                zarr_t[mode] = pmax
            else:
                print('----------------mode %d value equals %f' % (mode, pmax))
                zarr_final[mode] = 0
                zarr_t[mode] = 0
            
        
        self.res = results  
 
        print(zarr_final)
        recon, I_f = self.Inverse_I_function_HF(imgstack, self.xv2, self.yv2, self.wl, z_h, zarr_final, recon_dist)
        
        after_ao = recon
        fn1 = os.path.join(self.newfold,'After_sensorlessAO_recon.tif')
        tf.imsave(fn1, after_ao)
        df = pd.DataFrame(self.res)
        df.to_excel(self.newfold  + '/SNR.xlsx')
        df = pd.DataFrame(zarr_final)
        df.to_excel(self.newfold  + 'final_AO.xlsx')
        return I_f
         
        
        
    def Inverse_I_function_Fe(self, imgstack, x, y, wl, z_h, zarr, recon_dist, msk, z_s):               #compare with Man's paper
        phiin = self.getZArrWF(zarr)
        theta1 = 0 * np.pi / 3                               #Theta timed 2 cause there are two exp(i*theta)
        theta2 = 2 * np.pi / 3
        theta3 = 4 * np.pi / 3
        final_intensity = (imgstack[0] * (np.exp(-1j*theta3)-np.exp(-1j*theta2)) +
                            imgstack[1] * (np.exp(-1j*theta1)-np.exp(-1j*theta3)) +
                            imgstack[2] * (np.exp(-1j*theta2)-np.exp(-1j*theta1)))
        recon_temp = np.exp((1j*pi/self.wl/recon_dist)*((x)**2+(y)**2)) 
        g_1 = ifftshift(ifft2(fft2(final_intensity)*fft2(recon_temp)))
        Fe_g = fftshift(fft2(ifftshift(g_1)))
        # tf.imshow(np.abs(Fe_g))
        Fe_g = Fe_g * phiin
        g = fftshift(fft2(ifftshift(Fe_g)))
        return np.abs(g), Fe_g
    
    
    def Inverse_I_function_Fe3D(self, imgstack, x, y, wl, z_h, zarr, msk):               #compare with other paper

        z_depth = np.arange((z_s-0.01), (z_s+0.01), self.dz)                           #z_depth(-10um to 10um, step size:0.2 um)
        zr_depth = np.zeros(len(z_depth))
        for i in range(len(z_depth)):
            zr_depth[i] = self.recon_dist_calc_realsetting(z_depth[i], z_h)                          #z_depth : 20um
        Recon_3d_stack = np.zeros((len(zr_depth), self.nx*2, self.ny*2))
        intensity_stack = np.zeros((len(zr_depth), self.nx*2, self.ny*2))
        for i in range(len(zr_depth)):
            Recon_3d_stack[i,:,:], intensity_stack[i,:,:] =self.Inverse_I_function_Fe(imgstack, x, y, wl, z_h, zarr, zr_depth[i], msk, z_depth[i])
        return Recon_3d_stack
    
        
    def ao_optimize_snr_Fe(self, imgstack, z_s, z_h, amp_ini, amp_start, amp_stop, amp_step, mode_start, mode_stop, recon_dist):       # compare with other paper
        results = []
        results.append(('Mode','Amp','Metric(snr)'))
        
        modes = np.arange(mode_start,mode_stop,1)
        zarr_ini = amp_ini
        zarr_final = np.zeros(24)
        self.Holo_R = self.Hologram_radius_realsetting(z_s, z_h)
        # print( self.Holo_R)
        self.radius = self.Holo_R / self.dx
        msk = (self.xv ** 2 + self.yv ** 2) <= self.radius**2
        zarr_t = zarr_ini
        for mode in modes:
            amprange = np.arange(amp_start[mode],amp_stop[mode],amp_step[mode])
            dt = np.zeros(amprange.shape)
            dt_stack = np.zeros((24, amprange.size))
            for k, amp in enumerate(amprange):
                zarr_t[mode] = amp
                zarr = zarr_t
                recon, I_f = self.Inverse_I_function_Fe(imgstack, self.xv2, self.yv2, self.wl, z_h, zarr, recon_dist, msk, z_s)
                temp = recon
                
                fn = "zm%0.2d_amp%.4f" %(mode,amp)
                fn1 = os.path.join(self.newfold,fn+'.tif')
                tf.imsave(fn1, temp)
                
                dt[k]= self.snr(self.wl,temp,self.dx) # metric is snr
                
                results.append((mode,amp,dt[k]))
                print(k, amp, dt[k])
            pmax = self.peak(amprange, dt)
            dt_stack[mode] = dt
            if (pmax!=0.0):
                print('----------------setting mode %d at value of %f' % (mode, pmax))
                zarr_final[mode] = pmax
                zarr_t[mode] = pmax
                # print(zarr_final)
            else:
                print('----------------mode %d value equals %f' % (mode, pmax))
                zarr_final[mode] = 0
                zarr_t[mode] = 0
                
        self.res = results  

        recon, I_f = self.Inverse_I_function_Fe(imgstack, self.xv2, self.yv2, self.wl, z_h, zarr_final, recon_dist, msk, z_s)
        after_ao = recon
        fn1 = os.path.join(self.newfold,'After_sensorlessAO_recon.tif')
        tf.imsave(fn1, after_ao)
        # fn2 = os.path.join(self.newfold,'After_sensorlessAO_FinalCompHolo.tif')
        # tf.imshow(np.angle(I_f))
        # tf.imsave(fn2, I_f)
        df = pd.DataFrame(self.res)
        df.to_excel(self.newfold  + '/SNR.xlsx')
        return zarr_final
    
   

if __name__ == '__main__':
    sx = 0            
    sy = 0
    t = hologram_simulation()
    z_s = 3.0001
    z_h = 100
    m = 0.01        # step size or the AO searching


    amp_ini = [0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0]                           #initial correction mask for start searching
    
    amp_start = [0,    0,    0,     0,     -1.5,   -1.5,  
                -1.5,    -1.5,    -1.5,     -1.5,     -1.5,   -1.5,
                -1.5,  -1.5,  -1.5,   0,       0,      0.5, 
                  0,    0,    0,     0,       0,      0]              #start searching
     
    
    amp_stop  = [0,   0,   0,  0,   0.6,
                1.5,   1.5,   1.5,   1.5,   1.5,
                1.5,   1.5,   1.5,   1.5,   1.5,
                1.5,   1.5,   1.5,   1.5,    0,
                0,   0,   0,   0]                                    #end searching

    
    amp_step = [0,   0,   0,   0,   m,
                m,   m,   m,   m,   m,
                m,   m,   m,   m,   m,
                m,   m,   m,   m,   0,
                0,   0,   0,   0]                                #searching step size
    
    mode_start = 4
    mode_stop = 19

    t.zarr = [0,   0,   0,   0,   0,    0,  
              0,   0,   0,   0,   0,    0,   
              0,   0,   0,   0,   0,    0,  
              0,   0,   0,   0,   0,   0]                             #Input aberration 

    
    
############################ Generate Hologram with zarr #############################################
    
    z_r = t.recon_dist_calc_realsetting(z_s, z_h) 


    # t.generate_holoImgs(z_s, z_h, z_r, sx , sy, verbose = False)            # Generate single hologram at (sx,sy)
    # t.generate_holoImgs_4beads(z_s, z_h, z_r, verbose = False)              # Generate 4 hologram at four conners of the FOV
    t.generate_holoImgs_multiZ(z_h, verbose = False)                        # Generate 11 holograms distribute in the 3D volume
    

    # path = 'Holo_stack_4beads_pos1.tif'                                      # Load local data
    # t.imgstack = tf.imread(path)
    
# # ############################ Reconstuction without AO #############################################

#     H_f_noAO, recon_noAO = t.recon( z_r, t.imgstack, t.xv1, t.yv1)
#     # # tf.imshow(np.abs(H_f_noAO)) 
#     # tf.imshow(np.abs(recon_noAO)) 
#     # # # t.finch_recon(z_r,verbose=True)
#     t.finch_recon3D(H_f_noAO, z_s, z_h, 0)

# # ############################ AO searching algorithm #############################################
   
#     H_f_Ao = t.ao_optimize_snr_HF(t.imgstack, z_s, z_h, amp_ini, amp_start, amp_stop, amp_step, mode_start, mode_stop, z_r)
#     # t.finch_recon3D(H_f_Ao, 3.00001, z_h, 1)


# ######################### Reconstruction with AO, the input correction msak is defined in amp_ini #############################################    
#     recon_AO, H_f_ao = t.Inverse_I_function_HF(t.imgstack, t.xv1, t.yv1, t.wl, z_h, amp_ini, z_r) 
#     # start = time.time()
#     t.finch_recon3D(H_f_ao, z_s, z_h, 1)
#     # end = time.time()
#     # print(end - start)
    
    
# ############################ AO searching algorithm using Man's method #############################################
   
#     zarr_final = t.ao_optimize_snr_Fe(t.imgstack, z_s, z_h, amp_ini, amp_start, amp_stop, amp_step, mode_start, mode_stop, z_r)
#     Recon_3d_stack = t.Inverse_I_function_Fe3D(t.imgstack, t.xv1, t.yv1, t.wl, z_h, amp_ini, t.msk)
#     # tf.imshow(np.abs(recon_AO)) 
#     # fn = os.path.join(t.newfold, 'Recon_3d_stack_afterAO.tif')
#     # tf.imsave(fn,Recon_3d_stack) 
    
# ############################ Reconstruction with AO using Man's method, the input correction msak is defined in amp_ini#############################################
#     # start = time.time() 
#     Recon_3d_stack = t.Inverse_I_function_Fe3D(t.imgstack, t.xv1, t.yv1, t.wl, z_h, amp_ini, t.msk)
#     # tf.imshow(np.abs(recon_AO)) 
#     # fn = os.path.join(t.newfold, 'Recon_3d_stack_afterAO.tif')
#     # tf.imsave(fn,Recon_3d_stack) 
#     # end = time.time()
#     # print(end - start)
    
    
    
