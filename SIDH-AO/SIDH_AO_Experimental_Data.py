#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 01:36:09 2023

@author: shaohli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 21:34:08 2023
SNR searching for adding corection at final complex hologram
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
        self.dx = 0.016 #pixel size
        self.nx = 256
        self.ny = 256
        self.dp = 1 / (self.nx * 2 * self.dx)
        self.k = 2 * np.pi / self.wl
        
        self.dz = 0.0002                        # z stepsize 500nm
        
        self.f_o = 3       # Focal length of objective (mm)
        self.z_s = 3      # Distance between sample and objective (mm)
        self.f_slm = 300
          
        self.f_TL = 180.         #focal length of tube lens
        self.f_2 = 120.          #focal length of second lens
        self.d1 = 183.           #distance between objective and tube lens
        self.d2 = self.f_TL + self.f_2          #distance between tube lens and second lens
        self.f_3 = 200.          #focal length of third lens
        self.f_4 = 100.          #focal length of fourth lens
        self.d3 = self.f_2 + self.f_3         #distance between second lens and third lens
        # self.d4 = self.f_4 + self.f_3          #distance between third lens and fourth lens
        # self.d5 = self.f_4          #distance between fourth lens and interferometer
        
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
                     0,   0,   0,   0]
        
        t=time.localtime() 
        x = np.array([1e4,1e2,1])    
        t1 = int((t[0:3]*x).sum())
        t2 = int((t[3:6]*x).sum())
        fnt = "%s%s" %(t1,t2)
        self.newfold = fnt + '_ao_iteration' + '_snr' + '/'
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
        print("R_h")
        print(Holo_R_real)
        return Holo_R_real
   
    
    def Linear_phase_function(self, x, y, xs, ys, wl ,recon_dist):
        L = np.exp((1j * 2 * np.pi ) * (wl ** (-1)) * ((sx) * x + (sy) *y))
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
        ''' introduce the aberration at the interferometer pupil plane '''
        ph = np.zeros((self.nx*2, self.ny*2),dtype=np.float32)
        maskRadius, mask = self.Pupil_Mask(self.f_o)
        for j,m in enumerate(zarr):
            ph = ph + m*Z.Zm(j,rad=maskRadius,orig=None,Nx=self.nx*2)
        bpp = mask*np.exp(1j*ph)
        # tf.imshow(np.angle(bpp))
        return bpp
    
    def getZArr_HF(self, zarr,z_s,z_h):     
        ''' get the aberration at the complex hologram (H_F) after back propogate Z_r, the radius of the  H_F is 2 times of H_R'''
        ph = np.zeros((self.nx*2, self.ny*2),dtype=np.float32)
        # HF_R = 1.5 * self.Hologram_radius_realsetting(z_s, z_h)
        HF_R = 0.5 * self.Hologram_radius_realsetting(z_s, z_h)
        maskRadius = HF_R / self.dx
        mask = (self.xv ** 2 + self.yv ** 2) <= maskRadius**2
        for j,m in enumerate(zarr):
            ph = ph + m*Z.Zm(j,rad=maskRadius,orig=None,Nx=self.nx*2)
        bpp = mask*np.exp(1j*ph) 
        # tf.imshow(np.angle(bpp))
        return bpp, mask
    
    
    def Inverse_I_function_HF(self, imgstack, x, y, wl, z_h, z_s, zarr, recon_dist):               #correction applied at virtual pupil
        '''apply AO to complex hologram at BS'''              
        phiin, HF_mask = self.getZArr_HF(zarr, z_s, z_h)
        theta1 = 0 * np.pi / 3                               #Theta timed 2 cause there are two exp(i*theta)
        theta2 = 2 * np.pi / 3
        theta3 = 4 * np.pi / 3
        final_intensity = (imgstack[0] * (np.exp(-1j*theta3)-np.exp(-1j*theta2)) +
                            imgstack[1] * (np.exp(-1j*theta1)-np.exp(-1j*theta3)) +
                            imgstack[2] * (np.exp(-1j*theta2)-np.exp(-1j*theta1)))
        
        f_vir = 160
        
        HF_R_2 = 1.4 *  self.Pupil_Radius(z_s)
        maskRadius_2 = HF_R_2 / self.dx
        mask_2 = (self.xv ** 2 + self.yv ** 2) <= maskRadius_2**2
        # tf.imshow(mask_2)
        # tf.imshow(phiin)
        
        
        I_2 = self.Quadratic_phase_function(1/(self.f_slm+f_vir-z_h), wl, x, y) * mask_2
        I_3 = self.Quadratic_phase_function(-1/f_vir, wl, x, y) 
        I_4 = self.Quadratic_phase_function(1/(f_vir), wl, x, y)  * mask_2
        
        
        I_f =  self.iFT(self.FT(final_intensity) * self.FT(I_2))       # before the virtual lens

        I_f = I_f * I_3                                           
        I_f = self.iFT(self.FT(I_f) * self.FT(I_4)) * phiin         # at virtual pupil plane
        # tf.imshow(np.abs(I_f))
        
   
        I_b = self.Quadratic_phase_function(1/(self.f_slm+f_vir-z_h), wl, x, y) 
    
    
        I_f = self.iFT(self.FT(I_f) * np.conj( self.FT(I_4)))            # after the virtual lens
    
        
        I_f = I_f / I_3
        I_f = self.iFT(self.FT(I_f) * np.conj( self.FT(I_b)))       # at camera plane
        
        ######################################
        
        recon_temp = np.exp((1j*pi/self.wl/recon_dist)*((x)**2+(y)**2))
        g = ifftshift(ifft2(fft2(I_f)*fft2(recon_temp)))
        
        # # g  = np.roll(g, 2, axis=1) # right
        # # g  = np.roll(g, -2, axis=0) # right
        tf.imshow(np.abs(g))
        # tf.imshow(np.abs(g_ori))

        return np.abs(g), I_f
    
     
    def Inverse_I_function_HF_BS_nor(self, imgstack, x, y, wl, z_h, recon_dist):               #correction applied at virtual pupil
        '''apply AO to complex hologram at BS'''         
        zarr = [0,   0,   0,   0,      0,
                   0,   0,   0,   0,   0,
                   0,   0,   0,   0,   0,
                   0,   0,   0,   0,   0,
                   0,   0,   0,   0]
        z_s = 2.9968          
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
        # tf.imshow(np.abs(I_f))
        
   
        I_b = self.Quadratic_phase_function(1/(self.f_slm+f_vir-z_h), wl, x, y) 
    
    
        I_f = self.iFT(self.FT(I_f) * np.conj( self.FT(I_4)))            # after the virtual lens
    
        
        I_f = I_f / I_3
        I_f = self.iFT(self.FT(I_f) * np.conj( self.FT(I_b)))       # at camera plane
        
        ######################################
        
        recon_temp = np.exp((1j*pi/self.wl/recon_dist)*((x)**2+(y)**2))
        g = ifftshift(ifft2(fft2(I_f)*fft2(recon_temp)))
        
        # # g  = np.roll(g, 2, axis=1) # right
        # # g  = np.roll(g, -2, axis=0) # right
        tf.imshow(np.abs(g))
        # tf.imshow(np.abs(g_ori))

        return np.abs(g), I_f
           
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
  
   
    def snr(self, wl,img,dx):  #frequency filter
        nx,ny = img.shape
        self.Mt = self.Transverse_Magnification_realsetting(z_s, z_h)
        dk   = 1 / (nx * (self.dx/self.Mt) ) 
        maskRadius =  self.na / self.wl / dk
        radius = int(nx/6)
        # msk = (self.xv ** 2 + self.yv ** 2) <= (radius)**2
        img = img  
        tf.imshow(img)
        siglp = maskRadius * 0.05
        sighp = maskRadius * 0.7
        msk_1 =  (self.xv ** 2 + self.yv ** 2) <= (maskRadius)**2      
        lp = self.gaussianArr(shape=(nx,nx), sigma=siglp, peakVal=1, orig=None, dtype=np.float32)
        hp = 1-self.gaussianArr(shape=(nx,nx), sigma=sighp, peakVal=1, orig=None, dtype=np.float32)
        aft = fftshift(fft2(img))
        hpC = (np.abs(hp*aft*msk_1)).sum()
        lpC = (np.abs(lp*aft*msk_1)).sum()
        res = hpC/lpC
        return res
    
    
    # def snr(self, wl,img,dx):       #peak intensity
    #     nx,ny = img.shape
    #     radius = int(nx/6)
    #     msk = (self.xv ** 2 + self.yv ** 2) <= (radius)**2
    #     img = img*msk
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
        
    
    def ao_optimize_snr(self, imgstack, z_s, z_h, amp_ini, amp_start, amp_stop, amp_step, mode_start, mode_stop, recon_dist):
        # newfold = self.path + '/' + fnt + '_ao_iteration' + '_snr' + '/'
        results = []
        results.append(('Mode','Amp','Metric(snr)'))
        # print(amprange)
        modes = np.arange(mode_start,mode_stop,1)
        zarr_ini = amp_ini
        zarr_final = np.zeros(24)
        zarr_t = zarr_ini
        for mode in modes:
            amprange = np.arange(amp_start[mode],amp_stop[mode],amp_step[mode])
            dt = np.zeros(amprange.shape)
            dt_stack = np.zeros((24, amprange.size))
            for k, amp in enumerate(amprange):
                zarr_t[mode] = amp
                zarr = zarr_t
                # print(zarr)
                recon, I_f = self.Inverse_I_function_HF(self.imgstack, self.xv1, self.yv1, self.wl, z_h, z_s, zarr, recon_dist) 
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
        recon, I_f = self.Inverse_I_function_HF(self.imgstack, self.xv1, self.yv1, self.wl, z_h, z_s, zarr_final, recon_dist) 
        after_ao = recon
        fn1 = os.path.join(self.newfold,'After_sensorlessAO_recon.tif')
        tf.imsave(fn1, after_ao)
        # fn2 = os.path.join(self.newfold,'After_sensorlessAO_FinalCompHolo.tif')
        # tf.imshow(np.angle(I_f))
        # tf.imsave(fn2, I_f)
        df = pd.DataFrame(self.res)
        df.to_excel(self.newfold  + '/SNR.xlsx')
        df = pd.DataFrame(zarr_final)
        df.to_excel(self.newfold  + 'final_AO.xlsx')
        return I_f
  
    
    def recon(self, recon_dist, imgstack, xv, yv, msk):
        theta1 = 0 * np.pi / 3                               #Theta timed 2 cause there are two exp(i*theta)
        theta2 = 2 * np.pi / 3
        theta3 = 4 * np.pi / 3
        final_intensity = (imgstack[0,:,:] * (np.exp(-1j*theta3)-np.exp(-1j*theta2)) +
                           imgstack[1,:,:] * (np.exp(-1j*theta1)-np.exp(-1j*theta3)) +
                           imgstack[2,:,:] * (np.exp(-1j*theta2)-np.exp(-1j*theta1)))
        recon_temp = np.exp((1j*pi/self.wl/recon_dist)*((xv)**2+(yv)**2)) * msk
        g = ifftshift(ifft2(fft2(final_intensity)*fft2(recon_temp)))
        return  final_intensity, np.abs(g)
    
    
        
            
    def finch_recon(self,recon_dist, verbose=False):
        self.final_intensity, self.holo_recon = self.recon(recon_dist, self.imgstack, self.xv1, self.yv1, self.msk)
        if verbose:
            # tf.imshow(np.abs(self.final_intensity))
            # tf.imshow(np.angle(self.final_intensity))
            # tf.imshow(np.abs(self.holo_recon))
            fn = os.path.join(self.newfold,'Befroe_sensorlessAO.tif')
            tf.imsave(fn, np.abs(self.holo_recon))
        
            
    def finch_recon3D(self, HF, z_s, z_h, sx, sy,k):                                                         # 3D reconstruction of FINCH
        z_depth = np.arange((z_s-0.01), (z_s+0.01), self.dz)                           #z_depth(-10um to 10um, step size:1 um)
        zr_depth = np.zeros(len(z_depth))
        for i in range(len(z_depth)):
            zr_depth[i] = self.recon_dist_calc_realsetting(z_depth[i], z_h)                          #z_depth : 20um
        self.Recon_3d_stack = np.zeros((len(zr_depth), self.nx*2, self.ny*2))
        self.intensity_stack = np.zeros((len(zr_depth), self.nx*2, self.ny*2))
        for m in range(len(zr_depth)):
            recon_temp = np.exp((1j*pi/self.wl/zr_depth[m])*((self.xv1)**2+(self.yv1)**2)) * self.msk
            g = ifftshift(ifft2(fft2(HF)*fft2(recon_temp)))
            self.Recon_3d_stack[m,:,:] = np.abs(g)
        # tf.imwrite('Holo_final_intensity_stack' + '_' + td[-5:] + '.tif',self.intensity_stack)
        if k == 1:
            # tf.imwrite('Recon_3d_stack_afterAO.tif',self.Recon_3d_stack)
            fn = os.path.join(self.newfold, 'Recon_3d_stack_afterAO.tif')
            tf.imsave(fn,self.Recon_3d_stack) 
        elif k == 0:
            # tf.imwrite('Recon_3d_stack_beforeAO.tif',self.Recon_3d_stack)
            fn = os.path.join(self.newfold, 'Recon_3d_stack_beforeAO.tif')
            tf.imsave(fn,self.Recon_3d_stack) 
            
         
    def Inverse_I_function_Fe(self, imgstack, x, y, wl, z_h, zarr, recon_dist, msk, z_s):               #compare with other paper
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
        # newfold = self.path + '/' + fnt + '_ao_iteration' + '_snr' + '/'
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
                
        # plt.plot(amprange, dt_stack[4])
        # plt.plot(amprange, dt_stack[5])
        # plt.plot(amprange, dt_stack[6],'.')
        # plt.plot(amprange, dt_stack[7])
        
        self.res = results  
        # zarr_final[6]=-0.5
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
    z_s = 2.9998
    z_h = 130
    
    amp_ini = [0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,
               0,   0,   0,   0]
    
    amp_start = [0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,
                 0,   0,   0,   0]
    
    amp_stop  = [0,   0,   0,   0,  0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0]
    
    amp_step = [0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0]
    mode_start = 4
    mode_stop = 6
        

    path = '/Users/shaohli/Desktop/Jornal Paper/OE2024/Data/Fig4_StrehlRatio_overlapPSF /WF_200nmDR_800gain_2mW_AO/WF_200nmDR_800gain_2mW_AO_MMStack_Pos0.ome.tif'
    path1 = os.path.dirname(path)
    t.imgstack = tf.imread(path)

    
    z_r = t.recon_dist_calc_realsetting(z_s, z_h) 
############################ Reconstruction without AO #############################################
    
    I_f_noAO, recon_noAO = t.recon(z_r, t.imgstack, t.xv1, t.yv1, t.msk)
    # tf.imshow(np.abs(recon_noAO))
    t.finch_recon3D(I_f_noAO, z_s, z_h, sx, sy, 0)
    fn = os.path.join(t.newfold, 'Recon_3d_stack_noAO.tif')
    
############################ AO searching algorithm #############################################

    I_f_ao = t.ao_optimize_snr(t.imgstack, z_s, z_h, amp_ini, amp_start, amp_stop, amp_step, mode_start, mode_stop, z_r)
    # start = time.time()
    t.finch_recon3D(I_f_ao, z_s, z_h, sx, sy, 1)
    # end = time.time()
    # print(end - start)
    # fn = os.path.join(t.newfold, 'Recon_3d_stack_afterAO.tif')

############################AO searching algorithm using FET (Man's paper) #############################################
    # zarr_final = t.ao_optimize_snr_Fe(t.imgstack, z_s, z_h, amp_ini, amp_start, amp_stop, amp_step, mode_start, mode_stop, z_r)
    
    # start = time.time()
    # Recon_3d_stack = t.Inverse_I_function_Fe3D(t.imgstack, t.xv1, t.yv1, t.wl, z_h, amp_ini, t.msk)
    # tf.imshow(np.abs(recon_AO)) 
    # end = time.time()
    # print(end - start)
    # fn = os.path.join(t.newfold, 'Recon_3d_stack_afterAO.tif')
    # tf.imsave(fn,Recon_3d_stack) 
 
    

  