"""
Implement 2D and 3D phase retrieval using differential phase contrast (DPC)

Michael Chen
Jun 2, 2017

Edited by Zack Phillips
Nov 11, 2017
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter, gaussian_filter
from matplotlib_scalebar.scalebar import ScaleBar

# libwallerlab imports
# from libwallerlab.algorithms.iteralg import _norm,_softThreshold,_softThreshold_isoTV
# from libwallerlab.algorithms.iteralg import gradientDescent,FISTA,newton,lbfgs
from libwallerlab.utilities.display import getRoi
# from libwallerlab.utilities.transforms import Fourier
from libwallerlab.utilities import io

# Optics Algorithms imports
from ..abstract import OpticsAlgorithmOptions, OpticsAlgorithm

class DpcOptions(OpticsAlgorithmOptions):
    '''
    Dpc Options class, used to define options for the DpcSolver
    '''
    def __init__(self, from_dict=None):

        # DPC options
        self.dpc_count = 4
        self.algorithm = 'l2'
        self.plot_verbose = False
        self.phase_offset_method = 'positive'
        self.result_gaussian_filter_sigma = 30

        # Regularization params
        self.reg_u = 1e-6
        self.reg_p = 1e-6
        self.reg_TV = (1e-3,1e-3)
        self.rho = 1e-3
        self.bulk_refractive_index = 1.3
        self.regularization_order = 2
        self.max_tv_iterations = 5
        OpticsAlgorithmOptions.__init__(self, from_dict)

class DpcSolver(OpticsAlgorithm):
    '''
    Main DPC solver class
    '''
    def __init__(self, dataset, options=None):

        # Create obtions object if it hasn't been created already
        if options is None:
            options = DpcOptions()

        # Initialize the metaclass
        OpticsAlgorithm.__init__(self, dataset, options)
        self.dataset = dataset

        # Coordinate definitions
        self.xlin = self._genGrid(self.n, self.dataset.metadata.system.eff_pixel_size_um)
        self.ylin = self._genGrid(self.m, self.dataset.metadata.system.eff_pixel_size_um)
        self.dfx = 1.0 / (self.n * self.dataset.metadata.system.eff_pixel_size_um)
        self.dfy = 1.0 / (self.m * self.dataset.metadata.system.eff_pixel_size_um)
        self.fxlin = np.fft.ifftshift(self._genGrid(self.n, self.dfx))
        self.fylin = np.fft.ifftshift(self._genGrid(self.m, self.dfy))
        self.measurements_are_normalized = False

        # Instantiate DPC-related options
        self.Fobj = Fourier(self.frame_list[0].shape,(-1,-2))
        self.normalize()
        self.genPupil()
        self.genSource()
        self.genWotf()
        # self.f,self.ax = plt.subplots(1,3,figsize=(16,4))
        # plt.close()

    def _genGrid(self, size, dx):
        xlin = np.arange(size, dtype='complex128')
        return (xlin-size // 2)*dx

    def genSource(self):
        '''
        Function to generate source
        '''

        # This variable has the same length as frame_state_list and cntains a list of list with all leds and colors which are turned on during this acquisition
        self.source = []

        # Loop over frames
        max_val = 0
        for frame_index in range(self.n_frames):
            frame_state = self.frame_state_list[frame_index]
            frame_shift = []

            # Generate new source
            self.source.append({})
            for color_name in self.colors_used:
                self.source[-1][color_name] = np.zeros((self.m, self.n), dtype=np.complex128)

            # Looper over time points within frame exposure (flickering LEDs)
            for time_point_index in range(len(frame_state['illumination']['sequence'])):
                # Loop over leds in pattern
                for led_index in range(len(frame_state['illumination']['sequence'][time_point_index])):
                    # Loop over colors in LEDs
                    for color_name in frame_state['illumination']['sequence'][time_point_index][led_index]['value']:

                        # Extract values
                        value = frame_state['illumination']['sequence'][time_point_index][led_index]['value'][color_name] / ((2 ** self.dataset.metadata.illumination.bit_depth) - 1)
                        led_number = frame_state['illumination']['sequence'][time_point_index][led_index]['index']

                        # Only Add this led and color if it's turned on
                        if value > 0:

                            # Add this LED to list of LEDs which are on in this frame
                            pupil_shift_x = np.round(self.source_list_na[led_number][0] / self.wavelength_um[color_name] * self.metadata.system.eff_pixel_size_um * self.n)
                            pupil_shift_y = np.round(self.source_list_na[led_number][1] / self.wavelength_um[color_name] * self.metadata.system.eff_pixel_size_um * self.m)

                            # if np.abs(pupil_shift_x) <= self.n / 2 and np.abs(pupil_shift_x) <= self.m / 2:
                            self.source[-1][color_name][int(self.m // 2 + pupil_shift_y), int(self.n // 2 + pupil_shift_x)] = value
                            max_val = max(value, max_val)

            for color_name in self.colors_used:
                self.source[-1][color_name] /= max_val

    def genSource_old(self):
        self.source = []
        P = genPupil(self.metadata,self.metadata.NA,NA_in=self.metadata.NA_in)

        for rotIdx in range(self.options.dpc_count):
            self.source.append(np.zeros((self.metadata.dim)))
            rotdegree = self.metadata.rotation[rotIdx]
            if rotdegree <180:
                self.source[-1][self.metadata.fylin[:, np.newaxis]*np.cos(np.deg2rad(rotdegree))+1e-15>=
                                self.metadata.fxlin[ np.newaxis,:]*np.sin(np.deg2rad(rotdegree))] = 1.0
                self.source[-1] *= P
            else:
                self.source[-1][self.metadata.fylin[:, np.newaxis]*np.cos(np.deg2rad(rotdegree))+1e-15<
                                self.metadata.fxlin[ np.newaxis,:]*np.sin(np.deg2rad(rotdegree))] = -1.0
                self.source[-1] *= P
                self.source[-1] += P
        self.source = np.asarray(self.source)

    def genWotf(self):
        F = lambda x: self.Fobj.fourierTransform(x)
        IF = lambda x: self.Fobj.inverseFourierTransform(x)
        self.Hu = []
        self.Hp = []
        for frame_index in range(len(self.dataset.frame_state_list)):
            # self.Hu.append({})
            # self.Hp.append({})
            self.Hu.append(np.zeros((self.m, self.n), dtype=np.complex128))
            self.Hp.append(np.zeros((self.m, self.n), dtype=np.complex128))

            for color_name in self.colors_used:
                if np.sum(np.abs(self.source[frame_index][color_name])) > 0:
                    G = ((self.options.bulk_refractive_index / self.dataset.metadata.illumination.spectrum.center[color_name]) ** 2 - \
                         (self.fxlin[ np.newaxis, :] ** 2 + self.fylin[:,  np.newaxis] ** 2)) ** 0.5

                    FSP_cFPG = F(self.source[frame_index][color_name] * self.pupil[color_name]) * F(self.pupil[color_name] / G).conj()
                    I0 = (self.source[frame_index][color_name] * self.pupil[color_name] * self.pupil[color_name].conj()).sum()
                    # self.Hu[-1][color_name] = 2.0 * IF(FSP_cFPG.real) * G / I0
                    # self.Hp[-1][color_name] = 2.0 * 1j * IF(1j * FSP_cFPG.imag) * G / I0
                    Hu = 2.0 * IF(FSP_cFPG.real) * G / I0
                    Hp = 2.0 * 1j * IF(1j * FSP_cFPG.imag) * G / I0
                    self.Hu[-1] += np.nan_to_num(Hu)
                    self.Hp[-1] += np.nan_to_num(Hp)

        self.Hu = np.asarray(self.Hu)
        self.Hp = np.asarray(self.Hp)

    def genPupil(self):
        # Create grid in Fourier domain
        fy = np.arange(-self.m / 2, self.m / 2) / (self.eff_pixel_size * self.m)
        fx = np.arange(-self.n / 2, self.n / 2) / (self.eff_pixel_size * self.n)
        [fxx, fyy] = np.meshgrid(fx, fy)

        # Pupil initialization
        r = np.sqrt(fxx ** 2 + fyy ** 2)
        self.pupil      = {}
        self.pupil_mask = {}
        for color_name in self.colors_used:
            self.pupil[color_name]      = (r < (self.dataset.metadata.objective.na) / self.wavelength_um[color_name]).astype(np.complex128)
            self.pupil_mask[color_name] = self.pupil[color_name].copy()

    def normalize(self):
        if not self.measurements_are_normalized:
            # if self.dataset.metadata.background.roi is None or self.dataset.metadata.background.roi.y_start is None:
            #     self.dataset.metadata.background.roi = getRoi(self.dataset.frame_list[0,:,:]/np.max(self.dataset.frame_list[0,:,:]),'select a region without object')
            #
            # roi_size = self.dataset.metadata.background.roi.size()
            # assert all(roi_size), "Roi size must be >0!"

            for frame in self.frame_list:
                meanIntensity = np.mean(uniform_filter(frame, size=frame.shape[1] // 2))
                # meanIntensity = frame[self.dataset.metadata.background.roi.y_start:self.dataset.metadata.background.roi.y_end,
                #                       self.dataset.metadata.background.roi.x_start:self.dataset.metadata.background.roi.x_end,].sum() / np.prod(roi_size)
                assert meanIntensity > 0, "Mean frame intensity should be >0"
                frame /= meanIntensity        # normalize intensity with DC term
                frame -= 1.0                  # subtract the DC term

            self.measurements_are_normalized = True
        else:
            print("WARNING: skipping normalization since it has already been applied.")

    def plotResult(self,x,error):
        u_opt = x[:x.size//2].reshape(self.frame_list[0].shape)
        p_opt = x[x.size//2:].reshape(self.frame_list[0].shape)
        callback_amp_phase_error(self.f,self.ax,np.exp(u_opt+1j*p_opt),error,self.metadata.xlin.real,self.metadata.ylin.real)

    def afunc(self,x,forward_only=False,funcVal_only=False):
        F = lambda x:self.Fobj.fourierTransform(x)
        IF = lambda x:self.Fobj.inverseFourierTransform(x)
        Fu = F(x[:x.size//2].reshape(self.frame_list[0].shape))
        Fp = F(x[x.size//2:].reshape(self.frame_list[0].shape))
        Ax = self.Hu*Fu[ np.newaxis,:,:] + self.Hp*Fp[ np.newaxis,:,:]
        Ax = np.asarray([IF(img_rotation).real for img_rotation in Ax])
        if forward_only:
            Ax.shape = (Ax.size,1)
            return Ax
        res = Ax - self.frame_list_current
        funcVal = _norm(res)**2
        if funcVal_only:
            return funcVal
        else:
            Fres = np.asarray([F(img_rotation) for img_rotation in res])
            grad = [IF((self.Hu.conj()*Fres).sum(axis=0)).real,IF((self.Hp.conj()*Fres).sum(axis=0)).real]
            grad = np.append(grad[0].ravel(),grad[1].ravel())
            grad.shape = (grad.size,1)
            grad[:x.size//2] += self.reg_u*x[:x.size//2]
            grad[x.size//2:] += self.reg_p*x[x.size//2:]
            return grad,funcVal

    def hessian(self,d):
        F = lambda x:self.Fobj.fourierTransform(x)
        IF = lambda x:self.Fobj.inverseFourierTransform(x)
        d_u = d[:d.size//2].reshape(self.frame_list[0].shape)
        d_p = d[d.size//2:].reshape(self.frame_list[0].shape)
        Fd_u = F(d_u);Fd_p = F(d_p);
        Hd = np.append((IF(((self.Hu.conj()*self.Hu).sum(axis=0))*Fd_u)+\
                       IF(((self.Hu.conj()*self.Hp).sum(axis=0))*Fd_p)+self.reg_u*d_u).real.ravel(),
                       (IF(((self.Hp.conj()*self.Hu).sum(axis=0))*Fd_u)+\
                       IF(((self.Hp.conj()*self.Hp).sum(axis=0))*Fd_p)+self.reg_p*d_p).real.ravel())
        Hd.shape = (Hd.size,1)
        return Hd

    def l2Deconv(self, fIntensity, AHA, determinant):
        IF = lambda x: self.Fobj.inverseFourierTransform(x)
        AHy = [(self.Hu.conj()*fIntensity).sum(axis=0),(self.Hp.conj()*fIntensity).sum(axis=0)]
        u = IF((AHA[3]*AHy[0]-AHA[1]*AHy[1])/determinant).real
        p = IF((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant).real
        return u+1j*p

    def tvDeconv(self,fIntensity,AHA,determinant,fDx,fDy,order=1,TV_type='iso'):
        F = lambda x: self.Fobj.fourierTransform(x)
        IF = lambda x: self.Fobj.inverseFourierTransform(x)
        z_k = np.zeros((4,) + (self.m, self.n))
        u_k = np.zeros((4,) + (self.m, self.n))
        D_k = np.zeros((4,) + (self.m, self.n))

        for Iter in range(self.options.max_tv_iterations):
            y2 = [F(z_k[Idx] - u_k[Idx]) for Idx in range(self.options.dpc_count)]
            AHy = np.asarray([(self.Hu.conj() * fIntensity).sum(axis=0)+self.options.rho * (fDx.conj() * y2[0] + fDy.conj() * y2[1]),\
                              (self.Hp.conj() * fIntensity).sum(axis=0)+self.options.rho * (fDx.conj() * y2[2] + fDy.conj() * y2[3])])
            u = IF((AHA[3] * AHy[0] - AHA[1] * AHy[1]) / determinant).real
            p = IF((AHA[0] * AHy[1] - AHA[2] * AHy[0]) / determinant).real
            if Iter < self.options.max_tv_iterations-1:
                if order==1:
                    D_k[0] = u - np.roll(u,-1,axis=1)
                    D_k[1] = u - np.roll(u,-1,axis=0)
                    D_k[2] = p - np.roll(p,-1,axis=1)
                    D_k[3] = p - np.roll(p,-1,axis=0)
                elif order==2:
                    D_k[0] = u - 2 * np.roll(u,-1,axis=1) + np.roll(u,-2,axis=1)
                    D_k[1] = u - 2 * np.roll(u,-1,axis=0) + np.roll(u,-2,axis=0)
                    D_k[2] = p - 2 * np.roll(p,-1,axis=1) + np.roll(p,-2,axis=1)
                    D_k[3] = p - 2 * np.roll(p,-1,axis=0) + np.roll(p,-2,axis=0)
                elif order==3:
                    D_k[0] = u - 3 * np.roll(u,-1,axis=1) + 3 * np.roll(u,-2,axis=1) - np.roll(u,-3,axis=1)
                    D_k[1] = u - 3 * np.roll(u,-1,axis=0) + 3 * np.roll(u,-2,axis=0) - np.roll(u,-3,axis=0)
                    D_k[2] = p - 3 * np.roll(p,-1,axis=1) + 3 * np.roll(p,-2,axis=1) - np.roll(p,-3,axis=1)
                    D_k[3] = p - 3 * np.roll(p,-1,axis=0) + 3 * np.roll(p,-2,axis=0) - np.roll(p,-3,axis=0)
                z_k = D_k + u_k

                if TV_type == 'iso':
                    z_k[:2,:,:] = _softThreshold_isoTV(z_k[:2,:,:], Lambda=self.options.reg_TV[0] / self.options.rho)
                    z_k[2:,:,:] = _softThreshold_isoTV(z_k[2:,:,:], Lambda=self.options.reg_TV[1] / self.options.rho)
                elif TV_type == 'aniso':
                    z_k[:2,:,:] = _softThreshold(z_k[:2,:,:], Lambda=self.options.reg_TV[0] / self.options.rho)
                    z_k[2:,:,:] = _softThreshold(z_k[2:,:,:], Lambda=self.options.reg_TV[1] / self.options.rho)
                else:
                    print('no such type for total variation!')
                    raise
                u_k += D_k - z_k
        return u + 1j * p

    def solve(self, xini=None, **kwargs):

        # Load options values from kwargs if supplied
        io.loadDictRecursive(self.options, kwargs)

        F = lambda x: self.Fobj.fourierTransform(x)
        self.x_opt = []

        # Set plotting callback
        if self.options.plot_verbose:
            kwargs.update({'callback':self.plotResult})

        # Choose algorithm
        if self.options.algorithm == 'l2':
            AHA = [(self.Hu.conj()*self.Hu).sum(axis=0) + self.options.reg_u,(self.Hu.conj()*self.Hp).sum(axis=0),\
                   (self.Hp.conj()*self.Hu).sum(axis=0),(self.Hp.conj()*self.Hp).sum(axis=0)+self.options.reg_p]

            if 'order' in kwargs:
                tv_order = kwargs['order']
                assert isinstance(tv_order,int), "order should be an integer!"
                assert tv_order > 0, "order should be possitive!"
                fDx = np.zeros((self.m, self.n))
                fDy = np.zeros((self.m, self.n))
                fDx[0,0] = 1.0; fDx[0,-1] = -1.0; fDx = F(fDx);
                fDy[0,0] = 1.0; fDy[-1,0] = -1.0; fDy = F(fDy);
                if tv_order > 1:
                    fDx = fDx ** tv_order
                    fDy = fDy ** tv_order
                regTerm = fDx * fDx.conj() + fDy * fDy.conj()
                AHA[0] += self.options.reg_TV[0] * regTerm
                AHA[3] += self.options.reg_TV[1] * regTerm
            determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]
            for frame_index in range(self.frame_list.shape[0] // self.options.dpc_count):
                for dpc_index in range(self.options.dpc_count):
                    intensity_f = F(self.frame_list[frame_index * self.options.dpc_count + dpc_index])
                fIntensity = np.asarray([F(self.frame_list[frame_index * self.options.dpc_count + imgIdx]) for imgIdx in range(self.options.dpc_count)])
                self.x_opt.append(self.l2Deconv(fIntensity,AHA,determinant))

        elif self.options.algorithm == 'tv':
            fDx = np.zeros((self.m, self.n))
            fDy = np.zeros((self.m, self.n))
            fDx[0,0] = 1.0; fDx[0,-1] = -1.0; fDx = F(fDx);
            fDy[0,0] = 1.0; fDy[-1,0] = -1.0; fDy = F(fDy);
            if 'order' not in kwargs or kwargs['order']==1:
                pass
            elif kwargs['order']==2:
                fDx = fDx**2; fDy = fDy**2;
            elif kwargs['order']==3:
                fDx = fDx**3; fDy = fDy**3;
            else:
                print('tvDeconv does not support order higher than 3!')
                raise

            regTerm = self.options.rho * (fDx * fDx.conj() + fDy * fDy.conj())

            AHA = [(self.Hu.conj() * self.Hu).sum(axis=0) + regTerm + self.options.reg_u, (self.Hu.conj() * self.Hp).sum(axis=0),\
                   (self.Hp.conj() * self.Hu).sum(axis=0), (self.Hp.conj() * self.Hp).sum(axis=0) + regTerm + self.options.reg_p]
            determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]
            for frame_index in range(self.frame_list.shape[0] // self.options.dpc_count):
                fIntensity = np.asarray([F(self.frame_list[frame_index*self.options.dpc_count + imgIdx]) for imgIdx in range(self.options.dpc_count)])
                self.x_opt.append(self.tvDeconv(fIntensity, AHA, determinant, fDx, fDy))

        else:
            xini = np.zeros((2 * self.frame_list[0].size, 1)) if xini is None else xini
            error = []
            if self.options.plot_verbose:
                kwargs.update({'callback':self.plotResult})
            for frame_index in range(self.frame_list.shape[0] // self.options.dpc_count):
                self.dpc_dataset = self.frame_list[frame_index * self.options.dpc_count:(frame_index + 1) * self.options.dpc_count]
                if self.options.algorithm == 'gradientDescent':
                    self.x_opt_frame,error_frame = gradientDescent(self.afunc, xini, **kwargs)
                elif self.options.algorithm == 'FISTA':
                    self.x_opt_frame,error_frame = FISTA(self.afunc, xini, **kwargs)
                elif self.options.algorithm == 'newton':
                    self.x_opt_frame,error_frame = newton(self.afunc, xini, Hessian=self.hessian, **kwargs)
                else:
                    raise ValueError("Inavlid algorithm (%s)" % self.options.algorithm)
                self.x_opt.append(self.x_opt_frame[:self.x_opt_frame.size // 2].reshape(self.frame_list[0].shape)+\
                          1j * self.x_opt_frame[self.x_opt_frame.size // 2:].reshape(self.frame_list[0].shape))
                error.append(error_frame)

        # Append error if we're using another solver
        if self.options.algorithm not in ['l2', 'tv']:
            return (self.x_opt, error)
        else:
            return(self.x_opt)

    def normalizeResult(self, x_opt=None, phase_offset_method=None, gaussian_filter_sigma=None, write=True):
        if x_opt is None:
            assert self.x_opt is not None, "Must run algorithm first to normalize result!"
            x_opt = self.x_opt

        if phase_offset_method is None:
            phase_offset_method = self.options.phase_offset_method

        if gaussian_filter_sigma is None:
            gaussian_filter_sigma = self.options.result_gaussian_filter_sigma

        if gaussian_filter_sigma is not None:
            if gaussian_filter_sigma > 0:
                for idx in range(len(x_opt)):
                    x_opt[idx] = np.real(x_opt[idx]) + 1j * (np.imag(x_opt[idx]) - gaussian_filter(np.imag(x_opt[idx]), sigma=gaussian_filter_sigma))

        # Normalize phase
        if phase_offset_method == 'zero_mean':
            for idx in range(len(x_opt)):
                x_opt[idx] = np.real(x_opt[0]) + 1j * (np.imag(x_opt[0]) - np.mean(np.imag(x_opt[0])))

        elif phase_offset_method  == 'positive':
            for idx in range(len(x_opt)):
                x_opt[idx] = np.real(x_opt[0]) + 1j * (np.imag(x_opt[0]) - np.min(np.imag(x_opt[0])))

        if write:
            self.x_opt = x_opt

        return x_opt


    def showResult(self, figsize=(10,3)):
        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.imshow(np.real(self.x_opt[0]), cmap='gray')
        cb = plt.colorbar()
        cb.set_label('a.u.')
        plt.axis('off')
        plt.title('Absorption')

        scalebar = ScaleBar(self.dataset.metadata.system.eff_pixel_size_um, units="um")
        plt.gca().add_artist(scalebar)
        plt.subplot(122)
        plt.imshow(np.imag(self.x_opt[0]), cmap='gray')
        cb = plt.colorbar()
        cb.set_label('radians')
        plt.title('Phase')
        plt.axis('off')
        scalebar = ScaleBar(self.dataset.metadata.system.eff_pixel_size_um, units="um")
        plt.gca().add_artist(scalebar)
