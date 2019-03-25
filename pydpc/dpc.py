import numpy as np
import copy
import llops as yp
import matplotlib.pyplot as plt
import comptic

# Valid source types used in the functions below
valid_source_types = ['top', 'bottom', 'left', 'right',
                      'monopole_top', 'monopole_bottom', 'monopole_left', 'monopole_right',
                      'dipole_top', 'dipole_bottom', 'dipole_left', 'dipole_right',
                      'half', 'third', 'quadrant']

# Define system parameters for our system
_system_params_default = {
                             'pixel_count': (2580, 2180),
                             'objective_numerical_aperture': 0.25,
                             'objective_magnification': 10.2,
                             'system_magnification': 1.0,
                             'camera_pixel_size': 6.5e-6,
                             'camera_is_color': False,
                             'camera_readout_time': 0.032,  # seconds
                             'camera_quantum_efficency': 0.55,
                             'camera_max_counts': 65535,
                             'camera_dark_current': 0.9,
                             'camera_readout_noise': 3.85,
                             'illumination_wavelength': 0.53e-6,
                             'illumination_source_position_list_na': []
                         }


def getDefaultSystemParams(**kwargs):
    """Returns a dict of default optical system parameters."""
    params = copy.deepcopy(_system_params_default)
    for key in kwargs:
        if key in params:
            params[key] = kwargs[key]

    return params


def genDpcSourceList(shape,
                     dpc_type_list=('top', 'bottom', 'left', 'right'),
                     angle=0,
                     objective_numerical_aperture=0.25,
                     illumination_inner_na=0.0,
                     camera_pixel_size=6.5e-6,
                     objective_magnification=10,
                     system_magnification=1.0,
                     illumination_wavelength=0.53e-6,
                     center=True,
                     dtype=None,
                     backend=None,
                     **kwargs):
    """Generate list of rasterized DPC source patterns.

    Arguments:
        shape [2x1 tuple]:
            Shape of measurements

        dpc_type_list [list of str]:
            List of escriptors of DPC types. Can be one of:
                 "top": Standard half-circle, top orientation
                 "bottom": Standard half-circle, bottom orientation
                 "left": Standard half-circle, left orientation
                 "right": Standard half-circle, right orientation
                 "monopole_top": Monopole source, top orientation
                 "monopole_bottom": Monopole source, bottom orientation
                 "monopole_left": Monopole source, left orientation
                 "monopole_right, Monopole source, right orientation
                 "dipole_top": Dipole source, top orientation
                 "dipole_bottom": Dipole source, bottom orientation
                 "dipole_left": Dipole source, left orientation
                 "dipole_right, Dipole source, right orientation
                 "half": Source with 180 degree coverage, set orientation with angle kwarg
                 "third": Source with 120 degree coverage, set orientation with angle kwarg
                 "quadrant": Source with 90 degree coverage, set orientation with angle kwarg

        angle:
            Angle of source. Used onlt with "half", "third", or "quadrant" orientations

        objective_numerical_aperture:
            Objective numerical aperture

        illumination_inner_na:
            Inner NA cutoff of source

        camera_pixel_size:
            Pixel size of camera, usually microns

        objective_magnification:
            Magnification of objective. Total magnification is calculated as
            the product of this and system_magnification keyword argument.

        system_magnification:
            Magnification of system (not including objective). Total
            magnification is calculated as the product of this and
            system_magnification keyword argument.

        illumination_wavelength:
            Wavelength of illuminaiton source, usually microns

        center:
            Whether to center (fftshift) the output

        dtype:
            Desired datatype of the source list (valid options are provided by
            llops.valid_datatypes)

        backend:
            Desired backend of the source list (valid options are provided by
            llops.valid_backends)

    """

    # Parse source types
    if type(dpc_type_list) not in (tuple, list):
        dpc_type_list = [dpc_type_list]

    # Ensure all source types are valid
    for dpc_type in dpc_type_list:
        assert dpc_type.lower() in valid_source_types

    # Generate inner source
    source_inner = comptic.imaging.pupil(shape, camera_pixel_size=camera_pixel_size,
                            objective_magnification=objective_magnification,
                            system_magnification=system_magnification,
                            illumination_wavelength=illumination_wavelength,
                            objective_numerical_aperture=illumination_inner_na,
                            center=center, dtype=dtype, backend=backend)

    # Generate outer source
    source_outer = comptic.imaging.pupil(shape,
                            camera_pixel_size=camera_pixel_size,
                            objective_magnification=objective_magnification,
                            system_magnification=system_magnification,
                            illumination_wavelength=illumination_wavelength,
                            objective_numerical_aperture=objective_numerical_aperture,
                            center=center, dtype=dtype, backend=backend)

    # Get base source to filter for each source
    source_base = source_outer - source_inner

    # Generate coordinates
    yy, xx = yp.grid(shape)

    # Generate radius of pupil in px
    effective_pixel_size = camera_pixel_size / system_magnification / objective_magnification
    ps_fourier = 1 / effective_pixel_size / np.asarray(shape)
    pupil_radius_px = int(np.round((objective_numerical_aperture / illumination_wavelength) / ps_fourier)[0])

    # Generate LED source patterns
    source_list = []
    for dpc_type in dpc_type_list:
        if dpc_type.lower() == 'half':
            led_pattern = source_base * (yy * yp.sin(np.deg2rad(angle)) >= xx * yp.cos(np.deg2rad(angle)))
        elif dpc_type.lower() == 'top':
            led_pattern = source_base * (yy > 0)
        elif dpc_type.lower() == 'bottom':
            led_pattern = source_base * (yy < 0)
        elif dpc_type.lower() == 'left':
            led_pattern = source_base * (xx < 0)
        elif dpc_type.lower() == 'right':
            led_pattern = source_base * (xx > 0)
        elif dpc_type.lower() == 'monopole_left':
            furthest_index = [shape[0] // 2, shape[1] // 2 - pupil_radius_px]
            led_pattern = yp.zeros_like(source_base)
            led_pattern[furthest_index[0], furthest_index[1]] = 1.0
        elif dpc_type.lower() == 'monopole_right':
            furthest_index = [shape[0] // 2, shape[1] // 2 + pupil_radius_px]
            led_pattern = yp.zeros_like(source_base)
            led_pattern[furthest_index[0], furthest_index[1]] = 1.0
        elif dpc_type.lower() == 'monopole_top':
            furthest_index = [shape[0] // 2 - pupil_radius_px, shape[1] // 2]
            led_pattern = yp.zeros_like(source_base)
            led_pattern[furthest_index[0], furthest_index[1]] = 1.0
        elif dpc_type.lower() == 'monopole_bottom':
            furthest_index = [shape[0] // 2 + pupil_radius_px, shape[1] // 2]
            led_pattern = yp.zeros_like(source_base)
            led_pattern[furthest_index[0], furthest_index[1]] = 1.0
        elif dpc_type.lower() == 'dipole_left':
            led_pattern = yp.zeros_like(source_base)
            led_pattern[shape[0] // 2 - int(pupil_radius_px / np.sqrt(2)),
                        shape[1] // 2 - int(pupil_radius_px / np.sqrt(2))] = 1.0

            led_pattern[shape[0] // 2 + int(pupil_radius_px / np.sqrt(2)),
                        shape[1] // 2 - int(pupil_radius_px / np.sqrt(2))] = 1.0
        elif dpc_type.lower() == 'dipole_right':
            led_pattern = yp.zeros_like(source_base)
            led_pattern[shape[0] // 2 - int(pupil_radius_px / np.sqrt(2)),
                        shape[1] // 2 + int(pupil_radius_px / np.sqrt(2))] = 1.0

            led_pattern[shape[0] // 2 + int(pupil_radius_px / np.sqrt(2)),
                        shape[1] // 2 + int(pupil_radius_px / np.sqrt(2))] = 1.0
        elif dpc_type.lower() == 'dipole_top':
            led_pattern = yp.zeros_like(source_base)
            led_pattern[shape[0] // 2 - int(pupil_radius_px / np.sqrt(2)),
                        shape[1] // 2 + int(pupil_radius_px / np.sqrt(2))] = 1.0

            led_pattern[shape[0] // 2 - int(pupil_radius_px / np.sqrt(2)),
                        shape[1] // 2 - int(pupil_radius_px / np.sqrt(2))] = 1.0
        elif dpc_type.lower() == 'dipole_bottom':
            led_pattern = yp.zeros_like(source_base)
            led_pattern[shape[0] // 2 + int(pupil_radius_px / np.sqrt(2)),
                        shape[1] // 2 + int(pupil_radius_px / np.sqrt(2))] = 1.0

            led_pattern[shape[0] // 2 + int(pupil_radius_px / np.sqrt(2)),
                        shape[1] // 2 - int(pupil_radius_px / np.sqrt(2))] = 1.0

        elif dpc_type.lower() == 'half':
            led_pattern = source_base
            led_pattern *= (yy * yp.sin(np.deg2rad(angle + 180)) <= xx * yp.cos(np.deg2rad(angle + 180)))
            led_pattern *= (yy * yp.sin(np.deg2rad(angle)) >= xx * yp.cos(np.deg2rad(angle)))

        elif dpc_type.lower() == 'third':
            led_pattern = source_base
            led_pattern *= (yy * yp.sin(np.deg2rad(angle + 120)) <= xx * yp.cos(np.deg2rad(angle + 120)))
            led_pattern *= (yy * yp.sin(np.deg2rad(angle)) >= xx * yp.cos(np.deg2rad(angle)))

        elif dpc_type.lower() == 'quadrant':
            led_pattern = source_base
            led_pattern *= (yy * yp.sin(np.deg2rad(angle + 90)) <= xx * yp.cos(np.deg2rad(angle + 90)))
            led_pattern *= (yy * yp.sin(np.deg2rad(angle)) >= xx * yp.cos(np.deg2rad(angle)))
        else:
            raise ValueError('Invalid dpc type %s' % dpc_type)

        # Append to list
        source_list.append(led_pattern)

    # Return
    if len(source_list) == 1:
        return source_list[0]
    else:
        return source_list


def genDpcSourcePositionList(illumination_source_position_list_na, objective_numerical_aperture,
                      dpc_type_list=('top', 'bottom', 'left', 'right'), **kwargs):
    """Return a list of LED powers in each DPC pattern."""

    # Get LEDs within numerical aperture
    illumination_source_position_list_na = np.asarray(illumination_source_position_list_na)
    bf_led_mask = [np.sqrt(src[0] ** 2 + src[1] ** 2) <= objective_numerical_aperture for src in illumination_source_position_list_na]

    # Generate LED patterns
    led_pattern_list = []
    for dpc_type in dpc_type_list:
        if dpc_type.lower() == 'top':
            led_pattern = yp.cast(bf_led_mask, 'float32')
            led_pattern *= yp.cast(illumination_source_position_list_na[:, 0] >= 0, 'float32')
        elif dpc_type.lower() == 'bottom':
            led_pattern = yp.cast(bf_led_mask, 'float32')
            led_pattern *= yp.cast(illumination_source_position_list_na[:, 0] <= 0, 'float32')
        elif dpc_type.lower() == 'left':
            led_pattern = yp.cast(bf_led_mask, 'float32')
            led_pattern *= yp.cast(illumination_source_position_list_na[:, 1] <= 0, 'float32')
        elif dpc_type.lower() == 'right':
            led_pattern = yp.cast(bf_led_mask, 'float32')
            led_pattern *= yp.cast(illumination_source_position_list_na[:, 1] >= 0, 'float32')

        # Append to list
        led_pattern_list.append(led_pattern)

    # Return
    return led_pattern_list


def plotWotfList(Hr_list, Hi_list, source_names=None):
    """Plot pairs of WOTFs."""
    yp.listPlotFlat(Hr_list + [yp.imag(Hi) for Hi in Hi_list], source_names, max_width=4)


def plotSourcePositionList(led_pattern_list, illumination_source_position_list_na, objective_numerical_aperture=None, labels=None, background_color='k', zoom=True, **kwargs):
    """ Plot a list of LED patterns."""

    # Convert source position list to numpy array
    illumination_source_position_list_na = np.asarray(illumination_source_position_list_na)

    # Generate figure
    plt.figure(figsize=(14, 4))

    # Plot DPC patterns
    for index, pattern in enumerate(led_pattern_list):

        plt.subplot(141 + index)
        plt.scatter(illumination_source_position_list_na[:, 1], illumination_source_position_list_na[:, 0], c=background_color)
        plt.scatter(illumination_source_position_list_na[:, 1], illumination_source_position_list_na[:, 0], c=pattern)
        plt.axis('square')

        if zoom:
            plt.xlim((-1.5 * objective_numerical_aperture, 1.5 * objective_numerical_aperture))
            plt.ylim((-1.5 * objective_numerical_aperture, 1.5 * objective_numerical_aperture))
        else:
            plt.xlim((-1, 1))
            plt.ylim((-1, 1))

        if objective_numerical_aperture:
            circle = plt.Circle((0, 0), objective_numerical_aperture, color='r', fill=False, linewidth=4)
            plt.gca().add_artist(circle)

        if labels:
            plt.title(labels[index])


def genMeasurementNonLinear(field, led_pattern, illumination_source_position_list_na=[],
                            camera_pixel_size=6.5e-3,
                            objective_numerical_aperture=0.25, objective_magnification=10,
                            system_magnification=1.0, illumination_wavelength=0.53e-6,
                            **kwargs):

    # Generate pupil
    pupil = comptic.imaging.pupil(yp.shape(field),
                     camera_pixel_size=camera_pixel_size,
                     objective_magnification=objective_magnification,
                     system_magnification=system_magnification,
                     illumination_wavelength=illumination_wavelength,
                     objective_numerical_aperture=objective_numerical_aperture,
                     center=True,
                     dtype=yp.getDatatype(field),
                     backend=yp.getBackend(field))

    # Calculate Effective Pixel Size
    effective_pixel_size = camera_pixel_size / objective_magnification / system_magnification

    # Generate real-space grid
    yy, xx = yp.grid(yp.shape(field), effective_pixel_size)

    intensity = yp.zeros_like(field)
    for led_power, led_na in zip(led_pattern, illumination_source_position_list_na):
        # Only process non-zero LED powers
        if led_power > 0.0:

            # Generate Illumination
            illumination = np.exp(-1j * 2 * np.pi * (led_na[1] / illumination_wavelength * xx + led_na[0] / illumination_wavelength * yy))

            # Generate intensity
            intensity = intensity + led_power * yp.abs(yp.iFt(yp.Ft(illumination * field) * pupil)) ** 2.0

    return intensity


def genMeasurementsLinear(field, led_pattern_list, illumination_source_position_list_na=[],
                          camera_pixel_size=6.5e-3,
                          objective_numerical_aperture=0.25, objective_magnification=10,
                          system_magnification=1.0, illumination_wavelength=0.53e-6,
                          **kwargs):

    # Generate WOTFs
    Hr_list, Hi_list = wotfsFromSourcePositionList(yp.shape(field), led_pattern_list,
                                                   illumination_wavelength=illumination_wavelength,
                                                   illumination_source_position_list_na=illumination_source_position_list_na,
                                                   objective_magnification=objective_magnification,
                                                   objective_numerical_aperture=objective_numerical_aperture,
                                                   system_magnification=system_magnification,
                                                   camera_pixel_size=camera_pixel_size)

    # Generate Intensity
    intensity_list = []
    for (Hr, Hi) in zip(Hr_list, Hi_list):
        intensity_list.append(yp.real(1 + yp.iFt(Hr * yp.Ft(yp.abs(field)) + Hi * yp.Ft(yp.angle(field)))))

    return intensity_list


def rasterizeSource(shape, led_pattern, illumination_source_position_list_na=[],
                    camera_pixel_size=6.5e-3,
                    objective_numerical_aperture=0.25, objective_magnification=10,
                    system_magnification=1.0, illumination_wavelength=0.53e-6,
                    dtype=None, backend=None, **kwargs):

    # Calculate Effective Pixel Size
    effective_pixel_size = camera_pixel_size / objective_magnification / system_magnification

    # Generate fourier-space grid
    kyy, kxx = yp.grid(shape, 1 / effective_pixel_size / np.asarray(shape))

    # Generate source
    source = yp.zeros(shape, dtype, backend)
    for led_power, led_na in zip(led_pattern, illumination_source_position_list_na):

        # Calculate distance from illumination point
        dist = (kyy - led_na[0] / illumination_wavelength) ** 2 + (kxx - led_na[1] / illumination_wavelength) ** 2

        # Assign the correct position in the source
        source[yp.argmin(dist)] = led_power

    return source


def invert(measurement_list, Hr_list, Hi_list, reg_real=1e-8, reg_imag=1e-8):
    """ Perform DPC inversion"""

    # Build AHA
    AHA_11 = sum([yp.abs(Hr) ** 2 for Hr in Hr_list]) + reg_real
    AHA_12 = sum([yp.conj(Hr) * Hi for Hr, Hi in zip(Hr_list, Hi_list)])
    AHA_21 = sum([yp.conj(Hi) * Hr for Hr, Hi in zip(Hr_list, Hi_list)])
    AHA_22 = sum([yp.abs(Hi) ** 2 for Hi in Hi_list]) + reg_imag
    AHA = [AHA_11, AHA_12, AHA_21, AHA_22]

    # Calculate determinant
    determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]

    # Generate measurement list in Fourier domain
    measurement_list_fourier = np.asarray([yp.Ft(yp.dcopy(meas)) for meas in measurement_list])

    # Generate AHy
    AH_y_real = sum([yp.conj(Hr) * y_f for Hr, y_f in zip(Hr_list, measurement_list_fourier)])
    AH_y_imag = sum([yp.conj(Hi) * y_f for Hi, y_f in zip(Hi_list, measurement_list_fourier)])
    AHy = [AH_y_real, AH_y_imag]

    # Solve for absorption
    absorption = yp.real(yp.iFt((AHA[3] * AHy[0] - AHA[1] * AHy[1]) / determinant))

    # Solve for phase
    phase = yp.real(yp.iFt((AHA[0] * AHy[1] - AHA[2] * AHy[0]) / determinant))

    # Generate complex field
    field = absorption + 1.0j * phase

    # Return
    return field


def wotfsFromSourceList(source_list, pupil=None, **system_params):
    """Generate WOTFs given a list of sources."""
    # Generate pupil
    if pupil is None:
        pupil = comptic.imaging.pupil(yp.shape(source_list[0]), **system_params)

    # Generate WOTFs
    wotf_list = [genWotfs(source, pupil, **system_params) for source in source_list]
    Hr_list, Hi_list = [wotf_pair[0] for wotf_pair in wotf_list], [wotf_pair[1] for wotf_pair in wotf_list]

    # Return
    return (Hr_list, Hi_list)


def wotfsFromSourcePositionList(shape, led_pattern_list, **system_params):
    """Generate WOTF for a list of source positions."""
    # Generate sources
    source_list = [rasterizeSource(shape, led_pattern, **system_params) for led_pattern in led_pattern_list]

    # Generate WOTFs and return
    return wotfsFromSourceList(source_list, pupil=None, **system_params)


# Generate Hu and Hp
def genWotfs(source, pupil, illumination_wavelength, **kwargs):
    """Function which generates the Weak-Object Transfer Functions (WOTFs) for absorption (Hu) and phase (Hp) given a source and pupul list

    Args:
        source_list: 3D ndarray of sources in the Fourier domain, where first dimensions are k_x, k_y and third is the number of transfer functions to generate
        pupil_list: 3D ndarray of pupil functions in the Fourier domain, where first dimensions are k_x, k_y and third is the number of transfer functions to generate
        lambda_list: list of wavelengths corresponding to the third dimension in sourceList and pupilList
        shifted_output: Whether to perform a fftshift on the output before returning

    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (Na_x, NA_y)
    """

    # Calculate source power
    DC = yp.sum((np.abs(pupil) ** 2 * source))

    # Ensure source is non-zeros
    if DC == 0:
        print("WARNING: Source must be non-zero")

    # Generate WOTF
    M = yp.Ft(source * pupil) * yp.conj(yp.Ft(pupil))
    Hu = 2 * yp.iFt(np.real(M)) / DC
    Hp = 1j * 2 * yp.iFt(1j * np.imag(M)) / DC

    # Cast both as complex
    Hu = yp.astype(Hu, 'complex32')
    Hp = yp.astype(Hp, 'complex32')

    # Return
    return (Hu, Hp)


class DpcSolver:
    """DPC Solver class, for use on production-grade systems."""
    def __init__(self, measurement_list,
                       illumination_wavelength,
                       objective_na,
                       objective_magnification,
                       illumination_inner_na,
                       camera_pixel_size,
                       source_list,
                       regularization_amplitude=1e-4,
                       regularization_phase=1e-4,
                       system_magnification=1):

        # Get system parameters
        self.illumination_wavelength    = illumination_wavelength
        self.objective_na               = obejctive_na
        self.illumination_inner_na      = illumination_inner_na
        self.camera_pixel_size          = camera_pixel_size
        self.magnification              = objective_magnification * system_magnification
        self.regularization_amplitude   = regularization_amplitude
        self.regularization_phase       = regularization_phase

        # Initialize variables
        self.pupil, self.source_list, self.wotf_list = None, None, None

        # Parse measurements and preprocess
        self.measurement_list = measurement_list

        # Parse source list
        self.source_list = genDpcSourceListRaster(source_list)

        # Generate pupil
        self._comptic.imaging.pupil()

        # Generate WOTFs
        self._genWotfs()

    @property
    def measurement_list(self):
        return self._measurement_list

    @measurement_list.setter
    def measurement_list(self, new_measurement_list):

        # Perform normalization
        for img in new_measurement_list:
            img /= yp.mean(img)        # normalize intensity with DC term
            img -= 1.0                  # subtract the DC term

        # Copy to class variable
        self._measurement_list = copy.deepcopy(new_measurement_list)

    def _genWotfs(self):
        pass

    def solve(self, method='direct', initialization=None, plot_verbose=False, **kwargs):
        dpc_result  = []
        AHA         = [(self.Hu_list.conj() * self.Hu_list).sum(axis=0)+self.reg_u,            (self.Hu_list.conj()*self.Hp_list).sum(axis=0),\
                       (self.Hp_list.conj() * self.Hu_list).sum(axis=0)         , (self.Hp_list.conj()*self.Hp_list).sum(axis=0)+self.reg_p]
        determinant = AHA[0]*AHA[3]-AHA[1]*AHA[2]
        for frame_index in range(self.dpc_measurements.shape[0] // self.dpc_num):
            fIntensity = np.asarray([F(self.dpc_measurements[frame_index*self.dpc_num+image_index]) for image_index in range(self.dpc_num)])
            AHy        = np.asarray([(self.Hu_list.conj()*fIntensity).sum(axis=0), (self.Hp_list.conj()*fIntensity).sum(axis=0)])
            absorption = IF((AHA[3]*AHy[0]-AHA[1]*AHy[1])/determinant).real
            phase      = IF((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant).real
            dpc_result.append(absorption+1.0j*phase)

        return np.asarray(dpc_result)
