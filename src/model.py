import numpy as np
from sympy import finite_diff_weights as fd_w
import pytest

from devito import (Grid, SubDomain, Function, Constant, warning,
                    SubDimension, Eq, Inc, Operator, div, sin, Abs)
from devito.builtins import initialize_function, gaussian_smooth, mmax, mmin
from devito.tools import as_tuple

__all__ = ['SeismicModel', 'Model', 'ModelElastic',
           'ModelViscoelastic', 'ModelViscoacoustic']


def initialize_damp(damp, padsizes, spacing, abc_type="damp", fs=False):
    """
    Initialize damping field with an absorbing boundary layer.

    Parameters
    ----------
    damp : Function
        The damping field for absorbing boundary condition.
    nbl : int
        Number of points in the damping layer.
    spacing :
        Grid spacing coefficient.
    mask : bool, optional
        whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    """

    eqs = [Eq(damp, 1.0 if abc_type == "mask" else 0.0)]
    for (nbl, nbr), d in zip(padsizes, damp.dimensions):
        if not fs or d is not damp.dimensions[-1]:
            dampcoeff = 1.5 * np.log(1.0 / 0.001) / (nbl)
            # left
            dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                      thickness=nbl)
            pos = Abs((nbl - (dim_l - d.symbolic_min) + 1) / float(nbl))
            val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
            val = -val if abc_type == "mask" else val
            eqs += [Inc(damp.subs({d: dim_l}), val/d.spacing)]
        # right
        dampcoeff = 1.5 * np.log(1.0 / 0.001) / (nbr)
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=nbr)
        pos = Abs((nbr - (d.symbolic_max - dim_r) + 1) / float(nbr))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if abc_type == "mask" else val
        eqs += [Inc(damp.subs({d: dim_r}), val/d.spacing)]

    Operator(eqs, name='initdamp')()


class PhysicalDomain(SubDomain):

    name = 'physdomain'

    def __init__(self, so, fs=False):
        super().__init__()
        self.so = so
        self.fs = fs

    def define(self, dimensions):
        map_d = {d: d for d in dimensions}
        if self.fs:
            map_d[dimensions[-1]] = ('middle', self.so, 0)
        return map_d


class FSDomain(SubDomain):

    name = 'fsdomain'

    def __init__(self, so):
        super().__init__()
        self.size = so

    def define(self, dimensions):
        """
        Definition of the upper section of the domain for wrapped indices FS.
        """

        return {d: (d if not d == dimensions[-1] else ('left', self.size))
                for d in dimensions}


class GenericModel(object):
    """
    General model class with common properties
    """
    def __init__(self, origin, spacing, shape, space_order, nbl=20,
                 dtype=np.float32, subdomains=(), bcs="damp", grid=None,
                 fs=False):
        self.shape = shape
        self.space_order = space_order
        self.nbl = int(nbl)
        self.origin = tuple([dtype(o) for o in origin])
        self.fs = fs
        # Default setup
        origin_pml = [dtype(o - s*nbl) for o, s in zip(origin, spacing)]
        shape_pml = np.array(shape) + 2 * self.nbl

        # Model size depending on freesurface
        physdomain = PhysicalDomain(space_order, fs=fs)
        subdomains = subdomains + (physdomain,)
        if fs:
            fsdomain = FSDomain(space_order)
            subdomains = subdomains + (fsdomain,)
            origin_pml[-1] = origin[-1]
            shape_pml[-1] -= self.nbl

        # Origin of the computational domain with boundary to inject/interpolate
        # at the correct index
        if grid is None:
            # Physical extent is calculated per cell, so shape - 1
            extent = tuple(np.array(spacing) * (shape_pml - 1))
            self.grid = Grid(extent=extent, shape=shape_pml, origin=origin_pml,
                             dtype=dtype, subdomains=subdomains)
        else:
            self.grid = grid

        self._physical_parameters = set()
        self.damp = None
        self._initialize_bcs(bcs=bcs)

    def _initialize_bcs(self, bcs="damp"):
        # Create dampening field as symbol `damp`
        if self.nbl == 0:
            self.damp = 1 if bcs == "mask" else 0
            return

        # First initialization
        init = self.damp is None
        # Get current Function if already initialized
        self.damp = self.damp or Function(name="damp", grid=self.grid,
                                          space_order=self.space_order)
        if callable(bcs):
            bcs(self.damp, self.nbl)
        else:
            re_init = ((bcs == "mask" and mmin(self.damp) == 0) or
                       (bcs == "damp" and mmax(self.damp) == 1))
            if init or re_init:
                if re_init and not init:
                    bcs_o = "damp" if bcs == "mask" else "mask"
                    warning("Re-initializing damp profile from %s to %s" % (bcs_o, bcs))
                    warning("Model has to be created with `bcs=\"%s\"`"
                            "for this WaveSolver" % bcs)
                initialize_damp(self.damp, self.padsizes, self.spacing,
                                abc_type=bcs, fs=self.fs)
        self._physical_parameters.update(['damp'])

    @property
    def padsizes(self):
        """
        Padding size for each dimension.
        """
        padsizes = [(self.nbl, self.nbl) for _ in range(self.dim-1)]
        padsizes.append((0 if self.fs else self.nbl, self.nbl))
        return padsizes

    def physical_params(self, **kwargs):
        """
        Return all set physical parameters and update to input values if provided
        """
        known = [getattr(self, i) for i in self.physical_parameters]
        return {i.name: kwargs.get(i.name, i) or i for i in known}

    def _gen_phys_param(self, field, name, space_order, is_param=True,
                        default_value=0, **kwargs):
        if field is None:
            return default_value
        if isinstance(field, np.ndarray):
            function = Function(name=name, grid=self.grid, space_order=space_order,
                                parameter=is_param)
            initialize_function(function, field, self.padsizes)
        else:
            function = Constant(name=name, value=field, dtype=self.grid.dtype)
        self._physical_parameters.update([name])
        return function

    @property
    def physical_parameters(self):
        return as_tuple(self._physical_parameters)

    @property
    def dim(self):
        """
        Spatial dimension of the problem and model domain.
        """
        return self.grid.dim

    @property
    def spacing(self):
        """
        Grid spacing for all fields in the physical model.
        """
        return self.grid.spacing

    @property
    def space_dimensions(self):
        """
        Spatial dimensions of the grid
        """
        return self.grid.dimensions

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each `SpaceDimension`.
        """
        return self.grid.spacing_map

    @property
    def dtype(self):
        """
        Data type for all assocaited data objects.
        """
        return self.grid.dtype

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))


class SeismicModel(GenericModel):
    """
    The physical model used in seismic inversion processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    vp : array_like or float
        Velocity in km/s.
    nbl : int, optional
        The number of absorbin layers for boundary damping.
    bcs: str or callable
        Absorbing boundary type ("damp" or "mask") or initializer.
    dtype : np.float32 or np.float64
        Defaults to np.float32.
    epsilon : array_like or float, optional
        Thomsen epsilon parameter (0<epsilon<1).
    delta : array_like or float
        Thomsen delta parameter (0<delta<1), delta<epsilon.
    theta : array_like or float
        Tilt angle in radian.
    phi : array_like or float
        Asymuth angle in radian.
    b : array_like or float
        Buoyancy.
    vs : array_like or float
        S-wave velocity.
    qp : array_like or float
        P-wave attenuation.
    qs : array_like or float
        S-wave attenuation.
    """
    _known_parameters = ['vp', 'damp', 'vs', 'b', 'epsilon', 'delta',
                         'theta', 'phi', 'qp', 'qs', 'lam', 'mu']

    def __init__(self, origin, spacing, shape, space_order, vp, nbl=20, fs=False,
                 dtype=np.float32, subdomains=(), bcs="mask", grid=None, **kwargs):
        super().__init__(origin, spacing, shape, space_order, nbl,
                         dtype, subdomains, grid=grid, bcs=bcs, fs=fs)

        # Initialize physics
        self._initialize_physics(vp, space_order, **kwargs)

        # User provided dt
        self._dt = kwargs.get('dt')
        # Some wave equation need a rescaled dt that can't be infered from the model
        # parameters, such as isoacoustic OT4 that can use a dt sqrt(3) larger than
        # isoacoustic OT2. This property should be set from a wavesolver or after model
        # instanciation only via model.dt_scale = value.
        self._dt_scale = 1

    def _initialize_physics(self, vp, space_order, **kwargs):
        """
        Initialize physical parameters and type of physics from inputs.
        The types of physics supported are:
        - acoustic: [vp, b]
        - elastic: [vp, vs, b] represented through Lame parameters [lam, mu, b]
        - visco-acoustic: [vp, b, qp]
        - visco-elastic: [vp, vs, b, qs]
        - vti: [vp, epsilon, delta]
        - tti: [epsilon, delta, theta, phi]
        """
        params = []
        # Buoyancy
        b = kwargs.get('b', 1)

        # Initialize elastic with Lame parametrization
        if 'vs' in kwargs:
            vs = kwargs.pop('vs')
            self.lam = self._gen_phys_param((vp**2 - 2. * vs**2)/b, 'lam', space_order,
                                            is_param=True)
            self.mu = self._gen_phys_param(vs**2 / b, 'mu', space_order, is_param=True)
        else:
            # All other seismic models have at least a velocity
            self.vp = self._gen_phys_param(vp, 'vp', space_order)
        # Initialize rest of the input physical parameters
        for name in self._known_parameters:
            if kwargs.get(name) is not None:
                field = self._gen_phys_param(kwargs.get(name), name, space_order)
                setattr(self, name, field)
                params.append(name)

    @property
    def _max_vp(self):
        if 'vp' in self._physical_parameters:
            return mmax(self.vp)
        else:
            return np.sqrt(mmin(self.b) * (mmax(self.lam) + 2 * mmax(self.mu)))

    @property
    def _thomsen_scale(self):
        # Update scale for tti
        if 'epsilon' in self._physical_parameters:
            return np.sqrt(1 + 2 * mmax(self.epsilon))
        return 1

    @property
    def dt_scale(self):
        return self._dt_scale

    @dt_scale.setter
    def dt_scale(self, val):
        self._dt_scale = val

    @property
    def _cfl_coeff(self):
        """
        Courant number from the physics and spatial discretization order.
        The CFL coefficients are described in:
        - https://doi.org/10.1137/0916052 for the elastic case
        - https://library.seg.org/doi/pdf/10.1190/1.1444605 for the acoustic case
        """
        # Elasic coefficient (see e.g )
        if 'lam' in self._physical_parameters or 'vs' in self._physical_parameters:
            coeffs = fd_w(1, range(-self.space_order//2+1, self.space_order//2+1), .5)
            c_fd = sum(np.abs(coeffs[-1][-1])) / 2
            return np.sqrt(self.dim) / self.dim / c_fd
        a1 = 4  # 2nd order in time
        coeffs = fd_w(2, range(-self.space_order, self.space_order+1), 0)[-1][-1]
        return np.sqrt(a1/float(self.grid.dim * sum(np.abs(coeffs))))

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        dt = self._cfl_coeff * np.min(self.spacing) / (self._thomsen_scale*self._max_vp)
        dt = self.dtype("%.3e" % (self.dt_scale * dt))
        if self._dt:
            return self._dt
        return dt

    def update(self, name, value):
        """
        Update the physical parameter param.
        """
        try:
            param = getattr(self, name)
        except AttributeError:
            # No physical parameter with tha name, create it
            setattr(self, name, self._gen_phys_param(value, name, self.space_order))
            return
        # Update the physical parameter according to new value
        if isinstance(value, np.ndarray):
            if value.shape == param.shape:
                param.data[:] = value[:]
            elif value.shape == self.shape:
                initialize_function(param, value, self.nbl)
            else:
                raise ValueError("Incorrect input size %s for model" % value.shape +
                                 " %s without or %s with padding" % (self.shape,
                                                                     param.shape))
        else:
            param.data = value

    @property
    def m(self):
        """
        Squared slowness.
        """
        return 1 / (self.vp * self.vp)

    @property
    def dm(self):
        """
        Create a simple model perturbation from the velocity as `dm = div(vp)`.
        """
        dm = Function(name="dm", grid=self.grid, space_order=self.space_order)
        Operator(Eq(dm, div(self.vp)), subs=self.spacing_map)()
        return dm

    def smooth(self, physical_parameters, sigma=5.0):
        """
        Apply devito.gaussian_smooth to model physical parameters.

        Parameters
        ----------
        physical_parameters : string or tuple of string
            Names of the fields to be smoothed.
        sigma : float
            Standard deviation of the smoothing operator.
        """
        model_parameters = self.physical_params()
        for i in physical_parameters:
            gaussian_smooth(model_parameters[i], sigma=sigma)
        return


@pytest.mark.parametrize('shape', [(51, 51), (16, 16, 16)])
def test_model_update(shape):

    # Set physical properties as numpy arrays
    vp = np.full(shape, 2.5, dtype=np.float32)  # vp constant of 2.5 km/s
    qp = 3.516*((vp[:]*1000.)**2.2)*10**(-6)  # Li's empirical formula
    b = 1 / (0.31*(vp[:]*1000.)**0.25)  # Gardner's relation

    # Define a simple visco-acoustic model from the physical parameters above
    va_model = SeismicModel(space_order=4, vp=vp, qp=qp, b=b, nbl=10,
                            origin=tuple([0. for _ in shape]), shape=shape,
                            spacing=[20. for _ in shape],)

    # Define a simple acoustic model with vp=1.5 km/s
    model = SeismicModel(space_order=4, vp=np.full_like(vp, 1.5), nbl=10,
                         origin=tuple([0. for _ in shape]), shape=shape,
                         spacing=[20. for _ in shape],)

    # Define a velocity Function
    vp_fcn = Function(name='vp0', grid=model.grid, space_order=4)
    vp_fcn.data[:] = 2.5

    # Test 1. Update vp of acoustic model from array
    model.update('vp', vp)
    assert np.array_equal(va_model.vp.data, model.vp.data)

    # Test 2. Update vp of acoustic model from Function
    model.update('vp', vp_fcn.data)
    assert np.array_equal(va_model.vp.data, model.vp.data)

    # Test 3. Create a new physical parameter in the acoustic model from array
    model.update('qp', qp)
    assert np.array_equal(va_model.qp.data, model.qp.data)

    # Make a set of physical parameters from each model
    tpl1_set = set(va_model.physical_parameters)
    tpl2_set = set(model.physical_parameters)

    # Physical parameters in either set but not in the intersection.
    diff_phys_par = tuple(tpl1_set ^ tpl2_set)

    # Turn acoustic model (it is just lacking 'b') into a visco-acoustic model
    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(model.dim))
    for i in diff_phys_par:
        # Test 4. Create a new physical parameter in the acoustic model from function
        model.update(i, getattr(va_model, i).data[slices])
        assert np.array_equal(getattr(model, i).data, getattr(va_model, i).data)


# For backward compatibility
Model = SeismicModel
ModelElastic = SeismicModel
ModelViscoelastic = SeismicModel
ModelViscoacoustic = SeismicModel


import os
import numpy as np

def demo_model(preset, **kwargs):
    """
    Utility function to create preset `Model` objects for
    demonstration and testing purposes. The particular presets are ::

    * `constant-isotropic` : Constant velocity (1.5 km/sec) isotropic model
    * `constant-tti` : Constant anisotropic model. Velocity is 1.5 km/sec and
                      Thomsen parameters are epsilon=.3, delta=.2, theta = .7rad
                      and phi=.35rad for 3D. 2d/3d is defined from the input shape
    * 'layers-isotropic': Simple n-layered model with velocities ranging from 1.5 km/s
                 to 3.5 km/s in the top and bottom layer respectively.
                 2d/3d is defined from the input shape
    * 'layers-elastic': Simple n-layered model with velocities ranging from 1.5 km/s
                    to 3.5 km/s in the top and bottom layer respectively.
                    Vs is set to .5 vp and 0 in the top layer.
    * 'layers-viscoelastic': Simple two layers viscoelastic model.
    * 'layers-tti': Simple n-layered model with velocities ranging from 1.5 km/s
                    to 3.5 km/s in the top and bottom layer respectively.
                    Thomsen parameters in the top layer are 0 and in the lower layers
                    are scaled versions of vp.
                    2d/3d is defined from the input shape
    * 'circle-isotropic': Simple camembert model with velocities 1.5 km/s
                 and 2.5 km/s in a circle at the center. 2D only.
    * 'marmousi2d-isotropic': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``devitocodes/data`` repository
                    to be available on your machine.
    * 'marmousi2d-tti': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``devitocodes/data`` repository
                    to be available on your machine.
    * 'marmousi3d-tti': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``devitocodes/data`` repository
                    to be available on your machine.
    """
    space_order = kwargs.pop('space_order', 2)
    shape = kwargs.pop('shape', (101, 101))
    spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
    origin = kwargs.pop('origin', tuple([0. for _ in shape]))
    nbl = kwargs.pop('nbl', 10)
    dtype = kwargs.pop('dtype', np.float32)
    vp = kwargs.pop('vp', 1.5)
    nlayers = kwargs.pop('nlayers', 3)
    fs = kwargs.pop('fs', False)

    if preset.lower() in ['constant-elastic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5 km/s.
        vs = 0.5 * vp
        b = 1.0

        return SeismicModel(space_order=space_order, vp=vp, vs=vs, b=b,
                            origin=origin, shape=shape, dtype=dtype, spacing=spacing,
                            nbl=nbl, **kwargs)

    if preset.lower() in ['constant-viscoelastic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 2.2 km/s.
        qp = kwargs.pop('qp', 100.)
        vs = kwargs.pop('vs', 1.2)
        qs = kwargs.pop('qs', 70.)
        b = 1/2.

        return SeismicModel(space_order=space_order, vp=vp, qp=qp, vs=vs,
                            qs=qs, b=b, origin=origin, shape=shape,
                            dtype=dtype, spacing=spacing, nbl=nbl,
                            **kwargs)

    if preset.lower() in ['constant-isotropic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5 km/s.

        return SeismicModel(space_order=space_order, vp=vp, origin=origin, shape=shape,
                            dtype=dtype, spacing=spacing, nbl=nbl, fs=fs, **kwargs)

    if preset.lower() in ['constant-viscoacoustic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5 km/s.
        qp = kwargs.pop('qp', 100.)
        b = 1/2.

        return SeismicModel(space_order=space_order, vp=vp, qp=qp, b=b, nbl=nbl,
                            origin=origin, shape=shape, spacing=spacing, **kwargs)

    elif preset.lower() in ['constant-tti']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5 km/s.
        v = np.empty(shape, dtype=dtype)
        v[:] = 1.5
        epsilon = .3*np.ones(shape, dtype=dtype)
        delta = .2*np.ones(shape, dtype=dtype)
        theta = .7*np.ones(shape, dtype=dtype)
        phi = None
        if len(shape) > 2:
            phi = .35*np.ones(shape, dtype=dtype)

        return SeismicModel(space_order=space_order, vp=v, origin=origin, shape=shape,
                            dtype=dtype, spacing=spacing, nbl=nbl, epsilon=epsilon,
                            delta=delta, theta=theta, phi=phi, bcs="damp", **kwargs)

    elif preset.lower() in ['layers-isotropic']:
        # A n-layers model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 3.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        vp_i = np.linspace(vp_top, vp_bottom, nlayers)
        for i in range(1, nlayers):
            v[..., i*int(shape[-1] / nlayers):] = vp_i[i]  # Bottom velocity

        return SeismicModel(space_order=space_order, vp=v, origin=origin, shape=shape,
                            dtype=dtype, spacing=spacing, nbl=nbl, bcs="damp",
                            fs=fs, **kwargs)

    elif preset.lower() in ['layers-elastic']:
        # A n-layers model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 3.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        vp_i = np.linspace(vp_top, vp_bottom, nlayers)
        for i in range(1, nlayers):
            v[..., i*int(shape[-1] / nlayers):] = vp_i[i]  # Bottom velocity

        vs = 0.5 * v[:]
        b = 1 / (0.31 * (1e3*v)**0.25)
        b[v < 1.51] = 1.0
        vs[v < 1.51] = 0.0

        return SeismicModel(space_order=space_order, vp=v, vs=vs, b=b,
                            origin=origin, shape=shape,
                            dtype=dtype, spacing=spacing, nbl=nbl, **kwargs)

    elif preset.lower() in ['layers-viscoelastic', 'twolayer-viscoelastic',
                            '2layer-viscoelastic']:
        # A two-layer model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.6 km/s,
        # and the bottom part of the domain has 2.2 km/s.
        ratio = kwargs.pop('ratio', 3)
        vp_top = kwargs.pop('vp_top', 1.6)
        qp_top = kwargs.pop('qp_top', 40.)
        vs_top = kwargs.pop('vs_top', 0.4)
        qs_top = kwargs.pop('qs_top', 30.)
        b_top = kwargs.pop('b_top', 1/1.3)
        vp_bottom = kwargs.pop('vp_bottom', 2.2)
        qp_bottom = kwargs.pop('qp_bottom', 100.)
        vs_bottom = kwargs.pop('vs_bottom', 1.2)
        qs_bottom = kwargs.pop('qs_bottom', 70.)
        b_bottom = kwargs.pop('b_bottom', 1/2.)

        # Define a velocity profile in km/s
        vp = np.empty(shape, dtype=dtype)
        qp = np.empty(shape, dtype=dtype)
        vs = np.empty(shape, dtype=dtype)
        qs = np.empty(shape, dtype=dtype)
        b = np.empty(shape, dtype=dtype)
        # Top and bottom P-wave velocity
        vp[:] = vp_top
        vp[..., int(shape[-1] / ratio):] = vp_bottom
        # Top and bottom P-wave quality factor
        qp[:] = qp_top
        qp[..., int(shape[-1] / ratio):] = qp_bottom
        # Top and bottom S-wave velocity
        vs[:] = vs_top
        vs[..., int(shape[-1] / ratio):] = vs_bottom
        # Top and bottom S-wave quality factor
        qs[:] = qs_top
        qs[..., int(shape[-1] / ratio):] = qs_bottom
        # Top and bottom density
        b[:] = b_top
        b[..., int(shape[-1] / ratio):] = b_bottom

        return SeismicModel(space_order=space_order, vp=vp, qp=qp,
                            vs=vs, qs=qs, b=b, origin=origin,
                            shape=shape, dtype=dtype, spacing=spacing,
                            nbl=nbl, **kwargs)

    elif preset.lower() in ['layers-tti', 'layers-tti-noazimuth']:
        # A n-layers model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.\
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 3.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        vp_i = np.linspace(vp_top, vp_bottom, nlayers)
        for i in range(1, nlayers):
            v[..., i*int(shape[-1] / nlayers):] = vp_i[i]  # Bottom velocity

        epsilon = .3*(v - 1.5)
        delta = .2*(v - 1.5)
        theta = .5*(v - 1.5)
        phi = None
        if len(shape) > 2 and preset.lower() not in ['layers-tti-noazimuth']:
            phi = .25*(v - 1.5)

        model = SeismicModel(space_order=space_order, vp=v, origin=origin, shape=shape,
                             dtype=dtype, spacing=spacing, nbl=nbl, epsilon=epsilon,
                             delta=delta, theta=theta, phi=phi, bcs="damp", **kwargs)

        if kwargs.get('smooth', False):
            if len(shape) > 2 and preset.lower() not in ['layers-tti-noazimuth']:
                model.smooth(('epsilon', 'delta', 'theta', 'phi'))
            else:
                model.smooth(('epsilon', 'delta', 'theta'))

        return model

    elif preset.lower() in ['circle-isotropic']:
        # A simple circle in a 2D domain with a background velocity.
        # By default, the circle velocity is 2.5 km/s,
        # and the background veloity is 3.0 km/s.
        vp = kwargs.pop('vp_circle', 3.0)
        vp_background = kwargs.pop('vp_background', 2.5)
        r = kwargs.pop('r', 15)

        # Only a 2D preset is available currently
        assert(len(shape) == 2)

        v = np.empty(shape, dtype=dtype)
        v[:] = vp_background

        a, b = shape[0] / 2, shape[1] / 2
        y, x = np.ogrid[-a:shape[0]-a, -b:shape[1]-b]
        v[x*x + y*y <= r*r] = vp

        return SeismicModel(space_order=space_order, vp=v, origin=origin, shape=shape,
                            dtype=dtype, spacing=spacing, nbl=nbl, bcs="damp",
                            fs=fs, **kwargs)

    elif preset.lower() in ['marmousi-isotropic', 'marmousi2d-isotropic']:
        shape = (1601, 401)
        spacing = (7.5, 7.5)
        origin = (0., 0.)
        nbl = kwargs.pop('nbl', 20)

        # Read 2D Marmousi model from devitocodes/data repo
        data_path = kwargs.get('data_path', None)
        if data_path is None:
            raise ValueError("Path to devitocodes/data not found! Please specify with "
                             "'data_path=<path/to/devitocodes/data>'")
        path = os.path.join(data_path, 'Simple2D/vp_marmousi_bi')
        v = np.fromfile(path, dtype='float32', sep="")
        v = v.reshape(shape)

        # Cut the model to make it slightly cheaper
        v = v[301:-300, :]

        return SeismicModel(space_order=space_order, vp=v, origin=origin, shape=v.shape,
                            dtype=np.float32, spacing=spacing, nbl=nbl, bcs="damp",
                            fs=fs, **kwargs)

    elif preset.lower() in ['marmousi-tti2d', 'marmousi2d-tti',
                            'marmousi-tti3d', 'marmousi3d-tti']:

        shape = (201, 201, 70)
        nbl = kwargs.pop('nbl', 20)

        # Read 2D Marmousi model from devitocodes/data repo
        data_path = kwargs.pop('data_path', None)
        if data_path is None:
            raise ValueError("Path to devitocodes/data not found! Please specify with "
                             "'data_path=<path/to/devitocodes/data>'")
        path = os.path.join(data_path, 'marmousi3D/vp_marmousi_bi')

        # velocity
        vp = 1e-3 * np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiVP.raw'),
                                dtype='float32', sep="")
        vp = vp.reshape(shape)

        # Epsilon, in % in file, resale between 0 and 1
        epsilon = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiEps.raw'),
                              dtype='float32', sep="") * 1e-2
        epsilon = epsilon.reshape(shape)

        # Delta, in % in file, resale between 0 and 1
        delta = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiDelta.raw'),
                            dtype='float32', sep="") * 1e-2
        delta = delta.reshape(shape)

        # Theta, in degrees in file, resale in radian
        theta = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiTilt.raw'),
                            dtype='float32', sep="")
        theta = np.float32(np.pi / 180 * theta.reshape(shape))

        if preset.lower() in ['marmousi-tti3d', 'marmousi3d-tti']:
            # Phi, in degrees in file, resale in radian
            phi = np.fromfile(os.path.join(data_path, 'marmousi3D/Azimuth.raw'),
                              dtype='float32', sep="")
            phi = np.float32(np.pi / 180 * phi.reshape(shape))
        else:
            vp = vp[101, :, :]
            epsilon = epsilon[101, :, :]
            delta = delta[101, :, :]
            theta = theta[101, :, :]
            shape = vp.shape
            phi = None

        spacing = tuple([10.0]*len(shape))
        origin = tuple([0.0]*len(shape))

        return SeismicModel(space_order=space_order, vp=vp, origin=origin, shape=shape,
                            dtype=np.float32, spacing=spacing, nbl=nbl, epsilon=epsilon,
                            delta=delta, theta=theta, phi=phi, bcs="damp", **kwargs)

    elif preset.lower() in ['layers-viscoacoustic']:
        # A n-layers model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 3.5 km/s.

        # Define a velocity profile in km/s
        vp = np.empty(shape, dtype=dtype)
        qp = np.empty(shape, dtype=dtype)

        # Top and bottom P-wave velocity
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 3.5)

        # Define a velocity profile in km/s
        vp = np.empty(shape, dtype=dtype)
        vp[:] = vp_top  # Top velocity (background)
        vp_i = np.linspace(vp_top, vp_bottom, nlayers)
        for i in range(1, nlayers):
            vp[..., i*int(shape[-1] / nlayers):] = vp_i[i]  # Bottom velocity

        qp[:] = 3.516*((vp[:]*1000.)**2.2)*10**(-6)  # Li's empirical formula

        b = 1 / (0.31*(vp[:]*1000.)**0.25)  # Gardner's relation

        return SeismicModel(space_order=space_order, vp=vp, qp=qp, b=b, nbl=nbl,
                            origin=origin, shape=shape, spacing=spacing, **kwargs)

    else:
        raise ValueError("Unknown model preset name")
