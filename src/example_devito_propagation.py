import numpy as np
from devito import TimeFunction, VectorTimeFunction, Eq, solve, Operator

#NBVAL_IGNORE_OUTPUT
from model import Model, demo_model
from plotting import plot_velocity, plot_shotrecord, plot_image
from source import TimeAxis, RickerSource, Receiver

import matplotlib as cm
import matplotlib.pyplot as plt

N = (401, 401)

model = demo_model('layers-isotropic',             # A velocity model.
                   nlayers=6,
                   shape=N,  # Number of grid points.
                   spacing=(7.5, 7.5),  # Grid spacing in m.
                   nbl=80, space_order=8)      # boundary layer.
vp = model.vp.data

# NOT FOR MANUSCRIPT

# Quick plot of model.
plot_velocity(model)

# NOT FOR MANUSCRIPT
from devito import TimeFunction

t0 = 0.     # Simulation starts a t=0
tn = 3000.  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing

time = TimeAxis(start=t0, stop=tn, step=dt)

u = TimeFunction(name="u", grid=model.grid, 
                 time_order=2, space_order=8,
                 save=time.num)

# NOT FOR MANUSCRIPT
from devito import Eq, solve

pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

stencil = Eq(u.forward, solve(pde, u.forward))

# Src is halfway across model, at depth of 20 m.
x_extent, _ = model.domain_size
src_coords = [x_extent/2, 20]

f0 = 0.025  # kHz, peak frequency.
src = RickerSource(name='src', grid=model.grid, f0=f0, coordinates=src_coords, time_range=time)

src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)


# Recs are distributed across model, at depth of 20 m.
x_locs = np.linspace(0, x_extent, N[0])
rec_coords = [(x, 20) for x in x_locs]

rec = Receiver(name='rec', npoint=N[0], grid=model.grid, coordinates=rec_coords, time_range=time)

rec_term = rec.interpolate(u)

# NOT FOR MANUSCRIPT
# PLOTS HALF OF FIGURE 1.
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator

fig = plt.figure(figsize=(9,9))

extent = [model.origin[0], model.origin[0] + 1e-3 * model.shape[0] * model.spacing[0],
          model.origin[1] + 1e-3*model.shape[1] * model.spacing[1], model.origin[1]]

model_param = dict(vmin=1.5, vmax=3.5, cmap="GnBu", aspect=1, extent=extent)

ax0 = fig.add_subplot(111)
im = plt.imshow(np.transpose(vp), **model_param)
cb = plt.colorbar(shrink=0.8)
ax0.set_ylabel('Depth (km)',fontsize=20)
ax0.set_xlabel('X position (km)', fontsize=20)
cb.set_label('Velocity (km/s)', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
cb.ax.tick_params(labelsize=14)

plt.scatter(*(rec.coordinates.data[::8, :].T/1000), c='green', clip_on=False, zorder=100)
plt.text(*rec.coordinates.data[0].T/1000 + [0.02, 0.15], "receivers", color='green', size=14)
plt.scatter(*(src.coordinates.data.squeeze()/1000), c='red', s=60)
plt.text(*src.coordinates.data[0]/1000 + [0, 0.15], "source", color='red', size=14)
plt.scatter(0, 0, c='black', s=160, clip_on=False, zorder=101)
plt.text(-0.01, -0.03, "Origin", color='k', size=16, ha="right")
plt.title("Example velocity model", color='k', size=24)
plt.xlim((0, 3))
plt.ylim((3, 0))

minorLocator = MultipleLocator(1/100)
ax0.xaxis.set_minor_locator(minorLocator)
ax0.yaxis.set_minor_locator(minorLocator)

plt.grid(which='minor', alpha=0.3)

plt.savefig("model.pdf", dpi=400)
plt.savefig("model.png")
plt.show()

# NOT FOR MANUSCRIPT
from devito import Operator

op_fwd = Operator([stencil] + src_term + rec_term)

op_fwd(dt=model.critical_dt)

# NOT FOR MANUSCRIPT
# GENERATES FIGURE 2
from matplotlib import cm

fig1 = plt.figure(figsize=(10,10))
l = plt.imshow(rec.data, vmin=-.1, vmax=.1, cmap=cm.gray, aspect=1,
               extent=[model.origin[0], model.origin[0] + 1e-3*model.shape[0] * model.spacing[0],
                       1e-3*tn, t0])
plt.xlabel('X position (km)',  fontsize=20)
plt.ylabel('Time (s)',  fontsize=20)
plt.tick_params(labelsize=20)

plt.savefig("Figure2.png", dpi=400)
plt.savefig("Figure2.pdf")
plt.show()

# NOT FOR MANUSCRIPT
# GENERATES FIGURE 3
fig = plt.figure(figsize=(15, 5))

times = [400, 600, 800]

extent = [model.origin[0], model.origin[0] + 1e-3 * model.shape[0] * model.spacing[0],
          model.origin[1] + 1e-3*model.shape[1] * model.spacing[1], model.origin[1]]

data_param = dict(vmin=-1e-1, vmax=1e-1, cmap=cm.Greys, aspect=1, extent=extent, interpolation='none')
model_param = dict(vmin=1.5, vmax=3.5, cmap=cm.GnBu, aspect=1, extent=extent, alpha=.3)

ax0 = fig.add_subplot(131)
_ = plt.imshow(np.transpose(u.data[times[0],40:-40,40:-40]), **data_param)
_ = plt.imshow(np.transpose(vp), **model_param)
ax0.set_ylabel('Depth (km)',  fontsize=20)
ax0.text(0.75, 0.18, "t = {:.0f} ms".format(time.time_values[times[0]]), ha="center", color='k')

ax1 = fig.add_subplot(132)
_ = plt.imshow(np.transpose(u.data[times[1],40:-40,40:-40]), **data_param)
_ = plt.imshow(np.transpose(vp), **model_param)
ax1.set_xlabel('X position (km)',  fontsize=20)
ax1.set_yticklabels([])
ax1.text(0.75, 0.18, "t = {:.0f} ms".format(time.time_values[times[1]]), ha="center", color='k')

ax2 = fig.add_subplot(133)
_ = plt.imshow(np.transpose(u.data[times[2],40:-40,40:-40]), **data_param)
_ = plt.imshow(np.transpose(vp), **model_param)
ax2.set_yticklabels([])
ax2.text(0.75, 0.18, "t = {:.0f} ms".format(time.time_values[times[2]]), ha="center", color='k')

plt.savefig("Figure3.pdf")
plt.savefig("Figure3.png", dpi=400)
plt.show()


