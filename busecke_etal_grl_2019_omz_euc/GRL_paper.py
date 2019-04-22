############## Calculate EUC shape parameters ################
import numpy as np
import xarray as xr
import palettable
import matplotlib.pyplot as plt
import nc_time_axis
import warnings

from scipy.stats import linregress
from scipy.interpolate import interp1d

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import matplotlib.ticker as mticker


from xarrayutils import xr_linregress
from xarrayutils import filter_1D
from xarrayutils.plotting import center_lim, axis_arrow

# !!! this should be replaced by the xgcm version
from xarrayutils.weighted_operations import weighted_mean

fullwidth = 190/25.4 # agu full page fig size in inches
halfwidth = fullwidth / 2

###################### Preprocessing ###########################
def _debug_check(lst):
    print([a for a in list(lst) if a == 'dzt'])

def eq_mean(ds, roi=dict(yt_ocean=slice(-1,1)), model_processing=True, coord_include = ['dzt']):
    """Calculate the equatorial slices, defined by the weighted mean around the equator. 
    `model_processing` will add budget decompositions and interpolate all fields onto tracer grid."""
    ds = ds.copy()
    for co in coord_include:
        if co in list(ds.variables):
            ds[co+'_temp'] = ds[co]
            ds = ds.drop(co)
        # This works but seems clunky that I have to go through these lengths to assign any variable as a data_variable
    
    if model_processing:
        grid = Grid(ds)
        ds = interp_all(grid, ds)
        # This step is not conservative. Just a linear interpolation. But that only has a small effect on the velocities, since the budget terms are already on the tracer grid. Same for the other functions.
    ds = ds.sel(**roi)
    
    ds_mean = weighted_mean(ds, ds.dyt, dim='yt_ocean')
    
    for co in coord_include:
        tempname = co+'_temp'
        if tempname in list(ds_mean.data_vars):
            ds_mean.coords[co] = ds_mean[tempname]
            ds_mean = ds_mean.drop(tempname)
        
        
    ds_mean.attrs['averaged region'] = str(roi)
    
    return ds_mean

def depth_slice(ds, st_ocean=250, model_processing=True):
    """Calculate a depth slice via interpolation.
    `model_processing` will interpolate all fields onto tracer grid."""
    ds = ds.copy()
    if model_processing:
        grid = Grid(ds)
        ds = interp_all(grid, ds) 
    ds = ds.interp(st_ocean=st_ocean)
    ds.attrs['depth_slice'] = str(st_ocean)
    return ds

def lat_slice(ds, ref, model_processing=True):
    ds = ds.copy()
    if model_processing:
        grid = Grid(ds)
        ds = interp_all(grid, ds)
    ds = ds.interp(xt_ocean=ref.xt_ocean)
    return ds

# to xarrayutils?
def remove_seas_simple(ds):
    """Returns deseasonalized and standardized anomalies"""
    ds_clim_mean = ds.groupby('time.month').mean('time')    
    return xr.apply_ufunc(lambda x, m: (x - m), ds.groupby('time.month'), ds_clim_mean)


#################### EUC Parameters ############################
def _interp(x, y, c):
    try:
        return interp1d(y,x)(c)
    except ValueError:
        print('Something went wrong during interpolation')
        print('x', x)
        print('y', y)
        print('c', c)
        return np.nan

def _thickness(x,y,c):
    """finds distance along x between crossings of y(x) and a constant value c
    
    Example:
    x = profile.st_ocean.data
    y = profile.data
    c = 0.4
    tt = _thickness(x, y, c)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y)
    plt.plot(x, np.ones_like(x)*c)
    plt.plot(tt[1:3], np.ones(2)*c, 'ro')
    # Check if the thickness lines up
    plt.plot([tt[1], tt[1]+tt[0]], [c, c], linestyle='--')
    """
    # remove nans
    nan_idx = np.isnan(y)
    x = x[~nan_idx]
    y = y[~nan_idx]
    
    idx = np.argwhere(np.diff(np.sign(c-y))).flatten()
    segments = [([x[aa], x[aa+1]],[y[aa], y[aa+1]]) for aa in idx]
    crossings = np.array([_interp(x, y, c) for (x,y) in segments])
    
    # For now just ditch values that are not in pairs, e.g. have less/more than 2 crossings. 
    # Future improvements could look for a pair that has a max or min in the middle
    if len(crossings) != 2:
        thickness = np.nan
        first = np.nan
        last = np.nan
    else:
        thickness = np.diff(crossings)
        first = crossings[0]
        last = crossings[1]
    
    return np.array([thickness, first, last])

def extract_shape(da, c, dim='st_ocean'):
    """Calculates the shape of a local maximum (e.g. the EUC in zonal velocity)
    Extracts the maximum val/pos along dim
    and also calculates the length between two values=c along dim
    """
    
    if not isinstance(da, xr.DataArray):
        raise ValueError('Input for `da` must be xr.DataArray')
    
    mask = xr.ufuncs.isnan(da).all(dim)
    
    args = (da[dim], da, c)
    thick = xr.apply_ufunc(_thickness, *args, 
                         input_core_dims=[[dim], [dim], []],
                         output_core_dims=[['new']],
                         vectorize=True,
                         dask='parallelized',
                         output_dtypes=[da.dtype],
                         output_sizes={'new':3}).where(~mask)
    
    idx = da.fillna(0).argmax(dim).load()
    core_pos = da[dim][{dim:idx}]
    core_val = da[{dim:idx}]
        
    params = xr.Dataset({'value':core_val.where(~mask), 'position':core_pos.where(~mask),
                         'extent':thick.isel(new=0), 'inner':thick.isel(new=1),
                         'outer':thick.isel(new=2)})
    
    return params


def extract_EUC_stats(ds, c):
    """
    Extracts EUC statistics (high level wrapper for extract_shape)
    along depth and latitude
    """
    
    if 'yu_ocean' in ds.dims:
        roi = dict(yu_ocean=slice(-1, 1), st_ocean=slice(0,500))
        roi_wide = dict(yu_ocean=slice(-3, 3), st_ocean=slice(0,500))
        ds_sub=ds.sel(**roi)
        z_slice = weighted_mean(ds_sub.u, ds_sub.dyu, dim='yu_ocean')
    else:
        print('Warning: yu_ocean not found....check if it isnt labeled otherwise')
        roi = dict(st_ocean=slice(0,500))
        roi_wide = dict(st_ocean=slice(0,500))
        ds_sub=ds.sel(**roi)
        z_slice = ds_sub.u

    z_shape = extract_shape(z_slice, c, dim='st_ocean')
    
    ds_out = xr.Dataset({'core_value_z':z_shape.value,
                'core_pos_z':z_shape.position,
                'thickness':z_shape.extent,
                'upper':z_shape.inner,
                'lower':z_shape.outer,
                })
    
    if 'yu_ocean' in ds_sub.dims:
        # Get only the y position for the later masking
        y_slice = ds.sel(**roi_wide).u.max('st_ocean')
        y_shape = extract_shape(y_slice, c, dim='yu_ocean')

        ds_out['core_pos_y'] = y_shape.position
    
    ds_out.attrs['EUC_boundary'] = c
    return ds_out


def non_mono_idx(a):
    """Returns index of nonmonotonic values"""
    mon_idx = np.hstack((np.array([-1]), np.diff(a)))
    return mon_idx < 0

def _extract_contour(data, value, x, y, showplot):
    fig, ax = plt.subplots()
    if data.shape[0] != len(y):
        data = data.transpose()
    # plot contour
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hh = ax.contour(x, y, data, levels=[value])

            # regres contour...
            ch = hh.collections[0].get_paths()
            # Get the values
            verts = ch[0].vertices
            cx = verts[:,0]
            cy = verts[:,1]
            
            # interpolation will fail if the contours x values are non-monotonic. 
            # so remove those values
            
            # this effectively selects the upper boundary of the OMZ
            
            while any(~non_mono_idx(cx)):
                idx = non_mono_idx(cx)
                cx = cx[idx]
                cy = cy[idx]
                
            # this procedure can leave strong 'steps' in the found upper boundary, when the OMZ countour undulates.
            # remove values that show depth differences of more than 10m
            gap_idx = np.where(abs(np.diff(cy)) > 100)[0]
            if gap_idx.size != 0:
                cx = cx[0:gap_idx[0]]
                cy = cy[0:gap_idx[0]]
                
            
            
            interpolated = interp1d(cx, cy, bounds_error=False)(x)
                
    except IndexError:
        interpolated = np.ones(len(x)) * np.nan
    
    # plot results
    if showplot:
        ax.plot(cx, cy, marker='*')
        ax.plot(x, interpolated, color='y')
        plt.show()
    else:
        plt.close(fig)
    

    return interpolated

def xr_extract_contour(da, value, xdim, ydim, showplot=True):
    out = xr.apply_ufunc(_extract_contour, da, value, da[xdim], da[ydim], showplot,
                          input_core_dims = [[xdim, ydim], [], [xdim], [ydim], []],
                          output_core_dims = [[xdim]],
                          vectorize=True)
    return out

#### Misc convenience functions
def color_dict():
    cmap = palettable.cartocolors.qualitative.Vivid_10.mpl_colors

    cmip_models = ['NorESM1-ME', 'CESM1-BGC', 'MPI-ESM-LR', 'GFDL-ESM2M', 'MRI-ESM1',
           'IPSL-CM5A-LR', 'GFDL-ESM2G', 'MPI-ESM-MR']
    color_dict = {'CM21deg':'C0', 'CM26':'C1', 'Bianchi':'0.5', 'obs':'0.5'}

    for mi, mm in enumerate(cmip_models):
        color_dict[mm] = cmap[mi+2]
    return color_dict

def regress_slope(ds, regression_lon):
    ds = ds.copy()
    regression_roi = dict(xu_ocean=regression_lon)
    ds = ds.sel(**regression_roi)
    # interpolate over nans
    ds.interpolate_na('xu_ocean')
    return xr_linregress(ds.xu_ocean, ds, dim='xu_ocean', convert_to_dataset=False, nanmask=True)

def unwrap_CMIP_models(full_dict):
    added_dict = full_dict.copy()
    for kk, ds in full_dict.items():
        if 'model' in ds.dims:
            for mo in ds.model:
                added_dict[kk+'_'+str(mo.data)] = ds.sel(model=mo)
            del added_dict[kk]
    
    return added_dict


### This should go to mom_tools
from xgcm import Grid
# #!!! at the point of publication This needs to be packaged into a proper module, with version that can be included in the requirments
import sys 
sys.path.append('/home/Julius.Busecke/projects/mom_tools')
from mom_read import parse_xgcm_attributes

# For budget analysis
def compute_residual(ds, varlist, reference, res_name = 'residual'):
    """compute residual of given terms"""
    residual_list = [a for a in varlist if a not in [res_name, reference]]
    res = ds[reference] - sum([ds[v] for v in residual_list if v in ds.data_vars])
    res.name = res_name
    return res

def mom_recombine_budgets(ds):
    """Recombines some budget terms to a more intuitive form (e.g. impl_diffusion+KPP = vertical)"""
    ds = ds.copy()
    budget_vars = [a.replace('_tendency','') for a in ds.data_vars if 'tendency' in a]
    
    # can wrap this in a replace dictionary...but for now hardcode
    for var in budget_vars:
        if (var+'_vdiffuse_impl' in ds.data_vars) and (var+'_nonlocal_KPP' in ds.data_vars):
            ds[var+'_vertical_diff'] = ds[var+'_vdiffuse_impl'] + ds[var+'_nonlocal_KPP']
        else:
            print('No `vertical_diff` generated for %s' %var)
        
        if ('neutral_diffusion_'+var in ds.data_vars) and ('neutral_gm_'+var in ds.data_vars):
            ds[var+'_eddy_combined'] = ds['neutral_diffusion_'+var] + ds['neutral_gm_'+var]
        else:
            print('No `eddy_combined` generated for %s. dummy added.' %var)
            ds[var+'_eddy_combined'] = ds[var+'_tendency'] * 0
            
    if 'o2_advection' in ds.data_vars:
        #reconstruct mean advection from all components
        ds['o2_advection_recon'] = ds['o2_advection_x_recon'] + ds['o2_advection_y_recon'] + ds['o2_advection_z_recon']
        ds['o2_advection_eddy_recon'] = ds['o2_advection'] - ds['o2_advection_recon']
        ds['o2_advection_recon_w_diff'] = ds['o2_advection_recon'] + ds['o2_vertical_diff']
        ds['o2_advection_recon_test'] = ds['o2_advection_x'] + ds['o2_advection_y'] + ds['o2_advection_z']
#         ds['o2_advection_horizontal'] = ds['o2_advection_x'] + ds['o2_advection_y']
#         ds['o2_advection_horizontal_w_diff'] = ds['o2_advection_horizontal'] + ds['o2_vertical_diff']
#         ds['o2_advection_x_w_diff'] = ds['o2_advection_x'] + ds['o2_vertical_diff']
#         ds['o2_advection_w_eddy'] = ds['o2_eddy_combined'] + ds['o2_advection']
#         ds['o2_advection_horizontal_w_diff_eddy'] = ds['o2_advection_horizontal'] + ds['o2_vertical_diff'] + ds['o2_eddy_combined']

#         if 'o2_eddy_combined' in ds.data_vars:
#             ds['o2_advection_w_eddy'] = ds['o2_eddy_combined'] + ds['o2_advection']
#             ds['o2_advection_horizontal_w_diff_eddy'] = ds['o2_advection_horizontal'] + ds['o2_vertical_diff'] + ds['o2_eddy_combined']
#         else:
#             ds['o2_advection_w_eddy'] = ds['o2_advection']
#             ds['o2_advection_horizontal_w_diff_eddy'] = ds['o2_advection_horizontal'] + ds['o2_vertical_diff']

#         ds['o2_advection_w_diff_eddy'] = ds['o2_advection_w_eddy'] + ds['o2_vertical_diff']
    
    else:
        print('no budget terms found')
            
    return ds


############## plotting ############
def map_gridlines(ax, ticks, **kwargs):
    ax.outline_patch.set_linewidth(plt.rcParams['axes.linewidth'])
    # This shoul be executed twice if the dateline is crossed...
    #...see (https://github.com/SciTools/cartopy/issues/276)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, **kwargs)
    gl.xlocator = mticker.FixedLocator(ticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False

def label_lon_lat_axis(ax, lon=True, axis='x'):
    if axis == 'x':
        labels = np.array(ax.get_xticks())
    elif axis == 'y':
        labels = np.array(ax.get_yticks())
    else:
        raise RuntimeWarning('axis has to be `x` or `y`, got %s' %axis)
    
    if lon:
        formatter = LONGITUDE_FORMATTER
        labels[labels<(-180)] = 360 + labels[labels<(-180)] # this should be changed to <= to get 180degE instead of 180degW
    else:
        # this asumes latitude values
        formatter = LATITUDE_FORMATTER
    
    labels_converted = [formatter(la) for la in labels]
    
    if axis == 'x':
        ax.set_xticklabels(labels_converted)
    elif axis == 'y':
        ax.set_yticklabels(labels_converted)

def scatter_plot_base(ds, xvar, yvar, xparam, yparam, xop, yop, color='k', invertx=True, inverty=True,
                      ax = None, preaverage=False, line=False, **kwargs):
    #handle colors and lineproperties
    kwargs.update(dict(facecolors=[color], edgecolors=[color], linewidths=0.5))
    
    
    #handle labels
    xlabel = '%s %s %s' %(xop, xparam, xvar)
    ylabel = '%s %s %s' %(yop, yparam, yvar)
    
    def apply_op(op, data):
        if 'time' in ds.dims:
            if op == 'mean':
                return data.mean('time')
            elif op == 'std':
                return data.std('time')
            else:
                raise ValueError('op %s not recognized' %op)
        else:
            if op == 'mean':
                return data
            else:
                return data * np.nan
            
    # if std is chosen first average yearly to get comparable variability
    if preaverage and 'time' in ds.dims :
        ds = ds.groupby('time.year').mean('time').rename({'year':'time'})
    
    x = apply_op(xop, ds[xvar].sel(parameter=xparam))
    if invertx:
        x = -x
    y = apply_op(yop, ds[yvar].sel(parameter=yparam))
    
    if inverty:
        y = -y
    
    if 'run' in x.dims:
        kwargs['facecolors'] = ['none', color]
    
    if ax is None:
        ax = plt.gca()
    ax.scatter(x, y, **kwargs)
    
    if line:
        ax.axvline(x, linewidth=3, alpha=0.5, color='0.5')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return x, y

def scatter_wrapper(shape_regressed, xvar, yvar, xparam, yparam, xop, yop, ax, exclude=['tao', 'CMIP5_IPSL-CM5A-LR'], invertx=True, inverty=True):
    model_colors = color_dict()
    
    preaverage = False
    
    x = np.array([])
    y = np.array([])
    for kk in [a for a in shape_regressed.keys() if a not in exclude]:
        if kk == 'obs':
            line = True
        else:
            line = False
        ds = shape_regressed[kk]
        kwargs = dict(label=kk, s=8)
        
        co = model_colors[kk.replace('CMIP5_', '')]
        import matplotlib as mpl
        co = mpl.colors.to_hex(co)
        if kk in ['CM21deg', 'CM26', 'obs']:
            kwargs['s'] = 12
        if kk in ['CM21deg', 'CM26'] and 'std' in [xop, yop]:
            preaverage = True
            kwargs['label'] = kwargs['label'] + ' (from ann averages)'
        xx, yy = scatter_plot_base(ds, xvar, yvar, xparam, yparam,
                                   xop, yop, color = co,
                                   invertx=invertx, inverty=inverty, preaverage=preaverage, line=line, ax=ax, **kwargs)
        x = np.hstack((x, xx))
        y = np.hstack((y, yy))
    # ax.set_ylim([-10, 0])
    # mask nans
    idx = np.logical_or(np.isnan(x), np.isnan(y))
    reg = linregress(x[~idx], y[~idx])
    x_ax = np.array(ax.get_xlim())
    ax.text(0.7, 0.1, 'r: %.2f (%i%s)' %(reg.rvalue, (1-reg.pvalue)*100, '%'), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    ax.plot(x_ax, (reg.slope * x_ax) + reg.intercept)
    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    
def below_euc_plot(full_ds, shape_ds, highlight_lw=1):
    slim_ds = full_ds.drop([v for v in full_ds.data_vars if ('o2' not in v) and ('jo2' not in v)])

    # convert thickness weighted tendencies to concentration tend
    slim_ds = slim_ds / slim_ds.dzt
    units = 'mol/m^3/s'
    
    #mask between the omz boundary and the EUC core
    mask = xr.ufuncs.logical_and(full_ds.st_ocean >= shape_ds.core_pos_z, full_ds.o2 > 80e-6)
    dropcoords = [v for v in list(mask.coords) if v not in mask.dims]
    mask = mask.drop(dropcoords)
    ds_masked = slim_ds.where(mask)

    ds_mean = weighted_mean(ds_masked, full_ds.dzt, dim='st_ocean')
    fig, axarr = plt.subplots(ncols=4, figsize=[fullwidth *1.5, 1*2], dpi=600)
    
    #Testing how good the reconstruction of the full fluxes works
    ax = axarr.flat[0]
    ds_mean.o2_advection.mean('time').plot(ax=ax, label='advection', linewidth=highlight_lw, color='0.5')
    ds_mean.o2_advection_recon_test.mean('time').plot(ax=ax, label='advection from full fluxes')
    (ds_mean.o2_advection - ds_mean.o2_advection_recon_test).mean('time').plot(ax=ax, label='difference')
    ax.set_ylim(np.array([-0.2, 1]) * 2e-9)
    ax.legend()
    
    #Testing how good the reconstruction from monthly fluxes works
    ax = axarr.flat[1]
    ds_mean.o2_advection.mean('time').plot(ax=ax, label='advection')
    (ds_mean.o2_advection + ds_mean.o2_eddy_combined).mean('time').plot(ax=ax, label='advection+eddy')
    ds_mean.o2_eddy_combined.mean('time').plot(ax=ax, label='eddy')
    ds_mean.o2_advection_recon.mean('time').plot(ax=ax, label='advection (large scale)', linewidth=highlight_lw, color='0.5')
    (ds_mean.o2_advection - ds_mean.o2_advection_recon).mean('time').plot(ax=ax, label='advection - advection(large scale)')
    ax.set_ylim(np.array([-0.5, 1]) * 2e-9)
    ax.legend()
    
    #How does the slow varying reconstructed tendency split up in x-y-z components
    ax = axarr.flat[2]
    ds_mean.o2_advection_recon.mean('time').plot(ax=ax, label='advection (large scale)')
    ds_mean.o2_advection_x_recon.mean('time').plot(ax=ax, label='x advection (large scale)', linewidth=highlight_lw, color='0.5')
    ds_mean.o2_advection_y_recon.mean('time').plot(ax=ax, label='y advection (large scale)')
    ds_mean.o2_advection_z_recon.mean('time').plot(ax=ax, label='z advection (large scale)')
    ax.set_ylim(np.array([-1, 1]) * 3e-8)
    ax.legend()
    
    
    # What dominates the x component
    ax = axarr.flat[3]
    ds_mean.o2_advection_x_recon.mean('time').plot(ax=ax, label='x advection')
    ds_mean.o2_advection_x_recon_vel.mean('time').plot(ax=ax, label='x advection (divergence)', linewidth=highlight_lw, color='0.5')
    ds_mean.o2_advection_x_recon_tracer.mean('time').plot(ax=ax, label='x advection (tracer)')
    ax.set_ylim(np.array([-0.3, 1]) * 3e-8)
    ax.legend()
    
    for ax in axarr.flat:
        ax.axhline(0, color='0.5')
        ax.set_ylabel('o2 tendency [%s]' %units)
        ax.set_xlim(-180, -100)
        ax.set_title('')
        ax.set_xlabel('Longitude')
        ax.set_xticks(np.arange(-220, -60, 40))
        label_lon_lat_axis(ax, lon=True, axis='x')
    return fig

def budget_plot(ds_eq, var, metric, plot_kwargs, c_kwargs, c2_kwargs, islands, plot_roi, varlist):
    fig, axes = plt.subplots(ncols=2, figsize=np.array([7.2, 1]), dpi=600)
    for source, ax, dx in zip(['CM21deg', 'CM26'], axes, np.array([1, 1/10])*111e3):

        ds = ds_eq[source].copy()
        
        if var == 'residual':
            # Compute the residual based on input list
            ds['residual'] = compute_residual(ds, varlist, 'o2_tendency')

        if var in ds.data_vars:
            ds_plot = ds[var]
            ds_cont = ds['o2']
            ds_cont2 = ds['u']
            dzt = ds['dzt'].copy()

            # normalize the fluxes with the cell depth
            ds_plot = ds_plot / dzt

            ds_plot.name = var
            if metric == 'mean':
                plot_kwargs['vmax'] = 3e-9
                plot_kwargs['vmin'] = -plot_kwargs['vmax']
                ds_plot.mean('time').plot.contourf(levels=31, ax=ax, **plot_kwargs)
            elif metric == 'std':
                plot_kwargs['vmax'] = 6e-9
                plot_kwargs['vmin'] = -plot_kwargs['vmax']
                ds_plot.std('time').plot.contourf(levels=30, ax=ax, **plot_kwargs)


            plot_kwargs['vmin'] = -plot_kwargs['vmax']
            ds_cont.mean('time').plot.contourf(ax=ax, **c_kwargs)
            ds_cont2.mean('time').plot.contour(ax=ax, **c2_kwargs)

        else:
            print('`%s` not found in dataset %s' %(var, source))

        # indicate important islands
        arrow_dict = dict(width=plt.rcParams['lines.linewidth'],
                          headwidth=2,
                          headlength=2,
                          lw=plt.rcParams['lines.linewidth']
                         )
        [axis_arrow(ax, is_lon, is_name, arrowprops=arrow_dict) for is_name, is_lon in islands.items()]

        ax.set_xlim(plot_roi['xt_ocean'].start, plot_roi['xt_ocean'].stop)
        ax.set_ylim(plot_roi['st_ocean'].stop, plot_roi['st_ocean'].start)

        ax.set_title('')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Depth [m]')
        # set less ticks
        ax.set_xticks(np.arange(-220, -60, 40))
        label_lon_lat_axis(ax, lon=True, axis='x')
#     letter_subplots(axes, fontsize=8, box_color='w')
    return fig
    

def eof_plot(ds_eof, ds, var_factors = {'o2':1e6}, var_plot_kwargs={'o2':{'robust':True, 'yincrease':False},
                                                                    'u':{'robust':True, 'yincrease':False}}):
    ds = ds.copy()
    names_eofs = [a for a in ds_eof.data_vars if 'eof_' in a]
    n_eofs = len(names_eofs)
    n_modes = len(ds_eof.mode)
    
    fig, axarr = plt.subplots(nrows=n_eofs+1, ncols=n_modes, figsize=[7.2/3*n_modes, 0.75*(n_eofs+1)], dpi=600)
    for xi, var in enumerate(names_eofs):
        axes = axarr[xi,:]
        plot_ds = ds_eof[var]
        var_stripped = var.replace('eof_', '')
        
        # Apply conversion factor if given
        if var_stripped in var_factors.keys():
            plot_ds = plot_ds * var_factors[var_stripped]
        
        if var_stripped in var_plot_kwargs.keys():
            kwargs = var_plot_kwargs[var_stripped]
        else:
            kwargs = {}
            
        for ai, (mode, ax) in enumerate(zip(ds_eof.mode, axes)):
            h = plot_ds.sel(mode=mode).plot.contourf(ax=ax, levels=15,**kwargs)
            cb=h.colorbar
            # get the colorlimits from the first plot and apply to all modes
            if ai == 0:
                kwargs['vmax'] = cb.get_clim()[1]
            # remove the colorbar since they are now all the same as the first plot
            if ai != n_modes-1:
                cb.remove()
                
            ########## contour overlays
            o2_levels = np.array([80])*1e-6
            u_levels = np.arange(0.2,0.9,0.3)
            rho_levels = [1026,]
            rho_color = 'C6'
            o2_color = 'k'
            u_color = '0.5'
            
            cont_kwargs = dict(x='xt_ocean', yincrease=False, add_labels=False)
            for k in ['x', 'y']:
                if k in kwargs.keys():
                    cont_kwargs[k] = kwargs[k]
            
            ds['u'].mean('time').plot.contour(ax=ax, levels=u_levels, colors=u_color, **cont_kwargs)
            ds['o2'].mean('time').plot.contour(ax=ax, levels=o2_levels, colors=o2_color, **cont_kwargs)
            ds['pot_rho_0'].mean('time').plot.contour(ax=ax, levels=rho_levels, colors=rho_color, **cont_kwargs)
            
            regress_field_absolute(ds['o2'], ds_eof['pc'].sel(mode=mode)).plot.contour(ax=ax, levels=o2_levels, colors=o2_color, linestyles='--', **cont_kwargs)
            regress_field_absolute(ds['pot_rho_0'], ds_eof['pc'].sel(mode=mode)).plot.contour(ax=ax, levels=rho_levels, colors=rho_color, linestyles='--', **cont_kwargs)

            ax.text(0.05, 0.05, '%s percent' %int(abs(ds_eof['variance_fraction'].sel(mode=mode))*100),transform=ax.transAxes)
            ax.set_xticks(np.arange(-220, -60, 40))
            label_lon_lat_axis(ax, lon=True, axis='x')
            ax.set_title('')
    
    ############# Timeseries
    for ai, (mode, ax) in enumerate(zip(ds_eof.mode, axarr[-1,:])):
        plot_ds = ds_eof['pc'].sel(mode=mode)
        
        # fix time until the xarray fix is out...
        plot_ds = fix_time(plot_ds)
        plot_ds_filtered = filter_1D(plot_ds, 2, 'time')
        
        ax.plot(plot_ds.time, plot_ds ,color='0.5', linewidth=0.5)
        ax.plot(plot_ds_filtered.time, plot_ds_filtered ,color='0.5')
        center_lim(ax)
        ax.set_xlabel('Year')
        
        
        # add additional timeseries
        if 'NINO34' in ds.data_vars:
            ts_external = fix_time(ds).NINO34
            ts_external_co = 'C1'
            axtwin = ax.twinx()
            axtwin.plot(ts_external.time, ts_external, color=ts_external_co)
            axtwin.set_ylabel(ts_external.name, color=ts_external_co)
            center_lim(axtwin)
            axtwin.spines['right'].set_color('C1')
            axtwin.tick_params(axis='y', colors='C1')
            axtwin.yaxis.label.set_color('C1')

                
    #syncing all eof axes in x and y
    axarr[0,0].get_shared_x_axes().join(*axarr[0:-1,:].flat)
    
    # removing unneccesary labels
    [ax.set_xticklabels([]) for ax in axarr[0:-2,:].flat]
    [ax.set_xlabel('')  for ax in axarr[0:-2,:].flat]
    
    [ax.set_yticklabels([]) for ax in axarr[0:-1,1:].flat]
    [ax.set_ylabel('') for ax in axarr[0:-1,1:].flat]
    [ax.set_title('') for ax in axarr[1:,:].flat]
    
    # aling the timeseries axes with the above
    for ax, ref in zip(axarr[-1, :], axarr[-2,:]):
        x0 = ax.get_position().x0
        y0 = ax.get_position().y0
        w = ref.get_position().width
        h = ref.get_position().height * 0.7
        ax.set_position([x0, y0, w, h])
    
    for ax in axarr[:-1, :].flat:
        ax.set_xlabel('')
        
    for ax in axarr[:-1, :].flat:
        ax.set_ylabel('Depth [m]')
        
    return fig, axarr

    
    
    
################ Budget Calculations ################
    
def calculate_momentum_budget(ds):
    grid = Grid(ds)
    
    combo = xr.Dataset()
    
    combo['u'] = ds.u
    combo['v'] = ds.v
    
    combo['du_dx'] = grid.diff(ds.u, 'X') / ds.dxtn
    combo['du_dy'] = grid.diff(ds.u, 'Y') / ds.dyte
    
    combo['u_du_dx'] = grid.interp(-combo['du_dx'] * grid.interp(ds.u, 'X'),'Y')
    combo['v_du_dy'] = grid.interp(-combo['du_dy'] * grid.interp(ds.v, 'Y'), 'X')
    
    
    combo['hor'] = combo['u_du_dx'] + combo['v_du_dy']
    combo['hor'].attrs['long_name'] = 'Zonal Velocity tendency due to hor divergence of momentum'
    combo['hor'].attrs['units'] = 'm/s^(-2)'
    
    # Add tracer and vertical vel in there to get all relavant. Then drop again
    combo['wt'] = ds.wt # for now just to include 'sw_ocean'
    combo['temp'] = ds.temp
    combo = combo.drop(['wt', 'temp'])
    return combo

################## Needs to be refactored 

############################### EOF tools ####################################

from eofs.standard import Eof
from eofs.multivariate.standard import MultivariateEof

def fix_time(ds):
    ds = ds.copy()
    ds.time.data = [nc_time_axis.CalendarDateTime(item, "360_day") for item in ds.time.data]
    return ds

def rewrap_eof(ds, ds_old):
    # drop time dimension
    ds_old = ds_old.isel(time=0).drop('time').copy()
    n_mode = ds.shape[0]
    coords = ds_old.copy().coords
    coords = [('mode', np.arange(n_mode))] + [(k,v.data) for k,v in coords.items() if k in ds_old.dims]
    return xr.DataArray(ds, coords=coords)

def rewrap_pc(ds, time):
    ds = ds.transpose()
    n_mode = ds.shape[0]
    coords = time.copy().coords
    coords = [('mode', np.arange(n_mode))] + [(k,v.data) for k,v in coords.items() if k in time.dims]
    return xr.DataArray(ds, coords=coords)

def rewrap_eig(ds):
    n_mode = ds.shape[0]
    return xr.DataArray(ds, coords=[('mode', np.arange(n_mode))])

def eof_wrapper(ds, fields, weight_fields, n=4, normalize=True):
    
    ds = ds.fillna(0).copy()
    data = [ds[d].copy() for d in fields]
    weights = [ds[w] for w in weight_fields]
    
    if len(data) != len(weights):
        raise ValueError('fields and weight fields need to have the same number of inputs')
    
    # Check if input demands multivariate eof
    if len(data) > 1:
        multivariate = True
    else:
        multivariate = False
    
    if multivariate and not normalize:
        raise RuntimeError('For multivariate eofs it is suggested to normalize all variables with their global variance. \
                           Otherwise some variables might dominate the covariance matrix.')
        
    
    # Normalize the dataset befor the computation
    if normalize:
        mean = [d.mean('time') for d in data]
        std = [d.std() for d in data]
        data = [(d - m) / s for d,m,s in zip(data, mean, std)]
    
    # Broadcast the weight dataarrays
    weights_raw = [xr.ones_like(d)* w  for d,w in zip(data, weights)]
    
    # Convert xarray to numpy arrays
    data = [d.data for d in data]
    weights = [w.data for w in weights_raw]
    
    if multivariate:
        solver = MultivariateEof(data, weights=weights)
    else:
        solver = Eof(data[0], weights=weights[0])
    
    # We scale the eof by multiplying with eigenvalues and the pcs by divinding 
    # with eigenvalues to preserve total variance (is this right? Ask Sloan)
    eof = solver.eofs(neofs=n, eofscaling=2)
    pc = solver.pcs(npcs=n, pcscaling=1)
    variance_fraction = solver.varianceFraction(neigs=n)
    
    # Rewrap into xarrays
    if multivariate:
        eof = xr.Dataset({k:rewrap_eof(v, ds[k]) for k,v in zip(fields, eof)})
    else:
        eof = xr.Dataset({fields[0]:rewrap_eof(eof, ds[fields[0]])})
    
    pc = rewrap_pc(pc, ds.time).to_dataset(name='pc')
    variance_fraction = rewrap_eig(variance_fraction).to_dataset(name='variance_fraction')
    
    # Unweight the eofs
    for ff,w in  zip(fields, weights_raw):
        # Since the eofs are constructed on the weighted fields, I probably need to divide by that again to get concentration changes
        eof[ff] = eof[ff] / w.mean('time')
    
    # "un"normalize the eofs with the global std
    if normalize:
        for ff,st,w in zip(fields, std, weights_raw):
            eof[ff] = eof[ff] * st
    
    # Merge all output to dataset
    ds_out = xr.merge([eof, pc, variance_fraction])
    
    # Rename all eof fields for clarity
    ds_eof = ds_out.rename({k:'eof_'+k for k in fields})
    
    return ds_eof

def regress_field_absolute(da, ts, regdim='time'):
    da_mean = da.mean(regdim)
    da_an = da - da_mean
    da_reg = xr_linregress(ts, da_an)
    return da_mean + da_reg.slope




################### Frozen versions with evolving dependencies elsewhere (already refactored) #############
def interp_all(grid, ds, target="center"):
    """Interpolates all variables and coordinates in `ds` onto common dimensions,
    specified by target."""
    ds = ds.copy()
    ds_new = xr.Dataset()
    warnings.warn("Frozen Version: Up to date version is maintained in xarrayutils", DeprecationWarning)

    def _core_interp(da, grid):
        for ax in ["X", "Y", "Z"]:
            # Check if any dimension matches this axis
            match = [a for a in da.dims if a in grid.axes[ax].coords.values()]
            if len(match) > 0:
                pos = [
                    p for p, a in grid.axes[ax].coords.items() if a in match
                ]
                if target not in pos:
                    da = grid.interp(da, ax, to=target)
        return da

    for vv in ds.data_vars:
        ds_new[vv] = _core_interp(ds[vv], grid)
    return ds_new
