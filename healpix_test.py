import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

from pylab import cm
import matplotlib.pyplot as plt
import matplotlib as mpl


# define the function to be used
def cat_to_hpx(lon, lat, nside, radec=True):
    """
    Converts a catalogue to a HEALPix map of number density i.e. 
    Number of stars per square degrees of sky.

    Parameters
    ----------
    lon, lat : (ndarray, ndarray)
        Coordinates of the sources in degree. 
        If radec=True, assume input is in the icrs system,
        Otherwise assume input is galactic latitude and longitude.


    nside : int
        HEALPix nside of the target map, defines the number of pixels.

    radec : bool
        Switch between Ra/Dec and l/b (galactic) as input coordinate system.

    Return
    ------
    hpx_map : ndarray
        HEALPix map of the catalogue number counts in Galactic coordinates

    """

    npix = hp.nside2npix(nside)

    if radec:
        eq = SkyCoord(lon, lat, unit='deg')
        l, b = eq.galactic.l.value, eq.galactic.b.value
    else:
        l, b = lon, lat

    # convert to theta, phi -> galactic longitude and colatitude in sphererical system
    theta = np.radians(90. - b)
    phi = np.radians(l)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)

    indx, counts = np.unique(indices, return_counts=True)

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[indx] = counts

    return hpx_map

RA_gcns = (128.3,130.3)
DEC_gcns = (40.,40.5)

#input the nside, should be a number of 2**n
nside = 16

#call function to create Healpix map
hpx_map_gcns = cat_to_hpx(RA_gcns, DEC_gcns, nside=nside, radec=True)

#define a figure and size
f= plt.figure(figsize=(10,7))

#calculate the side of each pixel in square degrees
hpix_area = round(41253/len(np.arange(hp.nside2npix(nside),dtype=np.double)),1)

#find minimum and maximum to normalise the colorbar
mmin, mmax = min((hpx_map_gcns)),max((hpx_map_gcns)) -40 #contribution from Hyades was strong, thus subtracted by random number (-40)

#add the desired
cmap = cm.jet

#create a mollview projection from healpy and add custom colorbar and colormap
hp.mollview((hpx_map_gcns), hold=True, nest=False,cmap=cmap,min=mmin,max=mmax,notext=True,fig=1,
         cbar=False,title='$\mathrm{100 \ pc \ GCNS \ Gaia}$')

#add meridians and parallels
hp.graticule()

#create a colorbar axis
cax = f.add_axes([0.125, 0.1, 0.775, 0.03])

#create a colorbar and normalise it
cb = mpl.colorbar.ColorbarBase(cax,cmap=cmap,norm=mpl.colors.Normalize(vmin=mmin,vmax=mmax),
                               orientation='horizontal')

#add label to colorbar
cb.set_label(r'$\mathrm{ N, \ [}$'+str(hpix_area)+'$\mathrm{\ deg^2]^{-1}}$',fontsize=10)
cb.solids.set_edgecolor("face")
plt.show()