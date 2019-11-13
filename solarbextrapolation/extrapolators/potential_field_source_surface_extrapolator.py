import astropy
import astropy.coordinates as coord
import astropy.constants as const
import astropy.units as u

from sunpy.coordinates import frames
import sunpy.map.mapbase

import numpy as np
import scipy.linalg as la

HAS_NUMBA = False
try:
    import numba
    HAS_NUMBA = True
except Exception:
    pass

HAS_PFSSPY = False
try:
    import pfsspy as pfss
    HAS_PFSSPY = True
except Exception:
    print("Pfsspy is not installed!")

from ndcube import NDCube


class PFSSExtraploator(Extrapolators):
    """
	Draft of the Extrapolator class
	"""
    def __init__(self, map, rss, nr, nphi=None, ns=None, **kwargs):
        """
        Initialize the extrapolator
        
        Parameters
    	----------
    	map : NDCube
            Boundary magnetic field
    	
		rss : float
            Radius of the source surface

        n_r : int
            Grid of r
		
		n_s : int
            Grid of cos(theta)
        
        n_phi : int 
            Grid of phi
		
        """
        # TODO: Use NDCube or sunpy.map.mapbase.GenericMap as boundary condition?
        assert isinstance(map, NDCube), "Map is not NDCube!"
        # TODO: this means div(B) = 0, is it always true on the solar surface?
        self.br   = map.data - np.mean(map.data)
        self.rss  = rss
        self.nr   = nr
        self.input = pfss.Input(self.br, self.nr, self.rss)


    def plot_input(self, ax):
        return self.input.plot_input(ax)


    def calculate_pfss(self):
        return pfss.pfss(self.input)