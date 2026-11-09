"""neuro_rave: reduce domain (e.g. session) effects in time-series windows.

``RAVE`` learns a domain-invariant embedding from windows and domain IDs alone;
``RAVEPlus`` additionally uses whichever class labels are known. Both follow
Gonschorek et al., "Removing Inter-Experimental Variability from Functional
Data in Systems Neuroscience", NeurIPS 2021.
"""

from neuro_rave.models import RAVE, RAVEPlus

__all__ = ["RAVE", "RAVEPlus", "__version__"]

__version__ = "0.1.0"
