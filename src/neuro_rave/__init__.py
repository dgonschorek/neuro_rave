"""neuro_rave: reduce domain (e.g. session) effects in time-series data.

``RAVE`` learns a domain-invariant embedding from time series and domain IDs alone;
``RAVEPlus`` additionally uses whichever class labels are known. Both follow
Gonschorek et al., "Removing Inter-Experimental Variability from Functional
Data in Systems Neuroscience", NeurIPS 2021. ``VAEDANN`` is an experimental
variational baseline, and ``neuro_rave.evaluation`` scores any method's outputs
on held-out data.
"""

from neuro_rave.models import RAVE, VAEDANN, RAVEPlus

__all__ = ["RAVE", "RAVEPlus", "VAEDANN", "__version__"]

__version__ = "0.2.0"
