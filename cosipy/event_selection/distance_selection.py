from typing import Iterable

import numpy as np
import astropy.units as u
from astropy.units import Quantity

from cosipy.interfaces.data_interface import TimeTagEmCDSDistanceEventDataInSCFrameInterface
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.util.iterables import asarray


class DistanceSelector(EventSelectorInterface):

    event_data_type = TimeTagEmCDSDistanceEventDataInSCFrameInterface

    def __init__(self, distance_min: Quantity = 0 * u.cm, distance_max: Quantity = np.inf * u.cm):
        """
        Selects events whose distance between the first two hits falls
        within [distance_min, distance_max).

        Parameters
        ----------
        distance_min: Minimum distance [inclusive]. Default: 0 cm
        distance_max: Maximum distance [exclusive]. Default: infinity
        """

        self._distance_min_cm = distance_min.to_value(u.cm)
        self._distance_max_cm = distance_max.to_value(u.cm)

    def _select(self, events: TimeTagEmCDSDistanceEventDataInSCFrameInterface,
                early_stop: bool = True) -> Iterable[bool]:

        distance_cm = asarray(events.distance_cm, dtype=np.float64, force_dtype=False)

        return (distance_cm >= self._distance_min_cm) & (distance_cm < self._distance_max_cm)
