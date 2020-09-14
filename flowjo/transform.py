##############################################################################     ##    ######
#    A.J. Zwijnenburg                   2020-09-21           v1.4                 #  #      ##
#    Copyright (C) 2020 - AJ Zwijnenburg          GPLv3 license                  ######   ##
##############################################################################  ##    ## ######

## Copyright notice ##########################################################
# FlowJo Tools provides a python API into FlowJo's .wsp files.
# Copyright (C) 2020 - AJ Zwijnenburg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
##############################################################################

"""
Classes for the representation of data/scale transformations

:class: _Abstract
Abstract class providing transformations for major/minor ticks and labels (according to FlowJo)
.scale()    - scaling of a single value
.scaler()   - scaling of a list of values
.labels()   - returns a list of labels representative for the transform class
.major_ticks() - returns the scaled mayor ticks (corresponding to the labels)
.minor_ticks() - returns the scaled minor ticks

:class: Linear
The linear transform class
(see _Abstract for attributes/functions)

:class: Log
The Logarithmic (log10) transform class
(see _Abstract for attributes/functions)

:class: Biex
The biexponential transform class
(see _Abstract for attributes/functions)

"""
from __future__ import annotations

import numpy as np
import pandas as pd
import copy
import bisect

class _Abstract():
    """
    Abstract representation of a scale. This class returns the tick-marks / tick-labels
    on a x-range of 0-1023 for the scale on the original data range.
    In orther words. This class calculates the tick-marks as would be plotted in FlowJo.
    """
    _major_ticks: List[int] = []
    _labels: List[str] = []
    _minor_ticks: List[int] = []

    def __init__(self):
        self.start: int = None
        self.end: int = None

    def scale(self, data: float, start: int, end: int) -> float:
        """
        Scales a single value
            :param data: the data to scale
            :param start: the local space start
            :param end: the local space end
        """
        return self.scaler([data], start, end)[0]

    def scaler(self, data: List[float], start: int, end: int) -> List[float]:
        """
        The scaling function
            :param data: the data to scale
            :param start: the local space start
            :param end: the local space end
        """
        raise NotImplementedError("implement in child class")

    def labels(self, start: int=0, end: int=1023) -> List[str]:
        """
        Returns the label ticks location and tick label in plot x-coordinate
            :param start: plot minimum xlim
            :param end: plot maximum xlim
        """
        return self._labels
 
    def major_ticks(self, start: int=0, end: int=1023) -> List[float]:
        """
        Returns the major tick location in plot x-coordinate
            :param start: plot minimum xlim
            :param end: plot maximum xlim
        """
        ticks = copy.deepcopy(self._major_ticks)

        ticks = self.scaler(ticks, start, end)

        return ticks
    
    def minor_ticks(self, start: int=0, end: int=1023) -> List[float]:
        """
        Returns the minor tick location in plot x-coordinate
            :param start: plot minimum xlim
            :param end: plot maximum xlim
        """
        ticks = copy.deepcopy(self._minor_ticks)

        ticks = self.scaler(ticks, start, end)

        return ticks

class Linear(_Abstract):
    """
    Represents a linear scale between the start and end value
        :param start: start value
        :param end: end value (if None, end-value is determined based on data and has to be manually added!)
    """
    _major_ticks = [
        0,
        50_000,
        100_000,
        150_000,
        200_000,
        250_000
    ]
    _labels = [
        "0",
        "50K",
        "100K",
        "150K",
        "200K",
        "250K"
    ]
    _minor_ticks = [
        10_000, 20_000, 30_000, 40_000,
        60_000, 70_000, 80_000, 90_000,
        110_000, 120_000, 130_000, 140_000,
        160_000, 170_000, 180_000, 190_000,
        210_000, 220_000, 230_000, 240_000,
        260_000
    ]

    def __init__(self, start: float=0, end: float=262144, gain: float=1):
        super().__init__()
        self.start: float = start
        self.end: float = end
        self.gain: float = gain   #unused

    def scaler(self, data: List[float], start: int, end: int) -> List[float]:
        """
        The scaling function
            :param data: the data to scale
            :param start: the local space start
            :param end: the local space end
        """
        # Get scaling parameters
        scale_range = self.end - self.start
        step_size = (end - start) / scale_range

        # scale
        for i in range(0, len(data)):
            data[i] = ((data[i] - self.start) * step_size) + start

        return data

    def __repr__(self) -> str:
        return f"(LinearTransform:{self.start}-{self.end})"

class Log(_Abstract):
    """
    Represents a logarithmic scale between the start and end value
        :param start: start value
        :param end: end value
    """
    _major_ticks = [
        5, 10,
        50, 100,
        500, 1_000,
        5_000, 10_000,
        50_000, 100_000
    ]
    _labels = [
        "", "10¹",
        "", "10²",
        "", "10³",
        "", "10⁴",
        "", "10⁵"
    ]
    _minor_ticks = [
        2, 3, 4, 6, 7, 8, 9,
        20, 30, 40, 60, 70, 80, 90,
        200, 300, 400, 600, 700, 800, 900,
        2_000, 3_000, 4_000, 6_000, 7_000, 8_000, 9_000,
        20_000, 30_000, 40_000, 60_000, 70_000, 80_000, 90_000,
        200_000
    ]

    def __init__(self, start: float=3, end: float=262144):
        super().__init__()
        self.start: float = start
        self.end: float = end

        if self.start <= 0:
            raise ValueError("logarithmic scale have to start with a value >0")

    def scaler(self, data: List[float], start: int, end: int) -> List[float]:
        """
        The scaling function
            :param data: the data to scale
            :param start: the local space start
            :param end: the local space end
        """
        # Get scaling parameters
        scale_range = np.log10(self.end) - np.log10(self.start)
        step_size = (end - start) / scale_range

        # scale
        for i in range(0, len(data)):
            data[i] = ((np.log10(data[i]) - np.log10(self.start)) * step_size) + start

        return data

    def _repr__(self) -> str:
        return f"(LogTransform:{self.start}-{self.end})"

class Biex(_Abstract):
    """
    Represents a biexponential scale between the start and end value.
    Different from the others this scale generates a look-up table. This is faster for big matrix transforms. 
    But a bit waistfull for small number of iterations. Ow well.
        :param start: start value
        :param end: end value
    """
    _major_ticks = [
        -100_000, -50_000,
        -10_000, -5_000,
        -1_000, -500,
        -100, -50,
        -10, -5,
        0,
        5, 10,
        50, 100,
        500, 1_000,
        5_000, 10_000,
        50_000, 100_000
    ]
    _labels = [
        "-10⁵", "",
        "-10⁴", "",
        "-10³", "",
        "-10²", "",
        "-10¹", "",
        "0",
        "", "10¹",
        "", "10²",
        "", "10³",
        "", "10⁴",
        "", "10⁵"
    ]
    _minor_ticks = [
        -200_000,
        -90_000, -80_000, -70_000, -60_000, -40_000, -30_000, -20_000,
        -9_000, -8_000, -7_000, -6_000, -4_000, -3_000, -2_000,
        -900, -800, -700, -600, -400, -300, -200,
        -90, -80, -70, -60, -40, -30, -20,
        -9, -8, -7, -6, -4, -3, -2, -1,
        1, 2, 3, 4, 6, 7, 8, 9,
        20, 30, 40, 60, 70, 80, 90,
        200, 300, 400, 600, 700, 800, 900,
        2_000, 3_000, 4_000, 6_000, 7_000, 8_000, 9_000,
        20_000, 30_000, 40_000, 60_000, 70_000, 80_000, 90_000,
        200_000
    ]

    def __init__(self, end: float=262144, neg_decade: float=0, width: float=-100, pos_decade: float=4.418539922, length: int=256):
        super().__init__()
        self.length: int = length   #unused
        self.neg_decade: float = neg_decade
        self.width: float = width
        self.pos_decade: float = pos_decade
        self.end: float = end

        self.values: List[int] = None
        self.lookup: List[float] = None
        self._lookup_range: Tuple[int, int] = None

    def scaler(self, data: List[float], start: int, end: int) -> List[float]:
        """
        The scaling function. Values outside the lookup range will be placed outside the start-end range
            :param data: the data to scale
            :param start: the local space start
            :param end: the local space end
        """
        if self.lookup is None:
            self._build_lookup(start, end)
        elif self._lookup_range[0] != start or self._lookup_range[1] != end:
            self._build_lookup(start, end)

        for i in range(0, len(data)):
            index = bisect.bisect_right(self.lookup, data[i])

            if not index:
                value = start-1
            elif index == len(self.lookup):
                value = end+1
            else:
                value = self.values[index]

            data[i] = value

        return data

    def labels(self, start: int=0, end: int=1023) -> Tuple[List[float], List[str]]:
        """
        Returns the label ticks location and tick label in plot x-coordinate
            :param start: plot minimum xlim
            :param end: plot maximum xlim
        """
        labels = copy.deepcopy(self._labels)

        labels_x = copy.deepcopy(self._major_ticks)
        labels_x = self.scaler(labels_x, start, end)

        # remove too-close-to-zero labels
        zero = labels_x[10]
        for i in range(11, len(labels)):
            if (labels_x[i] - zero) < ((end - start) * 0.035):
                labels[i] = ""
                labels[len(labels) -i -1] = ""

        # cleanup unfound labels
        for i in range(len(labels_x), 0, -1):
            if labels_x[i-1] is None:
                labels.pop(i-1)

        return labels

    def major_ticks(self, start: int=0, end: int=1023) -> List[float]:
        """
        Returns the major tick location in plot x-coordinate
            :param start: plot minimum xlim
            :param end: plot maximum xlim
        """
        ticks = super().major_ticks(start, end)

        # cleanup unfound labels
        for i in range(len(ticks), 0, -1):
            if ticks[i-1] is None:
                ticks.pop(i-1)

        return ticks
    
    def minor_ticks(self, start: int=0, end: int=1023) -> List[float]:
        """
        Returns the minor tick location in plot x-coordinate
            :param start: plot minimum xlim
            :param end: plot maximum xlim
        """
        ticks = super().minor_ticks(start, end)

        # cleanup unfound labels
        for i in range(len(ticks), 0, -1):
            if ticks[i-1] is None:
                ticks.pop(i-1)

        return ticks

    def _build_lookup(self, start: int, end: int):
        """
        Builds the lookup table for the biexponential transform
            :param start: the local space start
            :param end: the local space end
        """
        # Source paper: David R. Parks, Mario Roederer, Wayne A. Moore.  A new “Logicle” display method avoids deceptive effects of logarithmic scaling for low signals and compensated data.  Cytometry Part A, Volume 69A, Issue 6, pages 541-551, June 2006.
        # Use the FlowJo implementation, as transformed by CytoLib: https://github.com/RGLab/cytolib/blob/master/src/transformation.cpp directly from the FlowJo java implementation

        decades = self.pos_decade
        width = np.log10(-self.width)   # log10 transform to make the data 'equivalent' to decades and extra
        extra = self.neg_decade
        channel_range = end - start

        # Make sure width is within expected values
        if width < 0.5 or width > 3.0:
            raise ValueError(f"width has to be >= 3.16 (10**0.5) and <= -1000 (10**3), not '{width}'")

        # Make sure extra is within expected values
        if extra < 0:
            raise ValueError(f"neg_decade has to be >=0, not '{extra}'")

        # Remove enough width from positive decades to be able to plot the negative decades
        decades -= (width * 0.5)

        # Set the negative decades
        extra += (width * 0.5)

        # Find the location for 0
        zero_channel = int(extra * channel_range / (extra + decades))
        zero_channel = min(zero_channel, channel_range // 2)              # 0 is at most at the middle of the plot

        if zero_channel > 0:
            # Get total positive decades
            decades = extra * channel_range / zero_channel
        width /= 2 * decades

        maximum = self.end
        # Calculate positive range (m) as specified in Logicle paper (in natural log units)
        positive_range = np.log(10.0) * decades

        minimum = maximum / np.exp(positive_range)
        negative_range = self._log_root(positive_range, width)

        # Amount of points to calculate
        n_points = channel_range + 1

        # Instantiate table
        positive = [0.0]*n_points
        negative = [0.0]*n_points
        vals = [0.0]*n_points

        # Fill table with natural logged 
        step = (channel_range) / (n_points-1)
        fraction = 1 / n_points
        for i in range(0, n_points, 1):
            vals[i] = i * step
            positive[i] = np.exp(i * fraction * positive_range)
            negative[i] = np.exp(i * fraction * (-negative_range))
        
        s = np.exp((positive_range + negative_range) * (width + extra / decades))
        for i in range(0, n_points, 1):
            negative[i] *= s

        s = positive[zero_channel] - negative[zero_channel]
        for i in range(zero_channel, n_points, 1):
            positive[i] = minimum * (positive[i] - negative[i] - s)

        # Force symmetry by negating the positive channel into the negative decades
        for i in range(0, zero_channel, 1):
            m: int = 2 * zero_channel - i
            positive[i] = -positive[m]

        self.values = vals
        self.lookup = positive
        self._lookup_range = (start, end)

    @staticmethod
    def _log_root(positive_range: float, width: float):
        """
        Approximates (?) logarithmic root (i think). See _scaler for source
        """
        x_low = 0
        x_high = positive_range
        
        x_mean = (x_low + x_high) * 0.5
        x_delta = abs(int(x_low - x_high))
        x_delta_last = x_delta

        f_b = -2.0 * np.log(positive_range) + width * positive_range
        f = 2.0 * np.log(x_mean) + width * positive_range + f_b
        d_f = 2.0 / (x_mean + width)

        if width == 0:
            return positive_range
        
        for i in range(0, 100, 1):
            if ((x_mean - x_high) * d_f - f) * ((x_mean - x_low) * d_f - f) >= 0 or abs(int(2 * f)) > abs(int(x_delta_last * d_f)):
                x_delta = (x_high - x_low) / 2
                x_mean = x_low + x_delta
            
                if x_mean == x_low:
                    return x_mean
            
            else:
                x_delta = f / d_f
                t = x_mean
                x_mean -= x_delta
            
                if x_mean == t:
                    return x_mean
            
            if abs(int(x_delta)) < 1.0e-12:
                return x_mean

            x_delta_last = x_delta
            f = 2 * np.log(x_mean) + width * x_mean+ f_b
            d_f = 2 / x_mean + width
            
            if f < 0:
                x_low = x_mean
            else:
                x_high = x_mean

        return x_mean

    def __repr__(self) -> str:
        return f"(BiexTransform:{self.end};{self.width:.1f};{self.pos_decade:.1f})"
