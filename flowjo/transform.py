##############################################################################     ##    ######
#    A.J. Zwijnenburg                   2021-05-15           v1.11                #  #      ##
#    Copyright (C) 2021 - AJ Zwijnenburg          GPLv3 license                  ######   ##
##############################################################################  ##    ## ######

## Copyright notice ##########################################################
# FlowJo Tools provides a python API into FlowJo's .wsp files.
# Copyright (C) 2021 - AJ Zwijnenburg
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
Classes for the representation of data/scale transformations. 
For conversions of data from the untransformed (global) scale to the transformed (local) scale.

:class: _AbstractGenerator
Abstract class provides a generator of an arbitrary range of ticks/label of a transformation
This abstract class provides an interface for the transform classes.
.labels()   - returns a list of labels representative for the transform class
.major_ticks() - returns the scaled mayor ticks (corresponding to the labels)
.minor_ticks() - returns the scaled minor ticks

:class: LinearGenerator
Generates the ticks/labels for a linear transform/scale

:class: Log10Generator
Generates the ticks/labels for a log10 transform/scale

:class: BiexGenerator
Generates the ticks/labels for a biexponential transform/scale

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

:class: Log10
The Logarithmic (log10) transform class
(see _Abstract for attributes/functions)

:class: Biex
The biexponential transform class
(see _Abstract for attributes/functions)

"""
from __future__ import annotations
from typing import Callable, List, Dict, Tuple

import numpy as np
import bisect
import copy
import sys

class _AbstractGenerator():
    """
    Abstract Generator. Abstract class for generators of labels, minor/major tick locations 
    for 'infinite' range transforms. Should be subclassed for a scale-specific implementation.
    Is defined to only generate labels/ticks in the specified (global) range.
    """
    _superscript: Dict[str, str]={
        "-":"⁻",
        "0":"⁰",
        "1":"¹",
        "2":"²",
        "3":"³",
        "4":"⁴",
        "5":"⁵",
        "6":"⁶",
        "7":"⁷",
        "8":"⁸",
        "9":"⁹"
    }

    def __init__(self):
        pass

    def labels(self, start: int, end: int) -> List[str]:
        """
        Returns a list of the labels between start and end
            :param start: the global space start
            :param end: the global space end
        """
        raise NotImplementedError("implement in child class")

    def major_ticks(self, start: int, end: int) -> List[int]:
        """
        Returns a list of the major ticks between start and end
            :param start: the global space start
            :param end: the global space end
        """
        raise NotImplementedError("implement in child class")
    
    def minor_ticks(self, start: int, end: int) -> List[int]:
        """
        Returns a list of the minor ticks between start and end
            :param start: the global space start
            :param end: the global space end
        """
        raise NotImplementedError("implement in child class")

    def __eq__(self, other) -> bool:
        """
        Test for equality. Should be able to handle all transform classes
        """
        raise NotImplementedError("implement in child class")

class LinearGenerator(_AbstractGenerator):
    """
    Generates labels, minor/major tick locations for a linear scale
    """
    def __init__(self):
        super().__init__()
        self.begin: int = 0
        self.stepsize_minor: int = 10_000
        self.stepsize_major: int = 50_000

    def labels(self, start: int, end: int) -> List[str]:
        """
        Returns a list of the labels between start and end
            :param start: the global space start
            :param end: the global space end
        """
        ticks = self.major_ticks(start, end)

        values: List[str] = []
        # each tick has to be changed into a string representation
        for tick in ticks:
            #handle decimals
            if isinstance(tick, float):
                value = "{:.2e}".format(tick)
            else:
                if tick//1000 != 0 and tick%1000 == 0:
                    value = "{}K".format(tick//1000)
                else:
                    value = "{}".format(tick)
            values.append(value)

        return values
    
    def major_ticks(self, start: int, end: int) -> List[int]:
        """
        Returns a list of the major tick locations between start and end
            :param start: the global space start
            :param end: the global space end
        """
        begin_i = (start - self.begin) / self.stepsize_major
        end_i = (end - self.begin) / self.stepsize_major

        values: List[int] = []

        if begin_i == end_i:
            if ((start - self.begin) % self.stepsize_major) == 0:
                values.append(start)

        elif begin_i < end_i:
            begin_i = int(begin_i) - 1
            end_i = int(end_i) + 1

            value = (begin_i * self.stepsize_major) + self.begin
            if value >= start and value <= end:
                values.append(value)

            for i in range(1, end_i - begin_i, 1):
                value = ((begin_i + i) * self.stepsize_major) + self.begin
                values.append(value)

            value = (end_i * self.stepsize_major) + self.begin
            if value >= start and value <= end:
                values.append(value)

        elif begin_i > end_i:
            begin_i = int(begin_i) + 1
            end_i = int(end_i) - 1

            value = (begin_i * self.stepsize_major) + self.begin
            if value <= start and value >= end:
                values.append(value)

            for i in range(begin_i - end_i -1, 0, -1):
                value = ((end_i + i) * self.stepsize_major) + self.begin
                values.append(value)

            value = (end_i * self.stepsize_major) + self.begin
            if value <= start and value >= end:
                values.append(value)

        return values

    def minor_ticks(self, start: int, end: int) -> List[int]:
        """
        Returns a list of the minor ticks between start and end
            :param start: the global space start
            :param end: the global space end
        """
        begin_i = (start - self.begin) / self.stepsize_minor
        end_i = (end - self.begin) / self.stepsize_minor

        values: List[int] = []

        if begin_i == end_i:
            if ((start - self.begin) % self.stepsize_minor) == 0:
                if ((start - self.begin) % self.stepsize_major) != 0:
                    values.append(start)

        elif begin_i < end_i:
            begin_i = int(begin_i) - 1
            end_i = int(end_i) + 1

            value = (begin_i * self.stepsize_minor) + self.begin
            if value >= start and value <= end:
                if ((value - self.begin) % self.stepsize_major) != 0:
                    values.append(value)

            for i in range(1, end_i - begin_i, 1):
                value = ((begin_i + i) * self.stepsize_minor) + self.begin
                if ((value - self.begin) % self.stepsize_major) != 0:
                    values.append(value)

            value = (end_i * self.stepsize_minor) + self.begin
            if value >= start and value <= end:
                if ((value - self.begin) % self.stepsize_major) != 0:
                    values.append(value)

        elif begin_i > end_i:
            begin_i = int(begin_i) + 1
            end_i = int(end_i) - 1

            value = (begin_i * self.stepsize_minor) + self.begin
            if value <= start and value >= end:
                if ((value - self.begin) % self.stepsize_major) != 0:
                    values.append(value)

            for i in range(begin_i - end_i -1, 0, -1):
                value = ((end_i + i) * self.stepsize_minor) + self.begin
                if ((value - self.begin) % self.stepsize_major) != 0:
                    values.append(value)

            value = (end_i * self.stepsize_minor) + self.begin
            if value <= start and value >= end:
                if ((value - self.begin) % self.stepsize_major) != 0:
                    values.append(value)

        return values

    def __eq__(self, other) -> bool:
        """
        Test for equality.
        """
        if not isinstance(other, LinearGenerator):
            return False
        
        if self.begin != other.begin:
            return False
        if self.stepsize_major != other.stepsize_major:
            return False
        if self.stepsize_minor != other.stepsize_minor:
            return False

        return True

class Log10Generator(_AbstractGenerator):
    """
    Generates labels, minor/major tick locations for a log10 scale
    """
    def __init__(self):
        super().__init__()

    def labels(self, start: int, end: int) -> List[str]:
        """
        Returns a list of the labels between start and end
            :param start: the global space start
            :param end: the global space end
        """
        ticks = self.major_ticks(start, end)

        values: List[str] = []
        # each tick has to be changed into a string representation
        
        for tick in ticks:
            decade = np.log10(tick)
            if decade % 1 != 0:
                value = ""

            else:
                value = str(int(decade))
                if value == "0":
                    value = "1"
                else:
                    value = "".join([self._superscript[x] for x in value])
                    value = "10{}".format(value)
            values.append(value)

        return values

    def major_ticks(self, start: int, end: int) -> List[int]:
        """
        Returns a list of the major tick locations between start and end
            :param start: the global space start
            :param end: the global space end
        """
        if start <= 0 or end <= 0:
            raise ValueError("cannot generate a log ticks <= 0")

        decade_start = np.log10(start)
        decade_end = np.log10(end)

        values: List[int] = []

        if start == end:
            decade_start = int(decade_start)
            decade = 10**decade_start

            if 1 * decade == start:
                values.append(start)
            elif 5 * decade == start:
                values.append(start)

        elif decade_start < decade_end:
            decade_start = int(decade_start) - 1
            decade_end = int(decade_end) + 1

            # iterate through decades
            for i in range(decade_start, decade_end, 1):
                decade = 10**i
                
                value = 1 * decade
                if value >= start and value <= end:
                    values.append(value)
            
                value = 5 * decade
                if value >= start and value <= end:
                    values.append(value)
            
        elif decade_start > decade_end:
            decade_start = int(decade_start) + 1
            decade_end = int(decade_end) - 1

            # iterate through decades
            for i in range(decade_start, decade_end, -1):
                decade = 10**i

                value = 5 * decade
                if value <= start and value >= end:
                    values.append(value)
    
                value = 1 * decade
                if value <= start and value >= end:
                    values.append(value)

        return values

    def minor_ticks(self, start: int, end: int) -> List[int]:
        """
        Returns a list of the minor tick locations between start and end
            :param start: the global space start
            :param end: the global space end
        """
        if start <=0 or end <= 0:
            raise ValueError("cannot generate a log ticks <= 0")

        decade_start = np.log10(start)
        decade_end = np.log10(end)

        values: List[int] = []

        if start == end:
            decade_start = int(decade_start)
            decade = 10**decade_start

            if 2 * decade == start:
                values.append(start)
            elif 3 * decade == start:
                values.append(start)
            elif 4 * decade == start:
                values.append(start)
            elif 6 * decade == start:
                values.append(start)
            elif 7 * decade == start:
                values.append(start)
            elif 8 * decade == start:
                values.append(start)
            elif 9 * decade == start:
                values.append(start)

        elif decade_start < decade_end:
            decade_start = int(decade_start) - 1
            decade_end = int(decade_end) + 1

            # iterate through decades
            for i in range(decade_start, decade_end, 1):
                decade = 10**i
                value = 2 * decade
                if value >= start and value <= end:
                    values.append(value)
            
                value = 3 * decade
                if value >= start and value <= end:
                    values.append(value)

                value = 4 * decade
                if value >= start and value <= end:
                    values.append(value)

                value = 6 * decade
                if value >= start and value <= end:
                    values.append(value)

                value = 7 * decade
                if value >= start and value <= end:
                    values.append(value)

                value = 8 * decade
                if value >= start and value <= end:
                    values.append(value)

                value = 9 * decade
                if value >= start and value <= end:
                    values.append(value)
            
        elif decade_start > decade_end:
            decade_start = int(decade_start) + 1
            decade_end = int(decade_end) - 1

            # iterate through decades
            for i in range(decade_start, decade_end, -1):
                decade = 10**i
                decade = 1 if decade == 0 else decade

                value = 9 * decade
                if value <= start and value >= end:
                    values.append(value)
    
                value = 8 * decade
                if value <= start and value >= end:
                    values.append(value)

                value = 7 * decade
                if value <= start and value >= end:
                    values.append(value)

                value = 6 * decade
                if value <= start and value >= end:
                    values.append(value)

                value = 4 * decade
                if value <= start and value >= end:
                    values.append(value)

                value = 3 * decade
                if value <= start and value >= end:
                    values.append(value)

                value = 2 * decade
                if value <= start and value >= end:
                    values.append(value)

        return values

    def __eq__(self, other) -> bool:
        """
        Test for equality.
        """
        if not isinstance(other, Log10Generator):
            return False
        
        return True

class BiexGenerator(_AbstractGenerator):
    """
    Generates labels, minor/major tick locations for a biexponential-like scales
    """
    def __init__(self):
        super().__init__()

    def labels(self, start: int, end: int) -> List[str]:
        """
        Returns a list of the labels between start and end
            :param start: the global space start
            :param end: the global space end
        """
        ticks = self.major_ticks(start, end)

        values: List[str] = []
        # each tick has to be changed into a string representation
        
        for tick in ticks:

            if tick == 0:
                values.append("0")
                continue

            elif tick < 0:
                is_negative = True
                decade = np.log10(-tick)
            else:
                is_negative = False
                decade = np.log10(tick)

            if decade % 1 != 0:
                value = ""

            else:
                value = str(int(decade))
                if value == "0":
                    value = "1"
                else:
                    value = "".join([self._superscript[x] for x in value])
                    if is_negative:
                        value = "-10{}".format(value)
                    else:
                        value = "10{}".format(value)
            values.append(value)

        return values

    def major_ticks(self, start: int, end: int) -> List[int]:
        """
        Returns a list of the major tick locations between start and end
            :param start: the global space start
            :param end: the global space end
        """
        # The ticks overlap with the log10 scale
        values: List[int] = []

        if start == end:
            if start == 0:
                values.append(0)
            elif start > 0:
                temp = self._major_ticks(start, end)
                values.extend(temp)
            elif start < 0:
                temp = self._major_ticks(-start, -end)
                temp = [-x for x in temp]
                values.extend(temp)
        
        else:
            if start < 0 and end < 0:
                temp = self._major_ticks(-start, -end)
                temp = [-x for x in temp]
                values.extend(temp)
            elif start > 0 and end > 0:
                temp = self._major_ticks(start, end)
                values.extend(temp)

            elif start < end:
                if start == 0:
                    values.append(0)
                    temp = self._major_ticks(np.nextafter(0.0, 1), end)
                    values.extend(temp)
                elif end == 0:
                    temp = self._major_ticks(-start, np.nextafter(0.0, 1))
                    temp = [-x for x in temp]
                    values.extend(temp)
                    values.append(0)
                else:
                    temp = self._major_ticks(-start, np.nextafter(0.0, 1))
                    temp = [-x for x in temp]
                    values.extend(temp)
                    values.append(0)
                    temp = self._major_ticks(np.nextafter(0.0, 1), end)
                    values.extend(temp)

            elif start > end:
                if start == 0:
                    values.append(0)
                    temp = self._major_ticks(np.nextafter(0.0, 1), -end)
                    temp = [-x for x in temp]
                    values.extend(temp)
                elif end == 0:
                    temp = self._major_ticks(start, np.nextafter(0.0, 1))
                    values.append(0)
                    values.extend(temp)
                else:
                    temp = self._major_ticks(start, np.nextafter(0.0, 1))
                    values.extend(temp)
                    values.append(0)
                    temp = self._major_ticks(np.nextafter(0.0, 1), -end)
                    temp = [-x for x in temp]
                    values.extend(temp)

        return values

    def _major_ticks(self, start: int, end: int) -> List[int]:
        """
        Helper function returns a list of log10 major tick locations between start and end
        Returns from 1 (but not included) onwards
            :param start: the global space start
            :param end: the global space end
        """
        decade_start = np.log10(start)
        decade_end = np.log10(end)

        values: List[int] = []

        if start == end:
            decade_start = int(decade_start)
            decade = 10**decade_start

            if 1 * decade == start:
                if start != 1:
                    values.append(start)
            elif 5 * decade == start:
                values.append(start)

        elif decade_start < decade_end:
            decade_start = int(decade_start) - 1
            decade_start = 0 if decade_start < 0 else decade_start
            decade_end = int(decade_end) + 1

            # iterate through decades
            for i in range(decade_start, decade_end, 1):
                decade = 10**i
                
                value = 1 * decade
                if value == 1:
                    continue

                if value >= start and value <= end:
                    values.append(value)
            
                value = 5 * decade
                if value >= start and value <= end:
                    values.append(value)
            
        elif decade_start > decade_end:
            decade_start = int(decade_start) + 1
            decade_end = int(decade_end) - 1
            decade_end = 0 if decade_end < 0 else decade_end

            # iterate through decades
            for i in range(decade_start, decade_end, -1):
                decade = 10**i

                value = 5 * decade
                if value <= start and value >= end:
                    values.append(value)
    
                value = 1 * decade
                if value == 1:
                    continue
                if value <= start and value >= end:
                    values.append(value)

        return values

    def minor_ticks(self, start: int, end: int) -> List[int]:
        """
        Returns a list of the minor tick locations between start and end
            :param start: the global space start
            :param end: the global space end
        """
        values: List[int] = []

        if start == end:
            if start == 0:
                pass
            elif start == 1:
                values.append(1)
            elif start == -1:
                values.append(-1)
            elif start > 0:
                temp = self._minor_ticks(start, end)
                values.extend(temp)
            elif start < 0:
                temp = self._minor_ticks(-start, -end)
                temp = [-x for x in temp]
                values.extend(temp)
        
        else:
            if start < 0 and end < 0:
                temp = self._minor_ticks(-start, -end)
                temp = [-x for x in temp]
                values.extend(temp)
            elif start > 0 and end > 0:
                temp = self._minor_ticks(start, end)
                values.extend(temp)

            elif start < end:
                if start == 0:
                    temp = self._minor_ticks(np.nextafter(0.0, 1), end)
                    values.extend(temp)
                elif end == 0:
                    temp = self._minor_ticks(-start, np.nextafter(0.0, 1))
                    temp = [-x for x in temp]
                    values.extend(temp)
                else:
                    temp = self._minor_ticks(-start, np.nextafter(0.0, 1))
                    temp = [-x for x in temp]
                    values.extend(temp)
                    temp = self._minor_ticks(np.nextafter(0.0, 1), end)
                    values.extend(temp)

            elif start > end:
                if start == 0:
                    temp = self._minor_ticks(np.nextafter(0.0, 1), -end)
                    temp = [-x for x in temp]
                    values.extend(temp)
                elif end == 0:
                    temp = self._minor_ticks(start, np.nextafter(0.0, 1))
                    values.extend(temp)
                else:
                    temp = self._minor_ticks(start, np.nextafter(0.0, 1))
                    values.extend(temp)
                    temp = self._minor_ticks(np.nextafter(0.0, 1), -end)
                    temp = [-x for x in temp]
                    values.extend(temp)

        return values

    def _minor_ticks(self, start: int, end: int) -> List[int]:
        """
        Helper function: Returns a list of the log10 minor tick locations between start and end
        Spikes in 1 as a minor tick location and doesnt returns for ticks <1.
            :param start: the global space start
            :param end: the global space end
        """
        decade_start = np.log10(start)
        decade_end = np.log10(end)

        values: List[int] = []

        if start == end:
            decade_start = int(decade_start)
            decade = 10**decade_start

            if start == 1:
                values.append(1)
            elif 2 * decade == start:
                values.append(start)
            elif 3 * decade == start:
                values.append(start)
            elif 4 * decade == start:
                values.append(start)
            elif 6 * decade == start:
                values.append(start)
            elif 7 * decade == start:
                values.append(start)
            elif 8 * decade == start:
                values.append(start)
            elif 9 * decade == start:
                values.append(start)

        elif decade_start < decade_end:
            decade_start = int(decade_start) - 1
            decade_start = 0 if decade_start < 0 else decade_start
            decade_end = int(decade_end) + 1

            # iterate through decades
            for i in range(decade_start, decade_end, 1):
                decade = 10**i

                value = 1 * decade
                if value == 1 and value >= start and value <= end:
                    values.append(value)

                value = 2 * decade
                if value >= start and value <= end:
                    values.append(value)
            
                value = 3 * decade
                if value >= start and value <= end:
                    values.append(value)

                value = 4 * decade
                if value >= start and value <= end:
                    values.append(value)

                value = 6 * decade
                if value >= start and value <= end:
                    values.append(value)

                value = 7 * decade
                if value >= start and value <= end:
                    values.append(value)

                value = 8 * decade
                if value >= start and value <= end:
                    values.append(value)

                value = 9 * decade
                if value >= start and value <= end:
                    values.append(value)
            
        elif decade_start > decade_end:
            decade_start = int(decade_start) + 1
            decade_end = int(decade_end) - 1
            decade_end = -1 if decade_end < -1 else decade_end

            # iterate through decades
            for i in range(decade_start, decade_end, -1):
                decade = 10**i
                decade = 1 if decade == 0 else decade

                value = 9 * decade
                if value <= start and value >= end:
                    values.append(value)
    
                value = 8 * decade
                if value <= start and value >= end:
                    values.append(value)

                value = 7 * decade
                if value <= start and value >= end:
                    values.append(value)

                value = 6 * decade
                if value <= start and value >= end:
                    values.append(value)

                value = 4 * decade
                if value <= start and value >= end:
                    values.append(value)

                value = 3 * decade
                if value <= start and value >= end:
                    values.append(value)

                value = 2 * decade
                if value <= start and value >= end:
                    values.append(value)

                value = 1 * decade
                if value == 1 and value <= start and value >= end:
                    values.append(value)

        return values

    def __eq__(self, other) -> bool:
        """
        Test for equality.
        """
        if not isinstance(other, BiexGenerator):
            return False
        
        return True

class _Abstract():
    """
    Abstract representation of a scale. This class returns the tick-marks / tick-labels
    of the global range projected onto the local (linear) range
    In other words: This class calculates the tick-marks as would be plotted in FlowJo.
    """
    def __init__(self):
        # start and end are in global space
        self.l_start: float = None
        self.l_end: float = None
        self.g_start: float = None
        self.g_end: float = None
        self.generator: _AbstractGenerator = None

    def scale(self, data: float) -> float:
        """
        Scales a single value
            :param data: the data to scale
        """
        return self.scaler([data])[0]

    def scaler(self, data: List[float]) -> List[float]:
        """
        The scaling function
            :param data: the data to scale
        """
        raise NotImplementedError("implement in child class")

    def labels(self) -> List[str]:
        """
        Returns the labels for the major ticks (for coordinates use major_ticks())
        """
        return self.generator.labels(self.g_start, self.g_end)
 
    def major_ticks(self) -> List[float]:
        """
        Returns the major tick locations
        """
        ticks = self.generator.major_ticks(self.g_start, self.g_end)

        ticks = self.scaler(ticks)

        return ticks
    
    def minor_ticks(self) -> List[float]:
        """
        Returns the minor tick locations
        """
        ticks = self.generator.minor_ticks(self.g_start, self.g_end)

        ticks = self.scaler(ticks)

        return ticks

    def __eq__(self, other) -> bool:
        """
        Test for equality. Should be able to handle all transform classes
        """
        raise NotImplementedError("implement in child class")

class Linear(_Abstract):
    """
    Represents a linear scale between the start and end value
        :param l_start: start local value
        :param l_end: end local value
        :param g_start: start global value
        :param g_end: end global value
        :param gain: (unused) gain
    """
    def __init__(self, l_start: float=0, l_end: float=1023, g_start: int=0, g_end: int=262144, gain: float=1):
        super().__init__()
        self.generator = LinearGenerator()

        self.l_start: float = l_start
        self.l_end: float = l_end
        self.g_start: float = g_start
        self.g_end: float = g_end
        self.gain: float = gain   #unused

    def scaler(self, data: List[float]) -> List[float]:
        """
        The scaling function
            :param data: the data to scale
        """
        data = copy.deepcopy(data)

        # Get scaling parameters
        scale_range = self.g_end - self.g_start
        step_size = (self.l_end - self.l_start) / scale_range

        # scale
        for i in range(0, len(data)):
            if data[i] is None:
                continue
            data[i] = ((data[i] - self.g_start) * step_size) + self.l_start

        return data

    def __eq__(self, other) -> bool:
        if not isinstance(other, _Abstract):
            raise ValueError("can only test equality against other transform classes")

        if not isinstance(other, Linear):
            return False
        
        if self.g_start != other.g_start:
            return False
        if self.g_end != other.g_end:
            return False
        if self.l_start != other.l_start:
            return False
        if self.l_end != other.l_end:
            return False
        
        if self.gain != other.gain:
            raise NotImplementedError("no clue how to handle gain in LinearTransform, please notify author")

        return True

    def __repr__(self) -> str:
        return f"(LinearTransform:[{self.g_start}-{self.g_end}]->[{self.l_start}-{self.l_end}])"

class Log10(_Abstract):
    """
    Represents a logarithmic scale between the start and end value
        :param l_start: start local value
        :param l_end: end local value
        :param g_start: global start value
        :param g_end: global end value
    """
    def __init__(self, l_start: float=0, l_end: float=1023, g_start: float=3, g_end: float=262144):
        super().__init__()
        self.generator = Log10Generator()

        self.l_start: float = l_start
        self.l_end: float = l_end
        self.g_start: float = g_start
        self.g_end: float = g_end

        if self.g_start <= 0:
            raise ValueError("logarithmic scale have to start with a value >0")

    def scaler(self, data: List[float]) -> List[float]:
        """
        The scaling function
            :param data: the data to scale
        """
        data = copy.deepcopy(data)

        # Get scaling parameters
        scale_range = np.log10(self.g_end) - np.log10(self.g_start)
        step_size = (self.l_end - self.l_start) / scale_range

        # scale
        for i in range(0, len(data)):
            if data[i] is None:
                continue

            # Not all values can be scaled by log; so those get turned to the minimum value
            if data[i] < 0:
                data[i] = self.l_start
            else:
                data[i] = ((np.log10(data[i]) - np.log10(self.g_start)) * step_size) + self.l_start

        return data

    def __eq__(self, other) -> bool:
        if not isinstance(other, _Abstract):
            raise ValueError("can only test equality against other transform classes")

        if not isinstance(other, Log10):
            return False
        
        if self.g_start != other.g_start:
            return False
        if self.g_end != other.g_end:
            return False
        if self.l_start != other.l_start:
            return False
        if self.l_end != other.l_end:
            return False

        return True

    def __repr__(self) -> str:
        return f"(Log10Transform:[{self.g_start}-{self.g_end}]->[{self.l_start}-{self.l_end}])"

class Biex(_Abstract): 
    """
    Represents a biexponential scale between the start and end value.
    This class generates a look-up table. This is faster for big matrix transforms. 
    But a bit waistfull for small number of iterations. Ow well.
        :param l_start: start local value
        :param l_end: end local value
        :param g_end: global end value
        :param neg_decade: the extra negative decades
        :param width: the biexponential width parameter
        :param pos_decade: the positive decades
        :param length: (unused) the lookup table resolution
    """
    def __init__(self, l_start: float=0, l_end: float=1023, g_end: float=262144, neg_decade: float=0, width: float=-100, pos_decade: float=4.418539922, length: int=256):
        super().__init__()
        self.generator = BiexGenerator()

        self.l_start: float = l_start
        self.l_end: float = l_end
        self.g_end: float = g_end
        self.neg_decade: float = neg_decade
        self.width: float = width
        self.pos_decade: float = pos_decade
        self.length: int = length   #unused - likely a reference flowjo's lookup resolution

        self.values: List[int] = None
        self.lookup: List[float] = None
        self._lookup_range: Tuple[int, int] = None # in local space

    def scaler(self, data: List[float]) -> List[float]:
        """
        The scaling function. Values outside the lookup range will be placed outside the start-end range
            :param data: the data to scale
        """
        self._check_lookup()

        data = copy.deepcopy(data)

        for i in range(0, len(data)):
            if data[i] is None:
                continue

            index = bisect.bisect_right(self.lookup, data[i])

            if not index:
                value = self.l_start
            elif index == len(self.lookup):
                value = self.l_end
            else:
                value = self.values[index]

            data[i] = value

        return data

    def labels(self) -> Tuple[List[float], List[str]]:
        """
        Returns the labels for the major ticks (for coordinates use major_ticks())
        """
        self._check_lookup()
        labels = self.generator.labels(self.lookup[0], self.lookup[-1])

        # remove too-close-to-zero labels
        # to prevent overlapping labels

        # find zero
        i_0 = None
        for i, label in enumerate(labels):
            if label == "0":
                i_0 = i
                break

        # No zero
        if i_0 is None:
            return labels

        labels_x = self.scaler(self.generator.major_ticks(self.lookup[0], self.lookup[-1]))

        x_0 = labels_x[i_0]
        for i in range(i_0 + 1, len(labels)):
            if (labels_x[i] - x_0) < ((self.l_end - self.l_start) * 0.035):
                labels[i] = ""
                labels[i_0 - (i-i_0)] = ""

        return labels

    def major_ticks(self) -> List[float]:
        """
        Returns the major tick locations
        """
        self._check_lookup()
        ticks = self.generator.major_ticks(self.lookup[0], self.lookup[-1])
        ticks = self.scaler(ticks)

        return ticks
    
    def minor_ticks(self) -> List[float]:
        """
        Returns the minor tick locations
        """
        self._check_lookup()
        ticks = self.generator.minor_ticks(self.lookup[0], self.lookup[-1])
        ticks = self.scaler(ticks)

        return ticks

    def _check_lookup(self) -> None:
        """
        Checks if (re)building of the lookup table is necessary and (re)builds ifso.
        """
        if self.lookup is None:
            self._build_lookup()

        elif self._lookup_range[0] != self.l_start or self._lookup_range[1] != self.l_end:
            self._build_lookup()

    def _build_lookup(self):
        """
        Builds the lookup table for the biexponential transform
        """
        # Source paper: David R. Parks, Mario Roederer, Wayne A. Moore.  A new “Logicle” display method avoids deceptive effects of logarithmic scaling for low signals and compensated data.  Cytometry Part A, Volume 69A, Issue 6, pages 541-551, June 2006.
        # Use the FlowJo implementation, as transformed by CytoLib: https://github.com/RGLab/cytolib/blob/master/src/transformation.cpp directly from the FlowJo java implementation

        decades = self.pos_decade
        width = np.log10(-self.width)   # log10 transform to make the data 'equivalent' to decades and extra
        extra = self.neg_decade
        channel_range = int(self.l_end - self.l_start)
        
        if channel_range < 1:
            raise ValueError(f"cannot build lookup table with local space range {self.l_start}-{self.l_end}")

        # Make sure width is within expected values
        if width < 0.5 or width > 3.0:
            raise ValueError(f"width has to be <= -3.16 (10**0.5) and >= -1000 (10**3), not '{width}'")

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

        maximum = self.g_end
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
        self._lookup_range = (self.l_start, self.l_end)

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

    def __eq__(self, other) -> bool:
        if not isinstance(other, _Abstract):
            raise ValueError("can only test equality against other transform classes")

        if not isinstance(other, Biex):
            return False
        
        if self.neg_decade != other.neg_decade:
            return False
        if self.width != other.width:
            return False
        if self.pos_decade != other.pos_decade:
            return False
        if self.g_end != other.g_end:
            return False
        if self.l_start != other.l_start:
            return False
        if self.l_end != other.l_end:
            return False

        return True

    def __repr__(self) -> str:
        return f"(BiexTransform:[{self.g_end};{self.width:.1f};{self.pos_decade:.1f}]->[{self.l_start}-{self.l_end}])"

class Fasinh(_Abstract):
    """
    Represents a inverse hyperblic sine transformation.
        :param l_start: start local value
        :param l_end: end local value
        :param t: global end value / top of scale
        :param m: the number of (positive) decades
        :param a: the number of additional negative decades
        :param w: (unused) the number of decades in the linear section
        :param length: (unused) the lookup table resolution
    """
    def __init__(self, l_start: float=0, l_end: float=1023, t: float=262144, m: float=5.418539922, a: float=0.5, w: float=-262144, length: int=256):
        super().__init__()
        self.generator = BiexGenerator()

        self.l_start: float = l_start
        self.l_end: float = l_end
        self.g_end: float = t   # for adherence to the _Abstract api, otherwise unused
        self.t: float = t
        self.m: float = m
        self.a: float = a
        self.w: float = w   #unused - likely a reference to flowjo's implementation
        self.length: int = length   #unused - likely a reference flowjo's implementation

    def scaler(self, data: List[float]) -> List[float]:
        """
        The scaling function
            :param data: the data to scale
        """
        data = copy.deepcopy(data)

        #                   asinh(x sinh(M ln(10)) / T) + A ln(10)
        # fasinh(x,T,M,A) = ──────────────────────────────────────
        #                               (M + A)ln(10)
        #
        #                   asinh(x sinh(M ln(10)) / T)      A ln(10)
        # fasinh(x,T,M,A) = ─────────────────────────── + ─────────────
        #                           (M + A)ln(10)         (M + A)ln(10)
        #
        #                          1             ┌     sinh(M ln(10)) ┐     A ln(10)
        # fasinh(x,T,M,A) = ───────────── * asinh│ x * ────────────── │ + ─────────────
        #                   (M + A)ln(10)        └           T        ┘   (M + A)ln(10)

        #
        # fasinh(x,T,M,A) = a * asinh(x * b) + c
        #
        #           1                 sinh(M ln(10))             
        # a = ─────────────       b = ──────────────       c = A ln(10) * a
        #     (M + A)ln(10)                 T                  

        # Constants
        ln10 = np.log(10)
        a = 1 / ( (self.m + self.a) * ln10)
        b = ( np.sinh(self.m * ln10) ) / self.t
        c = ( self.a * ln10) * a

        # scale
        for i in range(0, len(data)):
            if data[i] is None:
                continue

            data[i] = (a * np.arcsinh((data[i] * b)) ) + c
            data[i] = (data[i] * self.l_end) + self.l_start

        return data

    def labels(self) -> List[str]:
        """
        Returns the labels for the major ticks (for coordinates use major_ticks())
        """
        g_start = -10**self.a
        return self.generator.labels(g_start, self.t)
 
    def major_ticks(self) -> List[float]:
        """
        Returns the major tick locations
        """
        g_start = -10**self.a
        ticks = self.generator.major_ticks(g_start, self.t)

        ticks = self.scaler(ticks)

        return ticks
    
    def minor_ticks(self) -> List[float]:
        """
        Returns the minor tick locations
        """
        g_start = -10**self.a
        ticks = self.generator.minor_ticks(g_start, self.t)

        ticks = self.scaler(ticks)

        return ticks

    def __eq__(self, other) -> bool:
        if not isinstance(other, _Abstract):
            raise ValueError("can only test equality against other transform classes")

        if not isinstance(other, Fasinh):
            return False
        
        if self.t != other.t:
            return False
        if self.m != other.m:
            return False
        if self.a != other.a:
            return False
        if self.l_start != other.l_start:
            return False
        if self.l_end != other.l_end:
            return False

        return True

    def __repr__(self) -> str:
        return f"(FasinhTransform:[{self.t};{self.m:.1f};{self.a:.1f}]->[{self.l_start}-{self.l_end}])"

class Hyperlog(_Abstract):
    """
    Represents a hyperlog transformation.
    Implementation based on https://github.com/RGLab/flowCore/blob/master/src/Hyperlog.cpp
        :param l_start: start local value
        :param l_end: end local value
        :param t: global end value / top of scale
        :param m: the number of (positive) decades
        :param a: the number of additional negative decades
        :param w: the number of decades in the linear section
        :param length: (unused) the lookup table resolution
    """
    TAYLOR_LENGTH: int = 16
    HALLEY_ITERATIONS: int = 10

    def __init__(self, l_start: float=0.0, l_end: float=1023.0, t: float=262144.0, m: float=5.418539922, a: float=0.5, w: float=-262144, length: int=256):
        super().__init__()
        self.generator = BiexGenerator()

        raise Exception("HYPERLOG IS STILL INCORRECT!")

        self.l_start: float = l_start
        self.l_end: float = l_end
        self.g_end: float = t   # for adherence to the _Abstract api, otherwise unused

        # Standard parameters
        self.t: float = t
        self.m: float = m
        self.a: float = a
        self.w: float = w
        self.length: int = length   #unused - likely a reference flowjo's implementation

        # Actual parameters
        # Choose the data zero location and the width of the linearization region
        # to match the corresponding logicle scale
        self._w: float = self.w / (self.m + self.a)
        self._x2: float = self.a / (self.m + self.a)
        self._x1: float = self._x2 + self._w
        self._x0: float = self._x2 + (2 * self._w)
        self._b: float = (self.m + self.a) * np.log(10.0)
        e2bx0: float = np.exp(self._b * self._x0)
        c_a: float = e2bx0 / self._w
        f_a: float = np.exp(self._b * self._x1) + (c_a * self._x1)
        self._a: float = self.t / ((np.exp(self._b) + c_a) - f_a)
        self._c: float = c_a * self._a
        self._f: float = f_a * self._a

        # Use Taylor series near x1, i.e., data zero to avoid round off problems of formal definition
        self._taylor_x: float = self._x1 + (self._w / 4)

        # Compute coefficients of the Taylor series
        coef: float = self._a * np.exp(self._b * self._x1)
        self._taylor: List[float] = [0.0]*self.TAYLOR_LENGTH
        for i in range(0, self.TAYLOR_LENGTH):
            coef *= self._b / (i + 1)
            self._taylor[i] = coef

        self._taylor[0] += self._c # hyperlog condition
        self._inverse_x0 = self._inverse(self._x0)

    def scale(self, data: float) -> float:
        """
        Scales the value according to hyperlog transform
        """
        # Handle true zero separately
        if data == 0:
            #return self._x1
            return (self._x1 * (self.l_end - self.l_start)) + self.l_start

        # Reflect negative values
        negative: bool = data < 0
        if negative:
            data = -data

        # Initial guess at solution
        x: float = None
        if data < self._inverse_x0:
            # Use linear approximation in the quasi linear region
            x = self._x1 + data * self._w / self._inverse_x0
        else:
            # otherwise use ordinary logarithm
            x = np.log(data / self._a) / self._b
        
        # Try for double precision unless in extended range
        tolerance: float = 3 * sys.float_info.epsilon
        if x > 1:
            tolerance = 3 * x * sys.float_info.epsilon

        for i in range(0, self.HALLEY_ITERATIONS):
            # Compute the funciton and its first two derivatives
            ae2bx: float = self._a * np.exp(self._b * x)
            # double ce2mdx = self._c / np.exp(self._d * x)
            y: float = None
            if x < self._taylor_x:
                # near zero use the Taylor series
                t_x: float = x - self._x1
                t_sum: float = self._taylor[-1] * t_x
                for j in range(self.TAYLOR_LENGTH-2, -1, -1):
                    t_sum = (t_sum + self._taylor[j]) * t_x
                y = t_sum - data
            else:
                # This formulation has better roundoff behavior
                y = (ae2bx + self._c * x) - (self._f + data)

            abe2bx: float = self._b * ae2bx
            dy: float = abe2bx + self._c
            ddy: float = self._b * abe2bx

            # This is Halley's method with cubic convergence
            delta: float = y / (dy * (1 - y * ddy / (2 * dy * dy)))
            x -= delta

            # if we have reached the desired precision we are done
            if abs(delta) < tolerance:
                # Handle negative arguments
                if negative:
                    x = 2 * self._x1 - x
                    return (x * (self.l_end - self.l_start)) + self.l_start
                else:
                    #return x
                    return (x * (self.l_end - self.l_start)) + self.l_start

        raise ValueError("Hyperlog transform did not converge")

    def scaler(self, data: List[float]) -> List[float]:
        """
        The scaling function
            :param data: the data to scale
        """
        for i in range(0, len(data)):
            data[i] = self.scale(data[i])

        return data

    def _inverse(self, data: float) -> float:
        """
        Reverts the scaled data back into unscaled form.
        Expects a data range of 0-1
        """
        # Reflect negative scale regions
        negative: bool = data < self._x1
        if negative:
            data: float = 2 * self._x1 - data

        # Double inverse
        if data < self._taylor_x:
            # near x1, i.e., data zero use the series expansion
            t_x: float = data - self._x1
            t_sum: float = self._taylor[-1] * t_x
            for j in range(self.TAYLOR_LENGTH-2, -1, -1):
                t_sum = (t_sum + self._taylor[j]) * t_x
            inverse = t_sum
        else:
            # This formulation has better roundoff behavior
            inverse = (self._a * np.exp(self._b * data) + self._c * data) - self._f
        
        # Handle scale for negative values
        if negative:
            return -inverse
        else:
            return inverse

    def labels(self) -> List[str]:
        """
        Returns the labels for the major ticks (for coordinates use major_ticks())
        """
        g_start = self._inverse(0.0)
        labels = self.generator.labels(g_start, self.t)

        # remove too-close-to-zero labels
        # to prevent overlapping labels

        # find zero
        i_0 = None
        for i, label in enumerate(labels):
            if label == "0":
                i_0 = i
                break

        # No zero
        if i_0 is None:
            return labels

        labels_x = self.scaler(self.generator.major_ticks(g_start, self.t))

        x_0 = labels_x[i_0]
        for i in range(i_0 + 1, len(labels)):
            if (labels_x[i] - x_0) < ((self.l_end - self.l_start) * 0.035):
                labels[i] = ""
                labels[i_0 - (i-i_0)] = ""

        return labels
 
    def major_ticks(self) -> List[float]:
        """
        Returns the major tick locations
        """
        g_start = self._inverse(0.0)
        ticks = self.generator.major_ticks(g_start, self.t)

        ticks = self.scaler(ticks)

        return ticks
    
    def minor_ticks(self) -> List[float]:
        """
        Returns the minor tick locations
        """
        g_start = self._inverse(0.0)
        ticks = self.generator.minor_ticks(g_start, self.t)

        ticks = self.scaler(ticks)

        return ticks

    def __eq__(self, other) -> bool:
        if not isinstance(other, _Abstract):
            raise ValueError("can only test equality against other transform classes")

        if not isinstance(other, Hyperlog):
            return False
        
        if self.t != other.t:
            return False
        if self.m != other.m:
            return False
        if self.a != other.a:
            return False
        if self.l_start != other.l_start:
            return False
        if self.l_end != other.l_end:
            return False

        return True

    def __repr__(self) -> str:
        return f"(HyperlogTransform:[{self.t};{self.m:.1f};{self.a:.1f};{self.w:.1f}]->[{self.l_start}-{self.l_end}])"

class Logicle(_Abstract):
    """
    Represents the logicle transformation.
    Implementation based on https://github.com/RGLab/flowCore/src/Logicle.cpp
        :param l_start: start local value
        :param l_end: end local value
        :param t: global end value / top of scale
        :param m: the number of (positive) decades
        :param a: the number of additional negative decades
        :param w: the number of decades in the linear section
    """
    TAYLOR_LENGTH: int = 16
    HALLEY_ITERATIONS: int = 20

    def __init__(self, l_start: float=0, l_end: float=1023, t: float=262144, m: float=5.418539922, a: float=0.5, w: float=2.652246341):
        super().__init__()
        self.generator = BiexGenerator()

        self.l_start: float = l_start
        self.l_end: float = l_end
        self.g_end: float = t   # for adherence to the _Abstract api, otherwise unused
        
        # standard parameters
        self.t: float = t   # top of scale
        self.m: float = m   # number of decades of log component
        self.w: float = w   # number of decades of linear component
        self.a: float = a   # additional negative decades

        # Actual parameters, formulas from biexponential paper
        self._w: float = self.w / (self.m + self.a)
        self._x2: float = self.a / (self.m + self.a)
        self._x1: float = self._x2 + self._w
        self._x0: float = self._x2 + (2 * self._w)
        self._b: float = (self.m + self.a) * np.log(10.0)
        self._d: float = self._solve(self._b, self._w)
        c_a: float = np.exp(self._x0 * (self._b + self._d))
        mf_a: float = np.exp(self._b * self._x1) - (c_a / np.exp(self._d * self._x1))
        self._a: float = self.t / (np.exp(self._b) - mf_a - (c_a / np.exp(self._d)))
        self._c: float = c_a * self._a
        self._f: float = -mf_a * self._a

        # Use Taylor series near x1, i.e., the data near to zero
        # to avoid round-off problems of formal definition
        self._taylor_x: float = self._x1 + (self._w / 4)

        # Compute coefficients of the Taylor series
        pos_coef: float = self._a * np.exp(self._b * self._x1)
        neg_coef: float = -self._c / np.exp(self._d * self._x1)

        # 16 is enough for full precision of typical scales
        self._taylor: List[float] = [None]*self.TAYLOR_LENGTH
        for i in range(0, self.TAYLOR_LENGTH):
            pos_coef *= (self._b / (i + 1))
            neg_coef *= (-self._d / (i + 1))
            self._taylor[i] = pos_coef + neg_coef
        # Exact result of logicle condition
        self._taylor[1] = 0.0

    def _solve(self, b:float, w:float) -> float:
        """
        Approximate the root of the logicle transform
        """
        # w==0 means it's really arcsinh
        if w == 0:
            return b

        # precision is the same as that of b
        tolerance: float = 2 * b * sys.float_info.epsilon
        d_lo: float = 0.0 + sys.float_info.epsilon
        d_hi: float = b

        d: float = Logicle._R_zeroin(
            ax=d_lo,
            bx=d_hi,
            f=self._logicle,
            tol=tolerance,
            maxit=20
        )
        return d

    def _logicle(self, x: float) -> float:
        """
        Logicle function: f(w,b) = 2 * (ln(d) - ln(b)) + w * (b + d)
        """
        return 2 * ( np.log(x) - np.log(self._b)) + (self._w * (self._b + x))
    
    @staticmethod
    def _R_zeroin(ax: float, bx: float, f: Callable, tol: float, maxit: int) -> float:
        """
        Root finder routines, copied from stats/src/zeroin.c
            :param ax: left border of the search range
            :param bx: right border of the search range
            :param f: the function under investigation
            :param maxit: maximum number of iterations            
        """
        fa: float= f(ax)
        fb: float= f(bx)
        return Logicle._R_zeroin2(ax=ax, bx=bx, fa=fa, fb=fb, f=f, tol=tol, maxit=maxit)

    @staticmethod
    def _R_zeroin2(ax: float, bx: float, fa: float, fb: float, f: Callable, tol: float, maxit: int) -> float:
        """
        Root finder routine; faster for "expensive" f(), in those typical case where f(ax) and f(bx) are available anyway
            :param ax: left border of the search range
            :param bx: right border of the search range
            :param fa, fb: f(a), f(b)
            :param f: function under investigation
            :param tol: acceptable tolerance
            :param maxit: maximum number of iterations
        """
        a: float=ax
        b: float=bx
        c: float=a
        fc: float=fa
        maxit: int = maxit + 1

        # First test if we have found a root at an endpoint
        if fa == 0.0:
            #tol = 0.0, maxit = 0
            return a
        if fb == 0.0:
            #tol = 0.0, maxit = 0
            return b
        
        # Main iteration loop
        while maxit:
            maxit -= 1

            prev_step: float = b-a  # Distance from the last but one to the last approximation
            tol_act: float = None   # Actual tolerance
            p: float = None         # interpolation step is calculated in the form of p/q
            q: float = None         # division operation is delayed until the last moment
            new_step: float = None  # step at this iteration

            if abs(fc) < abs(fb):
                # Swap data for b to be the best approximation
                a=b
                b=c
                c=a
                fa=fb
                fb=fc
                fc=fa
            tol_act = 2 * sys.float_info.epsilon * abs(b) + tol/2
            new_step = (c-b)/2

            if abs(new_step) <= tol_act or fb == 0.0:
                #maxit -= maxit, tol = abs(c-b)
                return b # Acceptable approximation is found

            # Decide if the interpolation can be tried
            if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
                # if prev_step was large enough && in true direction
                # interpolation may be tried
                t1: float = None
                t2: float = None
                cb: float = None
                cb = c-b
                
                if a == c:
                    # if we have only two distinct points, linear interpolation can only be applied
                    t1 = fb/fa
                    p = cb * t1
                    q = 1.0 - t1
                else:
                    # else Quadratic inverse interpolation
                    q = fa/fc
                    t1 = fb/fc
                    t2 = fb/fa
                    p = t2 * (cb*q*(q-t1) - (b-a)*(t1-1.0))
                    q = (q-1.0) * (t1-1.0) * (t2-1.0)
                
                if p > 0.0:
                    # p was calculated with the opposite sign; make p positive and assign possible minus to q
                    q = -q
                else:
                    p = -p

                if p < (0.75 * cb * q - abs(tol_act * q) /2) and p < abs(prev_step*q/2):
                    # if b+p/q falls in [b,c] and isnt too large; it is accepted
                    # if p/q is too large then the bisection procedure can reduce [b,c] range to a higher extent
                    new_step = p/q
            
            if abs(new_step) < tol_act:
                # Adjust the step to be not less than tolerance
                if new_step > 0.0:
                    new_step = tol_act
                else:
                    new_step = -tol_act

            # Store the previous approximation
            a=b
            fa=fb
            # Step to a new approximation
            b += new_step
            fb = f(b)

            if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
                # Adjust c for it ot have a sign opposite to that of b
                c = a
                fc = fa

        # Failure to approximate
        # tol = abs(c-b), maxit = -1
        return b

    def scale(self, data: float) -> float:
        """
        Scales the value according to logicle transform
        """
        # Handle true zero separately
        if data == 0:
            #return self._x1
            return (self._x1 * (self.l_end - self.l_start)) + self.l_start
        
        # Reflect negative values
        negative: bool = data < 0
        if negative:
            data = -data

        # Initial guess at solution
        x: float = None
        if data < self._f:
            # Use linear approximation in the quasi linear region
            x = self._x1 + data / self._taylor[0]
        else:
            # otherwise use ordinary logarithm
            x = np.log(data / self._a) / self._b

        # try for double precision unless in extended range
        tolerance: float = 3 * sys.float_info.epsilon
        if x > 1:
            tolerance = 3 * x * sys.float_info.epsilon

        for i in range(0, self.HALLEY_ITERATIONS):
            # compute the function and its first two derivatives
            ae2bx: float = self._a * np.exp(self._b * x)
            ce2mdx: float = self._c / np.exp(self._d * x)

            y: float = None
            if x < self._taylor_x:

                # Near zero use the Taylor series
                # Taylor series is around x1
                t_x: float = x - self._x1
                # Note that taylor[1] should be identical to zero according to logical condition
                t_sum: float = self._taylor[-1] * t_x
                for j in range(self.TAYLOR_LENGTH-2, 1, -1):
                    t_sum = (t_sum + self._taylor[j]) * t_x
                y = ((t_sum * t_x + self._taylor[0]) * t_x) - data
            else:
                # This formulation bas better roundoff behavior
                y = (ae2bx + self._f) - (ce2mdx + data)
            
            abe2bx: float = self._b * ae2bx
            cde2mdx: float = self._d * ce2mdx
            dy: float = abe2bx + cde2mdx
            ddy: float = (self._b * abe2bx) - (self._d * cde2mdx)

            # This is Halley's method with cubic convergence
            delta: float = y / (dy * (1 - ((y * ddy) / (2 * dy * dy))))
            x -= delta

            # if we have reached the desired precision we're done
            if abs(delta) < tolerance:
                # handle negative arguments
                if negative:
                    value = 2 * self._x1 - x
                    #return value
                    return (value * (self.l_end - self.l_start)) + self.l_start
                else:
                    value = x
                    #return value
                    return (value * (self.l_end - self.l_start)) + self.l_start

        raise ValueError("Logicle transform did not converge")

    def scaler(self, data: List[float]) -> List[float]:
        """
        The scaling function
            :param data: the data to scale
        """
        for i in range(0, len(data)):
            data[i] = self.scale(data[i])

        return data

    def _inverse(self, data: float) -> float:
        """
        Reverts the scaled data back into unscaled form.
        Expects a data range of 0-1
        """
        # reflect negative scale regions
        negative: bool = data < self._x1
        if negative:
            data = 2 * self._x1 - data

        # compute the biexponetial
        inverse: float = None
        if data < self._taylor_x:
            # near x1, i.e., data zero uses the series expansion
            t_x: float = data - self._x1
            # Note that taylor[1] should be identical to zero according to logical condition
            t_sum: float = self._taylor[-1] * t_x
            for j in range(self.TAYLOR_LENGTH-2, 1, -1):
                t_sum = (t_sum + self._taylor[j]) * t_x
            inverse = ((t_sum * t_x + self._taylor[0]) * t_x) - data
        else:
            inverse = (self._a * np.exp(self._b * data) + self._f) - (self._c / np.exp(self._d * data))
        
        # handle scale for negative values
        if negative:
            return -inverse
        else:
            return inverse

    def labels(self) -> List[str]:
        """
        Returns the labels for the major ticks (for coordinates use major_ticks())
        """
        g_start = self._inverse(0.0)
        labels = self.generator.labels(g_start, self.t)

        # remove too-close-to-zero labels
        # to prevent overlapping labels

        # find zero
        i_0 = None
        for i, label in enumerate(labels):
            if label == "0":
                i_0 = i
                break

        # No zero
        if i_0 is None:
            return labels

        labels_x = self.scaler(self.generator.major_ticks(g_start, self.t))

        x_0 = labels_x[i_0]
        for i in range(i_0 + 1, len(labels)):
            if (labels_x[i] - x_0) < ((self.l_end - self.l_start) * 0.035):
                labels[i] = ""
                labels[i_0 - (i-i_0)] = ""

        return labels
 
    def major_ticks(self) -> List[float]:
        """
        Returns the major tick locations
        """
        g_start = self._inverse(0.0)
        ticks = self.generator.major_ticks(g_start, self.t)

        ticks = self.scaler(ticks)

        return ticks
    
    def minor_ticks(self) -> List[float]:
        """
        Returns the minor tick locations
        """
        g_start = self._inverse(0.0)
        ticks = self.generator.minor_ticks(g_start, self.t)

        ticks = self.scaler(ticks)

        return ticks
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, _Abstract):
            raise ValueError("can only test equality against other transform classes")

        if not isinstance(other, Logicle):
            return False
        
        if self.l_start != other.l_start:
            return False
        if self.l_end != other.l_end:
            return False
        if self.t != other.t:
            return False
        if self.m != other.m:
            return False
        if self.a != other.a:
            return False
        if self.w != other.w:
            return False

        return True

    def __repr__(self) -> str:
        return f"(LogicleTransform:[{self.t};{self.m:.2f};{self.a:.1f};{self.w:.1f}]->[{self.l_start}-{self.l_end}])"
