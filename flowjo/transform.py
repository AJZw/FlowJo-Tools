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
from typing import List, Dict, Tuple

import numpy as np
import bisect
import copy

class _AbstractGenerator():
    """
    Abstract Generator. Abstract class for generators of labels, minor/major tick locations 
    for 'infinite' range transforms. Should be subclassed for a scale-specific implementation.
    Is defined to only generate labels/ticks in the specified (global) range.
    """
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
    Generates labels, minor/major tick locations for a biexponential scale
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
                value = self.l_start-1.0
            elif index == len(self.lookup):
                value = self.l_end+1.0
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
            :param start: the local space start
            :param end: the local space end
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
