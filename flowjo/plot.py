##############################################################################     ##    ######
#    A.J. Zwijnenburg                   2020-09-11           v1.0                 #  #      ##
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
Classes for the plotting of FlowJo !Channel! data
Channel data is pre-scaled by FlowJo and binnen to the values 0-1023
As the scaling information is not exported the plot class cannot provide a default.
Therefore it is beneficial to set Plot.scales for proper results

:class: _condensor_density
Condensor function for Plot._bin(). Returns the amount of cells/density

:class: _condensor_mean
Condensor function for Plot._bin(). Returns the mean value for numeric data or the mode for categorical data

:class: _condensor_max
Condensor function for Plot._bin(). Returns the max value for numeric data or the mode for categorical data

:class: _condensor_min
Condensor function for Plot._bin(). Returns the min value for numeric data or the mode for categorical data

:class: _AbstractScale
Abstract class providing channel-scaled outputs for major/minor ticks and labels (according to FlowJo)

:class: LinearScale
The linear scale class

:class: LogScale
The Logarithmic (log10) scale class

:class: BiexScale
The biexponential scale class

:class: Plot
The main plotting class. Provides an interface for the convenient plotting of scatter and rasterized plots.
Rasterization is slow as proper statistics can be calculated per bin

"""

from __future__ import annotations

from .data import _Abstract
import pandas as pd
import numpy as np
import plotnine as p9
import copy
import bisect

#p9.options.figure_size=(12.8, 9.6)

def _condensor_density(column: pd.Series) -> Any:
    """
    Example condensor function for the _bin function. Must accept a row pd.Series and return a single value
    Both numeric and categorial data must be handled.
        :param column: the input data, will be a column, because of 'apply(axis="index")'
        :returns: categorical returns mode; numeric returns mean
    """
    return len(column.index)

def _condensor_mean(column: pd.Series) -> Any:
    """
    Example condensor function for the _bin function. Must accept a row pd.Series and return a single value
    Both numeric and categorial data must be handled.
        :param column: the input data, will be a column, because of 'apply(axis="index")'
        :returns: categorical returns mode; numeric returns mean
    """
    if pd.api.types.is_numeric_dtype(column):
        # numeric handling
        output = column.mean(skipna=True)
    else:
        # categorical handling
        output = column.mode(dropna=True)
        if output.empty:
            output = np.nan
        else:
            output = output.iloc[0]
    return output

def _condensor_max(column: pd.Series) -> Any:
    """
    Example condensor function for the _bin function. Must accept a row pd.Series and return a single value
    Both numeric and categorial data must be handled.
        :param column: the input data, will be a column, because of 'apply(axis="index")'
        :returns: categorical returns mode; numeric returns max
    """
    if pd.api.types.is_numeric_dtype(column):
        # numeric handling
        output = column.max(skipna=True)
    else:
        # categorical handling
        output = column.mode(dropna=True)
        if output.empty:
            output = np.nan
        else:
            output = output.iloc[0]
    return output

def _condensor_min(column: pd.Series) -> Any:
    """
    Example condensor function for the _bin function. Must accept a row pd.Series and return a single value
    Both numeric and categorial data must be handled.
        :param column: the input data, will be a column, because of 'apply(axis="index")'
        :returns: categorical returns mode; numeric returns min
    """
    if pd.api.types.is_numeric_dtype(column):
        # numeric handling
        output = column.min(skipna=True)
    else:
        # categorical handling
        output = column.mode(dropna=True)
        if output.empty:
            output = np.nan
        else:
            output = output.iloc[0]
    return output

class _AbstractScale():
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

    def _scaler(self, data: List[int], start: int, end: int) -> List[float]:
        """
        The scaling function
            :param data: the data to scale
            :param start: the local space start
            :param end: the local space end
        """
        raise NotImplementedError("implement in child class")

    def labels(self, start: int=0, end: int=1023) -> Tuple[List[float], List[str]]:
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

        ticks = self._scaler(ticks, start, end)

        return ticks
    
    def minor_ticks(self, start: int=0, end: int=1023) -> List[float]:
        """
        Returns the minor tick location in plot x-coordinate
            :param start: plot minimum xlim
            :param end: plot maximum xlim
        """
        ticks = copy.deepcopy(self._minor_ticks)

        ticks = self._scaler(ticks, start, end)

        return ticks

class LinearScale(_AbstractScale):
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

    def __init__(self, start: int=0, end: int=262144):
        super().__init__()
        self.start: int = start
        self.end: int = end

    def _scaler(self, data: List[int], start: int, end: int) -> List[float]:
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

class LogScale(_AbstractScale):
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

    def __init__(self, start: int=3, end: int=262144):
        super().__init__()
        self.start: int = start
        self.end: int = end

        if self.start <= 0:
            raise ValueError("logarithmic scale have to start with a value >0")

    def _scaler(self, data: List[int], start: int, end: int) -> List[float]:
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

class BiexScale(_AbstractScale):
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

    def __init__(self, end: int=262144, neg_decade: float=0, width: float=-100, pos_decade: float=4.42):
        super().__init__()
        self.neg_decade: float = neg_decade
        self.width: float = width
        self.pos_decade: float = pos_decade
        self.end: int = end

        self.values: List[int] = None
        self.lookup: List[float] = None

    def _scaler(self, data: List[int], start: int, end: int) -> List[float]:
        """
        The scaling function
            :param data: the data to scale
            :param start: the local space start
            :param end: the local space end
        """
        if self.lookup is None:
            self._build_lookup(start, end)

        for i in range(0, len(data)):
            index = bisect.bisect_right(self.lookup, data[i])

            if not index:
                value = None
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
        labels_x = self._scaler(labels_x, start, end)

        # remove too-close-to-zero labels
        zero = labels_x[10]
        for i in range(11, len(labels)):
            if (labels_x[i] - zero) < ((end - start) * 0.05):
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

class Plot():
    """
    Main plotting class. Load it with data and ask it to generate plots from that data.
    It will use (mainly) plotnine plots
        :param data: FlowJo data
    """
    def __init__(self, data: Union[pd.DataFrame, _Abstract]):
        self._data: pd.DataFrame=None

        self.color_na: str="#E3256B"
        self.color_map: str="magma"
        self.fill_na: str="#E3256B"
        self.fill_map: str="magma"
        self.is_channel: bool=True
        self.scale_lim: Tuple[int,int]=[0, 1023]
        self.scales: Dict[str, _Scale]={
            "FSC-A":LinearScale(start=0, end=262144),
            "FSC-W":LinearScale(start=0, end=262144),
            "FSC-H":LinearScale(start=0, end=262144),
            "SSC-A":LinearScale(start=0, end=262144),
            "SSC-W":LinearScale(start=0, end=262144),
            "SSC-H":LinearScale(start=0, end=262144),
            "Time":LinearScale(start=0, end=262144)
        }

        if isinstance(data, _Abstract):
            self._data = data.data
            self._check_scale()
        elif isinstance(data, pd.DataFrame):
            self._data = data
            self._check_scale()
        else:
            raise ValueError("plot must be instantiate with a pd.DataFrame or flowjo.data._Abstract class")
    
    @property
    def data(self) -> pd.DataFrame:
        """
        Getter for the internal data dataframe
        """
        return self._data

    def _plot_base(self, data: pd.Dataframe, x: str, y: str, color: str=None, fill: str=None) -> p9.ggplot:
        """
        Creates the data base for all plots
            :param data: the data table with all necessary plotting information
            :param x: the x-axis parameter
            :param y: the y-axis parameter
            :param color: for solid object the fill, for non-solid object the outline parameter
            :param fill: for non-solid objects the fill parameter
            :returns: the plot data base
            :raises ValueError: if parameters could not be found in .data
        """
        if not (data.columns == x).any():
            raise ValueError(f"x '{x}' does not specify columns in .data")

        if not pd.api.types.is_numeric_dtype(data[x]):
            raise ValueError(f"x '{x}' must be a numeric dtype")

        if not (data.columns == y).any():
            raise ValueError(f"y '{y}' does not specify columns in .data")
        
        if not pd.api.types.is_numeric_dtype(data[y]):
            raise ValueError(f"y '{y}' must be a numeric dtype")

        if color:
            if not (data.columns == color).any():
                raise ValueError(f"color '{color}' does not specify columns in .data")

        if fill:
            if not (data.columns == fill).any():
                raise ValueError(f"fill '{fill}' does not specify columns in .data")
        
        if color and fill:
            plot = p9.ggplot(
                data,
                p9.aes(x, y, fill=fill, color=color)
            )
        elif color:
            plot = p9.ggplot(
                data,
                p9.aes(x, y, color=color)
            )
        elif fill:
            plot = p9.ggplot(
                data,
                p9.aes(x, y, fill=fill)
            )
        else:
            plot = p9.ggplot(
                data,
                p9.aes(x, y)
            )

        return plot
   
    def _plot_theme(self, plot: p9.ggplot) -> p9.ggplot:
        """
        Adds the plot theme to the plot
            :param plot: the plot the theme
            :returns: themed plot
        """
        plot = plot + p9.theme_bw() + p9.theme(
            text=p9.element_text(family="sans-serif", weight="normal"),
            plot_title=p9.element_text(ha="center", weight="bold", size=14),
            axis_text_x=p9.element_text(ha="center", va="top"),
            axis_text_y=p9.element_text(ha="right", va="center"),
            #axis_ticks_major_x=p9.element_blank(),
            #axis_ticks_major_y=p9.element_blank(),
            #axis_ticks_minor_x=p9.element_blank(),
            #axis_ticks_minor_y=p9.element_blank(),
            axis_ticks_major_x=p9.element_line(color="#FFFFFFFF"),
            axis_ticks_major_y=p9.element_line(color="#FFFFFFFF"),
            axis_ticks_minor_x=p9.element_line(color="#FFFFFFFF"),
            axis_ticks_minor_y=p9.element_line(color="#FFFFFFFF"),
            panel_grid_major_x=p9.element_blank(),
            panel_grid_major_y=p9.element_blank(),
            panel_grid_minor_x=p9.element_blank(),
            panel_grid_minor_y=p9.element_blank(),
            panel_background=p9.element_rect(fill="#EEEEEEFF", color="#FFFFFFFF"),
            legend_title=p9.element_blank(),
            legend_key=p9.element_blank(),
            legend_key_width=8,
            legend_key_height=35,
            legend_entry_spacing_x=-10,
            legend_entry_spacing_y=-20
        )
        return plot

    def _plot_labels(self, plot: p9.ggplot, title: str=None, x: str=None, y: str=None) -> p9.ggplot:
        """
        Adds labels to the plot
        """
        if title:
            plot = plot + p9.ggtitle(
                title
            )

        if x is None:
            x = plot.mapping["x"]
        if y is None:
            y = plot.mapping["y"]

        plot = plot + p9.labs(
            x=x, 
            y=y
        )

        return plot

    def _plot_scale(self, plot: p9.ggplot, xlim: Tuple[int, int]=None, ylim: Tuple[int, int]=None) -> p9.ggplot:
        """
        Adds the scale limits to the plot
            :param plot: plot to add the scale to
            :param xlim: (optional) sets the scale limits and enables proper axis scale representation
            :param ylim: (optional) sets the scale limits and enables proper axis scale representation
        """
        if not self.is_channel:
            raise ValueError("you cannot use _plot_scale on non-channel data")

        # Fetch scale from self.scales (if available)
        if xlim is not None:
            x = plot.mapping["x"]
            
            try:
                scale_x = self.scales[x]
            except KeyError:
                plot = plot + p9.coords.coord_cartesian(xlim=xlim)

            plot = plot + p9.scale_x_continuous(
                breaks=scale_x.major_ticks(start=xlim[0], end=xlim[1]),
                minor_breaks=scale_x.minor_ticks(start=xlim[0], end=xlim[1]),
                labels=scale_x.labels(start=xlim[0], end=xlim[1]),
                expand=(0,0),
                limits=xlim
            )

        if ylim is not None:
            y = plot.mapping["y"]
            try:
                scale_y = self.scales[y]
            except KeyError:
                plot = plot + p9.coords.coord_cartesian(ylim=ylim)
            else:
                plot = plot + p9.scale_y_continuous(
                    breaks=scale_y.major_ticks(start=ylim[0], end=ylim[1]),
                    minor_breaks=scale_y.minor_ticks(start=ylim[0], end=ylim[1]),
                    labels=scale_y.labels(start=ylim[0], end=ylim[1]),
                    expand=(0,0),
                    limits=ylim
                )

        return plot

    def _plot_colorscale(self, plot: p9.ggplot, color_map: Dict[str, str]=None, rescale: bool=False) -> p9.ggplot:
        """
        Adds a color scale to the plot. Automatically detects whether the scale is discrete or continuous
            :color_map: if defined uses this color_map to manually assign colors to discrete values
            :rescale: whether to rescale the color_map to have min-max fall on the 2%-98% percentile
        """
        try:
            color = plot.mapping["color"]
        except KeyError:
            raise ValueError("cannot add a colorscale to a ggplot object without defined color mapping")

        if pd.api.types.is_numeric_dtype(plot.data[color]):
            # Continuous scale
            quantiles = plot.data[color].quantile([0.0, 0.02, 0.98, 1.0])
            if rescale:
                min_color = quantiles[0.02]
                max_color = quantiles[0.98]
            else:
                min_color = quantiles[0.0]
                max_color = quantiles[1.0]

            plot = plot + p9.scales.scale_color_cmap(
                cmap_name=self.color_map,
                limits=(min_color, max_color),
                guide=p9.guide_colorbar(
                    ticks=False
                ),
                na_value=self.color_na
            )

        elif pd.api.types.is_categorical_dtype(plot.data[color]) or pd.api.types.is_string_dtype(plot.data[color]):
            # Discrete
            
            # the tab10 & 20 discrete colorscales
            tab10 = ["#1f77b4","#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            tab20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]

            levels = plot.data[color].unique()

            if color_map:
                # Check if colormap covers all cases
                for level in levels:
                    if level not in color_map:
                        raise ValueError(f"level '{level}' undefined in color_map")

                plot = plot + p9.scales.scale_color_manual(
                    values=color_map,
                    na_value=self.color_na
                )
            elif len(levels) <= 10:
                plot = plot + p9.scales.scale_color_manual(
                    values = tab10,
                    na_value=self.color_na
                )
            else:
                # Use default
                pass   

        else:
            raise ValueError(f"unimplemented colorscale dtype {plot.data[color].dtype}")

        return plot
    
    def _plot_fillscale(self, plot: p9.ggplot, fill_map: Dict[str, str]=None, rescale: bool=False) -> p9.ggplot:
        """
        Adds a fill scale to the plot. Automatically detects whether the scale is discrete or continuous
            :fill_map: if defined uses this fill_map to manually assign colors to discrete values
            :rescale: whether to rescale the fill_map to have min-max fall on the 2%-98% percentile
        """
        try:
            fill = plot.mapping["fill"]
        except KeyError:
            raise ValueError("cannot add a fillscale to a ggplot object without defined fill mapping")

        if pd.api.types.is_numeric_dtype(plot.data[fill]):
            # Continuous scale
            quantiles = plot.data[fill].quantile([0.0, 0.02, 0.98, 1.0])
            if rescale:
                min_color = quantiles[0.02]
                max_color = quantiles[0.98]
            else:
                min_color = quantiles[0.0]
                max_color = quantiles[1.0]

            plot = plot + p9.scales.scale_fill_cmap(
                cmap_name=self.fill_map,
                limits=(min_color, max_color),
                guide=p9.guide_colorbar(
                    ticks=False
                ),
                na_value=self.fill_na
            )

        elif pd.api.types.is_categorical_dtype(plot.data[fill]) or pd.api.types.is_string_dtype(plot.data[fill]):
            # Discrete
            
            # the tab10 & 20 discrete colorscales
            tab10 = ["#1f77b4","#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            tab20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]

            levels = plot.data[fill].unique()

            if fill_map:
                # Check if fill_map covers all cases
                for level in levels:
                    if level not in fill_map:
                        raise ValueError(f"level '{level}' undefined in fill_map")

                plot = plot + p9.scales.scale_fill_manual(
                    values=fill_map,
                    na_value=self.fill_na
                )
            elif len(levels) <= 10:
                plot = plot + p9.scales.scale_fill_manual(
                    values = tab10,
                    na_value=self.fill_na
                )
            else:
                # Use default
                pass   

        else:
            raise ValueError(f"unimplemented fillscale dtype {plot.data[fill].dtype}")

        return plot

    def scatter(self, x: str, y: str, z: str, color_map: dict=None) -> p9.ggplot:
        """
        Creates a ggplot dotplot object with the correct data and axis
            :param x: the x dimension
            :param y: the y dimension
            :param z: the z dimension - used for color mapping
            :param color_map: only used for factorized color parameters. Uses the color_map to map the levels
        """
        plot = self._plot_base(self.data, x, y, color=z)
        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, x=x, y=y)
        plot = self._plot_scale(plot, xlim=self.scale_lim, ylim=self.scale_lim)
        plot = self._plot_colorscale(plot, rescale=True, color_map=color_map)
        plot = plot + p9.geom_point(na_rm="False")

        return plot

    def raster(self, x: str, y: str, z: str, z_stat: str="density", bin_size:int=2, color_map: dict=None) -> p9.ggplot:
        """
        Builds a raster plot of the specified data
            :param x: the x dimension
            :param y: the y dimension
            :param z: the z dimension - the parameter used for color mapping
            :param z_stat: the z statistic to calculate choose from ["density", "max", "min", "mean"]
            :param bin_size: effectively the size of the raster squares
            :param color_map: only used for factorized color parameters. Uses the color_map to map the levels
        """
        if z_stat not in ["density", "max", "min", "mean"]:
            raise ValueError(f"raster plotting has no implementation for z_stat '{z_stat}'")

        # Correct source data
        data = copy.deepcopy(self.data)
        data = self._reduce(data, bin_size)
        if z_stat == "density":
            data = self._bin(data, x, y, z=None, condensor=_condensor_density)
        elif z_stat == "max":
            data = self._bin(data, x, y, z=None, condensor=_condensor_max)
        elif z_stat == "min":
            data = self._bin(data, x, y, z=None, condensor=_condensor_min)
        elif z_stat == "mean":
            data = self._bin(data, x, y, z=None, condensor=_condensor_mean)

        data["__xmax"] = data[x] + 1
        data["__ymax"] = data[y] + 1

        # calculate new limits
        limits = copy.deepcopy(self.scale_lim)
        limits[0] = limits[0] // bin_size
        limits[1] = limits[1] // bin_size

        # Build the plot
        plot = self._plot_base(data, x, y, fill=z)
        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, f"{z_stat}({z})", x=x, y=y)
        plot = self._plot_scale(plot, xlim=limits, ylim=limits)
        plot = self._plot_fillscale(plot, rescale=True, fill_map=color_map)
        plot = plot + p9.geom_rect(
            p9.aes(
                xmin=x,
                xmax="__xmax",
                ymin=y,
                ymax="__ymax"
            )
        )

        return plot

    def _check_scale(self):
        """
        Checks whether the data is channel data (you cannot be 100% sure, but it will give an indication)
        """
        def find_min(column):
            if pd.api.types.is_numeric_dtype(column):
                output = column.min(skipna=True)
            else:
                return 0
        
        def find_max(column):
            if pd.api.types.is_numeric_dtype(column):
                output = column.max(skipna=True)
            else:
                return 0

        minimum = min(self.data.apply(find_min, axis="index"))
        maximum = max(self.data.apply(find_max, axis="index"))

        if minimum < 0 or maximum > 1023:
            self.is_channel = False
            print("It looks like the data consists of flowjo scale data. Please set the scaling and axis limits yourself.")

    @staticmethod
    def _reduce(data: pd.DataFrame, factor: int=2) -> pd.DataFrame:
        """
        Lowers the resolution of continuous data by dividing it by the factor
            :data: the data to reduce (in place)
            :param factor: resolution rescale factor
            :returns: reduced pd.DataFrame
        """
        if factor == 1:
            return

        if factor == 0:
            raise ValueError("cannot divide by zero")

        def __reduce(column: pd.Series, factor: int):
            if pd.api.types.is_numeric_dtype(column):
                return column.floordiv(factor)
            else:
                return column

        return data.apply(lambda x: __reduce(x, factor), axis="index")

    def reduce(self, factor: int=2) -> None:
        """
        Divides all categorical columns in .data with the factor. 
        If you want the DataFrame to be returned instead use the staticmethod version _reduce().
            :param factor: the integer division factor
        """
        self._data = self._reduce(self.data, factor)
        self.scale_lim[0] = self.scale_lim[0] // factor
        self.scale_lim[1] = self.scale_lim[1] // factor

    @staticmethod
    def _bin(data: pd.DataFrame, x:str, y:str=None, z:str=None, condensor: Callable[pd.Series]=_condensor_mean) -> pd.DataFrame:
        """
        Bins the pandas dataframe in 1(x), 2(x,y) or 3(x,y,z) dimensions.
        The value of the bin will be calculated using all data in that bin using the condensor function.
        The condensor function has to be able to accept categorial and discreet inputs
            :returns: the binned pd.DataFrame
        """
        group_x: List[pd.DataFrame] = [y for x, y in data.groupby(x, as_index=False)]
        
        # Store dtypes for recasting of numeric dtypes
        dtypes = data.dtypes

        binned: pd.DataFrame = None
        if y is None:
            # 1 Dimensional binning
            for i in range(0, len(group_x)):
                # Store x to prevent condensor from changing the value
                temp_x = group_x[i][x].iloc[0]
                group_x[i] = group_x[i].apply(lambda x: condensor(x), axis="index")
                group_x[i][x] = temp_x

            # delist into dataframe
            binned = pd.concat(group_x, axis="columns").T

        elif z is None:
            # 2 Dimensional binning
            for i in range(0, len(group_x)):
                group_y: List[pd.DataFrame] = [y for x, y in group_x[i].groupby(y, as_index=False)]
                for j in range(0, len(group_y)):
                    # store x, y to prevent condensor from changing the value
                    temp_xy = group_y[j][[x, y]].iloc[0]
                    group_y[j] = group_y[j].apply(lambda x: condensor(x), axis="index")
                    group_y[j][[x, y]] = temp_xy

                group_x[i] = pd.concat(group_y, axis="columns").T   
            binned = pd.concat(group_x, axis="index")

        else:
            # 3 Dimensional binning
            for i in range(0, len(group_x)):
                
                group_y: List[pd.DataFrame] = [y for x, y in group_x[i].groupby(y, as_index=False)]
                for j in range(0, len(group_y)):
                    
                    group_z: List[pd.DataFrame] = [y for x, y in group_y[j].groupby(z, as_index=False)]
                    for k in range(0, len(group_z)):
                        # store x,y,z to prevent condensor from changing the value
                        temp_xyz = group_z[k][[x, y, z]].iloc[0]
                        group_z[k] = group_z[k].apply(lambda x: condensor(x), axis="index")
                        group_z[k][[x, y, z]] = temp_xyz
                        
                    group_y[j] = pd.concat(group_z, axis="columns").T

                group_x[i] = pd.concat(group_y, axis="index")
    
            binned = pd.concat(group_x, axis="index")

        # restore numeric dtypes
        # The concat + transform removes numeric dtypes. Recast
        for column in binned.columns:
            # Only try to cast the originally numeric columns
            if not pd.api.types.is_numeric_dtype(dtypes[column]):
                continue

            try:
                binned[column] = pd.to_numeric(binned[column])
            except ValueError:
                pass

        return binned

    def bin(self, x:str, y:str=None, z:str=None, condensor: Callable[pd.Series]=_condensor_mean) -> None:
        """
        Splits the dataframe in x (and if applicable y, z). Each unique x(,y,z) split will be condensed
        into a single datapoint using the condensor functor. 
        If you want the DataFrame to be returned instead use the staticmethod version _bin().
            :param factor: the integer division factor
        """
        self._data = self._bin(self.data, x=x, y=y, z=z, condensor=condensor)
