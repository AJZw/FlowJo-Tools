##############################################################################     ##    ######
#    A.J. Zwijnenburg                   2020-09-24           v1.4                 #  #      ##
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

:class: Plotter
The main plotting class. Provides an interface for the convenient plotting of scatter and rasterized plots.
Rasterization is slow as proper statistics can be calculated per bin

"""

from __future__ import annotations

from .data import _Abstract
from .transform import Linear, Biex, Log
from PIL import Image
import pandas as pd
import numpy as np
import plotnine as p9
import matplotlib.pyplot as plt
import matplotlib
import os
import io
import copy

p9.options.figure_size=(7.0, 7.0)

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

def _condensor_blank(column: pd.Series) -> Any:
    """
    Example condensor function for the _bin function. Must accept a row pd.Series and return a single value
    Returns the first entree of the series, no manipulation is done.
        :param column: the input data, will be a column, because of 'apply(axis="index")'
        :returns: first entree of the series
    """
    return column.iloc[0]

class Plotter():
    """
    Main plotting class. Load it with data and ask it to generate plots from that data.
    It will use (mainly) plotnine plots
        :param data: FlowJo data
    """
    def __init__(self, data: Union[pd.DataFrame, _Abstract]):
        self.name: str=None
        self._data: pd.DataFrame=None

        self.linewidth_border: float=4
        self.linewidth_center: float=2
        self.arrow_width_border: float=0.15
        self.arrow_width_center: float=0.13
        self.color_na: str="#E3256B"
        self.color_map: str="magma"
        self.fill_na: str="#E3256B"
        self.fill_map: str="magma"
        self.is_channel: bool=True
        self.scale_lim: Tuple[int,int]=[0, 1023]

        self.scales: Dict[str, _Scale]={
            "FSC-A":Linear(start=0, end=262144),
            "FSC-W":Linear(start=0, end=262144),
            "FSC-H":Linear(start=0, end=262144),
            "SSC-A":Linear(start=0, end=262144),
            "SSC-W":Linear(start=0, end=262144),
            "SSC-H":Linear(start=0, end=262144),
            "Time":Linear(start=0, end=262144)
        }
        self.labels: Dict[str, str] = {
            "__sample":"sample"
        }
        self.levels: Dict[str, Dict[str,str]] = {}
        self.metadata: Dict[str, Any] = {}

        if isinstance(data, _Abstract):
            self.name = os.path.basename(data.path)
            self._data = data.data
            self._check_scale()
            # Add default labels
            for column in self._data.columns:
                if column in self.labels:
                    pass
                else:
                    self.labels[column] = column
    
        elif isinstance(data, pd.DataFrame):
            self._data = data
            self._check_scale()
            for column in self._data.columns:
                if column in self.labels:
                    pass
                else:
                    self.labels[column] = column

        else:
            raise ValueError("plot must be instantiate with a pd.DataFrame or flowjo.data._Abstract class")
    
    @property
    def data(self) -> pd.DataFrame:
        """
        Getter for the internal data dataframe
        """
        return self._data

    def _check_scale(self):
        """
        Checks whether the data is channel data (you cannot be 100% sure, but it will give an indication)
        """
        def find_min(column):
            if pd.api.types.is_bool_dtype(column):
                return 0
            elif pd.api.types.is_numeric_dtype(column):
                output = column.min(skipna=True)
            else:
                return 0
        
        def find_max(column):
            if pd.api.types.is_bool_dtype(column):
                return 0
            elif pd.api.types.is_numeric_dtype(column):
                output = column.max(skipna=True)
            else:
                return 0

        minimum = min(self.data.apply(find_min, axis="index"))
        maximum = max(self.data.apply(find_max, axis="index"))

        if minimum < 0 or maximum > 1023:
            self.is_channel = False
            print("It looks like the data consists of flowjo scale data. Please set the scaling and axis limits yourself.")

    ## abstract plotting functions

    def _plot_base(self, data: pd.Dataframe, x: str, y: str, color: str=None, fill: str=None) -> p9.ggplot:
        """
        Creates the data base for all plots
            :param data: the data table with all necessary plotting information. Assumes this is a deepcopy!
            :param x: the x-axis parameter
            :param y: the y-axis parameter
            :param color: for solid object the fill, for non-solid object the outline parameter
            :param fill: for non-solid objects the fill parameter
            :returns: the plot data base
            :raises ValueError: if parameters could not be found in .data
        """
        if id(data) == id(self.data):
            raise ValueError("make sure to call _plot_base with a deepcopy of data")

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
        
        # relevels the data
        for column in data.columns:
            if column in self.levels:
                data[column] = data[column].apply(lambda x: self.levels[column][x])

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
            legend_title=p9.element_text(ha="left"),
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
            :param plot: the plot to add the labels to
            :param title: (optional) overwrite the standard title
            :param x: (optional) overwrite the standard x label
            :param y: (optional) overwrite the standard y label
        """
        if title:
            plot = plot + p9.ggtitle(
                title
            )
        elif self.name:
            plot = plot + p9.ggtitle(
                self.name
            )

        if x is None:
            x = self.labels[plot.mapping["x"]]
        if y is None:
            y = self.labels[plot.mapping["y"]]

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
                plot = plot + p9.coords.coord_cartesian()
            else:
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
                plot = plot + p9.coords.coord_cartesian()
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

        if pd.api.types.is_numeric_dtype(plot.data[color]) and not pd.api.types.is_bool_dtype(plot.data[color]):
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

        elif pd.api.types.is_categorical_dtype(plot.data[color]) or pd.api.types.is_string_dtype(plot.data[color]) or pd.api.types.is_bool_dtype(plot.data[color]):
            # Discrete
            
            # the tab10 & 20 discrete colorscales
            tab10 = ["#1f77b4","#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            tab20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]

            levels = plot.data[color].unique()

            if color_map:
                # Check if colormap covers all cases
                for level in levels:
                    if level not in color_map:
                        # ignore np.nans, handled by plotnine
                        if pd.isnull(level):
                            pass
                        else:
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

        plot = plot + p9.labs(color=self.labels[color])

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

        if pd.api.types.is_numeric_dtype(plot.data[fill]) and not pd.api.types.is_bool_dtype(plot.data[fill]):
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

        elif pd.api.types.is_categorical_dtype(plot.data[fill]) or pd.api.types.is_string_dtype(plot.data[fill]) or pd.api.types.is_bool_dtype(plot.data[fill]):
            # Discrete
            
            # the tab10 & 20 discrete colorscales
            tab10 = ["#1f77b4","#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            tab20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]

            levels = plot.data[fill].unique()

            if fill_map:
                # Check if fill_map covers all cases
                for level in levels:
                    if level not in fill_map:
                        # ignore np.nans, handled by plotnine
                        if pd.isnull(level):
                            pass
                        else:
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

        plot = plot + p9.labs(fill=self.labels[fill])

        return plot

    def _plot_pca_loadings(self, plot: p9.ggplot, labels: bool=True):
        """
        Adds pca loadings to the plot
            :param plot: the plot to add the vectors to
            :param labels: whether to add labels to the vector
        """
        x = plot.mapping["x"]
        y = plot.mapping["y"]

        if "__pca_loadings" not in self.metadata:
            raise ValueError("please run .add_pca first. No loadings to add to the plot")

        data = copy.deepcopy(self.metadata["__pca_loadings"][[x, y]])
        data["__x"] = 0.0
        data["__y"] = 0.0

        # Scale loadings to make them fit nicely in the plot
        x_scaler = min(abs(min(plot.data[x])), abs(max(plot.data[x])))
        y_scaler = min(abs(min(plot.data[y])), abs(max(plot.data[y])))
        
        data[x] *= x_scaler * 1.5
        data[y] *= y_scaler * 1.5

        # Plot two lines, one for border color, one for fill color
        plot = plot + p9.geom_segment(
            mapping=p9.aes(x="__x", xend=x, y="__y", yend=y),
            data=data,
            color="#000000",
            size=self.linewidth_border,
            lineend="round",
            inherit_aes=False,
            arrow=p9.geoms.arrow(length=self.arrow_width_border, type="open")
        )
        plot = plot + p9.geom_segment(
            mapping=p9.aes(x="__x", xend=x, y="__y", yend=y),
            data=data,
            color="#FFFFFF",
            size=self.linewidth_center,
            lineend="round",
            inherit_aes=False,
            arrow=p9.geoms.arrow(length=self.arrow_width_center, type="open")
        )
        # Add labels
        if labels:
            # Only offsets
            data["__x_label"] = data[x] * 1.2
            data["__y_label"] = data[y] * 1.2
            data["__label"] = [self.labels[x] for x in data.index]

            plot = plot + p9.geom_text(
                mapping=p9.aes(label="__label", x="__x_label", y="__y_label"),
                data=data,
                inherit_aes=False,
                color="#000000",
                va="center",
                ha="center",
                fontweight="bold"
            )

        return plot

    ## plot implementations

    def scatter_pca(self, x: str, y: str, c: str, c_map: dict=None, loadings: bool=True, labels: bool=True) -> p9.ggplot:
        """
        Convenience wrapper around scatter plot for the plotting of pca plots. Make sure you have ran add_pca() first.
            :param x: the x dimension
            :param y: the y dimension
            :param c: the c dimension - used for color mapping
            :param c_map: only used for factorized color parameters. Uses the c_map to map the levels
            :param loadings: whether to plot the loadings
            :param labels: whether to plot the loading labels
        """
        plot = self.scatter(x, y, c, c_map)

        if loadings:
            plot = self._plot_pca_loadings(plot, labels)

        return plot

    def scatter(self, x: str, y: str, c: str, c_map: dict=None) -> p9.ggplot:
        """
        Creates a ggplot dotplot object with the correct data and axis
            :param x: the x dimension
            :param y: the y dimension
            :param c: the c dimension - used for color mapping
            :param c_map: only used for factorized color parameters. Uses the c_map to map the levels
        """
        data = copy.deepcopy(self.data[[x, y, c]])

        # Randomize data order
        data = data.sample(frac=1)

        plot = self._plot_base(data, x, y, color=c)
        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, title=self.name)
        plot = self._plot_scale(plot, xlim=self.scale_lim, ylim=self.scale_lim)
        plot = self._plot_colorscale(plot, rescale=True, color_map=c_map)
        plot = plot + p9.geom_point(na_rm="False")

        return plot

    def raster(self, x: str, y: str, c: str, c_stat: str="density", bin_size: int=4, c_map: dict=None) -> p9.ggplot:
        """
        Builds a raster plot of the specified data
            :param x: the x dimension
            :param y: the y dimension
            :param c: the c dimension - the parameter used for color mapping
            :param c_stat: the c statistic to calculate choose from ["density", "max", "min", "mean", "blank"]
            :param bin_size: effectively the size of the raster squares
            :param c_map: only used for categorical color parameters. Uses the c_map to map the levels
        """
        if c_stat not in ["density", "max", "min", "mean", "blank"]:
            raise ValueError(f"raster plotting has no implementation for c_stat '{c_stat}'")

        # Correct source data
        data = copy.deepcopy(self.data[[x, y, c]])
        data = self._reduce(data, bin_size)
        if c_stat == "density":
            data = self._bin(data, x, y, z=None, condensor=_condensor_density)
        elif c_stat == "max":
            data = self._bin(data, x, y, z=None, condensor=_condensor_max)
        elif c_stat == "min":
            data = self._bin(data, x, y, z=None, condensor=_condensor_min)
        elif c_stat == "mean":
            data = self._bin(data, x, y, z=None, condensor=_condensor_mean)
        elif c_stat == "blank":
            data = self._bin(data, x, y, z=None, condensor=_condensor_blank)

        data["__xmax"] = data[x] + 1
        data["__ymax"] = data[y] + 1

        # calculate new limits
        limits = copy.deepcopy(self.scale_lim)
        limits[0] = limits[0] / bin_size
        limits[1] = limits[1] / bin_size

        # build title
        if self.name:
            title = f"{self.name}: {c_stat}({c})"
        else:
            title = f"{c_stat}({c})"

        # Build the plot
        plot = self._plot_base(data, x, y, fill=c)
        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, title=title)
        plot = self._plot_scale(plot, xlim=limits, ylim=limits)
        plot = self._plot_fillscale(plot, rescale=True, fill_map=c_map)
        plot = plot + p9.geom_rect(
            data=plot.data,
            mapping=p9.aes(
                xmin=x,
                xmax="__xmax",
                ymin=y,
                ymax="__ymax"
            )
        )

        return plot

    def raster_3d(self, x: str, y: str, z: str, c: str, c_stat: str="density", xy_bin: int=4, z_bin: int=64) -> List[p9.ggplot]:
        """
        Creates a z-stack of x-y plots with z_start fill. If saved as gif give a 3dimensional representation 
            :param x: the x dimension
            :param y: the y dimension
            :param z: the z dimension - the parameter used for z-stack formation
            :param c: the c(olor) dimension - the parameter used for color mapping
            :param c_stat: the c statistic to calculate choose from ["density", "max", "min", "mean", "blank"]
            :param xy_bin: effectively the size of the raster squares, argument to _reduce()
            :param z_bin: determines the z-stack size, argument to _reduce()
        """
        if c_stat not in ["density", "max", "min", "mean", "blank"]:
            raise ValueError(f"raster plotting has no implementation for c_stat '{c_stat}'")

        if not pd.api.types.is_numeric_dtype(self.data[z]):
            raise ValueError(f"z '{z}' must be a numeric dtype to allow for raster_3d plotting")

        # Correct source data
        data = copy.deepcopy(self.data[[x, y, z, c]])

        # Custom reducing:
        data[x] = data[x].floordiv(xy_bin)
        data[y] = data[y].floordiv(xy_bin)
        data[z] = data[z].floordiv(z_bin)

        # Calculate new limits
        limits = copy.deepcopy(self.scale_lim)
        limits[0] = limits[0] // xy_bin
        limits[1] = limits[1] // xy_bin

        # 3 dimensional binning
        if c_stat == "density":
            data = self._bin(data, x, y, z=z, condensor=_condensor_density)
        elif c_stat == "max":
            data = self._bin(data, x, y, z=z, condensor=_condensor_max)
        elif c_stat == "min":
            data = self._bin(data, x, y, z=z, condensor=_condensor_min)
        elif c_stat == "mean":
            data = self._bin(data, x, y, z=z, condensor=_condensor_mean)
        elif c_stat == "blank":
            data = self._bin(data, x, y, z=z, condensor=_condensor_blank)
        
        # Add necessary rect information
        data["__xmax"] = data[x] + 1
        data["__ymax"] = data[y] + 1

        # Calculate color scale
        quantiles = data[c].quantile([0.0, 0.02, 0.98, 1.0])
        if True:
            min_color = quantiles[0.02]
            max_color = quantiles[0.98]
        else:
            min_color = quantiles[0.0]
            max_color = quantiles[1.0]

        # build title
        if self.name:
            title = f"{self.name}\n{z}[{i+1}/{len(z_stack)}] : {c_stat}({c})"
        else:
            title = f"{z}[{i+1}/{len(z_stack)}] : {c_stat}({c})"

        # Group based on z
        z_stack: List[pd.DataFrame] = [y for x, y in data.groupby(z, as_index=False)]
        plots: List[p9.ggplot] = []
        for i, frame in enumerate(z_stack):
            plot = self._plot_base(frame, x, y, fill=c)
            plot = self._plot_theme(plot)
            plot = self._plot_labels(plot, title=title)
            plot = self._plot_scale(plot, xlim=limits, ylim=limits)
            # force equal colorscale between frames by custom setting of limits
            plot = plot + p9.scales.scale_fill_cmap(
                cmap_name=self.fill_map,
                limits=(min_color, max_color),
                guide=p9.guide_colorbar(
                    ticks=False
                ),
                na_value=self.fill_na
            )
            plot = plot + p9.geom_rect(
                p9.aes(
                    xmin=x,
                    xmax="__xmax",
                    ymin=y,
                    ymax="__ymax"
                )
            )
            plots.append(plot)

        return plots

    def show_3d(self, x: str, y: str, z: str, c: str=None, c_stat="mean", bin_size: int=4, c_map: dict=None) -> None:
        """
        Creates a 3dimensional matplotlib figure object with the correct data and axis
            :param x: the x dimension
            :param y: the y dimension
            :param c: the c dimension - used for color mapping
            :param c_stat: the c statistic to calculate choose from ["density", "max", "min", "mean", "blank"]
            :param bin_size: effectively the size of the raster squares. by bin_size >12 you will start loosing accuracy on the scales.
            :param c_map: only used for factorized color parameters. Uses the c_map to map the levels
        """
        # Do some basic checks that normally would happen in _plot
        if not (self.data.columns == x).any():
            raise ValueError(f"x '{x}' does not specify columns in .data")
        if not (self.data.columns == y).any():
            raise ValueError(f"y '{y}' does not specify columns in .data")
        if not (self.data.columns == z).any():
            raise ValueError(f"z '{z}' does not specify columns in .data")
        if c and not (self.data.columns == c).any():
            raise ValueError(f"c '{c}' does not specify columns in .data")

        if not pd.api.types.is_numeric_dtype(self.data[x]):
            raise ValueError(f"x '{x}' must be a numeric dtype")
        if not pd.api.types.is_numeric_dtype(self.data[y]):
            raise ValueError(f"y '{y}' must be a numeric dtype")
        if not pd.api.types.is_numeric_dtype(self.data[z]):
            raise ValueError(f"z '{z}' must be a numeric dtype")
        
        if c_stat not in ["density", "max", "min", "mean", "blank"]:
            raise ValueError(f"binning has no implementation for c_stat '{c_stat}'")

        # Data manipulation here
        # Correct source data
        if not c:
            data = copy.deepcopy(self.data[[x, y, z]])
        elif c in (x,y,z):
            data = copy.deepcopy(self.data[[x, y, z]])
        else:
            data = copy.deepcopy(self.data[[x, y, z, c]])

        data = self._reduce(data, bin_size)

        # Calculate new limits
        limits = copy.deepcopy(self.scale_lim)
        limits[0] = limits[0] // bin_size
        limits[1] = limits[1] // bin_size

        # 3 dimensional binning
        if c_stat == "density":
            data = self._bin(data, x=x, y=y, z=z, condensor=_condensor_density)
        elif c_stat == "max":
            data = self._bin(data, x=x, y=y, z=z, condensor=_condensor_max)
        elif c_stat == "min":
            data = self._bin(data, x=x, y=y, z=z, condensor=_condensor_min)
        elif c_stat == "mean":
            data = self._bin(data, x=x, y=y, z=z, condensor=_condensor_mean)
        elif c_stat == "blank":
            data = self._bin(data, x=x, y=y, z=z, condensor=_condensor_blank)

        # work-around to allow for the plotting of x,y,z defined colors
        if c in (x,y,z):
            data["__c"] = data[c]
            c = "__c"
            cmap = "nipy_spectral"
        elif not c:
            c = "__c"
            data["__c"] = data[x]
            cmap = "nipy_spectral"
        else:
            cmap = "viridis"

        # manually set colors to allow for proper rescaling
        if pd.api.types.is_numeric_dtype(data[c]):
            quantiles = data[c].quantile([0.0, 0.02, 0.98, 1.0])
            if True:
                min_color = quantiles[0.02]
                max_color = quantiles[0.98]
            else:
                min_color = quantiles[0.0]
                max_color = quantiles[1.0]
            ratio_color = 1 / (max_color - min_color)

            colormap = plt.get_cmap(cmap)
            data[c] = data[c].apply(lambda x: (x - min_color) * ratio_color)
            data[c] = data[c].apply(lambda x: colormap(0 if x < 0 else (0.9999999 if x >= 1 else x), alpha=1))
        elif pd.api.types.is_string_dtype(data[c]):

            levels = data[c].unique()
            levels = levels[~pd.isnull(levels)]
            if c_map:
                # Check if colormap covers all cases
                for level in levels:
                    if level not in c_map:
                        raise ValueError(f"level '{level}' undefined in c_map")
                c_map["nan"] = self.color_na
                data[c] = data[c].apply(lambda x: self.color_na if pd.isnull(x) else c_map[x])

            elif len(levels) <= 10:
                c_map = plt.get_cmap("tab10")
                c_map = dict(zip(levels, c_map.colors[:len(levels)]))
                c_map["nan"] = self.color_na
                data[c] = data[c].apply(lambda x: self.color_na if pd.isnull(x) else c_map[x])
               
            else:
                # Use default
                pass

        # Approximate dot size
        dot_size = bin_size //4
        dot_size = 1 if dot_size < 1 else dot_size

        # construct matplotlib figure and axes objects
        figure = plt.figure(figsize=(12.8, 9.6))
        axes = figure.add_subplot(111, projection="3d", facecolor="#EEEEEEFF")
        axes.scatter(
            xs=data[x],
            ys=data[y],
            zs=data[z],
            c=data[c],
            zdir="y",
            depthshade=True,    # dont turn off - bug in matplotlib
            marker="s",
            s=dot_size,
            alpha=1
        )

        # Set axis ticks / scale / labels
        axes.set_xlim(limits)
        axes.set_ylim(limits)
        axes.set_zlim(limits)

        try:
            axis_scale = self.scales[x]
        except ValueError:
            pass
        else:
            # so apparently a specific plot-x can only have a single label
            major_ticks = np.array(axis_scale.major_ticks(limits[0], limits[1]))
            unique = np.unique(major_ticks, return_index=True)[1]
            labels = np.array(axis_scale.labels(limits[0], limits[1]))
            axes.set_xticks(ticks=major_ticks[unique], minor=False)
            axes.set_xticklabels(labels=labels[unique])
            axes.set_xticks(ticks=axis_scale.minor_ticks(limits[0], limits[1]), minor=True)
        axes.set_xlabel(self.labels[x])
        
        # Somehow y <-> z axis are swapped, correct for this
        try:
            axis_scale = self.scales[y]
        except ValueError:
            pass
        else:
            major_ticks = np.array(axis_scale.major_ticks(limits[0], limits[1]))
            unique = np.unique(major_ticks, return_index=True)[1]
            labels = np.array(axis_scale.labels(limits[0], limits[1]))
            axes.set_zticks(ticks=major_ticks[unique], minor=False)
            axes.set_zticklabels(labels=labels[unique])
            axes.set_zticks(ticks=axis_scale.minor_ticks(limits[0], limits[1]), minor=True)
        axes.set_zlabel(self.labels[y])
        
        try:
            axis_scale = self.scales[z]
        except ValueError:
            pass
        else:
            major_ticks = np.array(axis_scale.major_ticks(limits[0], limits[1]))
            unique = np.unique(major_ticks, return_index=True)[1]
            labels = np.array(axis_scale.labels(limits[0], limits[1]))
            axes.set_yticks(ticks=major_ticks[unique], minor=False)
            axes.set_yticklabels(labels=labels[unique])
            axes.set_yticks(ticks=axis_scale.minor_ticks(limits[0], limits[1]), minor=True)
        axes.set_ylabel(self.labels[z])

        # theming
        if self.name:
            axes.set_title(self.name)
        
        axes.grid(False)
        #dont think these parameters work... :(
        #axes.set_tick_params(which="major", direction="out", width=2.0, lenght=4.0)
        #axes.set_tick_params(which="minor", direction="out", width=1.0, length=2.0)
        axes.view_init(elev=0,azim=-90)
       
        plt.show()

    ## algorithms

    @staticmethod
    def _reduce(data: pd.DataFrame, factor: int=2) -> pd.DataFrame:
        """
        Lowers the resolution of continuous data by dividing it by the factor
            :data: the data to reduce (in place)
            :param factor: resolution rescale factor
            :returns: reduced pd.DataFrame
        """
        # Even run if factor == 1, non-integer input data needs to be 'integerized' for proper binning

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
    def _bin(data: pd.DataFrame, x: str, y: str=None, z: str=None, condensor: Callable[pd.Series]=_condensor_mean) -> pd.DataFrame:
        """
        Bins the pandas dataframe in 1(x), 2(x,y) or 3(x,y,z) dimensions.
        The value of the bin will be calculated using all data in that bin using the condensor function.
        The condensor function has to be able to accept categorial and discreet inputs
            :param x: the x-dimension
            :param y: (optional) the y-dimension
            :param z: (optional) the z-dimension
            :param condensor: the function to condense a bin into a single value
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

    def bin(self, x: str, y: str=None, z: str=None, condensor: Callable[pd.Series]=_condensor_mean) -> None:
        """
        Splits the dataframe in x (and if applicable y, z). Each unique x(,y,z) split will be condensed
        into a single datapoint using the condensor functor. 
        If you want the DataFrame to be returned instead use the staticmethod version _bin().
            :param x: the x-dimension
            :param y: (optional) the y-dimension
            :param z: (optional) the z-dimension
            :param condensor: the function to condense a bin into a single value
        """
        self._data = self._bin(self.data, x=x, y=y, z=z, condensor=condensor)

    def add_umap(self, parameters: List[str], q: Tuple[float, float]=(0.05, 0.95)) -> None:
        """
        Calculates the Uniform Manifold Approximation and Projection (UMAP) axis
        Adds the value of each datapoint under the column names "UMAP1" and "UMAP2"
            :param parameters: the parameters to use for umap calculation
            :param q: the quantile range to use for centering and scaling of the data (q_min, q_max)
        """
        for param in parameters:
            if not (param == self.data.columns).any():
                raise ValueError(f"parameter '{param}' does not specify a column in .data")

        # Scale data using quantiles to limit influence of outliers and to account for
        # a mixed gaussian distribution of the sample data

        # Scale data per sample
        data_samples = [y for x, y in self.data.groupby("__sample")]

        plots = []
        for sample in data_samples:
            # Standard/RobustScaler - they centralize based on mean/median. In flowcytometry data
            # we cannot assume that the distributions (generalize as a mix of two gaussians) shows equal/between sample
            # comparable sizes of the two guassians components. 
            # Secondly the data can have a lot of outliers, so scaling based on quantile will be more robust.
            # The mean of the quantiles will likely give a distribution agnostic centralisation.
            # Other option would be the geoMean, but that is quite influenced by the outliers

            # scale
            quantiles = sample[parameters].quantile(q=q)
            
            sample[parameters] = (sample[parameters] - quantiles.loc[q[0]]) / (quantiles.loc[q[1]] - quantiles.loc[q[0]])

            # Center
            quantiles = sample[parameters].quantile(q=q)
            q_mean = quantiles.mean()
            
            sample[parameters] = sample[parameters] - q_mean

        scaled_data = pd.concat(data_samples)

        import umap
        reducer = umap.UMAP()
        data_umap = pd.DataFrame(reducer.fit_transform(scaled_data[parameters]))

        # umap output data is in identical order to input data
        data_umap.columns = ["__UMAP1", "__UMAP2"]

        self.data["UMAP1"] = data_umap["__UMAP1"]
        self.data["UMAP2"] = data_umap["__UMAP2"]
        self.parameter_labels["UMAP1"] = "UMAP1"
        self.parameter_labels["UMAP2"] = "UMAP2"

    def add_tsne(self) -> None:
        """
        Calculates the t-distributed Stochastic Neighbouring Embedding axis
        Adds the value of each datapoint under the column names "tSNE1" and "tSNE2"
        """
        raise NotImplementedError("tSNE has yet to be implemented")

    def add_pca(self, parameters: List[str], q: Tuple[float, float]=(0.05, 0.95)):
        """
        Calculates the Principle Componet axis
        Adds the value of each datapoint loading under the column names "PCn" (replace n with number of PC)
            :param parameters: the parameters to use for umap calculation
            :param q: the quantile range to use for centering and scaling of the data (q_min, q_max)
        """
        for param in parameters:
            if not (param == self.data.columns).any():
                raise ValueError(f"parameter '{param}' does not specify a column in .data")

        # Scale data using quantiles to limit influence of outliers and to account for
        # a mixed gaussian distribution of the sample data

        # Scale data per sample
        data_samples = [y for x, y in self.data.groupby("__sample")]

        plots = []
        for sample in data_samples:
            # Standard/RobustScaler - they centralize based on mean/median. In flowcytometry data
            # we cannot assume that the distributions (generalize as a mix of two gaussians) shows equal/between sample
            # comparable sizes of the two guassians components. 
            # Secondly the data can have a lot of outliers, so scaling based on quantile will be more robust.
            # The mean of the quantiles will likely give a distribution agnostic centralisation.
            # Other option would be the geoMean, but that is quite influenced by the outliers

            # scale
            quantiles = sample[parameters].quantile(q=q)
            
            sample[parameters] = (sample[parameters] - quantiles.loc[q[0]]) / (quantiles.loc[q[1]] - quantiles.loc[q[0]])

            # Center
            quantiles = sample[parameters].quantile(q=q)
            q_mean = quantiles.mean()
            
            sample[parameters] = sample[parameters] - q_mean

        scaled_data = pd.concat(data_samples)

        import sklearn
        n_components = 4# len(parameters)
        pca = sklearn.decomposition.PCA(n_components=n_components)
        pca.fit(scaled_data[parameters])

        # Calculate transformation for sample plotting
        pca_data = pd.DataFrame(pca.transform(scaled_data[parameters]))
        pca_data.index = self.data.index
        pca_data.columns = [f"PC{i+1}" for i in range(0, n_components)]

        self._data[pca_data.columns] = pca_data
        self.labels.update({f"PC{i+1}":f"PC{i+1}: {round(x*100,2)}%" for i, x in enumerate(pca.explained_variance_)})

        # Store the components for vector plotting
        pca_vector = pd.DataFrame(pca.components_)
        pca_vector.index = [f"PC{i+1}" for i in range(0, n_components)]
        pca_vector.columns = parameters
        self.metadata["__pca_loadings"] = pca_vector.T

    ## saving

    def save_gif(self, path, x: str, y: str, z: str, c: str):
        """
        Saves a raster_3d of x-y with a z-stack and density of c
            :param path: the save directory (file name is generated automatically)
            :param x: the x dimension
            :param y: the y dimension
            :param z: the z dimension - the parameter used for z-stack formation
            :param c: the c(olor) dimension - the parameter used for color mapping
            :raises ValueError: if function cannot be completed
        """
        if path and not os.path.isdir(path):
            raise ValueError(f"path '{path}' doesnt point to existing directory")

        # Temporarily turn off plot view
        plt.ioff()
        
        plots = self.raster_3d(x=x, y=y, z=z, c=c, c_stat="density", xy_bin=4, z_bin=64)

        images = []
        buffers = []
        for plot in plots:
            figure = plot.draw()
            buffers.append(io.BytesIO())
            figure.savefig(buffers[-1], format="png")
            buffers[-1].seek(0)
            images.append(Image.open(buffers[-1]))

        images[0].save(
            os.path.join(path, f"{x}_{y}_{z}.gif"),
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0
        )

        for buffer in buffers:
            buffer.close()

        # Close all in the background drawn plots and renable plotview
        plt.close("all")
        plt.show()

    def save_png(self, path, x:str, y:str, c:str, c_stat:str="density", c_map:Dict[str, str]=None):
        """
        Saves a raster of x-y with color coding of the c_stat of c
            :param path: the save directory (file name is generated automatically)
            :param x: the x dimension
            :param y: the y dimension
            :param c: the c(olor) dimension - the parameter used for color mapping
            :param c_stat: the color statistic to plot
            :param c_map: only used for categorical color parameters. Uses the color_map to map the levels
            :raises ValueError: if function cannot be completed
        """
        if path and not os.path.isdir(path):
            raise ValueError(f"path '{path}' doesnt point to existing directory")

        plt.ioff()

        plot = self.raster(x=x, y=y, c=c, c_stat=c_stat, color_map=c_map)

        plot.save(
            filename=f"{x}_{y}_{c_stat}[{c}].png",
            format="png",
            path=path,
            verbose=False
        )

        plt.close("all")
        plt.show()
