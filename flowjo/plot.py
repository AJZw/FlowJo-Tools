##############################################################################     ##    ######
#    A.J. Zwijnenburg                   2021-07-07          v1.15                 #  #      ##
#    Copyright (C) 2023 - AJ Zwijnenburg          GPLv3 license                  ######   ##
##############################################################################  ##    ## ######

## Copyright notice ##########################################################
# FlowJo Tools provides a python API into FlowJo's .wsp files.
# Copyright (C) 2023 - AJ Zwijnenburg
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
Channel data is pre-scaled by FlowJo and binned to the values 0-1023
As the scaling information is not exported the plot class cannot provide a default.
Therefore it is beneficial to set Plot.transforms for proper results
"""

from __future__ import annotations
from typing import Any, List, Dict, Union, Tuple, Optional

import io
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import scipy

from . import transform
from .data import _Abstract

# For screen
p9.options.figure_size=(7.0, 7.0)
# For PowerPoint
#p9.options.figure_size=(3.5, 3.5)

## Static functions
def save_raster(plot: Union[p9.ggplot, matplotlib.figure.Figure], name:str, path:str="") -> None:
    """
    Saves a plot object as raster image (.png)

    Args:
        plot: the plot object
        name: the save name of the plot
        path: the path to the save directory

    Raises:
        ValueError: when plot type is not supported

    Note:
        To prevent loads of pop-ups, this function clears matplotlib.pyplot interactive mode
    """
    if path and not os.path.isdir(path):
        raise ValueError(f"path '{path}' doesnt point to existing directory")

    temp = p9.options.figure_size
    p9.options.figure_size=(3.5, 3.5)

    # Temporarily turn off plot view
    plt.ioff()
    
    if isinstance(plot, p9.ggplot):
        p9.ggsave(plot, os.path.join(path, f"{name}.png"), dpi=600)
    elif isinstance(plot, matplotlib.figure.Figure):
        plot.savefig(os.path.join(path, f"{name}.png"), dpi=600)
    else:
        raise ValueError(f"unsupported class '{plot.__class__.__name__}'")

    # Close all in the background drawn plots and renable plotview
    plt.close("all")
    plt.show()

    p9.options.figure_size=temp

def save_vector(plot: Union[p9.ggplot, matplotlib.figure.Figure], name:str, path:str="") -> None:
    """
    Saves a plot object as vector image (.svg)

    Args:
        plot: the plot object
        name: the save name of the plot
        path: the path to the save directory

    Raises:
        ValueError: when plot type is not supported

    Note:
        To prevent loads of pop-ups, this function clears matplotlib.pyplot interactive mode
    """
    if path and not os.path.isdir(path):
        raise ValueError(f"path '{path}' doesnt point to existing directory")

    temp = p9.options.figure_size
    p9.options.figure_size=(3.5, 3.5)

    # Temporarily turn off plot view
    plt.ioff()

    if isinstance(plot, p9.ggplot):
        p9.ggsave(plot, os.path.join(path, f"{name}.svg"), dpi=300)
    elif isinstance(plot, matplotlib.figure.Figure):
        plot.savefig(os.path.join(path, f"{name}.svg"), dpi=600)
    else:
        raise ValueError(f"unsupported class '{plot.__class__.__name__}'")


    # Close all in the background drawn plots and renable plotview
    plt.close("all")
    plt.show()

    p9.options.figure_size=temp

def add_polygon(plot: p9.ggplot, gate: pd.DataFrame) -> p9.ggplot:
    """
    Adds the gate polygon to the plot. This function doesnt check the axis-labels, so make sure axis are equivalent.

    Args:
        plot: the ggplot to add the gate polygon to
        gate: the gate coordinates

    Returns:
        the plot + gate polygon

    Raises:
        AttributeError: when plot is missing x and/or y mapping
        ValueError: when gate cannot be added to plot 
    """
    # I have to align the gate to the mapping of the plot
    try:
        x_label = plot.labels["x"]
    except KeyError:
        x_label = None

    try:
        x_mapping = plot.mapping["x"]
    except KeyError:
        raise AttributeError(f"plot doesnt contain a mapping for the x-axis") from None

    if x_label is None:
        x_label = x_mapping

    try:
        y_label = plot.labels["y"]
    except KeyError:
        y_label = None

    try:
        y_mapping = plot.mapping["y"]
    except KeyError:
        raise AttributeError(f"plot doesnt contain a mapping for the y-axis") from None

    if y_label is None:
        y_label = y_mapping

    # dont modify the referenced gate object
    gate = gate.copy()

    if len(gate.columns) == 1:
        if not gate.columns.isin([x_label]).any():
            raise ValueError(f"the mono-dimensional gate with axis '{gate.columns[0]}' cannot be mapped on plot with axis '{x_label}','{y_label}'")
        
        # Add missing column
        if gate.columns[0] == x_label:
            axis_unbound = "y"
        else:
            axis_unbound = "x"

        # I hope this works! I dont have time to test this now
        for scale in plot.scales:
            if axis_unbound == "x":
                if isinstance(scale, p9.scales.scale_x_continuous) or isinstance(scale, p9.scales.scale_x_date) or isinstance(scale, p9.scales.scale_x_datetime) or isinstance(scale, p9.scales.scale_x_discrete) or isinstance(scale, p9.scales.scale_x_log10) or isinstance(scale, p9.scales.scale_x_reverse) or isinstance(scale, p9.scales.scale_x_sqrt) or isinstance(scale, p9.scales.scale_x_timedelta):
                    gate[x_label] = [scale._limits[0], scale._limits[0], scale._limits[1], scale._limits[1], scale._limits[0]]
                    break
            elif axis_unbound == "y":
                if isinstance(scale, p9.scales.scale_y_continuous) or isinstance(scale, p9.scales.scale_y_date) or isinstance(scale, p9.scales.scale_y_datetime) or isinstance(scale, p9.scales.scale_y_discrete) or isinstance(scale, p9.scales.scale_y_log10) or isinstance(scale, p9.scales.scale_y_reverse) or isinstance(scale, p9.scales.scale_y_sqrt) or isinstance(scale, p9.scales.scale_y_timedelta):
                    gate[y_label] = [scale._limits[0], scale._limits[0], scale._limits[1], scale._limits[1], scale._limits[0]]
                    break

    else:
        if not gate.columns.isin([x_label, y_label]).all():
            raise ValueError(f"the gate with axis '{gate.columns[0]}','{gate.columns[1]}' cannot be mapped to a plot with axis '{x_label}','{y_label}'")

    # remap the x-y labels
    if gate.columns[0] == x_label:
        gate.columns = [x_mapping, y_mapping]
    else:
        gate.columns = [y_mapping, x_mapping]

    plot = plot + p9.geom_path(
        mapping=p9.aes(
            x=x_mapping,
            y=y_mapping
        ),
        inherit_aes=False,
        data=gate,
        color="#000000ff",
        size=1.0
        )

    return plot

class Plotter():
    """
    Main plotting class. Load it with data and ask it to generate plots from that data.
    Plotter always assumes the data is provided and stored transformed. 
    
    Args:
        data: FlowJo data or a dataframe

    Attributes:
        TAB10: the tab10 colormap
        TAB20: the tab20 colormap
        DARK2: the dark2 colormap
        name: the title of the generated plots
        line_color_border: the linecolor of arrow borders
        line_color_center: the linecolor of arrow centers
        linewidth_border: the linewidth of the border arrow stem
        linewidth_center: the linewidth of the center arrow stem
        arrow_width_border: the width of the border arrow arrowhead
        arrow_width_center: the width of the center arrow arrowhead
        color_na: the (outline) color of NA events
        color_map: the (outline) color continuous value mapping
        fill_na: the fill color of NA events
        fill_map: the fill color of continuous values
        point_size: the size of a single eventpoint
        is_channel: data-specifyer, should always be channel, scale not implemneted
        transforms: dictionary containing parameter-name transform mapping
        labels: dictionary containing parameter-name to label mapping
        levels: dictionary mapping the levels of a parameter to different label
        metadata: metadata storage
        mask: boolean mask to mask data for removal from plotting; True marks point to keep.
        mask_type: the removal method, eg 'remove' or 'outline'

    Raises:
        ValueError: if the provided data is of incorrect type

    """
    TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    TAB20 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a",  "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]
    DARK2 = ["#1b9e77", "#d95f22", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]

    def __init__(self, data: Union[pd.DataFrame, _Abstract]):
        self.name: str=None
        self._data: pd.DataFrame=None

        self.line_color_border: str="#000000"
        self.line_color_center: str="#ffffff"
        self.linewidth_border: float=4
        self.linewidth_center: float=2
        self.arrow_width_border: float=0.15
        self.arrow_width_center: float=0.13
        self.color_na: str="#e3256b"
        self.color_map: str="magma"
        self.fill_na: str="#e3256b"
        self.fill_map: str="magma"
        self.point_size: float=1.5
        self.is_channel: bool=True

        # Stores the local and global spaces of the data parameters
        self.transforms: Dict[str, transform._Abstract]={
            "FSC-A":transform.Linear(l_start=0, l_end=1023, g_start=0, g_end=262144),
            "FSC-W":transform.Linear(l_start=0, l_end=1023, g_start=0, g_end=262144),
            "FSC-H":transform.Linear(l_start=0, l_end=1023, g_start=0, g_end=262144),
            "SSC-A":transform.Linear(l_start=0, l_end=1023, g_start=0, g_end=262144),
            "SSC-W":transform.Linear(l_start=0, l_end=1023, g_start=0, g_end=262144),
            "SSC-H":transform.Linear(l_start=0, l_end=1023, g_start=0, g_end=262144),
            "Time":transform.Linear(l_start=0, l_end=1023, g_start=0, g_end=262144)
        }

        # The parameter are renamed according to the labels dictionary
        self.labels: Dict[str, str] = {
            "__sample":"sample"
        }

        # The levels of a parameter are renamed according to the levels dictionary
        self.levels: Dict[str, Dict[str,str]] = {}

        # Metadata storage
        self.metadata: Dict[str, Any] = {}

        # Specifies which data point to mask
        self.mask: pd.Series = None
        self.mask_type: str = "remove" # choose from ["remove", "outline"]

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

        Returns:
            the internal data dataframe
        """
        return self._data

    def _check_scale(self) -> None:
        """
        Checks whether the data is channel data (you cannot be 100% sure, but it will give an indication).
        Modifies self.is_channel.
        """
        def find_min(column):
            if pd.api.types.is_bool_dtype(column):
                return 0
            elif pd.api.types.is_numeric_dtype(column):
                output = column.min(skipna=True)
                return output
            else:
                return 0
        
        def find_max(column):
            if pd.api.types.is_bool_dtype(column):
                return 0
            elif pd.api.types.is_numeric_dtype(column):
                output = column.max(skipna=True)
                return output
            else:
                return 0

        minimum = min(self.data.apply(find_min, axis="index"))
        maximum = max(self.data.apply(find_max, axis="index"))

        if minimum < 0 or maximum > 1023:
            self.is_channel = False
            print("It looks like the data contains values outside of flowjo channel data. Please set the scaling and axis limits yourself.")

    ## abstract plotting functions
    def _plot_check(self, data: pd.Dataframe, x: str, y: Optional[str]=None, color: Optional[str]=None, fill: Optional[str]=None) -> None:
        """
        Checks for existance and correctness of the plotting parameters

        Args:
            data: the data table with all necessary plotting information.
            x: the x-axis parameter
            y: the y-axis parameter
            color: for solid object the fill, for non-solid object the outline parameter
            fill: for non-solid objects the fill parameter

        Raises:
            ValueError: if plotting parameters cannot be found or are of incorrect dtype
        """
        if not (data.columns == x).any():
            raise ValueError(f"x '{x}' does not specify columns in .data")

        if not pd.api.types.is_numeric_dtype(data[x]):
            raise ValueError(f"x '{x}' must be a numeric dtype")

        if y:
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

    def _plot_base(self, data: pd.Dataframe, x: str, y: str, color: str=None, fill: str=None) -> p9.ggplot:
        """
        Creates the data base for all plots

        Args:
            data: the data table with all necessary plotting information. Assumes this is a deepcopy!
            x: the x-axis parameter
            y: the y-axis parameter
            color: for solid object the fill, for non-solid object the outline parameter
            fill: for non-solid objects the fill parameter
        
        Returns:
            the plot data base

        Raises:
            ValueError: if parameters could not be found in .data
        """
        if id(data) == id(self.data):
            raise ValueError("make sure to call _plot_base with a deepcopy of data")

        # relevels the data
        for column in data.columns:
            if column in self.levels:
                try:
                    data[column] = data[column].apply(lambda x: self.levels[column][x])
                except KeyError as error:
                    raise ValueError(f"Plotter.levels specifies that column '{column}' has to be releveled. Missing key '{error.args[0]}'")

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

        Args:
            plot: the plot the theme

        Returns:
            themed plot
        """
        plot = plot + p9.theme_bw() + p9.theme(
            text=p9.element_text(family="sans-serif", weight="normal"),
            plot_title=p9.element_text(ha="center", weight="bold", size=14),
            axis_text_x=p9.element_text(ha="center", va="top", color="#000000", weight="bold"),
            axis_text_y=p9.element_text(ha="right", va="center", color="#000000", weight="bold"),
            axis_ticks_major_x=p9.element_line(color="#ffffffff", size=1.5),
            axis_ticks_length_major=4.0,
            axis_ticks_length_minor=2.5,
            axis_ticks_major_y=p9.element_line(color="#ffffffff", size=1.5),
            axis_ticks_minor_x=p9.element_line(color="#ffffffff", size=1.5),
            axis_ticks_minor_y=p9.element_line(color="#ffffffff", size=1.5),
            panel_grid_major_x=p9.element_blank(),
            panel_grid_major_y=p9.element_blank(),
            panel_grid_minor_x=p9.element_blank(),
            panel_grid_minor_y=p9.element_blank(),
            #panel_background=p9.element_rect(fill="#f8f8f8ff", color="#eeeeeeff"),
            panel_background=p9.element_rect(fill="#f8f8f8ff", color="#ffffffff"),
            panel_border=p9.element_rect(fill=None, color="#000000ff", size=1.5),
            legend_title=p9.element_text(ha="left"),
            #legend_key=p9.element_blank(),
            #legend_key_width=8,
            #legend_key_height=35,
            #legend_entry_spacing_x=-10,
            #legend_entry_spacing_y=-20
        )
        return plot

    def _plot_labels(self, plot: p9.ggplot, title: Optional[str]=None, x: Optional[str]=None, y: Optional[str]=None, 
                     color: Optional[str]=None, fill: Optional[str]=None) -> p9.ggplot:
        """
        Adds labels to the plot

        Args:
            plot: the plot to add the labels to
            title: (optional) overwrite the standard title
            x: (optional) overwrite the standard x label
            y: (optional) overwrite the standard y label
            color: (optional) overwrite the standard color label
            fill: (optional) overwrite the standard fill label

        Returns:
            plot with proper labels
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
            try:
                x = self.labels[plot.mapping["x"]]
            except KeyError:
                x = plot.mapping["x"]

        if y is None:
            try:
                y = self.labels[plot.mapping["y"]]
            except KeyError:
                y = plot.mapping["y"]


        plot = plot + p9.labs(
            x=x, 
            y=y
        )

        if color:
            plot = plot + p9.labs(
                color=color
            )

        if fill:
            plot = plot + p9.labs(
                fill=fill
            )

        return plot

    def _plot_scale(self, plot: p9.ggplot, xlim: bool=True, ylim: bool=True, x: Optional[str]=None, y: Optional[str]=None) -> p9.ggplot:
        """
        Adds the scale limits to the plot

        Args:
            plot: plot to add the scale to
            xlim: whether to enable axis scale representation according to x-transform
            ylim: whether to enable axis scale representation according to y-transform
            x: overrides the parameter to base the x-transform/scale on
            y: overrides the parameter to base the y-transform/scale on

        Returns:
            plot with proper scalelimits
        """
        #if not self.is_channel:
        #    raise ValueError("you cannot use _plot_scale on non-channel data")

        # Fetch scale from self.transforms (if available)
        if xlim is True:
            if not x:
                x = plot.mapping["x"]

            try:
                scale_x = self.transforms[x]
            except KeyError:
                plot = plot + p9.coords.coord_cartesian()
            else:
                plot = plot + p9.scale_x_continuous(
                    breaks=scale_x.major_ticks(),
                    minor_breaks=scale_x.minor_ticks(),
                    labels=scale_x.labels(),
                    expand=(0,0),
                    limits=(scale_x.l_start, scale_x.l_end)
                )

        if ylim is True:
            if not y:
                y = plot.mapping["y"]

            try:
                scale_y = self.transforms[y]
            except KeyError:
                plot = plot + p9.coords.coord_cartesian()
            else:
                plot = plot + p9.scale_y_continuous(
                    breaks=scale_y.major_ticks(),
                    minor_breaks=scale_y.minor_ticks(),
                    labels=scale_y.labels(),
                    expand=(0,0),
                    limits=(scale_y.l_start, scale_y.l_end)
                )

        return plot

    def _plot_colorscale(self, plot: p9.ggplot, color_map: Optional[Dict[str, str]]=None, rescale: bool=False) -> p9.ggplot:
        """
        Adds a color scale to the plot. Automatically detects whether the scale is discrete or continuous

        Args:
            color_map: if defined uses this color_map to manually assign colors to discrete values
            rescale: whether to rescale the color_map to have min-max fall on the 2%-98% percentile

        Raises:
            ValueError: if color mapping is undefined

        Returns:
            plot with proper colorscale
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
                    values = self.TAB10,
                    na_value=self.color_na
                )
            elif len(levels) <= 20:
                plot = plot + p9.scales.scale_color_manual(
                    values = self.TAB20,
                    na_value=self.color_na
                )
            else:
                # Use default
                pass   

        else:
            raise ValueError(f"unimplemented colorscale dtype {plot.data[color].dtype}")

        try:
            plot = plot + p9.labs(color=self.labels[color])
        except KeyError:
            plot = plot + p9.labs(color=color)

        return plot
    
    def _plot_fillscale(self, plot: p9.ggplot, fill_map: Optional[Dict[str, str]]=None, rescale: bool=False) -> p9.ggplot:
        """
        Adds a fill scale to the plot. Automatically detects whether the scale is discrete or continuous

        Args:
            fill_map: if defined uses this fill_map to manually assign colors to discrete values
            rescale: whether to rescale the fill_map to have min-max fall on the 2%-98% percentile

        Raises:
            ValueError: if fill mapping is undefined

        Returns:
            plot with proper fillscale
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
                    values = self.TAB10,
                    na_value=self.fill_na
                )
            elif len(levels) <= 20:
                plot = plot + p9.scales.scale_fill_manual(
                    values = self.TAB20,
                    na_value=self.fill_na
                )
            else:
                # Use default
                pass   

        else:
            raise ValueError(f"unimplemented fillscale dtype {plot.data[fill].dtype}")

        try:
            plot = plot + p9.labs(color=self.labels[fill])
        except KeyError:
            plot = plot + p9.labs(color=fill)

        return plot

    def _plot_pca_loadings(self, plot: p9.ggplot, labels: bool=True) -> p9.ggplot:
        """
        Adds pca loadings to the plot
        
        Args:
            plot: the plot to add the vectors to
            labels: whether to add labels to the vector

        Returns:
            plot with added pca loadings arrows
        """
        x = plot.mapping["x"]
        y = plot.mapping["y"]

        if "__pca_loadings" not in self.metadata:
            raise ValueError("please run .add_pca first. No loadings to add to the plot")

        data = self.metadata["__pca_loadings"][[x, y]].copy()
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
            color=self.line_color_border,
            size=self.linewidth_border,
            lineend="round",
            inherit_aes=False,
            arrow=p9.geoms.arrow(length=self.arrow_width_border, type="open")
        )
        plot = plot + p9.geom_segment(
            mapping=p9.aes(x="__x", xend=x, y="__y", yend=y),
            data=data,
            color=self.line_color_center,
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

    def scatter_pca(self, x: str, y: str, c: str, c_map: Optional[Dict[str, str]]=None, loadings: bool=True, labels: bool=True) -> p9.ggplot:
        """
        Convenience wrapper around scatter plot for the plotting of pca plots. Make sure you have ran add_pca() first.

        Args:
            x: the x dimension
            y: the y dimension
            c: the c dimension - used for color mapping
            c_map: only used for factorized color parameters. Uses the c_map to map the levels
            loadings: whether to plot the loadings
            labels: whether to plot the loading labels

        Returns:
            pca scatter plot
        """
        plot = self.scatter(x, y, c, c_map)

        if loadings:
            plot = self._plot_pca_loadings(plot, labels)

        return plot

    def scatter(self, x: str, y: str, c: str, c_map: Optional[Dict[str,str]]=None) -> p9.ggplot:
        """
        Creates a ggplot dotplot object with the correct data and axis

        Args:
            x: the x dimension
            y: the y dimension
            c: the c dimension - used for color mapping
            c_map: only used for factorized color parameters. Uses the c_map to map the levels

        Returns:
            scatter plot

        Raises:
            ValueError: if mask_type is unknown
        """
        self._plot_check(self.data, x, y, c, fill=None)

        params = pd.array([x,y,c]).dropna().unique()
        data = self.data[params].copy()

        # Randomize data order
        data = data.sample(frac=1)

        plot = self._plot_base(data, x, y, color=c)
        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, title=self.name)
        plot = self._plot_scale(plot, xlim=True, ylim=True)

        # to enable the plotting of 'pure' outline graphs
        if self.mask is not None and sum(~self.mask) == 0:
            pass
        else:
            plot = self._plot_colorscale(plot, rescale=True, color_map=c_map)

        if self.mask is not None:
            if self.mask_type == "remove":
                # mask data
                data_mask = plot.data.loc[~self.mask]
                plot = plot + p9.geom_point(
                    data=data_mask,
                    na_rm=False,
                    size=self.point_size,
                )
        
            elif self.mask_type == "outline":
                plot = plot + p9.geom_point(
                    color="#000000",
                    fill="#ffffff",
                    size=self.point_size,
                )
                plot = plot + p9.geom_point(
                    color="#00000000",
                    fill="#ffffffff",
                    size=self.point_size,
                )
                data_mask = data.loc[~self.mask]
                plot = plot + p9.geom_point(
                    data=data_mask,
                    size=self.point_size,
                )
            else:
                raise ValueError(f"unknown mask_type {self.mask_type}, has to be one of ['remove', 'outline']")

        else:
            plot = plot + p9.geom_point(na_rm=False,size=self.point_size)

        return plot

    def raster(self, x: str, y: str, bins: int=256) -> p9.ggplot:
        """
        Builds a density raster plot. This function is much more efficient then raster_special(c_stat="density"). 

        Args:
            x: the x dimension
            y: the y dimension
            bins: the number of bins per dimension

        Returns:
            raster plot

        Raises:
            ValueError: is masking is undefined
        """
        self._plot_check(self.data, x, y, color=None, fill=None)

        # Get source data of unique params
        params = pd.array([x,y]).dropna().unique()
        data = self.data[params].copy()

        if self.mask is not None:
            if self.mask_type != "remove":
                raise ValueError(f"rasterized plots only allow for 'remove' mask_type'")
            data = data.loc[~self.mask]

        # Calculate bins
        data["__x_bin"] = self._bin(x, bins=bins)
        data["__y_bin"] = self._bin(y, bins=bins)

        # Count bins
        data_count = data[["__x_bin", "__y_bin"]].value_counts(sort=False)
        data_count = data_count.reset_index(name="__density")
        data_count["__x_bin"] = data_count["__x_bin"].astype("float64")
        data_count["__y_bin"] = data_count["__y_bin"].astype("float64")

        # Make polygon
        data_count["__x_max"] = data_count["__x_bin"] + ((self.transforms[x].l_end-self.transforms[x].l_start) / bins)
        data_count["__y_max"] = data_count["__y_bin"] + ((self.transforms[y].l_end-self.transforms[y].l_start) / bins)

        # build title
        if self.name:
            title = f"{self.name}: density"
        else:
            title = f"density"

        # Build the plot
        plot = self._plot_base(data_count, "__x_bin", "__y_bin", fill="__density")
        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, title=title, x=x, y=y, fill="density")
        plot = self._plot_scale(plot, xlim=True, ylim=True, x=x, y=y)
        plot = self._plot_fillscale(plot, fill_map=None, rescale=False)
        plot = plot + p9.geom_rect(
            data=plot.data,
            mapping=p9.aes(
                xmin="__x_bin",
                xmax="__x_max",
                ymin="__y_bin",
                ymax="__y_max"
            )
        )

        return plot

    def density_overlay(self, x: str, y: str, c: str, c_level: Optional[str]=None, c_map: Optional[Dict[str,str]]=None, levels: int=15) -> p9.ggplot:
        """
        Builds a density raster plot while overlaying the color parameter as a density map.

        Args:
            x: the x dimension
            y: the y dimension
            c: the c dimension
            c_level: (optional) the c level/category to plot
            c_map: (optional) uses the c_map to map the levels. Only used if c_level is not defined
            levels: the number of geometry levels

        Returns:
            plot with overlayed density lines

        Raises:
            ValueError: when input parameters are incorrect
        """
        self._plot_check(self.data, x=x, y=y, color=c, fill=None)

        if not (pd.api.types.is_categorical_dtype(self.data[c]) or pd.api.types.is_string_dtype(self.data[c]) or pd.api.types.is_bool_dtype(self.data[c])):
            raise ValueError(f"can only generate a density overlay of a categorical category, data in '{c}' is not categorical")

        if c_level is not None and c_level not in self.data[c].unique():
            raise ValueError(f"the level '{c_level}' cannot be found in '{c}'")

        if pd.api.types.is_bool_dtype(self.data[c]):
            if c_level is None:
                c_level = True
            c_name = c
        else:
            if c_level is None:
                c_name = c
            else:
                c_name = c_level        

        # get unique data parameters
        params = pd.array([x,y]).dropna().unique()
        data = self.data[params].copy()
        data = data.sample(frac=1)

        # Make a base of outlined dotplot
        plot = self._plot_base(data, x, y, color=c)
        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, title=self.name)
        plot = self._plot_scale(plot, xlim=True, ylim=True)

        plot = plot + p9.geom_point(
            color="#000000",
            fill="#ffffff"
        )
        plot = plot + p9.geom_point(
            color="#00000000",
            fill="#ffffffff"
        )

        # The kernel 2dimensional density statistics effectively calculate density probabilities
        # The outhermost probability, generally falls outside the viewport. Sometimes it is partially
        # within the viewport though. This shows up as a partially filled 'artefactly' circle. 
        # One can just hide it with the alpha channel but we should put this past the boss

        # Set alpha scale range to (0.0, 1.0) to effectively hide this

        # Apply mask
        if self.mask is not None:
            data = self.data[~self.mask]
        else:
            data = self.data

        if c_level is not None:
            overlay_data = data.loc[data[c] == c_level, params].copy()
            plot += p9.stat_density_2d(
                data=overlay_data,
                mapping=p9.aes(
                    x=x,
                    y=y,
                    fill="..level..",
                    alpha="..level.."
                ),
                inherit_aes=False,
                geom="polygon",
                n=64,
                levels=levels,
                contour=True,
                package="scipy"     
            ) 

            plot += p9.scales.scale_fill_cmap(
                cmap_name=self.fill_map,
                guide=p9.guide_colorbar(
                    ticks=False
                ),
                na_value=self.fill_na
            )
        else:
            for i, factor in enumerate(data[c].unique()):
                # Skip NaNs
                if pd.isnull(factor):
                    continue

                overlay_data = data.loc[data[c] == factor, params].copy()

                if c_map is None:
                    plot += p9.stat_density_2d(
                        data=overlay_data,
                        mapping=p9.aes(
                            x=x,
                            y=y,
                            alpha="..level.."
                        ),
                        fill=self.TAB20[i],
                        inherit_aes=False,
                        geom="polygon",
                        n=64,
                        levels=levels,
                        contour=True,
                        package="scipy"     
                    ) 
                else:
                    plot += p9.stat_density_2d(
                        data=overlay_data,
                        mapping=p9.aes(
                            x=x,
                            y=y,
                            alpha="..level.."
                        ),
                        fill=c_map[factor],
                        color=c_map[factor],
                        size=0.1,
                        inherit_aes=False,
                        geom="polygon",
                        n=64,
                        levels=levels,
                        contour=True,
                        package="scipy"     
                    )

        plot += p9.scales.scale_alpha_continuous(
            range=(0.1,1),   #doesnt work?
            expand=(0,0),
            guide=False
        )
        plot += p9.labs(
            fill=c_name
        ) 

        return plot

    def raster_pca(self, x: str, y: str, c: Optional[str]=None, c_stat: str="mean", c_map: Optional[Dict[str,str]]=None, bins: int=256, loadings: bool=True, labels: bool=True) -> p9.ggplot:
        """
        Convenience wrapper around raster plot for the plotting of pca plots. Make sure you have ran add_pca() first.

        Args:
            x: the x dimension
            y: the y dimension
            c: the c dimension - the parameter used for color mapping; if None a density calculation is performed
            c_stat: the c statistic to calculate, see raster_special() for options
            c_map: only used for categorical color parameters. Uses the c_map to map the levels
            bins: the number of bins to rasterize into
            loadings: whether to plot the loadings
            labels: whether to plot the loading labels

        Returns:
            Rasterized PCA density plot
        """
        if c is None:
            plot = self.raster(
                x=x, 
                y=y,
                bins=bins
            )
        else:
            plot = self.raster_special(
                x=x,
                y=y,
                c=c, 
                c_stat=c_stat, 
                c_map=c_map,
                bins=bins
            )

        if loadings:
            plot = self._plot_pca_loadings(plot, labels)

        return plot

    def raster_special(self, x: str, y: str, c: str, c_stat: str="mean", bins: int=256, c_map: Optional[Dict[str,str]]=None) -> p9.ggplot:
        """
        Builds a raster plot using the c_stat(istic) to calculate the color-value

        Args:
            x: the x dimension
            y: the y dimension
            c: the c dimension - the parameter used for color mapping
            c_stat: the c statistic to calculate choose from ["max", "min", "sum", "mean", "median", "mode", "var", "std"]
            bins: the number of bins per dimension
            c_map: only used for categorical color parameters. Uses the c_map to map the levels

        Returns:
            Raster plot with per cell a parameterized statistic

        Raises:
            ValueError: incorrect parameters

        Note: the statistics are calculated on the transformed (=channel) data.
        """
        self._plot_check(self.data, x, y, c, fill=None)

        if c_stat not in ["max", "min", "sum", "mean", "median", "mode", "var", "std"]:
            raise ValueError(f"raster plotting has no implementation for c_stat '{c_stat}'")

        # Get source data of unique params
        params = pd.array([x,y,c]).dropna().unique()
        data = self.data[params].copy()

        # mask the data
        if self.mask is not None:
            if self.mask_type != "remove":
                raise ValueError(f"rasterized plots only allow for 'remove' mask_type'")
            data = data.loc[~self.mask]

        # Calculate bins
        data["__x_bin"] = self._bin(x, bins=bins)
        data["__y_bin"] = self._bin(y, bins=bins)

        data = data[["__x_bin","__y_bin",c]]

        # Calculate per group
        data_indexed = data.groupby(by=["__x_bin","__y_bin"], axis=0, sort=False, dropna=True)

        if c_stat == "max":
            c_name = f"__max({c})"
            c_rescale = False
            data_stat = data_indexed.max()
        elif c_stat == "min":
            c_name = f"__min({c})"
            c_rescale = False
            data_stat = data_indexed.min()
        elif c_stat == "sum":
            c_name = f"__sum({c})"
            c_rescale = False
            data_stat = data_indexed.sum()
        elif c_stat == "mean":
            c_name = f"__mean({c})"
            c_rescale = True
            data_stat = data_indexed.mean()
        elif c_stat == "median":
            c_name = f"__median({c})"
            c_rescale = True
            data_stat = data_indexed.mean()
        elif c_stat == "mode":
            c_name = f"__mode({c})"
            c_rescale = True
            data_stat = data_indexed.mean()
        elif c_stat == "var":
            c_name = f"__var({c})"
            c_rescale = True
            data_stat = data_indexed.std()
        elif c_stat == "std":
            c_name = f"__std({c})"
            c_rescale = True
            data_stat = data_indexed.std()
        else:
            raise ValueError(f"'{c_stat}' c_stat is an unknown operation")
        c_label = c_name[2:]

        # Remove multi-index
        data_stat.columns = [c_name]
        data_stat = data_stat.reset_index()
        data_stat = data_stat.loc[~data_stat[c_name].isna()]

        data_stat["__x_bin"] = data_stat["__x_bin"].astype("float64")
        data_stat["__y_bin"] = data_stat["__y_bin"].astype("float64")

        # Make polygon
        data_stat["__x_max"] = data_stat["__x_bin"] + ((self.transforms[x].l_end-self.transforms[x].l_start) / bins)
        data_stat["__y_max"] = data_stat["__y_bin"] + ((self.transforms[y].l_end-self.transforms[y].l_start) / bins)

        # build title
        if self.name:
            title = f"{self.name}: {c_stat}({c})"
        else:
            title = f"{c_stat}({c})"

        # Build the plot
        plot = self._plot_base(data_stat, "__x_bin", "__y_bin", fill=c_name)
        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, title=title, x=x, y=y, fill=c_label)
        plot = self._plot_scale(plot, xlim=True, ylim=True, x=x, y=y)
        plot = self._plot_fillscale(plot, rescale=c_rescale, fill_map=c_map)
        plot = plot + p9.geom_rect(
            data=data_stat,
            mapping=p9.aes(
                xmin="__x_bin",
                xmax="__x_max",
                ymin="__y_bin",
                ymax="__y_max"
            )
        )

        return plot

    def raster_special_3d(self, x: str, y: str, z: str, c: str=None, c_stat: str="mean", xy_bins: int=256, z_bins: int=8) -> List[p9.ggplot]:
        """
        Creates a z-stack of x-y plots with z_start fill. If saved as gif gives a 3dimensional representation

        Args:
            x: the x dimension
            y: the y dimension
            z: the z dimension - the parameter used for z-stack formation
            c: the c(olor) dimension - the parameter used for color mapping, if None calculates the density
            c_stat: the c statistic to calculate, choose from ["max", "min", "sum", "mean", "median", "mode", "var", "std"]
            xy_bins: the number of bins in the xy dimension
            z_bins: the number of bins in the z dimension

        Returns:
            A 3 dimensional raster plot

        Raises:
            ValueError: incorrect parameters

        Note: the statistics are calculated on the transformed (=channel) data.
        """
        self._plot_check(self.data, x, y, c, fill=None)

        if not (self.data.columns == z).any():
            raise ValueError(f"z '{z}' does not specify columns in .data")

        if not pd.api.types.is_numeric_dtype(self.data[z]):
            raise ValueError(f"z '{z}' must be a numeric dtype")
        
        if c_stat not in ["max", "min", "sum", "mean", "median", "mode", "var", "std"]:
            raise ValueError(f"raster plotting has no implementation for c_stat '{c_stat}'")

        # Get source data of unique params
        params = pd.array([x,y,z,c]).dropna().unique()
        data = self.data[params].copy()

        # mask the data
        if self.mask is not None:
            if self.mask_type != "remove":
                raise ValueError(f"rasterized plots only allow for 'remove' mask_type'")
            data = data.loc[~self.mask]

        # Cut into bins
        data["__x_bin"] = self._bin(x, bins=xy_bins)
        data["__y_bin"] = self._bin(y, bins=xy_bins)
        data["__z_bin"] = self._bin(z, bins=z_bins)

        if c is None:
            data = data[["__x_bin", "__y_bin", "__z_bin"]]
        else:
            data = data[["__x_bin", "__y_bin", "__z_bin", c]]

        # Calculate per group
        data_indexed = data.groupby(by=["__x_bin","__y_bin", "__z_bin"], axis=0, sort=False, dropna=True)

        if c is None:
            c_name = "__density"
            c_rescale = True
            data_stat = data.value_counts(sort=False)
            data_stat.name = c_name
        elif c_stat == "max":
            c_name = f"__max({c})"
            c_rescale = False
            data_stat = data_indexed.max()
        elif c_stat == "min":
            c_name = f"__min({c})"
            c_rescale = False
            data_stat = data_indexed.min()
        elif c_stat == "sum":
            c_name = f"__sum({c})"
            c_rescale = False
            data_stat = data_indexed.sum()
        elif c_stat == "mean":
            c_name = f"__mean({c})"
            c_rescale = True
            data_stat = data_indexed.mean()
        elif c_stat == "median":
            c_name = f"__median({c})"
            c_rescale = True
            data_stat = data_indexed.mean()
        elif c_stat == "mode":
            c_name = f"__mode({c})"
            c_rescale = True
            data_stat = data_indexed.mean()
        elif c_stat == "var":
            c_name = f"__var({c})"
            c_rescale = True
            data_stat = data_indexed.std()
        elif c_stat == "std":
            c_name = f"__std({c})"
            c_rescale = True
            data_stat = data_indexed.std()
        else:
            raise ValueError(f"'{c_stat}' c_stat is an unknown operation")
        c_label = c_name[2:]

        # Remove multi-index
        data_stat.columns = [c_name]
        data_stat = data_stat.reset_index()
        data_stat = data_stat.loc[~data_stat[c_name].isna()]

        data_stat["__x_bin"] = data_stat["__x_bin"].astype("float64")
        data_stat["__y_bin"] = data_stat["__y_bin"].astype("float64")
        data_stat["__z_bin"] = data_stat["__z_bin"].astype("float64")

        # Make polygon
        data_stat["__x_max"] = data_stat["__x_bin"] + ((self.transforms[x].l_end-self.transforms[x].l_start) / xy_bins)
        data_stat["__y_max"] = data_stat["__y_bin"] + ((self.transforms[y].l_end-self.transforms[y].l_start) / xy_bins)
        #data_stat["__z_max"] = data_stat["__z_bin"] + ((self.transforms[z].l_end-self.transforms[z].l_start) / z_bins)
        
        # Calculate color scale
        quantiles = data_stat[c_name].quantile([0.0, 0.02, 0.98, 1.0])
        if c_rescale:
            min_color = quantiles[0.02]
            max_color = quantiles[0.98]
        else:
            min_color = quantiles[0.0]
            max_color = quantiles[1.0]

        # Get x&y name
        try:
            x_name = self.labels[x]
        except KeyError:
            x_name = x
        try:
            y_name = self.labels[y]
        except KeyError:
            y_name = y

        # Group based on z
        z_stack: List[pd.DataFrame] = [y for x, y in data_stat.groupby("__z_bin", as_index=False)]
        plots: List[p9.ggplot] = []
        for i, frame in enumerate(z_stack):
            # build title
            if self.name:
                if c is None:
                    title = f"{self.name}\n{z}[{i+1}/{len(z_stack)}] : density"
                else:
                    title = f"{self.name}\n{z}[{i+1}/{len(z_stack)}] : {c_stat}({c})"
            else:
                if c is None:
                    title = f"{z}[{i+1}/{len(z_stack)}] : density"
                else:
                    title = f"{z}[{i+1}/{len(z_stack)}] : {c_stat}({c})"

            plot = self._plot_base(frame, "__x_bin", "__y_bin", fill=c_name)
            plot = self._plot_theme(plot)
            plot = self._plot_labels(plot, title=title, x=x_name, y=y_name, fill=c_label)
            plot = self._plot_scale(plot, xlim=True, ylim=True, x=x, y=y)
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
                    xmin="__x_bin",
                    xmax="__x_max",
                    ymin="__y_bin",
                    ymax="__y_max"
                )
            )
            plots.append(plot)

        return plots

    def correlation(self, x: str, y: str, c: str="__sample", y_stat: str="mean", summarize: bool=False, x_merge: Optional[Tuple[int,int]]=None, 
                    smooth: bool=True, bins: int=64, min_events: int=1, min_size: int=5, window: int=15) -> p9.ggplot:
        """
        Plots a correlation line graph of x versus y. If group is defined, will make a line per level in group.

        Args:
            x: the x dimension
            y: the y dimension
            c: which groups to split the data into, and plot separately
            y_stat: the condensor the apply to the y dimension, choose from ["max", "min", "sum", "mean", "median", "mode", "var", "std"]
            summarize: whether to summarize the data into a mean with standard deviations
            x_merge: (optional) specifies a binning area to consider as one for the purpose of calculating fraction
            smooth: whether to apply Savitzky-Golay smoothing on the average curves
            bins: the number of bins for the x dimension
            min_events: the minimum amount of events in a bin
            min_size: the minimum amount of bins each factor in c requires
            window: the smoothing window size, must be smaller then bins and uneven

        Returns:
            Correlation plot

        Raises:
            ValueError: incorrect parameters

        Note: the statistics are calculated on the transformed (=channel) data.
        """
        self._plot_check(self.data, x, y, color=c, fill=None)

        if c is not None:
            if pd.api.types.is_categorical_dtype(self.data[c]) or pd.api.types.is_string_dtype(self.data[c]) or pd.api.types.is_bool_dtype(self.data[c]):
                pass
            else:
                raise ValueError(f"c '{c}' must be a categorical dtype")

        if y_stat not in ["max", "min", "sum", "mean", "median", "mode", "var", "std"]:
            raise ValueError(f"correlation plotting has no implementation for y_stat '{y_stat}'")

        if not isinstance(window, int):
            raise ValueError("window must be an integer type")

        if window%2==0:
            raise ValueError("window must be uneven")

        # Get source data of unique params
        params = pd.array([x, y, c]).dropna().unique()
        data: pd.DataFrame = self.data[params].copy()

        # mask the data
        if self.mask is not None:
            if self.mask_type != "remove":
                raise ValueError(f"correlation plots only allow for 'remove' mask_type'")
            data = data.loc[~self.mask]
        
        ## Determine the coordinates of the bins
        try:
            trans_x = self.transforms[x]
        except KeyError:
            raise ValueError(f"no transform available for '{x}', unknown local limits") from None
        bins_x = np.linspace(trans_x.l_start, trans_x.l_end+1, num=bins+1, endpoint=True)
        bins_x = list(bins_x)
        bin_size = ((trans_x.l_end+1) - trans_x.l_start) / bins
        bin_size *= 0.5
        bins_x_names = np.linspace(trans_x.l_start + bin_size, trans_x.l_end + bin_size + 1, num=bins, endpoint=False)
        bins_x_names = list(bins_x_names)

        if x_merge:
            # Cut-out the area provided by x_merge, the bins are shifted to fit fully next to the merged area
            import bisect
            i_start = bisect.bisect_left(bins_x, x_merge[0])
            i_end = bisect.bisect_right(bins_x, x_merge[1])
            i_end = i_end - 1

            # Shift bins to the left or right to fit in the merged area
            x_diff_left = bins_x[i_start] - x_merge[0]
            x_diff_right = x_merge[1] - bins_x[i_end]

            # Calculate center of merged area
            x_merge_name = ((x_merge[1] - x_merge[0]) / 2) + x_merge[0]

            # Construct new bins
            new_x_left = [x - x_diff_left for x in bins_x[:i_start+1]]
            new_x_right = [x + x_diff_right for x in bins_x[i_end:]]
            new_x = new_x_left
            new_x.extend(new_x_right)
   
            # construct new names
            new_x_names_left = [x - x_diff_left for x in bins_x_names[:i_start]]
            new_x_names_right = [x + x_diff_right for x in bins_x_names[i_end:]]
            new_x_names = new_x_names_left
            new_x_names.append(x_merge_name)
            new_x_names.extend(new_x_names_right)

            bins_x = new_x
            bins_x_names = new_x_names

        # Assign data to the bins
        data["__x"] = pd.cut(
            data[x], 
            bins_x,
            labels=bins_x_names,
            ordered=False, 
            include_lowest=True
        )

        # Sometimes pd.cut returns a categorical
        data["__x"] = data["__x"].astype("float64")

        # Remove all rows that didnt pass the minimum event filter
        if c is None:
            data = data[["__x", y]]
            data_indexed = data.groupby(by=["__x"], axis=0, sort=False, dropna=True)
            # filter on minimum events
            if min_events > 1:
                data_indexed = data_indexed.filter(lambda x: len(x) > min_events)
                data_indexed = data_indexed.groupby(by=["__x"], axis=0, sort=True, dropna=True)
        else:
            data = data[["__x", y, c]]
            data_indexed = data.groupby(by=[c, "__x"], axis=0, sort=True, dropna=True)
            if min_events > 1:
                data_indexed = data_indexed.filter(lambda x: len(x) > min_events)
                data_indexed = data_indexed.groupby(by=[c, "__x"], axis=0, sort=True, dropna=True)

        # calculate stats
        if y_stat == "max":
            y_name = f"__max({y})"
            data_stat = data_indexed.max()
        elif y_stat == "min":
            y_name = f"__min({y})"
            data_stat = data_indexed.min()
        elif y_stat == "sum":
            y_name = f"__sum({y})"
            data_stat = data_indexed.sum()
        elif y_stat == "mean":
            y_name = f"__mean({y})"
            data_stat = data_indexed.mean()
        elif y_stat == "median":
            y_name = f"__median({y})"
            data_stat = data_indexed.median()
        elif y_stat == "mode":
            y_name = f"__mode({y})"
            data_stat = data_indexed.mode()
        elif y_stat == "var":
            y_name = f"__var({y})"
            data_stat = data_indexed.var()
        elif y_stat == "std":
            y_name = f"__std({y})"
            data_stat = data_indexed.std()
        else:
            raise ValueError(f"'{y_stat}' y_stat is an unknown operation")

        # Remove NaNs
        data_stat.columns = [y_name]
        data_stat = data_stat.loc[~data_stat[y_name].isna()]

        # Filter per repeat minimum size & remove multi-index
        c_select = data_stat.groupby(by=c).count() > min_size
        c_select = c_select[y_name]

        # Generate warning message
        for i in c_select.index:
            if c_select.loc[i] == False:
                print(f"data '{i}' has fewer then {min_size} datapoints; data is excluded from plotting")

        data_stat = data_stat.reset_index(level=1)
        data_stat = data_stat.loc[c_select]
        data_stat = data_stat.reset_index()

        # Store range for interpolation
        x_min = data_stat["__x"].min()
        x_max = data_stat["__x"].max()

        # Create new x-coordinates adjusted to the min-max range of the data
        x_axis = bins_x_names[bins_x_names.index(x_min):bins_x_names.index(x_max)+1]

        def smooth_curve(curve: pd.DataFrame, x: str, y: str, window: int, mode: str) -> pd.DataFrame:
            """
            Smooths a cureve using a non-uniform Savitsky Goley smoother

            Args:
                curve: the curve to smooth
                x: the column name defining the x-values
                y: the column name defining the y-values
                window: the window size used for savitsky-golay smoothing
                mode: the method use for edge-cases.

            Returns:
                Smoothed curve          
            """
            smoothed_x = np.array(curve[x])
            smoothed_y = self._non_uniform_savgol(
                x=smoothed_x, 
                y=np.array(curve[y]), 
                window=window,
                polynom=3,
                mode=mode
            )
            #smoothed_y = pd.Series(smoothed_y)
            smoothed = pd.concat([pd.Series(smoothed_x), pd.Series(smoothed_y)], axis=1, ignore_index=True)
            smoothed.columns = [x, y]
            return smoothed

        def linear_interp(axis: pd.Series, curve: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
            """
            Performs linear interpolation
            
            Args:
                axis: the x-coordinates to interpolate the curve data into
                curve: the curve to interpolate
                x: the column name defining the x-values
                y: the column name defining the y-values

            Returns:
                A interpolated curve
            """
            if curve.empty:
                return curve
            curve_interp = np.interp(axis, curve[x], curve[y])
            curve_interp = pd.DataFrame({x:axis, y:curve_interp})
            return curve_interp

        data_smooth = {i:k for i, k in data_stat.groupby(c)}
        for i in data_smooth:
            curve_data = data_smooth[i]

            if x_merge:
                # For proper smoothing split up the area
                merge_i = curve_data[curve_data["__x"] == x_merge_name].index[0]
                merge_i_loc = curve_data.index.get_loc(merge_i)
                merge_i_axis = x_axis.index(x_merge_name)

                # If enough data smooth before merge area
                if smooth and merge_i_loc > window:
                    before = smooth_curve(curve_data.loc[:merge_i-1], "__x", y_name, window, "mirror")
                else:
                    before = curve_data.loc[:merge_i-1, ["__x", y_name]]
                before = linear_interp(x_axis[:merge_i_axis-1], before, "__x", y_name)

                # Get merged area
                mid = curve_data.loc[[merge_i], ["__x", y_name]]
                mid = pd.concat([mid]*2, ignore_index=True)
                mid.iloc[0]["__x"] = x_merge[0]
                mid.iloc[1]["__x"] = x_merge[1]

                # If enough data get area after merge area
                if smooth and merge_i_loc < (len(curve_data["__x"]) - window - 1):
                    after = smooth_curve(curve_data.loc[merge_i+1:], "__x", y_name, window, "mirror")
                else:
                    after = curve_data.loc[merge_i+1:, ["__x", y_name]]
                after = linear_interp(x_axis[merge_i_axis+1:], after, "__x", y_name)

                data_smooth[i] = pd.concat([before, mid, after], ignore_index=True)

            elif smooth:
                # Smooth entire curve
                if len(curve_data.index) > window:
                    smoothed = smooth_curve(curve_data.loc[:], "__x", y_name, window, "mirror")
                    smoothed = linear_interp(x_axis[:], smoothed, "__x", y_name)
                else:
                    smoothed = linear_interp(x_axis[:], curve_data.loc[:], "__x", y_name)

                data_smooth[i] = smoothed

            else:
                smoothed = linear_interp(x_axis[:], curve_data.loc[:], "__x", y_name)
                data_smooth[i] = smoothed
            data_smooth[i][c] = i

        data_stat = pd.concat(data_smooth.values(), ignore_index=True)

        #########
        # build title
        if self.name:
            title = f"{self.name}: {y_stat}({y})"
        else:
            title = f"{y_stat}({y})"

        # Build the plot
        plot = self._plot_base(data_stat, "__x", y_name, c)       
        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, title=title, x=x, y=f"{y_stat}({y})")
        plot = self._plot_scale(plot, xlim=True, ylim=True, x=x, y=y)

        if summarize:   
            # Calculate mean and standard-deviation
            data_indexed = data_stat.groupby(by="__x", axis=0, sort=False)

            data_sum = data_indexed.mean()
            data_sum.columns = ["__mean"]
            data_sum["__std"] = data_indexed.std()
            data_sum.sort_index(inplace=True)
            data_sum["__x"] = data_sum.index

            # statistics
            # take statistic before summarisation to have a more relevant R value
            r_value, p_value = scipy.stats.pearsonr(data_stat["__x"], data_stat[y_name])
            title += f": r={r_value:.3f}, p={p_value:.4f}"

            # Add plotting parameters
            data_sum["__+std"] = data_sum["__mean"] + data_sum["__std"]
            data_sum["__-std"] = data_sum["__mean"] - data_sum["__std"]

            # Transform data into expected format for plotting. 
            data_group = {}
            data_group["mean"] = data_sum[["__x", "__mean"]].copy()
            data_group["mean"].columns = ["__x", y_name]
            data_group["mean"]["__stat"] = "mean"
            data_group["mean"].reset_index(drop=True, inplace=True)
    
            data_group["+std"] = data_sum[["__x", "__+std"]].copy()
            data_group["+std"].columns = ["__x", y_name]
            data_group["+std"]["__stat"] = "+std"
            data_group["+std"].reset_index(drop=True, inplace=True)
            # remove NaNs - caused by 

            data_group["-std"] = data_sum[["__x", "__-std"]].copy()
            data_group["-std"].columns = ["__x", y_name]
            data_group["-std"]["__stat"] = "-std"
            data_group["-std"].reset_index(drop=True, inplace=True)

            # Get polygon coordinates
            polygon_x = data_group["+std"]["__x"].copy()
            polygon_x = pd.concat([polygon_x, data_group["-std"]["__x"][::-1]].copy())
            polygon_y = data_group["+std"][y_name].copy()
            polygon_y = pd.concat([polygon_y, data_group["-std"][y_name][::-1]].copy())

            polygon = pd.concat([polygon_x, polygon_y], axis="columns")

            # Construct plot
            plot = plot + p9.ggtitle(
                title
            )

            plot = plot + p9.scales.scale_color_manual(
                values={"mean":"#f00000", "+std":"#000000", "-std":"#000000"}, 
                na_value=self.color_na
            )
            plot += p9.labs(color=f"mean±std({y_stat}({y}))")

            plot += p9.geom_polygon(
                data=polygon,
                mapping=p9.aes(x="__x", y=y_name),
                color=None,
                fill="#c0c0c0",
                alpha=0.5,
                inherit_aes=False
            )

            for name in ["+std", "-std", "mean"]:
                data = data_group[name]
                data.sort_values("__x")
                plot += p9.geom_path(
                    data=data,
                    mapping=p9.aes(x="__x", y=y_name, color="__stat"),
                    inherit_aes=False,
                    size=1.0
                )

        elif c:
            plot = self._plot_colorscale(plot)
        
            data_grouped = {x:y for x, y in data_stat.groupby(by=c, sort=False)}

            for name in data_grouped:
                data = data_grouped[name]
                data.sort_values(by="__x", inplace=True)
                plot += p9.geom_path(
                    data=data,
                    mapping=p9.aes(x="__x", y=y_name, color=c),
                    inherit_aes=False,
                    size=1.0
                )
        else:
            data_stat.sort_values(by="__x", inplace=True)
            plot += p9.geom_path(
                data=data_stat,
                mapping=p9.aes(x="__x", y=y_name),
                inherit_aes=False
            )

        # visualize merged area
        if x_merge:
            plot += p9.geom_vline(xintercept=x_merge[0])
            plot += p9.geom_vline(xintercept=x_merge[1])

        return plot

    def lowess(self, x: str, y: str, c: Optional[str]="__sample", summarize: bool=False, fraction: float=0.10) -> p9.ggplot:
        """
        Plots a lowess smoothed correlation line graph of x versus y. If group is defined, will make a line per level in group

        Args:
            x: the x dimension
            y: the y dimension
            c: (optional) which groups to split the data into, and plot separately
            summarize: whether to summarize the data into a mean with standard deviations
            fraction: the fraction of the data to use for per point during lowess smoothing

        Returns:
            A lowess smoothed line graph

        Raises:
            ValueError: incorrect parameters
            NotImplementedError: summarization is not yet implemented

        Note: the statistics are calculated on the transformed (=channel) data.

        TODO: implement summarization of lowess smoothed plot
        """
        self._plot_check(self.data, x, y, color=c, fill=None)

        if summarize and c is None:
            raise ValueError("cannot summarize if the data is not grouped")

        # Get source data of unique params
        params = pd.array([x, y, c]).dropna().unique()
        data = self.data[params].copy()

        # mask the data
        if self.mask is not None:
            if self.mask_type != "remove":
                raise ValueError(f"rasterized plots only allow for 'remove' mask_type'")
            data = data.loc[~self.mask]
        
        data = data[[x, y, c]]
        data_indexed = data.groupby(by=[c], axis=0, sort=False, dropna=True)

        # calculate stats
        data_dict = {i_c:i_data for i_c, i_data in data_indexed}
        data_lowess = {}
        for i_c in data_dict:
            i_data = data_dict[i_c]
            data_lowess[i_c] = Plotter._lowess(
                i_data,
                x=x,
                y=y,
                fraction=fraction,
                it=10,
                delta=None
            )
            data_lowess[i_c][c] = i_c

        data_lowess = pd.concat([data_lowess[x] for x in data_lowess])

        #########
        # build title
        if self.name:
            title = f"{self.name}: lowess({x}, {y})"
        else:
            title = f"lowess({x}, {y})"

        # Build the plot
        plot = self._plot_base(data_lowess, x, y, c)
        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, title=title, x=x, y=y)
        plot = self._plot_scale(plot, xlim=True, ylim=True, x=x, y=y)

        if summarize:
            raise NotImplementedError("summarization for lowess smoothing not yet implemented")
            # Calculate mean and standard-deviation
            data_indexed = data_lowess.groupby(by="__x_bin", axis=0, sort=False)

            data_repeat = data_indexed.count()
            data_sum = data_indexed.mean()
            data_sum.columns = ["__mean"]
            data_sum["__std"] = data_indexed.std()
            # remove NaNs (std of 1 value will return NaN)
            data_sum["__std"][pd.isnull(data_sum["__std"])] = 0.0
            data_sum.sort_index(inplace=True)
            data_sum["__x_bin"] = data_sum.index

            # filter for minimum amount of repeats
            data_sum = data_sum.loc[data_repeat.iloc[:,0] >= min_repeats]

            # statistics
            # take statistic before summarisation to have a more relevant R value
            #r_value, p_value = scipy.stats.pearsonr(data_sum["__x_bin"], data_sum["__mean"])
            r_value, p_value = scipy.stats.pearsonr(data_stat["__x_bin"], data_stat[y_name])
            title += f": r={r_value:.3f}, p={p_value:.4f}"

            # Add plotting parameters
            data_sum["__+std"] = data_sum["__mean"] + data_sum["__std"]
            data_sum["__-std"] = data_sum["__mean"] - data_sum["__std"]

            # Transform data into expected format for plotting. 
            data_group = {}
            data_group["mean"] = data_sum[["__x_bin", "__mean"]].copy()
            data_group["mean"].columns = ["__x_bin", y_name]
            data_group["mean"]["__stat"] = "mean"
            data_group["mean"].reset_index(drop=True, inplace=True)
    
            data_group["+std"] = data_sum[["__x_bin", "__+std"]].copy()
            data_group["+std"].columns = ["__x_bin", y_name]
            data_group["+std"]["__stat"] = "+std"
            data_group["+std"].reset_index(drop=True, inplace=True)
            # remove NaNs - caused by 

            data_group["-std"] = data_sum[["__x_bin", "__-std"]].copy()
            data_group["-std"].columns = ["__x_bin", y_name]
            data_group["-std"]["__stat"] = "-std"
            data_group["-std"].reset_index(drop=True, inplace=True)

            # Get polygon coordinates
            polygon_x = data_group["+std"]["__x_bin"].copy()
            polygon_x = pd.concat([polygon_x, data_group["-std"]["__x_bin"][::-1]].copy())
            polygon_y = data_group["+std"][y_name].copy()
            polygon_y = pd.concat([polygon_y, data_group["-std"][y_name][::-1]].copy())

            polygon = pd.concat([polygon_x, polygon_y], axis="columns")

            # Construct plot
            plot = plot + p9.ggtitle(
                title
            )

            plot = plot + p9.scales.scale_color_manual(
                values={"mean":"#f00000", "+std":"#000000", "-std":"#000000"}, 
                na_value=self.color_na
            )
            plot += p9.labs(color=f"mean±std({y_stat}({y}))")

            plot += p9.geom_polygon(
                data=polygon,
                mapping=p9.aes(x="__x_bin", y=y_name),
                color=None,
                fill="#c0c0c0",
                alpha=0.5,
                inherit_aes=False
            )

            for name in ["+std", "-std", "mean"]:
                data = data_group[name]
                data.sort_values("__x_bin")
                plot += p9.geom_path(
                    data=data,
                    mapping=p9.aes(x="__x_bin", y=y_name, color="__stat"),
                    inherit_aes=False,
                    size=1.0
                )

        else:
            if c:
                plot = self._plot_colorscale(plot)
            
                data_grouped = {x:y for x, y in data_lowess.groupby(by=c, sort=False)}

                for name in data_grouped:
                    data = data_grouped[name]
                    data.sort_values(by=x, inplace=True)
                    plot += p9.geom_path(
                        data=data,
                        mapping=p9.aes(x=x, y=y, color=c),
                        inherit_aes=False,
                        size=1.0
                    )
            else:
                data_lowess.sort_values(by=x, inplace=True)
                plot += p9.geom_path(
                    data=data_lowess,
                    mapping=p9.aes(x=x, y=y),
                    inherit_aes=False
                )

        return plot

    def histogram(self, x: str, c: Optional[str]=None, c_map: Optional[Dict[str,str]]=None, y_stat: str="fraction", bins: int=256, smooth: bool=True) -> p9.ggplot:
        """
        Creates a ggplot lineplot object representing a histogram with the correct data and axis

        Args:
            x: the x dimension
            c: (optional) the color dimension; the histogram will be split into multiple lines to represent the c dimension
            c_map: (optional) uses the c_map to map the c-levels
            y_stat: the y representation, should be: "fraction", "absolute", "local".
            bins: the number of bins per dimension
            smooth: whether to apply savitzky-golay smoothing on the curves

        Returns:
            Histogram

        Raises:
            ValueError: incorrect parameters
        """
        self._plot_check(self.data, x, y=None, color=c, fill=None)

        if y_stat not in ("fraction", "absolute", "local"):
            raise ValueError(f"unknown y_stat parameter '{y_stat}' should be 'fraction', 'absolute', or 'local'.")

        if c is not None:
            if pd.api.types.is_categorical_dtype(self.data[c]) or pd.api.types.is_string_dtype(self.data[c]) or pd.api.types.is_bool_dtype(self.data[c]):
                #categorical
                pass
            else:
                raise ValueError(f"c '{c}' must be a categorical dtype")

        if c is None:
            data = self.data[[x]].copy()
        else:
            data = self.data[[x, c]].copy()

        if smooth:
            import scipy.signal

        # mask the data
        if self.mask is not None:
            if self.mask_type != "remove":
                raise ValueError(f"categorized plots only allow for 'remove' mask_type'")
            data = data.loc[~self.mask]
        
        if c is None:
            data = {x:data}
        else:
            data = {a:b[[x]] for a, b in data.groupby(data[c].astype("category"))}

        # Calculate bins
        try:
            trans_x = self.transforms[x]
        except KeyError:
            raise ValueError(f"no transform available for '{x}', unknown local limits") from None
        bins_x = np.linspace(trans_x.l_start, trans_x.l_end+1, num=bins+1, endpoint=True)
        bins_x = list(bins_x)
        bin_size = (trans_x.l_end - trans_x.l_start + 1) / bins
        bin_size *= 0.5
        bins_x_names = np.linspace(trans_x.l_start + bin_size, trans_x.l_end + bin_size + 1, num=bins, endpoint=False)
        bins_x_names = list(bins_x_names)
        
        for i_key in data:
            i_data = data[i_key]
            i_bin = pd.cut(
                i_data[x], 
                bins_x,
                labels=bins_x_names,
                ordered=False, 
                include_lowest=True
            )
            i_count = i_bin.value_counts(sort=False)

            if y_stat == "fraction":
                i_sum = sum(i_count)
                i_count /= i_sum

            i_data = pd.DataFrame(i_count)
            i_data["__x"] = pd.to_numeric(i_data.index)

            if smooth:
                  i_data[x] = scipy.signal.savgol_filter(i_data[x], 21, 3, mode="mirror")

            data[i_key] = i_data

        # Build plot base
        plot = p9.ggplot(
            mapping=p9.aes(x)
        )

        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, x=x, y=y_stat)

        # custom _plot_scale()
        try:
            scale_x = self.transforms[x]
        except KeyError:
            plot = plot + p9.coords.coord_cartesian()
        else:
            plot = plot + p9.scale_x_continuous(
                breaks=scale_x.major_ticks(),
                minor_breaks=scale_x.minor_ticks(),
                labels=scale_x.labels(),
                expand=(0,0),
                limits=(scale_x.l_start, scale_x.l_end)
            )

        plot = plot + p9.scale_y_continuous(
            expand=(0,0,0.1,0)
        )

        # plot the line and polygons according to color mapping
        for i, i_key in enumerate(list(data.keys())):
            if c_map:
                plot += p9.geom_path(
                    data=data[i_key],
                    mapping=p9.aes(x="__x", y=x),
                    color=c_map[i_key],
                    inherit_aes=False,
                    size=1.0,
                    show_legend=True
                )

            else:
                if len(data.keys()) <= 20:
                    plot += p9.geom_path(
                        data=data[i_key],
                        mapping=p9.aes(x="__x", y=x),
                        color=self.TAB20[i],
                        inherit_aes=False,
                        size=1.0,
                        show_legend=True
                    )
                else:
                    plot += p9.geom_path(
                        data=data[i_key],
                        mapping=p9.aes(x="__x", y=x),
                        color="#000000",
                        inherit_aes=False,
                        size=1.0,
                        show_legend=True
                    )

            poly_end = data[i_key].iloc[[-1,0]].copy()
            poly_end[x] = [0.0, 0.0]
            poly_data = pd.concat([data[i_key].copy(), poly_end], axis=0)

            if c_map:
                plot += p9.geom_polygon(
                    data = poly_data,
                    mapping=p9.aes(x="__x", y=x),
                    color=None,
                    fill=c_map[i_key],
                    alpha=0.32,
                    inherit_aes=False
                )
            else:
                if len(data.keys()) <= 20:
                    plot += p9.geom_polygon(
                        data = poly_data,
                        mapping=p9.aes(x="__x", y=x),
                        color=None,
                        fill=self.TAB20[i] + "50",
                        inherit_aes=False
                    )    
                else:
                    plot += p9.geom_polygon(
                        data = poly_data,
                        mapping=p9.aes(x="__x", y=x),
                        color=None,
                        fill="#00000050",
                        inherit_aes=False
                    )              

        return plot

    def violin(self, y: str, c: Optional[str]=None, c_map: Optional[Dict[str,str]]=None) -> p9.ggplot:
        """
        Creates a ggplot violin object with the correct data and axis

        Args:
            y: the y dimension
            c: (optional) the color dimension; the violin plot will be split into multiple violins to represent the c dimension
            c_map: (optional) uses the c_map to map the c-levels

        Returns:
            violin plot

        Raises:
            ValueError: incorrect parameters
        """
        self._plot_check(self.data, y, y=None, color=c, fill=None)

        if c is not None:
            if pd.api.types.is_categorical_dtype(self.data[c]) or pd.api.types.is_string_dtype(self.data[c]) or pd.api.types.is_bool_dtype(self.data[c]):
                #categorical
                pass
            else:
                raise ValueError(f"c '{c}' must be a categorical dtype")

        if c is None:
            data = self.data[[y]].copy()
            c = "__mock"
            data[c] = True
            data[c] = data[c].astype("category")
            x_label = ""
        else:
            data = self.data[[y, c]].copy()
            x_label = c

        # Order categories
        if c_map is not None and c is not None:
            if pd.api.types.is_string_dtype(data[c]) or pd.api.types.is_bool_dtype(data[c]):
                data[c] = data[c].astype("category")
            data[c] = data[c].cat.reorder_categories(list(c_map.keys()))

        # mask the data
        if self.mask is not None:
            if self.mask_type != "remove":
                raise ValueError(f"categorized plots only allow for 'remove' mask_type'")
            data = data.loc[~self.mask]

        # Build plot base
        plot = p9.ggplot(
            data=data
        )

        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, x=x_label, y=y)

        # custom _plot_scale()
        try:
            scale_x = self.transforms[y]
        except KeyError:
            plot = plot + p9.coords.coord_cartesian()
            plot = plot + p9.scale_y_continuous(
                expand=(0,0,0.1,0)
            )
        else:
            plot = plot + p9.scale_y_continuous(
                breaks=scale_x.major_ticks(),
                minor_breaks=scale_x.minor_ticks(),
                labels=scale_x.labels(),
                expand=(0,0),
                limits=(scale_x.l_start, scale_x.l_end)
            )

        if c_map:
            plot += p9.geom_violin(
                mapping=p9.aes(x=c, y=y, color=c, fill=c),
                inherit_aes=False,
                size=1.0,
                show_legend=True,
                #style="full",
            ) 
            plot += p9.scale_color_manual(c_map)
            plot += p9.scale_fill_manual(c_map)

        else:
            plot += p9.geom_violin(
                mapping=p9.aes(x=c, y=y, color=c, fill=c),
                inherit_aes=False,
                size=1.0,
                show_legend=True,
                #style="full",
            )
            if len(data[c].unique()) <= 10:
                plot += p9.scale_color_manual(self.TAB10)
                plot += p9.scale_fill_manual(self.TAB10)
            elif len(data[c].unique()) <= 20:
                plot += p9.scale_color_manual(self.TAB20)
                plot += p9.scale_fill_manual(self.TAB20)

        return plot

    def contribution(self, x: str, c: str, c_map: Optional[Dict[str,str]]=None, x_merge: Optional[Tuple[int,int]]=None, smooth: bool=True,
                     bins: int=64, min_events: int=1, min_size: int=5, window: int=15) -> p9.ggplot:
        """
        Creates a contribution plot by binning along the x-dimension and factorizing the c dimension per x-bin.

        Args:
            x: the x dimension
            c: the color dimension; the multiple dimensions will be added together and form together 100%
            c_map: (optional) uses the c_map to map the c-levels
            x_merge: (optional) specifies a binning area to consider as one for the purpose of calculating fraction
            smooth: whether to apply Savitzky-Golay smoothing on the average curves
            bins: the number of bins for the x dimension
            min_events: the minimum amount of events in a bin
            min_size: the minimum amount of bins each factor in c requires
            window: the smoothing window size, must be smaller then bins and uneven

        Returns:
            Contribution plot

        Raises:
            ValueError: incorrect parameters
        """
        self._plot_check(self.data, x, y=None, color=c, fill=None)

        if pd.api.types.is_categorical_dtype(self.data[c]) or pd.api.types.is_string_dtype(self.data[c]) or pd.api.types.is_bool_dtype(self.data[c]):
            pass
        else:
            raise ValueError(f"c '{c}' must be a categorical dtype")

        if not isinstance(window, int):
            raise ValueError("window must be an integer type")

        if window%2==0:
            raise ValueError("window must be uneven")

        params = pd.Series([x, c,"__sample"]).dropna().unique()
        data: pd.DataFrame = self.data[params].copy()

        # Special treatment for boolean types
        if pd.api.types.is_bool_dtype(self.data[c]):
            data[c] = data[c].astype("str")
            if c_map is None:
                c_map = {"True":self.TAB10[0], "False":"#ffffff"}

        categories = pd.Series(data[c].unique())
        if c_map:
            if not categories.isin(c_map.keys()).all():
                raise ValueError(f"c_map must contain all levels of '{c}'")
            categories = list(c_map.keys())

        # mask the data
        if self.mask is not None:
            if self.mask_type != "remove":
                raise ValueError(f"rasterized plots only allow for 'remove' mask_type'")
            data = data.loc[~self.mask]

        ## Determine the coordinates of the bins
        try:
            trans_x = self.transforms[x]
        except KeyError:
            raise ValueError(f"no transform available for '{x}', unknown local limits") from None
        bins_x = np.linspace(trans_x.l_start, trans_x.l_end+1, num=bins+1, endpoint=True)
        bins_x = list(bins_x)
        bin_size = ((trans_x.l_end+1) - trans_x.l_start) / bins
        bin_size *= 0.5
        bins_x_names = np.linspace(trans_x.l_start + bin_size, trans_x.l_end + bin_size + 1, num=bins, endpoint=False)
        bins_x_names = list(bins_x_names)

        if x_merge:
            # Cut-out the area provided by x_merge, the bins are shifted to fit fully next to the merged area
            import bisect
            i_start = bisect.bisect_left(bins_x, x_merge[0])
            i_end = bisect.bisect_right(bins_x, x_merge[1])
            i_end = i_end - 1

            # Shift bins to the left or right to fit in the merged area
            x_diff_left = bins_x[i_start] - x_merge[0]
            x_diff_right = x_merge[1] - bins_x[i_end]

            # Calculate center of merged area
            x_merge_name = ((x_merge[1] - x_merge[0]) / 2) + x_merge[0]

            # Construct new bins
            new_x_left = [x - x_diff_left for x in bins_x[:i_start+1]]
            new_x_right = [x + x_diff_right for x in bins_x[i_end:]]
            new_x = new_x_left
            new_x.extend(new_x_right)
   
            # construct new names
            new_x_names_left = [x - x_diff_left for x in bins_x_names[:i_start]]
            new_x_names_right = [x + x_diff_right for x in bins_x_names[i_end:]]
            new_x_names = new_x_names_left
            new_x_names.append(x_merge_name)
            new_x_names.extend(new_x_names_right)

            bins_x = new_x
            bins_x_names = new_x_names

        # Calculate the c dimension fractions (per bin) per sample
        data_sample = {a:b[[x, c]] for a, b in data.groupby(data["__sample"].astype("category"))}

        ## Bin and calculate contributions within bin
        
        # Store the data range for interpolation
        x_min = trans_x.l_end
        x_max = trans_x.l_start
        for i_key in data_sample:
            i_data = data_sample[i_key]

            # Assign data to the bins
            i_data["__x"] = pd.cut(
                i_data[x], 
                bins_x,
                labels=bins_x_names,
                ordered=False, 
                include_lowest=True
            )

            # Sometimes pd.cut returns a categorical
            i_data["__x"] = i_data["__x"].astype("float64")

            # Calculate fraction / bin
            output = pd.DataFrame(np.nan, index=bins_x_names, columns=categories, dtype="float")
            output.sort_index(inplace=True)
            for j_key, j_data in i_data.groupby(by="__x"):
                # Minimum event filter
                if len(j_data.index) < min_events:
                    continue

                total = j_data[c].count()
                cats = j_data[c].value_counts()
                cats = cats / total
                
                output.loc[j_key] = cats
                output.loc[j_key].fillna(0.0, inplace=True)

            # Remove all rows that didnt pass the minimum event filter
            output = output.loc[~pd.isnull(output.iloc[:,0])]
            output["__x"] = output.index
            output["__sample"] = i_key

            # Minimum size filter
            if len(output.index) < min_size:
                print(f"data '{i_key}' has fewer then {min_size} datapoints; data is excluded from plotting")
                continue

            # Store range for interpolation
            if float(output["__x"].min()) < x_min:
                x_min = output["__x"].min()
            if float(output["__x"].max()) > x_max:
                x_max = output["__x"].max()

            data_sample[i_key] = output

        data_bins = pd.concat(data_sample.values(), ignore_index=True)

        # Create new x-coordinates adjusted to the min-max range of the data
        x_axis = bins_x_names[bins_x_names.index(x_min):bins_x_names.index(x_max)+1]

        def smooth_curve(curve: pd.DataFrame, x: str, y: str, window: int, mode: str) -> pd.DataFrame:
            """
            Smooths curve using non-univerofrm Savitsky Golay smoothing

            Args:
                curve: the curve to smooth
                x: the column name defining the x-values
                y: the column name defining the y-values
                window: the window size used for savitsky-golay smoothing
                mode: the method use for edge-cases

            Returns:
                Smoothed curve       
            """
            smoothed_x = np.array(curve[x])
            smoothed_y = self._non_uniform_savgol(
                x=smoothed_x, 
                y=np.array(curve[y]), 
                window=window,
                polynom=3,
                mode=mode
            )
            #smoothed_y = pd.Series(smoothed_y)
            smoothed = pd.concat([pd.Series(smoothed_x), pd.Series(smoothed_y)], axis=1, ignore_index=True)
            smoothed.columns = [x, y]
            return smoothed

        def linear_interp(axis: pd.Series, curve: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
            """
            Performs linear interpolation
            
            Args:
                axis: the x-coordinates to interpolate the curve data into
                curve: the curve to smooth
                x: the column name defining the x-values
                y: the column name defining the y-values

            Returns:
                Linear interpolated curve
            """
            if curve.empty:
                return curve
            curve_interp = np.interp(axis, curve[x], curve[y])
            curve_interp = pd.DataFrame({x:axis, y:curve_interp})
            return curve_interp

        curves = {}
        for cat in categories:
            curve = data_bins[["__sample", "__x", cat]]
            data_smooth = {i:k for i, k in curve.groupby("__sample")}

            for i in data_smooth:
                curve_data = data_smooth[i]

                if x_merge:
                    # For proper smoothing split up the area
                    merge_i = curve_data[curve_data["__x"] == x_merge_name].index[0]
                    merge_i_loc = curve_data.index.get_loc(merge_i)
                    merge_i_axis = x_axis.index(x_merge_name)

                    # If enough data smooth before merge area
                    if smooth and merge_i_loc > window:
                        before = smooth_curve(curve_data.loc[:merge_i-1], "__x", cat, window, "mirror")
                    else:
                        before = curve_data.loc[:merge_i-1, ["__x", cat]]
                    before = linear_interp(x_axis[:merge_i_axis-1], before, "__x", cat)

                    # Get merged area
                    mid = curve_data.loc[[merge_i], ["__x", cat]]
                    mid = pd.concat([mid]*2, ignore_index=True)
                    mid.iloc[0]["__x"] = x_merge[0]
                    mid.iloc[1]["__x"] = x_merge[1]

                    # If enough data get area after merge area
                    if smooth and merge_i_loc < (len(curve_data["__x"]) - window - 1):
                        after = smooth_curve(curve_data.loc[merge_i+1:], "__x", cat, window, "mirror")
                    else:
                        after = curve_data.loc[merge_i+1:, ["__x", cat]]
                    after = linear_interp(x_axis[merge_i_axis+1:], after, "__x", cat)

                    data_smooth[i] = pd.concat([before, mid, after], ignore_index=True)

                elif smooth:
                    # Smooth entire curve
                    if len(curve_data.index) > window:
                        smoothed = smooth_curve(curve_data.loc[:], "__x", cat, window, "mirror")
                        smoothed = linear_interp(x_axis[:], smoothed, "__x", cat)
                    else:
                        smoothed = linear_interp(x_axis[:], curve_data.loc[:], "__x", cat)

                    data_smooth[i] = smoothed

                else:
                    smoothed = linear_interp(x_axis[:], curve_data.loc[:], "__x", cat)
                    data_smooth[i] = smoothed

            # Average curves
            data_smooth = pd.concat(data_smooth.values(), ignore_index=True)
            curve_interp = data_smooth.groupby("__x").mean().iloc[:,0]
            
            curves[cat] = curve_interp

        curves = pd.DataFrame(curves)
        curves["__x"] = curves.index.astype("float")
        curves.reset_index(drop=True, inplace=True)

        # Stack curves
        for i, cat in enumerate(categories):
            if i==0:
                continue
            curves[cat] += curves[categories[i-1]]

        # Build polygons
        polygon = curves.shift(periods=1, axis=1, fill_value=0.0)
        polygon["__x"] = curves["__x"]
        polygon = polygon.reindex(index=polygon.index[::-1])
        polygon.reset_index(drop=True, inplace=True)
        polygon = pd.concat([curves, polygon], ignore_index=True)

        # Build plot base
        plot = p9.ggplot(
            data=curves,
            mapping=p9.aes("__x")
        )

        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, x=x, y=f"Fraction")

        # custom _plot_scale()
        try:
            scale_x = self.transforms[x]
        except KeyError:
            plot = plot + p9.coords.coord_cartesian()
        else:
            plot = plot + p9.scale_x_continuous(
                breaks=scale_x.major_ticks(),
                minor_breaks=scale_x.minor_ticks(),
                labels=scale_x.labels(),
                expand=(0,0),
                limits=(scale_x.l_start, scale_x.l_end)
            )

        plot = plot + p9.scale_y_continuous(
            breaks=(0, 0.25, 0.5, 0.75, 1.0),
            expand=(0,0,0.1,0)
        )

        # plot the line and polygons according to color mapping
        for i, cat in enumerate(categories):
            if c_map:
                plot += p9.geom_path(
                    #data=curves[cat],
                    mapping=p9.aes(x="__x", y=cat),
                    color=c_map[cat],
                    inherit_aes=False,
                    size=1.0,
                    show_legend=True
                )

            else:
                if len(curves.keys()) <= 20:
                    plot += p9.geom_path(
                        #data=curves[cat],
                        mapping=p9.aes(x="__x", y=cat),
                        color=self.TAB20[i],
                        inherit_aes=False,
                        size=1.0,
                        show_legend=True
                    )
                else:
                    plot += p9.geom_path(
                        #data=curves[cat],
                        mapping=p9.aes(x="__x", y=cat),
                        color="#000000",
                        inherit_aes=False,
                        size=1.0,
                        show_legend=True
                    )

            if c_map:
                plot += p9.geom_polygon(
                    data = polygon,
                    mapping=p9.aes(x="__x", y=cat),
                    #color=c_map[cat],
                    fill=c_map[cat],
                    alpha=0.8,
                    inherit_aes=False                    
                )
            else:
                if len(data.keys()) <= 20:
                    plot += p9.geom_polygon(
                        data = polygon,
                        mapping=p9.aes(x="__x", y=cat),
                        #color="#00000000",
                        fill=self.TAB20[i] + "BB",
                        inherit_aes=False
                    )
                else:
                    plot += p9.geom_polygon(
                        data = polygon,
                        mapping=p9.aes(x="__x", y=cat),
                        #color="#00000000",
                        fill="#000000BB",
                        inherit_aes=False
                    )

        # visualize merged area
        if x_merge:
            plot += p9.geom_vline(xintercept=x_merge[0])
            plot += p9.geom_vline(xintercept=x_merge[1])

        return plot

    def show_3d(self, x: str, y: str, z: str, c: str=None, c_stat: str="mean", bins: int=128, c_map: Optional[Dict[str,str]]=None) -> None:
        """
        Creates a 3dimensional matplotlib figure object with the correct data and opens an interactive window to show the plot

        Args:
            x: the x dimension
            y: the y dimension
            z: the z dimension
            c: the c dimension - used for color mapping. If None will represent the event density
            c_stat: the c statistic to calculate, choose from ["max", "min", "sum", "mean", "median", "mode", "var", "std"]
            bins: the number of bins per xyz dimension.
            c_map: only used for factorized color parameters. Uses the c_map to map the levels

        Raises:
            ValueError: incorrect parameters
        """
        self._plot_check(self.data, x, y, c, fill=None)

        if not (self.data.columns == z).any():
            raise ValueError(f"z '{z}' does not specify columns in .data")
        if not pd.api.types.is_numeric_dtype(self.data[z]):
            raise ValueError(f"z '{z}' must be a numeric dtype")
        
        if c_stat not in ["max", "min", "sum", "mean", "median", "mode", "var", "std"]:
            raise ValueError(f"binning has no implementation for c_stat '{c_stat}'")

        # Get all data from unique parameters
        params = pd.array([x,y,z,c]).dropna().unique()
        data = self.data[params].copy()

        # mask the data
        if self.mask is not None:
            if self.mask_type != "remove":
                raise ValueError(f"rasterized plots only allow for 'remove' mask_type'")
            data = data.loc[~self.mask]

        # Cut into bins
        data["__x_bin"] = self._bin(x, bins=bins)
        data["__y_bin"] = self._bin(y, bins=bins)
        data["__z_bin"] = self._bin(z, bins=bins)

        if c is None:
            data = data[["__x_bin", "__y_bin", "__z_bin"]]
        else:
            data = data[["__x_bin", "__y_bin", "__z_bin", c]]

        # Calculate per group
        data_indexed = data.groupby(by=["__x_bin","__y_bin", "__z_bin"], axis=0, sort=False, dropna=True)

        if c is None:
            c_name = "__density"
            c_rescale = True
            data_stat = data.value_counts(sort=False)
            data_stat.name = c_name
        elif c_stat == "max":
            c_name = f"__max({c})"
            c_rescale = False
            data_stat = data_indexed.max()
        elif c_stat == "min":
            c_name = f"__min({c})"
            c_rescale = False
            data_stat = data_indexed.min()
        elif c_stat == "sum":
            c_name = f"__sum({c})"
            c_rescale = False
            data_stat = data_indexed.sum()
        elif c_stat == "mean":
            c_name = f"__mean({c})"
            c_rescale = True
            data_stat = data_indexed.mean()
        elif c_stat == "median":
            c_name = f"__median({c})"
            c_rescale = True
            data_stat = data_indexed.mean()
        elif c_stat == "mode":
            c_name = f"__mode({c})"
            c_rescale = True
            data_stat = data_indexed.mean()
        elif c_stat == "var":
            c_name = f"__var({c})"
            c_rescale = True
            data_stat = data_indexed.std()
        elif c_stat == "std":
            c_name = f"__std({c})"
            c_rescale = True
            data_stat = data_indexed.std()
        else:
            raise ValueError(f"'{c_stat}' c_stat is an unknown operation")

        # Remove multi-index
        data_stat.columns = [c_name]
        data_stat = data_stat.reset_index()
        data_stat = data_stat.loc[~data_stat[c_name].isna()]

        data_stat["__x_bin"] = data_stat["__x_bin"].astype("float64")
        data_stat["__y_bin"] = data_stat["__y_bin"].astype("float64")
        data_stat["__z_bin"] = data_stat["__z_bin"].astype("float64")

        cmap = "nipy_spectral"

        # manually set colors to allow for proper rescaling
        if pd.api.types.is_numeric_dtype(data_stat[c_name]):
            quantiles = data_stat[c_name].quantile([0.0, 0.02, 0.98, 1.0])
            if c_rescale:
                min_color = quantiles[0.02]
                max_color = quantiles[0.98]
            else:
                min_color = quantiles[0.0]
                max_color = quantiles[1.0]
            ratio_color = 1 / (max_color - min_color)

            colormap = plt.get_cmap(cmap)
            data_stat[c_name] = data_stat[c_name].apply(lambda x: (x - min_color) * ratio_color)
            data_stat[c_name] = data_stat[c_name].apply(lambda x: colormap(0 if x < 0 else (0.9999999 if x >= 1 else x), alpha=1))
        elif pd.api.types.is_string_dtype(data[c_name]):

            levels = data_stat[c_name].unique()
            levels = levels[~pd.isnull(levels)]
            if c_map:
                # Check if colormap covers all cases
                for level in levels:
                    if level not in c_map:
                        raise ValueError(f"level '{level}' undefined in c_map")
                c_map["nan"] = self.color_na
                data_stat[c_name] = data_stat[c_name].apply(lambda x: self.color_na if pd.isnull(x) else c_map[x])

            elif len(levels) <= 10:
                c_map = plt.get_cmap("tab10")
                c_map = dict(zip(levels, c_map.colors[:len(levels)]))
                c_map["nan"] = self.color_na
                data_stat[c_name] = data_stat[c_name].apply(lambda x: self.color_na if pd.isnull(x) else c_map[x])
               
            else:
                # Use default
                pass

        # Approximate dot size
        dot_size = (self.transforms[x].l_end - self.transforms[x].l_start) / bins
        dot_size = 1 if dot_size < 1 else dot_size

        # construct matplotlib figure and axes objects
        figure = plt.figure(figsize=(12.8, 9.6))
        axes = figure.add_subplot(111, projection="3d", facecolor="#EEEEEEFF")
        axes.scatter(
            xs=data_stat["__x_bin"],
            ys=data_stat["__y_bin"],
            zs=data_stat["__z_bin"],
            c=data_stat[c_name],
            zdir="y",
            depthshade=True,    # dont turn off - bug in matplotlib
            marker="s",
            s=dot_size,
            alpha=1
        )

        # Set axis ticks / scale / labels
        axes.set_xlim((self.transforms[x].l_start, self.transforms[x].l_end))
        axes.set_ylim((self.transforms[y].l_start, self.transforms[y].l_end))
        axes.set_zlim((self.transforms[z].l_start, self.transforms[z].l_end))

        try:
            axis_transform = self.transforms[x]
        except ValueError:
            pass
        else:
            # so apparently a specific plot-x can only have a single label
            major_ticks = np.array(axis_transform.major_ticks())
            labels = np.array(axis_transform.labels())
            axes.set_xticks(ticks=major_ticks, minor=False)
            axes.set_xticklabels(labels=labels)
            axes.set_xticks(ticks=axis_transform.minor_ticks(), minor=True)
        try:
            axes.set_xlabel(self.labels[x])
        except KeyError:
            axes.set_xlabel(x)
        
        # Somehow y <-> z axis are swapped, correct for this
        try:
            axis_transform = self.transforms[y]
        except ValueError:
            pass
        else:
            major_ticks = np.array(axis_transform.major_ticks())
            unique = np.unique(major_ticks, return_index=True)[1]
            labels = np.array(axis_transform.labels())
            axes.set_zticks(ticks=major_ticks[unique], minor=False)
            axes.set_zticklabels(labels=labels[unique])
            axes.set_zticks(ticks=axis_transform.minor_ticks(), minor=True)
        try:
            axes.set_zlabel(self.labels[y])
        except KeyError:
            axes.set_zlabel(y)
        
        try:
            axis_transform = self.transforms[z]
        except ValueError:
            pass
        else:
            major_ticks = np.array(axis_transform.major_ticks())
            unique = np.unique(major_ticks, return_index=True)[1]
            labels = np.array(axis_transform.labels())
            axes.set_yticks(ticks=major_ticks[unique], minor=False)
            axes.set_yticklabels(labels=labels[unique])
            axes.set_yticks(ticks=axis_transform.minor_ticks(), minor=True)
        try:
            axes.set_ylabel(self.labels[z])
        except KeyError:
            axes.set_ylabel(z)

        # theming
        if self.name:
            axes.set_title(self.name)
        
        axes.grid(True, which="major")
        axes.grid(False, which="minor")
        #dont think these parameters work... :(
        #axes.set_tick_params(which="major", direction="out", width=2.0, lenght=4.0)
        #axes.set_tick_params(which="minor", direction="out", width=1.0, length=2.0)
        axes.view_init(elev=0,azim=-90)
       
        plt.show()

    def heatmap(self, y: List[str], c: Optional[str]=None, c_map: Optional[Dict[str, str]]=None, rescale: bool=True) -> p9.ggplot:
        """
        Creates a ggplot tiles object with the correct data and axis. All rows of the datatable will be columns of the heatmap

        Args:
            y: a list defining the parameters on the rows of the heatmap
            c: (optional) the category dimension; the heatmap plot will be split and grouped around the c dimension
            c_map: (optional) uses the c_map to map the c-levels, only use the keys
            rescale: whether to show full range of expression or cap the visual colorscale at 0.02-0.98 quantile

        Returns:
            a heatmap

        Raises:
            ValueError: incorrect parameters
        """
        for param in y:
            if not (self.data.columns == param).any():
                raise ValueError(f"y '{param}' does not specify columns in .data")

            if not pd.api.types.is_numeric_dtype(self.data[param]):
                raise ValueError(f"y '{param}' must be a numeric dtype")

        if c:
            if not (self.data.columns == c).any():
                raise ValueError(f"color '{c}' does not specify columns in .data")

            if pd.api.types.is_categorical_dtype(self.data[c]) or pd.api.types.is_string_dtype(self.data[c]) or pd.api.types.is_bool_dtype(self.data[c]):
                c_type = "category"
                #categorical
                pass
            else:
                c_type = "continuous"

        if c_map and c_type == "continuous":
                raise ValueError(f"when using a c_map, c '{c}' must be a categorical dtype")

        if c is None:
            data = self.data[y].copy()
            c = "__mock"
            c_type = "category"
            data[c] = True
            data[c] = data[c].astype("category")
        elif c_type == "category":
            y_c = [*y, c]
            data = self.data[y_c].copy()
            if pd.api.types.is_string_dtype(data[c]) or pd.api.types.is_bool_dtype(data[c]):
                data[c] = data[c].astype("category")
        else:
            y_c = [*y, c]
            data = self.data[y_c].copy()

        # mask the data
        if self.mask is not None:
            if self.mask_type != "remove":
                raise ValueError(f"categorized plots only allow for 'remove' mask_type'")
            data = data.loc[~self.mask]

        # Order categories
        if c_map is not None:
            data[c] = data[c].cat.reorder_categories(list(c_map.keys()))
            data = data.sort_values(by=c)
        elif c_type == "category":
            data[c] = data[c].cat.reorder_categories(data[c].unique())
            data = data.sort_values(by=c)
        else:
            data = data.sort_values(by=c, ascending=True)

        # Normalize to 0 - 1 for each parameter based on quantile
        # This to unify the colorcoding
        quantiles = data[y].quantile(q=(0, 1), axis=0)
        data[y] = (data[y] - quantiles.iloc[0]) / (quantiles.iloc[1] - quantiles.iloc[0])

        #for param in y:
        #    data.loc[data[param] < 0.0, param] = 0.0
        #    data.loc[data[param] > 1.0, param] = 1.0

        # Add heatmap coordinate system
        data["__x"] = range(0, len(data.index))
        data["__x"] += 0.5

        # Get cutoffs between c
        if c_type == "category":
            cutoffs = data[[c,"__x"]].groupby(c).max()
            cutoff_center = data[[c,"__x"]].groupby(c).mean()
            cutoffs = cutoffs.iloc[:-1]
            cutoffs += 0.5
        else:
            cutoffs = pd.DataFrame(columns=["__x"])
            cutoff_center = pd.DataFrame([data[[c,"__x"]].mean()], columns=["__x"], index=[c])

        data = pd.melt(data, id_vars=["__x", c])

        # Should be autosorted in the order of y; if not, sort here
        y_dict = dict(zip(y, range(len(y), 0, -1)))
        y_dict = {x:(y_dict[x]-0.5) for x in y_dict}
        data["__y"] = data["variable"].map(y_dict)

        # Build plot base
        plot = p9.ggplot(
            data=data
        )
        plot = self._plot_theme(plot)
        plot = self._plot_labels(plot, x="event", y="", fill="value")
        plot += p9.geom_tile(mapping=p9.aes(x="__x", y="__y", fill="value"))

        plot += p9.scale_x_continuous(breaks=cutoff_center["__x"], labels=cutoff_center.index)
        plot += p9.scale_y_continuous(breaks=[y_dict[x] for x in y_dict], labels=[x for x in y_dict])

        if pd.api.types.is_numeric_dtype(plot.data["value"]) and not pd.api.types.is_bool_dtype(plot.data["value"]):
            # Continuous scale
            quantiles = plot.data["value"].quantile([0.0, 0.02, 0.98, 1.0])
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

        for i in cutoffs["__x"]:
            data_line = pd.DataFrame([[i,-0.2],[i,len(y)+0.2]], index=[0, 1], columns=["__x", "__y"])
            plot += p9.geom_line(data=data_line, mapping=p9.aes(x="__x", y="__y"), inherit_aes=False)

        return plot

    ## algorithms
    def _bin(self, x: str, bins: int) -> pd.Categorical:
        """
        Bins the data of parameter x
        
        Args:
            x: the parameter to bin
            bins: the number of x bins

        Raises:
            ValueError: if transform doesnt contain local limits
        
        Returns:
            List defining the bins of x
        """
        # Calculate bins
        try:
            trans = self.transforms[x]
        except KeyError:
            raise ValueError(f"no transform available for '{x}', unknown local limits") from None
        bins = np.linspace(trans.l_start, trans.l_end, num=bins+1, endpoint=True)
        bins = list(bins)

        # Cut into bins
        temp = pd.cut(
            self.data[x], 
            bins,
            labels=bins[:-1],
            ordered=False, 
            include_lowest=True
        )

        return temp

    @staticmethod
    def _lowess(data: pd.DataFrame, x: str, y: str, fraction: float=0.1, it:int=1, delta: float=None) -> pd.DataFrame:
        """
        Performs LOcally WEighted Scatterplot Smoothing on the y value at the given x value

        Args:
            data: the input dataframe
            x: the x-axis
            y: the y-axis
            fraction: the fraction of the data to use when estimating each y-value
            it: the number of residual-based reweightings to perfrom
            delta: distance within which to use linear-interpolaition instead of weighted regression. If None delta is approximated.
            
        Returns
            pd.Dataframe of x and smoothed y
        """
        from statsmodels import nonparametric

        if delta is None:
            delta = 0.01 * 1024

        lowess = nonparametric.smoothers_lowess.lowess(
            endog=data[y],
            exog=data[x],
            frac=fraction,
            it=it,
            delta=delta,
            is_sorted=False,
            missing="drop",
            return_sorted=True
        )
        lowess = pd.DataFrame(lowess, columns=[x, y])

        if(pd.isnull(lowess[y]).all()):
            print("WARNING: lowess smoothing returned NA: insufficient variability. Consider increasing the estimation fraction.")

        lowess.drop_duplicates(inplace=True)

        return lowess

    @staticmethod
    def _non_uniform_savgol(x: np.array, y: np.array, window: int, polynom: int, mode: str="interp") -> np.array:
        """
        Applies a Savitzky-Golay filter to y with non-uniform spacing as defined in x

        This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
        The borders are interpolated like scipy.signal.savgol_filter would do

        Args:
            x: List of floats representing the x values of the data
            y: List of floats representing the y values. Must have same length as x
            window: Window length of datapoints. Must be odd and smaller than x
            polynom: The order of polynom used. Must be smaller than the window size
            mode: Must be 'mirror', 'nearest' or 'interp'

        Returns:
            The smoothed y values

        Raises:
            ValueError: incorrect parameters
            TypeError: if parameters are of incorrect type
        """
        if len(x) != len(y):
            raise ValueError('"x" and "y" must be of the same size')

        if len(x) < window:
            raise ValueError('The data size must be larger than the window size')

        if type(window) is not int:
            raise TypeError('"window" must be an integer')

        if window % 2 == 0:
            raise ValueError('The "window" must be an odd integer')

        if type(polynom) is not int:
            raise TypeError('"polynom" must be an integer')

        if polynom >= window:
            raise ValueError('"polynom" must be less than "window"')
        
        if mode not in ["mirror", "nearest", "interp"]:
            raise ValueError('"mode" must be "mirror", "nearest", "interp"')

        half_window = window // 2
        polynom += 1

        if mode in ["mirror", "nearest"]:
            # Inpute x-values
            b_i = x[1] - x[0]
            b = [x[0]-(i*b_i) for i in range(half_window, 0, -1)]

            b_i = x[len(x)-1] - x[len(x)-2]
            e = [x[len(x)-1]+(i*b_i) for i in range(1, half_window+1)]
            x = np.concatenate([b,x,e])
            
            # Define y-values
            if mode == "mirror":
                b = np.flip(y[1:half_window])
                e = np.flip(y[len(y)-half_window-2:len(y)-1])
                y = np.concatenate([b,y,e])
            elif mode == "nearest":
                b = np.array(y[0]*half_window)
                e = np.array(y[len(y)-1]*half_window)
                y = np.concatenate([b,y,e])

        # Initialize variables
        A = np.empty((window, polynom))     # Matrix
        tA = np.empty((polynom, window))    # Transposed matrix
        t = np.empty(window)                # Local x variables
        y_smoothed = np.full(len(y), np.nan)

        # Start smoothing
        for i in range(half_window, len(x) - half_window, 1):
            # Center a window of x values on x[i]
            for j in range(0, window, 1):
                t[j] = x[i + j - half_window] - x[i]

            # Create the initial matrix A and its transposed form tA
            for j in range(0, window, 1):
                r = 1.0
                for k in range(0, polynom, 1):
                    A[j, k] = r
                    tA[k, j] = r
                    r *= t[j]

            # Multiply the two matrices
            tAA = np.matmul(tA, A)

            # Invert the product of the matrices
            tAA = np.linalg.inv(tAA)

            # Calculate the pseudoinverse of the design matrix
            coeffs = np.matmul(tAA, tA)

            # Calculate c0 which is also the y value for y[i]
            y_smoothed[i] = 0
            for j in range(0, window, 1):
                y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

            # If at the end or beginning, store all coefficients for the polynom
            if i == half_window:
                first_coeffs = np.zeros(polynom)
                for j in range(0, window, 1):
                    for k in range(polynom):
                        first_coeffs[k] += coeffs[k, j] * y[j]
            elif i == len(x) - half_window - 1:
                last_coeffs = np.zeros(polynom)
                for j in range(0, window, 1):
                    for k in range(polynom):
                        last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

        if mode == "interp":
            # Interpolate the result at the left border
            for i in range(0, half_window, 1):
                y_smoothed[i] = 0
                x_i = 1
                for j in range(0, polynom, 1):
                    y_smoothed[i] += first_coeffs[j] * x_i
                    x_i *= x[i] - x[half_window]

            # Interpolate the result at the right border
            for i in range(len(x) - half_window, len(x), 1):
                y_smoothed[i] = 0
                x_i = 1
                for j in range(0, polynom, 1):
                    y_smoothed[i] += last_coeffs[j] * x_i
                    x_i *= x[i] - x[len(x)-half_window - 1]
        
        elif mode in ["mirror", "nearest"]:
            # slice out the added x and y values
            y_smoothed = y_smoothed[half_window:len(y_smoothed)-half_window]

        return y_smoothed

    ## Dimensional reduction

    def add_umap(self, parameters: List[str], q: Tuple[float, float]=(0.05, 0.95), seed: int=None) -> None:
        """
        Calculates the Uniform Manifold Approximation and Projection (UMAP) axis
        Adds the value of each datapoint under the column names "UMAP1" and "UMAP2".
        Performs percentile-based scaling on the data

        Args:
            parameters: the parameters to use for umap calculation
            q: the quantile range to use for centering and scaling of the data (q_min, q_max)
            seed: the seed used; if 'None' the seed is randomly generated

        Raises:
            ValueError: incorrect parametres
        """
        for param in parameters:
            if not (param == self.data.columns).any():
                raise ValueError(f"parameter '{param}' does not specify a column in .data")

        # Scale data using quantiles to limit influence of outliers and to account for
        # a mixed gaussian distribution of the sample data

        # Scale data per sample
        data_samples = [y for x, y in self.data.groupby("__sample")]

        umap_norm = {}
        for sample in data_samples:
            # Standard/RobustScaler - they centralize based on mean/median. In flowcytometry data
            # we cannot assume that the distributions (generalize as a mix of two gaussians) shows equal/between sample
            # comparable sizes of the two guassians components. 
            # Secondly the data can have a lot of outliers, so scaling based on quantile will be more robust.
            # The mean of the quantiles will likely give a distribution agnostic centralisation.
            # Other option would be the geoMean, but that is quite influenced by the outliers

            # Get sample id for metadata storage
            sample_id = sample["__sample"].iloc[0]
            sample_scale = pd.DataFrame(columns=("q_0", "q_1", "q_mean"))

            # scale
            quantiles = sample[parameters].quantile(q=q)
            sample_scale["q_0"] = quantiles.loc[q[0]]
            sample_scale["q_1"] = quantiles.loc[q[1]]

            sample[parameters] = sample[parameters]

            a = (sample[parameters] - quantiles.loc[q[0]]) / (quantiles.loc[q[1]] - quantiles.loc[q[0]])
            sample[parameters] = a

            # Center
            quantiles = sample[parameters].quantile(q=q)
            q_mean = quantiles.mean()
            sample_scale["q_mean"] = q_mean
            
            sample[parameters] = sample[parameters] - q_mean

            # Add normalization metadata
            umap_norm[sample_id] = sample_scale

        self.metadata["__umap_scalers"] = umap_norm

        scaled_data = pd.concat(data_samples)

        import umap
        reducer = umap.UMAP(random_state=seed)
        data_umap = pd.DataFrame(reducer.fit_transform(scaled_data[parameters]))
        data_umap.index = scaled_data.index

        # umap output data is in identical order to input data
        data_umap.columns = ["UMAP1", "UMAP2"]

        self._data = pd.concat([self._data, data_umap], axis=1)

        # Add label data
        self.labels["UMAP1"] = "UMAP1"
        self.labels["UMAP2"] = "UMAP2"

        # Add transform data
        for i in ("UMAP1", "UMAP2"):
            i_min = self.data[i].min()
            i_max = self.data[i].max()
            i_range = i_max - i_min
            start = i_min - (0.1 * i_range)
            end = i_max + (0.1 * i_range)

            i_generator = transform.LinearGenerator()
            i_generator.stepsize_minor = 2.5
            i_generator.stepsize_major = 5
            i_transform = transform.Linear(l_start=start, l_end=end, g_start=start, g_end=end)
            i_transform.generator = i_generator
            self.transforms[i] = i_transform

        # Add metadata
        self.metadata["__umap_params"] = parameters

    def add_tsne(self) -> None:
        """
        Calculates the t-distributed Stochastic Neighbouring Embedding axis
        Adds the value of each datapoint under the column names "tSNE1" and "tSNE2"

        Raises:
            NotImplementedError

        TODO: implement
        """
        raise NotImplementedError("tSNE has yet to be implemented")

    def add_pca(self, parameters: List[str], q: Tuple[float, float]=(0.05, 0.95), seed: int=None):
        """
        Calculates the Principle Component axis after percentile based scaling
        Adds the value of each datapoint loading under the column names "PCn" (replace n with number of PC)

        Args:
            parameters: the parameters to use for pca calculation
            q: the quantile range to use for centering and scaling of the data (q_min, q_max)
            seed: the seed used; if 'None' the seed is randomly generated

        Raises:
            ValueError: incorrect parameters
        """
        for param in parameters:
            if not (param == self.data.columns).any():
                raise ValueError(f"parameter '{param}' does not specify a column in .data")

        # Scale data using quantiles to limit influence of outliers and to account for
        # a mixed gaussian distribution of the sample data

        # Scale data per sample
        data_samples = [y for x, y in self.data.groupby("__sample")]

        pca_norm = {}
        for sample in data_samples:
            # Standard/RobustScaler - they centralize based on mean/median. In flowcytometry data
            # we cannot assume that the distributions (generalize as a mix of two gaussians) shows equal/between sample
            # comparable sizes of the two guassians components. 
            # Secondly the data can have a lot of outliers, so scaling based on quantile will be more robust.
            # The mean of the quantiles will likely give a distribution agnostic centralisation.
            # Other option would be the geoMean, but that is quite influenced by the outliers

            # Get sample id for metadata storage
            sample_id = sample["__sample"].iloc[0]
            sample_scale = pd.DataFrame(columns=("q_0", "q_1", "q_mean"))
                        
            # scale
            quantiles = sample[parameters].quantile(q=q)
            sample_scale["q_0"] = quantiles.loc[q[0]]
            sample_scale["q_1"] = quantiles.loc[q[1]]
            
            sample[parameters] = (sample[parameters] - quantiles.loc[q[0]]) / (quantiles.loc[q[1]] - quantiles.loc[q[0]])

            # Center
            quantiles = sample[parameters].quantile(q=q)
            q_mean = quantiles.mean()
            sample_scale["q_mean"] = q_mean
            
            sample[parameters] = sample[parameters] - q_mean

            # Add normalization metadata
            pca_norm[sample_id] = sample_scale

        self.metadata["__pca_scalers"] = pca_norm

        scaled_data = pd.concat(data_samples)

        import sklearn
        n_components = len(parameters)
        pca = sklearn.decomposition.PCA(n_components=n_components, random_state=seed)
        pca.fit(scaled_data[parameters])

        # Calculate transformation for sample plotting
        pca_data = pd.DataFrame(pca.transform(scaled_data[parameters]))
        pca_data.index = scaled_data.index
        pca_data.columns = [f"PC{i+1}" for i in range(0, n_components)]

        self._data[pca_data.columns] = pca_data
        self.labels.update({f"PC{i+1}":f"PC{i+1}: {round(x*100,2)}%" for i, x in enumerate(pca.explained_variance_)})

        # Store component mean for value overlaying
        pca_mean = pd.Series(pca.mean_)
        pca_mean.index = parameters
        self.metadata["__pca_means"] = pca_mean

        # Store the components for vector plotting
        pca_vector = pd.DataFrame(pca.components_)
        pca_vector.index = [f"PC{i+1}" for i in range(0, n_components)]
        pca_vector.columns = parameters
        self.metadata["__pca_loadings"] = pca_vector.T

        # Add transform data
        for i in pca_data.columns:
            i_min = pca_data[i].min()
            i_max = pca_data[i].max()
            i_range = i_max - i_min
            start = i_min - (0.1 * i_range)
            end = i_max + (0.1 * i_range)

            i_generator = transform.LinearGenerator()
            i_generator.stepsize_minor = 0.5
            i_generator.stepsize_major = 1
            i_transform = transform.Linear(l_start=start, l_end=end, g_start=start, g_end=end)
            i_transform.generator = i_generator
            self.transforms[i] = i_transform

    ## saving # originally designed for z-stack-like figures - left here for compatibility

    def save_3d_gif(self, path: str, x: str, y: str, z: str, c: str) -> None:
        """
        DEPRECATED: Saves a raster_3d of x-y with a z-stack and density of c

        Args:
            path: the save directory (file name is generated automatically)
            x: the x dimension
            y: the y dimension
            z: the z dimension - the parameter used for z-stack formation
            c: the c(olor) dimension - the parameter used for color mapping

        Raises:
            ValueError: if path doesnt point to an existing directory
        """
        from PIL import Image

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
