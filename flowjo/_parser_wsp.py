##############################################################################     ##    ######
#    A.J. Zwijnenburg                   2021-03-08           v1.8                 #  #      ##
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
Reads FlowJo workspace (.wsp) files. The workspace information is directly
parsed from the FlowJo wsp/XML files into storage classes. These objects are
not ment for direct usage as they represent the xml version very closely. Use
the wsp Module as interface.

Flow Cytometry data has to be handled in the following order:
Get Channel data - raw data as stored in the fcs file
Transform into Scale data - the data as extracted from the fcs file 
    (especially log-transformed values can be stored in special order)
Apply compensation - the data compensated using a compensation matrix
Apply transformations - linear/log/biexponential/etc transforms into a integer range of 0-1023
    This is called Channel data by FlowJo!
Apply gating - apply the gates to the transformated data

Attributes:
    CHANNEL_MIN: FlowJo's minimum channel value
    CHANNEL_MAX: FlowJo's maximum channel value
"""

from __future__ import annotations
from typing import Dict, List, Union, Optional

import os
import copy

from lxml import etree
import matplotlib.path as mpl_path
import numpy as np
import pandas as pd

from .matrix import MTX
from .transform import _Abstract as _AbstractTransform
from .transform import Biex as BiexTransform
from .transform import Fasinh as FasinhTransform
from .transform import Hyperlog as HyperlogTransform
from .transform import Linear as LinearTransform 
from .transform import Log10 as Log10Transform
from .transform import Logicle as LogicleTransform

CHANNEL_MIN: int = 0
CHANNEL_MAX: int = 1023

class AbstractGate:
    """
    Abstract representation of a gate node

    Args:
        element: gate node element

    Attributes:
        name: gate name
        annotation: Unknown/Unused in gate nodes
        owning_group: Unknown/Unused in gate nodes
        gates: the child gate nodes, indexed on gate id
    """
    def __init__(self, element: etree._Element) -> None:
        self.name: str = None
        self.annotation: str = None
        self.owning_group: str = None

        self.gates: Dict[str, Union[AbstractGate, StatGate]] = {}

        AbstractGate._parse(self, element)

    def _parse(self, element: etree._Element) -> None:
        """
        Parses all abstract gate properties

        Args:
            element: a gate node element

        Raises:
            NotImplementedError: when encountering an unimplemented gate node type
        """
        self.name = element.attrib["name"]
        self.annotation = element.attrib["annotation"]
        self.owning_group = element.attrib["owningGroup"]

        # Find subgates
        gates: List[etree._Element] = element.find("Subpopulations")

        if gates is None:
            return

        # Parse all subelements as they all should contain gate information, boolean transforms have unique tags
        for gate in gates:
            if gate.tag == "Population":
                gate = Gate(gate)
                self.gates[gate.id] = gate
            elif gate.tag == "NotNode":
                gate = NotGate(gate)
                self.gates[gate.id] = gate
            elif gate.tag == "OrNode":
                gate = OrGate(gate)
                self.gates[f"Or_{len(self.gates)}"] = gate
            elif gate.tag == "AndNode":
                gate = AndGate(gate)
                self.gates[f"And_{len(self.gates)}"] = gate
            elif gate.tag == "Statistic":
                stat = StatGate(gate)
                self.gates[f"Stat_{len(self.gates)}"] = stat
            else:
                raise NotImplementedError(f"Unimplemented node type '{gate.tag}' in '{self.name}'. Please contact author.")

    def __repr__(self) -> str:
        return "(AbstractGate)"

class Gate(AbstractGate):
    """
    A representation of a node / 'Population' element

    Args:
        element: the Population element

    Attributes:
        id: the gate id
        parent_id: the parent node gate id
        name: the gate name
        annotation: the gate annotation
        owning_group: the group that owns this gate
        count: the number of events in the gate
        gates: the sub gate nodes indexed by gate id
        x: the gate x-dimension
        y: the gate y-dimension
    """
    def __init__(self, element: etree._Element) -> None:
        super().__init__(element)
        self.id: str = None
        self.parent_id: Optional[str] = None

        self.count: int = None

        self._gating: _AbstractGating = None

        Gate._parse(self, element)

    @property
    def x(self) -> str:
        """
        Getter of the x dimension ID

        Returns:
            the x dimension ID
        """
        return self._gating.dimension_x

    @property
    def y(self) -> str:
        """
        Getter of the y dimension ID

        Returns:
            the y dimension ID
        """
        return self._gating.dimension_y

    def _parse(self, element: etree._Element) -> None:
        """
        Parses all abstract gate properties

        Args:
            element: 'Population' element or boolean node elements

        Raises:
            NotImplementedError: for unimplemented gating types
        """
        self.count = int(element.attrib["count"])

        gate = element.find("Gate")
        self.id = gate.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/gating}id"]
        
        try:
            self.parent_id = gate.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/gating}parent_id"]
        except KeyError:
            # No parent
            self.parent_id = None
        
        # Find gating type
        if gate[0].tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/gating}EllipsoidGate":
            self._gating = _EllipsoidGating(gate[0])
        elif gate[0].tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/gating}PolygonGate":
            self._gating = _PolygonGating(gate[0])
        elif gate[0].tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/gating}RectangleGate":
            self._gating = _RectangleGating(gate[0])
        else:
            raise NotImplementedError(f"unimplemented gating type '{gate[0].tag}' in {self.name}")

    def __repr__(self) -> str:
        if isinstance(self._gating, _RectangleGating):
            return f"(RectGate:{self.id})"
        elif isinstance(self._gating, _EllipsoidGating):
            return f"(EllipseGate:{self.id})"
        elif isinstance(self._gating, _PolygonGating):
            return f"(PolygonGate:{self.id})"
        else:
            return f"(UnknownGate:{self.id})"

class NotGate(Gate):
    """
    A representation of a not node / 'NotNode' element
    
    Args:
        element: the NotNode element

    Attributes:
        dependents: list of gate id's this NotGate depends on
    """
    def __init__(self, element: etree._Element) -> None:
        super().__init__(element)
        self.dependents: List[str] = []

        NotGate._parse(self, element)

    def _parse(self, element: etree._Element) -> None:
        """
        Parses all abstract gate properties
        
        Args:
            param element: 'NotNode' element
        """
        self.dependents = [element.attrib["name"] for element in element.findall("Dependents/Dependent")]

    def __repr__(self) -> str:
        return f"(NotGate:{self.id})"

class OrGate(AbstractGate):
    """
    A representation of a OrGate node

    Args:
        element: the gate node element

    Attributes:
        count: the number of events included in this gate
        dependents: list of gate id's this NotGate depends on
    """
    def __init__(self, element: etree._Element) -> None:
        super().__init__(element)
        self.count: int = None
        self.dependents: List[str] = []

        OrGate._parse(self, element)

    def _parse(self, element: etree._Element) -> None:
        """
        Parses the OrGate properties

        Args:
            element: a OrGate node element
        """
        self.count = int(element.attrib["count"])
        self.dependents = [element.attrib["name"] for element in element.findall("Dependents/Dependent")]

    def __repr__(self) -> str:
        return f"(OrGate)"

class AndGate(AbstractGate):
    """
    A representation of a AndGate node

    Args:
        element: the gate node element

    Attributes:
        count: the number of events included in this gate
        dependents: list of gate id's this AndGate depends on
    """
    def __init__(self, element: etree._Element) -> None:
        super().__init__(element)
        self.count: int = None
        self.dependents: List[str] = []

        AndGate._parse(self, element)

    def _parse(self, element: etree._Element) -> None:
        """
        Parses the AndGate properties
            
        Args:
            element: a AndGate node element
        """
        self.count = int(element.attrib["count"])
        self.dependents = [element.attrib["name"] for element in element.findall("Dependents/Dependent")]

    def __repr__(self) -> str:
        return "(AndGate)"

class StatGate(AbstractGate):
    """
    A representation of a stat node / 'Statistic' element

    Args:
        element: a 'Statistics' element

    Attributes:
        name: the statistics name
        annotation: the statistics annotation
        owning_group: the group that owns this statistics
        value: the calculated statistic
        ancestor: (optional) the ancestor used for statistic calculation
        id: (optional) the parameter the statistic is calculated on
        percent: (optional) the percentile the percentile statistic is calculated on
    """
    def __init__(self, element: etree._Element) -> None:
        super().__init__(element)

        # Statistics type/name dependent values
        self.ancestor: Optional[str] = None
        self.id: Optional[str] = None
        self.percent: Optional[float] = None

        #FlowJo calculated value
        self.value: float = None

        StatGate._parse(self, element)

    def _parse(self, element: etree._Element) -> None:
        """
        Parses all statistics node properties

        Args:
            element: 'Statistic' element
        """
        self.ancestor = element.attrib["ancestor"]
        self.value = float(element.attrib["value"])

        # Get statistics type/name specific values
        try:
            self.id = element.attrib["id"]
        except KeyError:
            pass

        if self.name == "Percentile":
            self.percent = float(element.attrib["percent"])

    def __repr__(self) -> str:
        if self.name == "fj.stat.freqofgrandparent":
            return "(Stat:Freq Of Grandparent)"
        elif self.name == "fj.stat.freqofparent":
            return "(Stat:Freq Of Parent)"
        elif self.name == "fj.stat.freqof":
            return "(Stat:Freq Of)"
        elif self.name == "fj.stat.freqoftotal":
            return "(Stat:Freq Of Total)"
        else:
            return f"(Stat:{self.name})"

class _AbstractGating:
    """
    An abstract gating class. 
    Contains the geometric data of a (geometric) gate node

    Args:
        element: a 'gating' element

    Attributes:
        space: defines if this gate is specified in 'scale' or 'channel' data. Depends on type of gate
        user_defined: whether the gate was defined by the user
        contains_events: whether the gate contains any events
        annotation_offset_x: the x-offset of the gate name annotation
        annotation_offset_y: the y-offset of the gate name annotation
        tint: the fill color of the gate
        is_tinted: whether the gate is tinted
        line_weight: the thickness of the gate border
    """
    def __init__(self, element: etree._Element) -> None:
        self.space: str = None
        self.user_defined: bool = None
        self.contains_events: bool = None
        
        self.annotation_offset_x: int = None
        self.annotation_offset_y: int = None

        self.tint: str = None
        self.is_tinted: bool = None
        self.line_weight: str = None

        _AbstractGating._parse(self, element)

    def _parse(self, element: etree._Element) -> None:
        """
        Parses the default gating attributes

        Args:
            element: the gating element
        """
        self.user_defined = bool(int(element.attrib["userDefined"]))
        self.contains_events = bool(int(element.attrib["eventsInside"]))
        
        self.annotation_offset_x = int(element.attrib["annoOffsetX"])
        self.annotation_offset_y = int(element.attrib["annoOffsetY"])

        self.tint = element.attrib["tint"]
        self.is_tinted = bool(int(element.attrib["isTinted"]))
        self.line_weight = element.attrib["lineWeight"]

    def contains(self, data: pd.DataFrame, transform_x: _AbstractTransform, transform_y: _AbstractTransform) -> pd.Series[bool]:
        """
        Returns an array defining if a x-y point is contained by the gating. Border of the gates is seen as part of the gate.
        For some gatings x and y need to be transformed first. Making the transformations essential information.
        
        Args:
            data: the data containing the x-y dimension in channel format
            transform_x: a transformation object defining the transformation of the x-dimension
            transform_y: a transformation object defining the transformation of the y-dimension

        Returns:
            a boolean series. True if contained by the gate
        """
        raise NotImplementedError("to be implemented in inherited class")

    def polygon(self, transform_x: _AbstractTransform, transform_y: _AbstractTransform) -> pd.DataFrame:
        """
        Returns a polygon path defining the gate border

        Args:
            transform_x: a transformation object defining the transformation of the x-dimension
            transform_y: a transformation object defining the transformation of the y-dimension
        
        Returns:
            row[n vectors] x col[x,y] dataframe in channel space
        """
        raise NotImplementedError("to be implemented in inherited class")

class _RectangleGating(_AbstractGating):
    """
    A representation of a rectangle gate. This also includes the Square and Quad gate

    Args:
        element: RectangleGate element.

    Attributes:
        space: defines if this gate is specified in 'scale' or 'channel' data.
        percent_x: Unknown
        percent_y: Unknown
        dimension_x: x parameter name
        min_x: the minimum x coordinate of the gate
        max_x: the maximum x coordinate of the gate
        dimension_y: y parameter name
        min_y: the minimum y coordinate of the gate
        max_y: the maximum y coordinate of the gate
    """
    def __init__(self, element: etree._Element) -> None:
        super().__init__(element)
        self.space = "scale"
        self.percent_x: float = None
        self.percent_y: float = None
        
        self.dimension_x: str = None
        self.min_x: float = None    # in channel space as parsed from wsp file
        self.max_x: float = None    # in channel space as parsed from wsp file

        self.dimension_y: Optional[str] = None
        self.min_y: Optional[float] = None    # in channel space as parsed from wsp file
        self.max_y: Optional[float] = None    # in channel space as parsed from wsp file

        self._parse(element)

    def _parse(self, element: etree._Element) -> None:
        """
        Parses the data of the element

        Args:
            element: RectangleGate element.

        Raises:
            NotImplementedError: for >2 dimensional gates
        """
        try:
            self.percent_x = float(element.attrib["percentX"])
        except KeyError:
            self.percent_x = None

        try:
            self.percent_y = float(element.attrib["percentY"])
        except KeyError:
            self.percent_x = None

        dimension_elements = element.findall("{http://www.isac-net.org/std/Gating-ML/v2.0/gating}dimension")
        
        if len(dimension_elements) > 2:
            raise NotImplementedError("No implementation for rectangle gate with more then 2 dimensinos")

        self.dimension_x = dimension_elements[0][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name"]
        try:
            self.min_x = float(dimension_elements[0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/gating}min"])
        except:
            self.min_x = None
        try:
            self.max_x = float(dimension_elements[0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/gating}max"])
        except:
            self.max_x = None

        # Gates on histograms are only 1 dimensional
        try:
            self.dimension_y = dimension_elements[1][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name"]
        except IndexError:
            self.dimension_y = None
            self.min_y = None
            self.max_y = None
            return

        try:
            self.min_y = float(dimension_elements[1].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/gating}min"])
        except:
            self.min_y = None
        try:
            self.max_y = float(dimension_elements[1].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/gating}max"])
        except:
            self.max_y = None

    def __str__(self) -> str:
        return f"(RectangleGating:\n {self.dimension_x}({self.min_x}-{self.max_x})\n {self.dimension_y}({self.min_y}-{self.max_y})\n)"

    def contains(self, data: pd.DataFrame, transform_x: _AbstractTransform, transform_y: _AbstractTransform) -> pd.Series[bool]:
        """
        Returns an array staing whether an x-y point is contained by the gating (border of the gates is excluded of the gate, as does FlowJo)
        For rectangle gatings x and y need to be transformed first.

        Args:
            data: the data containing the x-y dimension in channel format
            transform_x: (optional) a transformation object defining the transformation of the x-dimension
            transform_y: (optional) a transformation object defining the transformation of the y-dimension
        
        Raises:
            ValueError: if data does not contain the correct dimensions

        Returns:
            A Boolean series. True if contained by the gate
        """
        if not (data.columns == self.dimension_x).any():
            raise ValueError(f"data doesnt contain a column for dimension x '{self.dimension_x}'")

        if transform_y is not None:
            if not (data.columns == self.dimension_y).any():
                raise ValueError(f"data doesnt contain a column for dimension y '{self.dimension_y}'")

        boolean_result = np.array([True]*len(data.index))
        if self.max_x:
            scaled_max_x = transform_x.scale(self.max_x)
            boolean_result = boolean_result & (data[self.dimension_x] < scaled_max_x)
        
        if self.min_x:
            scaled_min_x = transform_x.scale(self.min_x)
            boolean_result = boolean_result & (data[self.dimension_x] >= scaled_min_x)

        if self.max_y:
            scaled_max_y = transform_y.scale(self.max_y)
            boolean_result = boolean_result & (data[self.dimension_y] < scaled_max_y)

        if self.min_y:
            scaled_min_y = transform_y.scale(self.min_y)
            boolean_result = boolean_result & (data[self.dimension_y] >= scaled_min_y)

        return boolean_result

    def polygon(self, transform_x: _AbstractTransform, transform_y: Optional[_AbstractTransform]) -> pd.DataFrame:
        """
        Returns a polygon path defining the gate border

        Args:
            transform_x: a transformation object defining the transformation of the x-dimension
            transform_y: a transformation object defining the transformation of the y-dimension
        
        Returns:
            row[n vectors] x col[x,Optional[y]] dataframe in channel space
        """
        x = [self.min_x, self.max_x, self.max_x, self.min_x, self.min_x]
        channel_x = transform_x.scaler(x)

        if self.min_x is None:
            channel_x[0] = transform_x.l_start
            channel_x[3] = transform_x.l_start
            channel_x[4] = transform_x.l_start
        if self.max_x is None:
            channel_x[1] = transform_x.l_end
            channel_x[2] = transform_x.l_end
           
        if transform_y is not None:
            y = [self.max_y, self.max_y, self.min_y, self.min_y, self.max_y]
            channel_y = transform_y.scaler(y)
            if self.min_y is None:
                channel_y[2] = transform_y.l_start
                channel_y[3] = transform_y.l_start
            if self.max_y is None:
                channel_y[0] = transform_y.l_end
                channel_y[1] = transform_y.l_end
                channel_y[4] = transform_y.l_end
        
            polygon = pd.DataFrame(list(zip(channel_x, channel_y)))
            polygon.columns = [self.dimension_x, self.dimension_y]

        else:
            polygon = pd.DataFrame([channel_x]).transpose()
            polygon.columns = [self.dimension_x]

        return polygon 

class _PolygonGating(_AbstractGating):
    """
    A representation of a polygon gate. This also includes the MultiSelect and Pencil gates

    Args:
        element: PolygonGate element.

    Attributes:
        space: defines if this gate is specified in 'scale' or 'channel' data.
        quad_id: unknown, likely id to polygon cache
        gate_resolution: the resolution FlowJo used for gate containment calculations
        dimension_x: x parameter name
        dimension_y: y parameter name
        coordinates_x: the gate vectors x coordinates
        coordinates_y: the gate vectors y coordinates
    """
    def __init__(self, element: etree._Element) -> None:
        super().__init__(element)
        self.space = "scale"

        self.quad_id: str = None
        self.gate_resolution: int = None

        self.dimension_x: str = None
        self.dimension_y: str = None
        self.coordinates_x: List[float] = []
        self.coordinates_y: List[float] = []

        self._parse(element)

    def _parse(self, element: etree._Element) -> None:
        """
        Parses the data of the element

        Args:
            element: PolygonGate element.

        Raises:
            NotImplementedError: >2 dimensional gate
        """
        self.quad_id = element.attrib["quadId"]
        self.gate_resolution = int(element.attrib["gateResolution"])
        
        dimension_elements = element.findall("{http://www.isac-net.org/std/Gating-ML/v2.0/gating}dimension")
        if len(dimension_elements) > 2:
            raise NotImplementedError("no implementation for polygon gates with more then 2 dimensinos")

        self.dimension_x = dimension_elements[0][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name"]
        self.dimension_y = dimension_elements[1][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name"]

        for item in element.iterfind("{http://www.isac-net.org/std/Gating-ML/v2.0/gating}vertex"):
            self.coordinates_x.append(float(item[0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"]))
            self.coordinates_y.append(float(item[1].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"]))

    def __str__(self) -> str:
        output = "(PolygonGating:\n"
        output += f" {self.dimension_x}:{self.dimension_y}"
        for i in range(0, len(self.coordinates_x)):
            output += f" {self.coordinates_x[i]}:{self.coordinates_y[i]}\n"
        output += ")"
        
        return output

    def polygon(self, transform_x: _AbstractTransform, transform_y: _AbstractTransform) -> pd.DataFrame:
        """
        Returns a polygon path defining the gate border

        Args:
            transform_x: a transformation object defining the transformation of the x-dimension
            transform_y: a transformation object defining the transformation of the y-dimension

        Returns:
            A row[n vectors] x col[x,y] dataframe in channel space
        """
        channel_x = copy.deepcopy(self.coordinates_x)
        channel_y = copy.deepcopy(self.coordinates_y)
        channel_x = transform_x.scaler(channel_x)
        channel_y = transform_y.scaler(channel_y)

        polygon = pd.DataFrame(list(zip(channel_x, channel_y)))
        polygon = polygon.append(polygon.iloc[0], ignore_index=True)
        polygon.columns = [self.dimension_x, self.dimension_y]

        return polygon        

    def contains(self, data: pd.DataFrame, transform_x: _AbstractTransform, transform_y: _AbstractTransform) -> pd.Series[bool]:
        """
        Returns an array staing whether an x-y point is contained by the gating (border of the gates is excluded of the gate, as does FlowJo)
        For rectangle gatings x and y need to be transformed first.

        Args:
            data: the data containing the x-y dimension in channel format
            transform_x: (optional) a transformation object defining the transformation of the x-dimension
            transform_y: (optional) a transformation object defining the transformation of the y-dimension
        
        Raises:
            ValueError: if data does not contain the correct dimensions

        Returns:
            A Boolean series. True if contained by the gate
        """
        if not (data.columns == self.dimension_x).any():
            raise ValueError(f"data doesnt contain a column for dimension x '{self.dimension_x}'")
        if not (data.columns == self.dimension_y).any():
            raise ValueError(f"data doesnt contain a column for dimension y '{self.dimension_y}'")

        # The coordinates_x and coordinates_y are in scale format. Apply transformation to transform to channel format.
        channel_x = copy.deepcopy(self.coordinates_x)
        channel_y = copy.deepcopy(self.coordinates_y)
        channel_x = transform_x.scaler(channel_x)
        channel_y = transform_y.scaler(channel_y)

        polygon = mpl_path.Path(list(zip(channel_x, channel_y)), closed=False)

        contained = pd.Series(polygon.contains_points(data[[self.dimension_x, self.dimension_y]]))
        
        # polygon contains drops index; add index
        contained.index = data.index

        return contained

class _EllipsoidGating(_AbstractGating):
    """
    A representation of an ellips gate

    Attributes that describe the polygon:

                   v_c
                    |
    v_a --- f_a --------- f_b ---- v_b
                    |
                   v_d

    Args:
        element: EllipsoidGate element.

    Attributes:
        space: defines if this gate is specified in 'scale' or 'channel' data.
        resolution: the channel resolution flowjo uses to calculate containment
        distance: the distance between any point on the ellipsoid and the two focal points
        dimension_x: the name of the x dimension
        dimension y: the name of the y dimension
        foci_a_x: focus point a, x coordinate
        foci_a_y: focus point a, y coordinate
        foci_b_x: focus point b, x coordinate
        foci_b_y: focus point b, y coordinate
        vertex_a_x: vertex point a, x coordinate
        vertex_a_y: vertex point a, y coordinate
        vertex_b_x: vertex point b, x coordinate
        vertex_b_y: vertex point b, y coordinate
        vertex_c_x: vertex point c, x coordinate
        vertex_c_y: vertex point c, y coordinate
        vertex_d_x: vertex point d, x coordinate
        vertex_d_y: vertex point d, y coordinate
    """
    def __init__(self, element: etree._Element) -> None:
        super().__init__(element)
        self.space = "channel"
        self.resolution = 256

        self.distance: float = None

        self.dimension_x: str = None
        self.dimension_y: str = None
        
        self.foci_a_x: float = None
        self.foci_a_y: float = None
        self.foci_b_x: float = None
        self.foci_b_y: float = None

        self.vertex_a_x: float = None
        self.vertex_a_y: float = None
        self.vertex_b_x: float = None
        self.vertex_b_y: float = None
        self.vertex_c_x: float = None
        self.vertex_c_y: float = None
        self.vertex_d_x: float = None
        self.vertex_d_y: float = None

        self._parse(element)

    def _parse(self, element: etree._Element) -> None:
        """
        Parses the data of the element

        Args:
            element: PolygonGate element

        Raises:
            NotImplementedError: If ellipsoid is defined in >2 dimensions
        """
        self.distance = float(element.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/gating}distance"])
        
        dimension_elements = element.findall("{http://www.isac-net.org/std/Gating-ML/v2.0/gating}dimension")
        if len(dimension_elements) > 2:
            raise NotImplementedError("EllipsoidGating is not implemented for more then 2 dimension")
        self.dimension_x = dimension_elements[0][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name"]
        self.dimension_y = dimension_elements[1][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name"]

        foci_element = element.find("{http://www.isac-net.org/std/Gating-ML/v2.0/gating}foci")
        self.foci_a_x = float(foci_element[0][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])
        self.foci_a_y = float(foci_element[0][1].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])
        self.foci_b_x = float(foci_element[1][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])
        self.foci_b_y = float(foci_element[1][1].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])

        vertex_element = element.find("{http://www.isac-net.org/std/Gating-ML/v2.0/gating}edge")
        self.vertex_a_x = float(vertex_element[0][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])
        self.vertex_a_y = float(vertex_element[0][1].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])
        self.vertex_b_x = float(vertex_element[1][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])
        self.vertex_b_y = float(vertex_element[1][1].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])
        self.vertex_c_x = float(vertex_element[2][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])
        self.vertex_c_y = float(vertex_element[2][1].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])
        self.vertex_d_x = float(vertex_element[3][0].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])
        self.vertex_d_y = float(vertex_element[3][1].attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}value"])

    def __str__(self) -> str:
        output = "(EllipsoidGating:\n"

        output += f" {self.dimension_x}-{self.dimension_y}\n"
        output += f" Foci: {self.foci_a_x}-{self.foci_a_y}\n"
        output += f" Foci: {self.foci_b_x}-{self.foci_b_y}\n"
        output += f" Vertex: {self.vertex_a_x}-{self.vertex_a_y}\n"
        output += f" Vertex: {self.vertex_b_x}-{self.vertex_b_y}\n"
        output += f" Vertex: {self.vertex_c_x}-{self.vertex_c_y}\n"
        output += f" Vertex: {self.vertex_d_x}-{self.vertex_d_y}\n"
        output += ")"

        return output

    def contains(self, data: pd.DataFrame, transform_x: _AbstractTransform, transform_y: _AbstractTransform) -> pd.Series[bool]:
        """
        Returns an array staing whether an x-y point is contained by the gating (border of the gates is excluded of the gate, as does FlowJo)
        For rectangle gatings x and y need to be transformed first.

        Args:
            data: the data containing the x-y dimension in channel format
            transform_x: (optional) a transformation object defining the transformation of the x-dimension
            transform_y: (optional) a transformation object defining the transformation of the y-dimension
        
        Raises:
            ValueError: if data does not contain the correct dimensions

        Returns:
            A Boolean series. True if contained by the gate
        """
        # Ellipse have the property that the outer line is always identical distance away from F_A and F_B (=self.distance)

        if not (data.columns == self.dimension_x).any():
            raise ValueError(f"data doesnt contain a column for dimension x '{self.dimension_x}'")
        if not (data.columns == self.dimension_y).any():
            raise ValueError(f"data doesnt contain a column for dimension y '{self.dimension_y}'")

        distance = np.sqrt((self.vertex_a_x - self.vertex_b_x)**2 + (self.vertex_a_y - self.vertex_b_y)**2)
        
        # The ellipse resolution is 0-255, adjust this to the actual local/channel resolution
        channel_scale_x = (self.resolution / (transform_x.l_end - transform_x.l_start + 1))
        channel_scale_y = (self.resolution / (transform_y.l_end - transform_y.l_start + 1))

        def ellipse_contain(data: pd.Series, distance, channel_scale_x, channel_scale_y) -> bool:
            """
            If the distance between the to-check point and ellipsoid focuspoints is smaller or equal to the distance 
            of the focuspoints to the ellipsoid border the point must be contained, and vice versa.

            Args:
                distance: the distance between the focuspoints and the ellipsoid border
                channel_scale_x: the x scaling factor
                channel_scale_y: the y scaling factor

            Returns:
                Contained?            
            """
            
            x = data.iloc[0] * channel_scale_x
            y = data.iloc[1] * channel_scale_y

            distance_f_a = np.sqrt((self.foci_a_x-x)**2 + (self.foci_a_y-y)**2)
            distance_f_b = np.sqrt((self.foci_b_x-x)**2 + (self.foci_b_y-y)**2)

            if distance_f_a + distance_f_b <= distance:
                return True
            else:
                return False

        return data[[self.dimension_x, self.dimension_y]].apply(lambda x: ellipse_contain(x, distance, channel_scale_x, channel_scale_y), axis=1)

    def polygon(self, transform_x: _AbstractTransform, transform_y: _AbstractTransform) -> pd.DataFrame:
        """
        Returns a polygon path defining the gate border

        Args:
            transform_x: a transformation object defining the transformation of the x-dimension
            transform_y: a transformation object defining the transformation of the y-dimension

        Returns:
            A row[n vectors] x col[x,y] dataframe in channel space
        """
        # Thanks stack overflow
        # https://stackoverflow.com/questions/4467121/solutions-for-y-for-a-rotated-ellipse - Dr. belisarius

        # for t from 0-2 * np.pi
        def ellips_x(a, b, rotation, t):
            return (a * np.cos(t) * np.cos(rotation)) - (b * np.sin(t) * np.sin(rotation))

        def ellips_y(a, b, rotation, t):
            return (b * np.cos(rotation) * np.sin(t)) + (a * np.cos(t) * np.sin(rotation))

        a = np.sqrt((self.vertex_a_x - self.vertex_b_x)**2 + (self.vertex_a_y - self.vertex_b_y)**2) /2
        b = np.sqrt((self.vertex_c_x - self.vertex_d_x)**2 + (self.vertex_c_y - self.vertex_d_y)**2) /2

        center_x = ((self.foci_b_x - self.foci_a_x) * 0.5) + self.foci_a_x
        center_y = ((self.foci_b_y - self.foci_a_y) * 0.5) + self.foci_a_y
        
        try:
            rotation = np.arctan((self.foci_b_y - self.foci_a_y)/(self.foci_b_x - self.foci_a_x))
        except ZeroDivisionError:
            rotation = 0
        
        channel_x = []
        channel_y = []
        for t in np.arange(0, 2*np.pi, (2*np.pi / 100)):
            channel_x.append(ellips_x(a, b, rotation, t))
            channel_y.append(ellips_y(a, b, rotation, t))

        # Modify to channel resolution
        channel_scale_x = ((transform_x.l_end - transform_x.l_start + 1) / self.resolution)
        channel_scale_y = ((transform_y.l_end - transform_y.l_start + 1) / self.resolution)

        channel_x = [channel_scale_x * (x + center_x) for x in channel_x]
        channel_y = [channel_scale_y * (y + center_y) for y in channel_y]

        # Close curve
        channel_x.append(channel_x[0])
        channel_y.append(channel_y[0])

        polygon = pd.DataFrame(list(zip(channel_x, channel_y)))
        polygon.columns = [self.dimension_x, self.dimension_y]

        return polygon 

class Cytometer():
    """
    A data class representing a cytometer

    Args:
        element: a 'Cytometer' element from a wsp file

    Attributes:
        name: the cytometer name
        cyt: the cytometer as specified in $CYT keyword
        manufacturer: the manifacturer
        homepage: the FlowJo webpage of the cytometer
        icon: the icon FlowJo uses with this cytometer
        serial_number: the serial number of the flow cytometer
        use_fcs3: DEFAULT setting for parsing of data with the cytometer; whether to use fcs3 (if not, uses fcs2)
        use_biex_transform: DEFAULT setting for parsing of data with the cytometer; whether by default biexponential transform is used
        transform_type: DEFAULT setting for parsing of data with the cytometer; the default transformation of the data
        use_gain: Likely whether gain amplification is used (like with avalanche detectors)
        linear_rescale: whether linear parameters are scaled according to user specified parameters
        linear_from_keyword: whether the linear transformation is acquired from the fcs keywords
        linear_minimum: DEFAULT minimum value on a linear scale
        linear_maximum: DEFAULT maximum value on a linear scale
        log_rescale: whether log parameters are scaled according to user specified parameters
        log_from_keyword: whether the log transformation is acquired from the fcs keywords
        log_minimum: DEFAULT minimum value on a log scale
        log_maximum: DEFAULT maximum value on a log scale
        biex_extra_negs: DEFAULT extra negative decades for biexponential transformations
        biex_width_basis: DEFAULT width basis for biexponential transformations
        always_linear: the always linear parameters
        always_log: the always log parameters
        transforms: the default transformation for any parameter
    """
    def __init__(self, element: etree._Element):
        # cytometer information
        self.name: str = None
        self.cyt: str = None
        self.manufacturer: str = None
        self.homepage: str = None
        self.icon: str = None
        self.serial_number: str = None

        # default parsing and transformation setting
        self.use_fcs3: bool = None
        self.use_biex_transform: bool = None    # If by default biex transform is used
        self.transform_type: str = None         # Default transform (if use_biex_transform is True -> 'BIEX' here too)
        
        self.use_gain: bool = None              # Likely whether gain amplification is used (like with avalanche detectors)

        # general transformation details
        self.linear_rescale: bool = None
        self.linear_from_keyword: bool = None
        self.linear_minimum: int = None
        self.linear_maximum: int = None
        
        self.log_rescale: bool = None
        self.log_from_keyword: bool = None
        self.log_minimum: float = None
        self.log_maximum: float = None

        self.biex_extra_negs: float = None
        self.biex_width_basis: float = None

        self.always_linear: List[str] = []
        self.always_log: List[str] = []
        # matrix id (identical to compensation matrix): parameter name: scale
        self.transforms: Dict[str, Dict[str, _AbstractTransform]] = {}

        self._parse(element)

    def _parse(self, element: etree._Element) -> None:
        """
        Parse the cytometer data

        Args:
            the cytometer element

        Raises:
            ValueError: when transformation matrix parsing failes.
        """
        self.name = element.attrib["name"]
        self.cyt = element.attrib["cyt"]
        self.manufacturer = element.attrib["manufacturer"]
        self.homepage = element.attrib["homepage"]
        self.icon = element.attrib["icon"]
        self.serial_number = element.attrib["serialnumber"]

        self.use_fcs3 = bool(int(element.attrib["useFCS3"]))
        self.use_biex_transform = bool(int(element.attrib["useTransform"]))
        self.transform_type = element.attrib["transformType"]

        self.use_gain = bool(int(element.attrib["useGain"]))              

        self.linear_rescale = bool(int(element.attrib["linearRescale"]))
        self.linear_from_keyword = bool(int(element.attrib["linFromKW"]))
        self.linear_minimum = int(element.attrib["linMin"])
        self.linear_maximum = int(element.attrib["linMax"])
        
        self.log_rescale = bool(int(element.attrib["logRescale"]))
        self.log_from_keyword = bool(int(element.attrib["logFromKW"]))
        self.log_minimum = float(element.attrib["logMin"])
        self.log_maximum = float(element.attrib["logMax"])

        self.biex_extra_negs = float(element.attrib["extraNegs"])
        self.biex_width_basis = float(element.attrib["widthBasis"])

        # parse always_linear and always_log
        for entree in element.find("LinParams").iterfind("Param"):
            self.always_linear.append(entree.text)

        for entree in element.find("LogParams").iterfind("Param"):
            self.always_linear.append(entree.text)

        # for filter in element.find("FilterParams").iterfind(""):
        # something something

        for transforms in element.find("TransformStore").iterfind("MatrixID"):
            identifier = transforms.attrib["matrixId"]
            self.transforms[identifier] = {}
            
            for transform in transforms.find("Transforms"):
                if transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}linear":
                    scale = LinearTransform(
                        l_start=CHANNEL_MIN,
                        l_end=CHANNEL_MAX,
                        g_start=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}minRange"]),
                        g_end=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}maxRange"]),
                        gain=float(transform.attrib["gain"])
                    )
                elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}log":
                    scale = Log10Transform(
                        l_start=CHANNEL_MIN,
                        l_end=CHANNEL_MAX,
                        g_start=int(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}offset"]),
                        g_end=int(10**float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}decades"]))
                    )
                elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}biex":
                    scale = BiexTransform(
                        l_start=CHANNEL_MIN,
                        l_end=CHANNEL_MAX,
                        g_end=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}maxRange"]),
                        neg_decade=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}neg"]),
                        width=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}width"]),
                        pos_decade=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}pos"]),
                        length=int(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}length"])
                    )
                elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}fasinh":
                    scale = FasinhTransform(
                        l_start=CHANNEL_MIN,
                        l_end=CHANNEL_MAX,
                        t=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}T"]),
                        m=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}M"]),
                        a=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}A"]),
                        w=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}W"]),
                        length=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}length"])
                    )
                elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}hyperlog":
                    scale = HyperlogTransform(
                        l_start=CHANNEL_MIN,
                        l_end=CHANNEL_MAX,
                        t=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}T"]),
                        m=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}M"]),
                        a=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}A"]),
                        w=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}W"]),
                        length=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}length"])
                    )
                elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}logicle":
                    scale = LogicleTransform(
                        l_start=CHANNEL_MIN,
                        l_end=CHANNEL_MAX,
                        t=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}T"]),
                        m=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}M"]),
                        a=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}A"]),
                        w=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}W"])
                    )
                elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}miltenyi":
                    print("WARNING: miltenyi transformation is not implemented. Transformation is ignored.")
                    continue
                else:
                    raise ValueError(f"cannot parse transform of type {transform.tag}")

                name = transform.find("{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}parameter").attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name"]
                self.transforms[identifier][name] = scale

    def __repr__(self) -> str:
        return f"(Cytometer:{self.cyt})"

class Group():
    """
    A data class representing a group of samples (with gating information)

    Args:
        element: A group element

    Attributes:
        name: the name of a group
        annotation: the annotation of a group
        owning_group: the group owner
        samples: the sample id's belonging to this group
        gates: the group-owned gates
    """
    def __init__(self, element: etree._Element) -> None:
        self.name: str = None
        self.annotation: str = None
        self.owning_group: str = None

        self.samples: List[str] = []
        self.gates: Dict[str, Gate] = {}

        self._parse(element)

    def _parse(self, element):
        """
        Parse the Group data

        Args:
            element: the 'GroupNode' element
        """
        self.name = element.attrib["name"]
        self.annotation = element.attrib["annotation"]
        self.owning_group = element.attrib["owningGroup"]

        # Add samples
        for sample in element.iterfind(".//SampleRefs/SampleRef"):
            self.samples.append(sample.attrib["sampleID"])

        # Add gates
        gates = element.find("Subpopulations")

        if gates is None:
            return

        for gate in gates.iterfind("Population"):
            gate = Gate(gate)
            self.gates[gate.id] = gate

    def __repr__(self) -> str:
        return f"(Group:{self.name}[{len(self.samples)}])"

class Sample():
    """
    A container class for a sample embedded in the wsp

    Args:
        element: the 'Sample' element

    Attributes:
        id: the sample id
        name: the sample name
        path: the sample path to fcs file
        owning_group: the group that owns this sample
        annotation: annotation
        count: the amount of cells in this sample
        compensation: the compensation matrix applied to this sample
        transforms: the transformation applied to each parameter of this sample
        keywords: the keywords (and values) of this sample
        gates: the gates applied to this sample
    """
    def __init__(self, element: etree._Element) -> None:
        self.id: str = None
        self.name: str = None
        self.path: str = None
        self.owning_group: str = None
        self.annotation: str = None
        self.count: int = None

        self.compensation: MTX = None
        self.transforms: Dict[str, _AbstractTransform] = {}
        self.keywords: Dict[str, str] = {}
        self.gates: Dict[str, Gate] = {}

        self._parse(element)

    def _parse(self, element: etree._Element) -> None:
        """
        Parse the sample data

        Args:
            element: the Sample element

        Raises:
            ValueError: when transformation cannot be parsed
            NotImplementedError: when unimplemented gate nodes are encountered
        """
        # Metadata
        dataset = element.find("DataSet")
        self.id = dataset.attrib["sampleID"]
        self.path = os.path.join(dataset.attrib["uri"][6:].replace("%20", " "))

        # compensation matrix
        matrix = element.find("{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}spilloverMatrix")
        if matrix is not None:
            self.compensation = MTX.from_wsp(matrix)
          
        # scale transforms
        for transform in element.find("Transformations"):
            if transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}linear":
                scale = LinearTransform(
                    l_start=CHANNEL_MIN,
                    l_end=CHANNEL_MAX,
                    g_start=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}minRange"]),
                    g_end=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}maxRange"]),
                    gain=float(transform.attrib["gain"])
                )
            elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}log":
                scale = Log10Transform(
                    l_start=CHANNEL_MIN,
                    l_end=CHANNEL_MAX,
                    g_start=int(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}offset"]),
                    g_end=int(10**float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}decades"]))
                )
            elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}biex":
                scale = BiexTransform(
                    l_start=CHANNEL_MIN,
                    l_end=CHANNEL_MAX,
                    g_end=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}maxRange"]),
                    neg_decade=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}neg"]),
                    width=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}width"]),
                    pos_decade=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}pos"]),
                    length=int(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}length"])
                )
            elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}fasinh":
                scale = FasinhTransform(
                    l_start=CHANNEL_MIN,
                    l_end=CHANNEL_MAX,
                    t=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}T"]),
                    m=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}M"]),
                    a=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}A"]),
                    w=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}W"]),
                    length=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}length"])
                )
            elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}hyperlog":             
                scale = HyperlogTransform(
                    l_start=CHANNEL_MIN,
                    l_end=CHANNEL_MAX,
                    t=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}T"]),
                    m=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}M"]),
                    a=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}A"]),
                    w=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}W"]),
                    length=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}length"])
                )
            elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}logicle":
                scale = LogicleTransform(
                    l_start=CHANNEL_MIN,
                    l_end=CHANNEL_MAX,
                    t=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}T"]),
                    m=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}M"]),
                    a=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}A"]),
                    w=float(transform.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}W"])
                )
            elif transform.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}miltenyi":
                print("WARNING: miltenyi transformation is not implemented. Transformation is ignored.")
                continue
            else:
                raise ValueError(f"Cannot parse transform of type {transform.tag}")

            name = transform.find("{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}parameter").attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name"]
            self.transforms[name] = scale

        # Keywords
        keywords = element.find("Keywords")
        for keyword in keywords.iterfind("Keyword"):
            self.keywords[keyword.attrib["name"]] = keyword.attrib["value"]

        # Group data etc
        sample_node = element.find("SampleNode")

        self.name = sample_node.attrib["name"]
        self.annotation = sample_node.attrib["annotation"]
        self.owning_group = sample_node.attrib["owningGroup"]
        self.count = sample_node.attrib["count"]

        # Add gates
        gates: List[etree._Element] = sample_node.find("Subpopulations")

        if gates is None:
            return

        # Parse all subelements as they all should contain gate information, boolean transforms have unique tags
        for gate in gates:
            if gate.tag == "Population":
                gate = Gate(gate)
                self.gates[gate.id] = gate
            elif gate.tag == "NotNode":
                gate = NotGate(gate)
                self.gates[gate.id] = gate
            elif gate.tag == "OrNode":
                gate = OrGate(gate)
                self.gates[f"Or_{len(self.gates)}"] = gate
            elif gate.tag == "AndNode":
                gate = AndGate(gate)
                self.gates[f"And_{len(self.gates)}"] = gate
            elif gate.tag == "Statistic":
                stat = StatGate(gate)
                self.gates[f"Stat_{len(self.gates)}"] = stat
            else:
                raise NotImplementedError(f"Unimplemented node type '{gate.tag}' in '{self.name}'. Please contact author.")

    def __repr__(self) -> str:
        return f"(Sample:{self.id}:{self.name})"

class Parser():
    """
    Class for the reading of FlowJo workspace (.wsp) files

    Args:
        path: path to the workspace file

    Attributes:
        name: the name of the workspace
        version: the wsp XML version (I think, not sure, its 20.0 when I am writing this :P )
        flowjo_version: the FlowJo version
        modified_data: whether the data is modified
        keywords: the user-specified keywords
        matrices: the compensation matrixes included in the wsp
        cytometers: the cytometers included in the wsp
        groups: the groups included in the wsp
        samples: the samples included in the wsp
        path: the path to the workspace file
    """
    def __init__(self, path: str=None):
        self._path: str = None
        self.name: str = None

        # general metadata
        self.version: str = None
        self.flowjo_version: str = None
        self.modified_data: str = None

        # general data
        self.keywords: List[str] = []
        # matrix id: matrix
        self.matrices: Dict[str, MTX] = {}
        self.cytometers: List[Cytometer] = []
        # group name: group data
        self.groups: Dict[str, Group] = {}
        # sample id: sample data
        self.samples: Dict[str, Sample] = {}

        self.path = path

    @property
    def path(self) -> str:
        """
        Getter for the workspace path
        
        Returns:
            workspace path
        """
        return self._path

    @path.setter
    def path(self, path: str) -> None:
        """
        Setter for the workspace path

        Args:
            path: the path a wsp file
        
        Raises:
            ValueError: if path cannot be found or doesnt point to a wsp file
        """
        path = os.path.join(path)

        if not os.path.isfile(path):
            raise ValueError(f"path {path} doesnt point to a file")

        if os.path.basename(path).split(os.extsep, 1)[1] != "wsp":
            raise ValueError(f"path {path} doesnt point to a wsp file")

        self._path = path
        self.name = os.path.basename(self.path)

        self.parse()

    def parse(self) -> None:
        """
        Parses the workspace data
        
        Raises:
            ValueError: if self.path is not a proper xml file
        """
        with open(self.path, mode='rb') as xml_file:
            xml_data = xml_file.read()

        if not xml_data:
            raise ValueError(f"path '{self.path}' refers to an empty file")

        parser = etree.XMLParser()
        try:
            tree = etree.fromstring(xml_data, parser)
        except etree.XMLSyntaxError as error:
            raise ValueError(f"xml file at path '{self.path}' contains invalid xml") from error

        # Check if workspace file
        if tree.tag != "Workspace":
            raise ValueError(f"xml file at path '{self.path}' is not a workspace file")

        # Get general metadata
        self.version = tree.attrib["version"]
        self.flowjo_version = tree.attrib["flowJoVersion"]
        self.date_modified = tree.attrib["modDate"]

        # Get main components of xml file
        # Most are useless for the current purposes and are commented out
        columns = tree.find("Columns")
        matrices = tree.find("Matrices")
        cytometers = tree.find("Cytometers")
        groups = tree.find("Groups")
        sample_list = tree.find("SampleList")
    
        #window_position = tree.find("WindowPosition")
        #text_traits = tree.find("TextTraits")
        #table_editor = tree.find("TableEditor")
        #layout_editor = tree.find("LayoutEditor")
        #scripts = tree.find("Scripts")
        #exports = tree.find("Exports")
        #sops = tree.find("SOPS")
        #experiment = tree.find("experiment")
        #weights = tree.find("weights")

        # forward parsing to relevant function
        self._parse_columns(columns)
        self._parse_matrices(matrices)
        self._parse_cytometers(cytometers)
        self._parse_groups(groups)
        self._parse_sample_list(sample_list)

    def _parse_columns(self, element: etree._Element) -> None:
        """
        Parses the workspace element

        Args:
            element: the 'Columns' element
        """
        # Clear old list
        self.keywords = []

        # Append entrees to list
        for entree in element.iterfind(".//Keyword"):
            self.keywords.append(entree.attrib["name"])
    
    def _parse_matrices(self, element: etree._Element) -> None:
        """
        Parses the workspace transformation matrices (= compensation matrix)

        Args:
            element: the 'Matrices' element
        """
        # Clear old list
        self.matrices = {}

        # Append entrees to list
        for matrix in element:
            matrix = MTX.from_wsp(matrix)
            self.matrices[matrix.id] = matrix

    def _parse_cytometers(self, element: etree._Element) -> None:
        """
        Parses the workspace cytometers data

        Args:
            element: the 'Cytometers' element
        """
        # Clear old data
        self.cytometers = []

        for cytometer_xml in element.iterfind("Cytometer"):
            self.cytometers.append(Cytometer(cytometer_xml))

    def _parse_groups(self, element: etree._Element) -> None:
        """
        Parses the workspace groups data
        
        Args:
            element: the 'Groups' element
        """
        # Clear old data
        self.groups = {}

        for group in element.iterfind("GroupNode"):
            data = Group(group)
            self.groups[data.name] = data

    def _parse_sample_list(self, element: etree._Element) -> None:
        """
        Parses the workspace sample list data

        Args:
            element: the 'SampleList' element
        """
        # Clear old data
        self.samples = {}

        for group in element.iterfind("Sample"):
            data = Sample(group)
            self.samples[data.id] = data

    def __repr__(self) -> str:
        return f"(WSP:{self.name})"
