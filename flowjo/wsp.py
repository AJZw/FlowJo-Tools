##############################################################################     ##    ######
#    A.J. Zwijnenburg                   2021-08-18           v1.18                #  #      ##
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
Provides a convenience interface over the flowjo workspace parser

:class: AbstractGate
A abstract gate node, for implementation see SampleGate, GroupGate, SampleStat, GroupStat

:class: SampleGate
A gate node representing the data included in the gate
.sample     - returns the Sample object this gate belongs to
.parent     - (if applicable) the parent gate
.id         - the gate's unique identifier
.name       - the gate's name
.x          - the gate's x dimension
.y          - the gate's y dimension
.data()     - returns the data of all cells included in the gate (deepcopy)
.gate_data()- returns the data of all cells included in the gate, with annotated gate membership (deepcopy)
.has_data() - whether the data has been loaded
.gates      - returns a list of all direct subgates
.count      - returns the amount of cells included in this gate
.path       - returns the full subgate structure of the current gate node
.transforms() - returns the dictionary of parameter transforms (shallow copy)
.polygon()  - returns a polygon representation of the gate
.[]         - returns the specified subgate
.__len__    - returns the amount of direct subgates
.__contains__ - checks whether the specified gate exists
.__str__    - returns a pretty print of the gate structure

:class: GroupGate
A group gate node representing the data included in a group gate
Keep in mind that the gate on the actual sample can be different!
The data() and gate_data() always return the cells as gated on sample-specific gates
.group      - returns the Group object this gate belongs to
.parent     - (if applicable) the parent gate
.id         - the gate's unique identifier
.name       - the gate's name
.x          - the gate's x dimension
.y          - the gate's y dimension
.data()     - returns the data of all cells included in the gate (deepcopy)
.gate_data()- returns the data of all cells included in the gate, with annotated gate membership (deepcopy)
.has_data() - whether the data has been loaded
.gates      - returns a list of all direct subgates
.count      - returns the amount of cells included in this gate
.path       - returns the full subgate structure of the current gate node
.transforms() - returns the dictionary of parameter transforms (shallow copy)
.polygon()  - returns a polygon representation of the gate
.[]         - returns the specified subgate
.__len__    - returns the amount of direct subgates
.__contains__ - checks whether the specified gate exists
.__str__    - returns a pretty print of the gate structure

:class: SampleStat
A read-only statistics node. Shows the statistics as calculated by FlowJo
.name       - the stat's name
.x          - the stat's dimension (if used)
.value      - the stat's value

:class: GroupStat
A read-only statistics of a group-owned node. Shows the statistics as calculated by FlowJo
.name       - the stat's name
.x          - the stat's dimension (if used)
.value      - the stat's values

:class: Sample
A class representing a single sample and all its components
.id         - the sample id
.name       - the sample name
.path_fcs   - the path to the source fcs file
.path_data  - the path to the loaded data
.data()     - the data of this sample (deepcopy)
.gate_data()- the data of this sample, with annotated gate membership (deepcopy)
.has_data() - whether the data has been loaded
.data_format - whether the internal data is in 'scale' or 'channel' units
.is_compensated - whether the internal data is compensated
.cytometer  - the cytometer this data is acquired on
.compensation - the compensation matrix applied to this sample
.transforms() - the data parameters transformation (shallow copy)
.keywords   - a dictionary of all fcs keywords
.gates      - the gate data of this sample
.count      - the amount of events in this sample's data
.load_data()- loads 'scale' or 'channel' data (in .csv format) into the sample
.subsample()- subsamples the data

:class: _Gates
Provides a nice 'attributy' getter of gates data
.gates      - returns a list of all root gates
.[]         - returns the specified gate node

:class: Group
A class representing a group of samples
.id         - the unique id of the group (identical to .name)
.name       - the name of the group
.gates      - the group gates (does not have to be identical to the gates of each individual sample)
.data()     - the concatenated data of all samples in the group (deepcopy)
.gate_data()- the concatenated data of all samples in the group, including gate membership (deepcopy)
.keywords() - the specified keyword(s) of all samples in the group
.transforms() - the transforms for the data parameters
.ids        - the identifiers of the samples included in this group
.names      - the names of the samples included in this group
.gates      - the group gate structure of this group
.__len__    - returns the amount of samples in this group
.[]         - returns the specified sample (first lookup by id, then by name)
.__contains__ - whether the group contains the specified sample (first lookup by id, then by name)

:class: Cytometer
A class representing a cytometer. Compensation and transforms belong together to a matrix identifier.
.id         - the unique id of the cytometer (identical to .name)
.name       - the name of the cytometer
.compensation - all compensation matrixes defined in the fcs-files for this cytometer ('Acquisition-defined')
.transforms - the cytometer's DEFAULT transformations

:class: Workspace
A class representing a FlowJo workspace
.path       - the path to a FlowJo .wsp file
.cytometers - the cytometer data stored in the workspace
.samples    - the sample data stored in the workspace
.groups     - the group data stored in the workspace
.compensation - the compensation matrixes stored in the workspace

:class: _Samples
Provides a nice 'attributy' getter of sample data
.ids        - the Sample unique identifiers
.names      - the Sample names (do not have to be unique, in that case use identifiers for lookup)
.data       - returns the Sample, indexed by the sample id
.__len__    - amount of Sample(s)
.[]         - getter for a specific Sample, use Sample id or name as index
.__contains__ - whether the sample exists

:class: _Groups
Provides a nice 'attributy' getter of group data
.ids        - the Group unique identifiers
.names      - the Group names (identical to ids)
.data       - returns the Group, indexed by the group id
.__len__    - amount of Group(s)
.[]         - getter for a specific group, use Group id or name as index
.__contains__ - whether the group exists

:class: _Cytometers
Provides a nice 'attributy' getter of cytometer data
.ids        - the Cytometer unique identifiers
.names      - the Cytometer names (identical to ids)
.data       - returns the Cytometer, indexed by the cytometer id
.__len__    - amount of Cytometer(s)
.[]         - getter for a specific cytometer, use cytometer id or name as index
.__contains__ - whether the cytometer exists

:class: _Compensation
Provides a nice 'attributy' getter of compensation matrixes
.ids        - the compensation matrix unique identifiers
.names      - the compensation matrix names
.data       - returns the compensation matrix, indexed by the compensation matrix id
.__len__    - amount of compensation matrix(es)
.[]         - getter for a specific compensation matrix, use compensation matrix id or name as index
.__contains__ - whether the compensation matrix exists

"""

from __future__ import annotations
from typing import Dict, List, Union

from ._parser_wsp import Parser as _Parser, _AbstractGating, CHANNEL_MIN, CHANNEL_MAX
from ._parser_wsp import Cytometer as _Cytometer, Sample as _Sample, Group as _Group
from ._parser_wsp import AbstractGate as _AbstractGate, Gate as _Gate, NotGate as _NotGate, OrGate as _OrGate, AndGate as _AndGate, StatGate as _StatGate
from .matrix import MTX
from .transform import _Abstract as _AbstractTransform

import pandas as pd
import numpy as np
import os
import copy
import sys

class AbstractGate:
    """
    An abstract class for gate nodes 
        :param gate_data: the parser gate object
        :param parent: the parent gate
    """
    def __init__(self, gate_data: _AbstractGate, parent: AbstractGate=None) -> None:
        self._gate: _AbstractGate = gate_data
        self.parent: AbstractGate = parent
        self.name: str = self._gate.name

        self._gates: Dict[str, AbstractGate] = {}

        self.__iter: int = None

    @property
    def path(self) -> str:
        """
        Returns the full gating path ('/' separated gate structure)
        """
        gate_path = self.name
        parent = self.parent
        while True:
            if parent is None:
                break
            gate_path = f"{parent.name}/{gate_path}"
            parent = parent.parent

        return gate_path

    @property
    def gates(self) -> List[str]:
        return list(self._gates.keys())

    def __len__(self) -> int:
        """
        Amount of subgates in this gate node
        """
        return len(self._gates)

    def __contains__(self, gate: str) -> bool:
        """
        Checks if this gate contains a subgate with the specified name
        """
        gates = gate.split("/", 1)

        gate = gates[0]

        if gate not in self._gates:
            return False
        elif len(gates) > 1:
            return self._gates[gate].__contains__(gates[1])
        else:
            return True

    def __getitem__(self, gate: str) -> AbstractGate:
        """
        Returns the specified Gate. Accepts chained gates (gate chains separated by '/')
            :param gate: the sample id or name
        """
        if not isinstance(gate, str):
            raise KeyError(f"gate index should inherit str not '{gate.__class__.__name__}'")

        gates = gate.split("/", 1)

        if gates[0] == "":
            raise KeyError(f"gate node '{gates[0]}' in '{self.path}/{gates[0]}' is empty")

        try:
            gate = self._gates[gates[0]]
        except KeyError:
            raise KeyError(f"gate node '{self.path}/{gates[0]}' cannot be found'") from None

        if len(gates) > 1:
            return gate[gates[1]]
        else:
            return gate

    def __iter__(self):
        self.__iter = 0
        return self

    def __next__(self):
        keys = list(self._gates.keys())
        
        if len(keys) <= self.__iter:
            raise StopIteration
        
        key = keys[self.__iter]
        self.__iter += 1
        
        return self._gates[key]

    def _repr_name(self, padding: int) -> str:
        """
        Generates a pretty representation of the current gate
            :param padding: the value to pad space to before placing count data
        """
        raise NotImplementedError("Implement in child class")

    def _repr_tree(self, prefix: str, padding: int) -> str:
        """
        Generate a pretty representation of the gate tree for __str__. 
        Iterative as it needs to scan the entire gate structure.
        The root gate's prefix is always default ("").
            :param prefix: the values to be added to the beginning of the gate representation.
            :param padding: the value to pad space to before placing count data
        """
        node_repr = self._repr_name(padding)

        # Calculate padding
        subnode_pad = 0
        for gate_id in self._gates.keys():
            if len(gate_id) > subnode_pad:
                subnode_pad = len(gate_id)
        subnode_pad += 2

        if prefix == "":
            for i, gate_id in enumerate(list(self._gates.keys())):
                # The last entree needs a fancy cap
                if len(self._gates) == i + 1:
                    subnode_prefix = "  "
                    subnode_repr = f" └{self._gates[gate_id]._repr_tree(subnode_prefix, subnode_pad)}"
                else:
                    subnode_prefix = " │"
                    subnode_repr = f" ├{self._gates[gate_id]._repr_tree(subnode_prefix, subnode_pad)}"

                node_repr += "\n" + subnode_repr

        else:
            node_repr = f"╴{node_repr}"

            for i, gate_id in enumerate(list(self._gates.keys())):
                # The last entree needs a fancy cap
                if len(self._gates) == i + 1:
                    subnode_prefix = f"{prefix}   "
                    subnode_repr = f"{prefix}  └{self._gates[gate_id]._repr_tree(subnode_prefix, subnode_pad)}"
                else:
                    subnode_prefix = f"{prefix}  │"
                    subnode_repr = f"{prefix}  ├{self._gates[gate_id]._repr_tree(subnode_prefix, subnode_pad)}"

                node_repr += "\n" + subnode_repr

        return node_repr

    def __repr__(self) -> str:
        # Append parent gates
        if self.parent is not None:
            output = self.parent.path + "/"
        else:
            output = ""
        
        output += self._repr_tree(prefix="", padding=len(self.name) + 2)
        return output

class SampleGate(AbstractGate):
    """
    A gate node representation for a gate describing a sample population
        :param sample: the sample this gate belongs to
        :param gate_data: the parser gate object
        :param parent: the parent gate
    """
    def __init__(self, sample: Sample, gate_data: _AbstractGate, parent: AbstractGate=None) -> None:
        super().__init__(gate_data, parent)
        self._sample: Sample = sample

        # Gate specific info
        self._gating: _AbstractGating = None
        self._in_gate: pd.Series = None     # Note: the _in_gate.name is undetermined and will be modified without warning

        self.id = None
        self.x = None
        self.y = None
       
        if isinstance(self._gate, _Gate):
            self._gating = self._gate._gating
            self.id = self._gate.id
            self.x = self._gate.x
            self.y = self._gate.y

        for gate in self._gate.gates:
            gate = self._gate.gates[gate]
            if isinstance(gate, _AbstractGate):
                self._gates[gate.name] = SampleGate(self._sample, gate, self)
            elif isinstance(gate, _StatGate):
                stat = SampleStat(self._sample, gate, self)
                self._gates[stat.name] = stat
            else:
                raise NotImplementedError(f"Unknown gate node '{gate}'. Please contact the author.")

    @property
    def sample(self) -> Sample:
        """
        Returns the sample this gate belongs to
        """
        return self._sample

    @property
    def count(self) -> int:
        """
        Returns the amount of cells within this gate
        """
        if self._in_gate is None:
            raise ValueError("Gate has not been applied to the sample data. Make sure to load_data() or apply_gates()")

        return sum(self._in_gate)

    def data(self, translate: bool=True) -> pd.DataFrame:
        """
        Returns the data for all events contained in this gate (this takes the entire gate structure into account)
        The data is deepcopied.
            :param translate: whether to change the column identifiers into the column names
        """
        if self._sample._data is None:
            raise ValueError("sample does not contain any data. Make sure to load_data()")

        # Make use of _data attribute to not make unnecessary deepcopies
        data = copy.deepcopy(self._sample._data.loc[self._in_gate])

        # Translate column names from identifier to names
        if translate:
            column_names = []
            for column in data.columns:
                try:
                    name = self._sample._parameter_names[column]
                except KeyError:
                    name = column
                if name == "":
                    name = column
                column_names.append(name)

            data.columns = column_names

        # Add sample identifyer
        data["__sample"] = self._sample.name

        return data

    def gate_data(self, factor: Dict[str, Dict[str, str]]=None, translate: bool=True) -> pd.DataFrame:
        """
        Getter for the data with gate annotations. Makes a deepcopy of the data
            :param factor: specifyer for factorization of gates membership. Dict[factor_column_name, Dict[gate_id, factor_level_name]]
            :param translate: whether to change the column identifiers into the column names
        """
        # This data is already 'gated'
        data = self.data(translate=translate)

        remove = self.path + "/"

        # Mitigation for 'PerformanceWarning: DataFrame is highly fragmented'
        # Acquire references to the _in_gates within the gating tree, then concat
        in_gate: List[pd.Series] = []
        for gate in self._gates:
            gate = self._gates[gate]
            if isinstance(gate, SampleGate):
                gate._attach_gate(in_gate, remove)

        # Gate data is necessary for factorization; this causes NaN's for events which are outside of self._in_gate
        data = pd.concat([data, *in_gate], axis=1)

        # factorize gate columns
        if factor is not None:
            # Mitigation for 'PerformanceWarning: DataFrame is highly fragmented'
            # First collect all new factors, then concat at once
            new_factors: List[pd.Series] = []
            #redundant = []
            for factor_name in factor:
                factor_levels = factor[factor_name]

                # Check if all factor_levels are available if not, generate warning
                if sum(data.columns.isin(factor_levels)) != len(factor_levels):
                    missing_levels = []
                    for factor_level in factor_levels:
                        if factor_level not in data.columns:
                            missing_levels.append(factor_level)

                    print(f"while factorizing '{factor_name}' not all levels were found {missing_levels}")

                factor_slice = pd.Series(np.nan, index=data.index, name=factor_name, dtype="object")

                for factor_level in factor_levels:
                    try:
                        factor_slice.loc[data[factor_level]] = factor_levels[factor_level]
                    except KeyError:
                        pass

                    #redundant.append(factor_level)

                new_factors.append(factor_slice)

            data = pd.concat([data, *new_factors], axis=1)
            # Remove now redundant columns
            #data.drop(columns=redundant, inplace=True)

        # Reapply current gate
        data = data[self._in_gate]
        
        return data

    def has_data(self) -> bool:
        """
        Returns whether the gate's data has been loaded
        """
        if self._sample._data is None:
            return False
        return True

    def transforms(self) -> Dict[str, _AbstractTransform]:
        """
        Returns the sample's parameter transforms. Returns a shallow copy.
        """
        return self._sample.transforms()

    def polygon(self) -> pd.DataFrame:
        """
        Returns a polygon representation of the _gating. Useful for plotting the gate.
        """
        transform_x = self._sample._transforms[self.x]
        if self.y:
            transform_y = self._sample._transforms[self.y]
        else:
            transform_y = None

        polygon = self._gating.polygon(transform_x, transform_y)

        column_names = []
        for column in polygon.columns:
            try:
                name = self._sample._parameter_names[column]
            except KeyError:
                column_names.append(column)
            
            if name == "":
                column_names.append(column)
            else:
                column_names.append(name)
        polygon.columns = column_names

        return polygon

    def _attach_gate(self, in_gate: List[pd.Series], remove: str=None) -> None:
        """
        Adds the True/False annotation of self._in_gate to the data. Makes sure all indexes are available.
        Recurses into the child-gates
            :param in_gate: a list of (named) gating masks
            :param remove: the prefixed gatenodes to remove in the output column headers
            :returns: the data with attached gate.
        """
        gate = self.path.split(remove, 1)
        if len(gate) < 2:
            gate = gate[0]
        else:
            gate = gate[1]

        #data[gate] = self._in_gate
        in_gate.append(self._in_gate)
        in_gate[-1].name = gate
        
        for gate in self._gates:
            gate = self._gates[gate]
            if isinstance(gate, SampleGate):
                gate._attach_gate(in_gate, remove)

    def _apply_gates(self) -> None:
        """
        Applies the gating structure to the dataset. Build boolean masks for each gate defining which cells fall in within the gate
        """
        # this needs to be refactored for proper handling of AND and OR Boolean gates
        # currently it assumes only a parent is needed for containment calculation. That is not true anymore when 
        # working with boolean AND, OR gates.

        if isinstance(self._gate, _OrGate) or isinstance(self._gate, _AndGate):
            raise NotImplementedError("Or and And Boolean gates are not yet implemented")

        # For the gate be to applied properly the gate might need to be transformed to the correct dimensions
        transform_x = self._sample._transforms[self.x]
        # One-dimensional plots (like histograms), do not have a y-transform
        if self.y:
            transform_y = self._sample._transforms[self.y]
        else:
            transform_y = None

        in_gating = self._gating.contains(self._sample._data, transform_x, transform_y)

        # Apply boolean gating here
        if isinstance(self._gate, _NotGate):
            in_gating = ~in_gating 

        # Append to parent gates (if available) to adhere to gate heirarchy
        if self.parent:
            self._in_gate = self.parent._in_gate & in_gating
        else:
            self._in_gate = in_gating

        # Forward signal
        for gate in self._gates:
            gate = self._gates[gate]

            # No need to apply gates on a stats node
            if isinstance(gate, SampleGate):
                gate._apply_gates()

    def _repr_name(self, padding: int) -> str:
        """
        Generates a pretty representation of the current gate
            :param padding: the value to pad space to before placing count data
        """
        if self._in_gate is None:
            node_name = self.name
        else:
            node_padding = padding - len(self.name)
            node_name = f"{self.name}{' '*node_padding}[{self.count}]"

        return node_name

class GroupGate(AbstractGate):
    """
    A gate node representation for a gate describing a group population
        :param group: the group this gate belongs to
        :param gate_data: the parser gate object
        :param parent: the parent gate
    """
    def __init__(self, group: Group, gate_data: _AbstractGate, parent: AbstractGate=None) -> None:
        super().__init__(gate_data, parent)
        self._group: Group = group

        # Gate specific info
        self._gating: _AbstractGating = None
        self._in_gate: pd.Series = None

        self.id = None
        self.x = None
        self.y = None
       
        if isinstance(self._gate, _Gate):
            self._gating = self._gate._gating
            self.id = self._gate.id
            self.x = self._gate.x
            self.y = self._gate.y

        for gate in self._gate.gates:
            gate = self._gate.gates[gate]
            if isinstance(gate, _AbstractGate):
                self._gates[gate.name] = GroupGate(self._group, gate, self)
            elif isinstance(gate, _StatGate):
                stat = GroupStat(self._group, gate, self)
                self._gates[stat.name] = stat
            else:
                raise NotImplementedError(f"Unknown gate node '{gate}'. Please contact the author.")

    @property
    def group(self) -> Group:
        """
        Returns the group this stat belongs to
        """
        return self._group

    @property
    def count(self) -> int:
        """
        Returns the total amount of cells within this gate. Sums the count of all samples
        """
        counts = 0

        for sample in self._group:
            counts += sample.gates[self.path].count

        return counts

    def data(self, translate: bool=True) -> pd.DataFrame:
        """
        Returns the data for all events contained in this gate (this takes the entire gate structure into account)
        The data is deepcopied.
            :param translate: whether to change the column identifiers into the column names
        """
        data = self._group.data(
            start_node=self.path,
            translate=translate
        )
        
        return data

    def gate_data(self, factor: Dict[str, Dict[str, str]]=None, translate: bool=True) -> pd.DataFrame:
        """
        Getter for the data with gate annotations. Makes a deepcopy of the data
            :param factor: specifyer for factorization of gates membership. Dict[factor_column_name, Dict[gate_id, factor_level_name]]
            :param translate: whether to change the column identifiers into the column names
        """
        data = self._group.gate_data(
            self.path,
            factor=factor,
            translate=translate
        )

        return data

    def has_data(self) -> bool:
        """
        Returns whether all samples in the group have been loaded with data
        Returns a dictionary of sample id and has_data
        """
        data = True

        for sample in self._group:
            if sample.has_data() is False:
                data = False
                break

        return data

    def transforms(self) -> Dict[str, _AbstractTransform]:
        """
        Returns the sample's parameter transforms. Returns a shallow copy.
        It is assumed all samples in a group have the same transforms.
        """
        return self._group.transforms()

    def polygon(self) -> pd.DataFrame:
        """
        Returns a polygon representation of the _gating. Useful for plotting the gate.
        """
        sample_id = self._group.ids[0]
        transform_x = self._group[sample_id]._transforms[self.x]

        if self.y:
            transform_y = self._group[sample_id]._transforms[self.y]
        else:
            transform_y = None

        polygon = self._gating.polygon(transform_x, transform_y)

        column_names = []
        for column in polygon.columns:
            try:
                name = self._group[self._group.ids[0]]._parameter_names[column]
            except KeyError:
                column_names.append(column)
            
            if name == "":
                column_names.append(column)
            else:
                column_names.append(name)
        polygon.columns = column_names

        return polygon

    def _repr_name(self, padding: int) -> str:
        """
        Generates a pretty representation of the current gate
            :param padding: the value to pad space to before placing count data
        """
        node_padding = padding - len(self.name)
        node_name = f"{self.name}{' '*node_padding}[{self.count}]"

        return node_name

class SampleStat(AbstractGate):
    """
    Convenience wrapper around a sample statistics gate node
    This is a pure read-only class. Most calculations cannot be redone due to requiring scale data (which is currently not implemented). 
        :param sample: the sample this gate belongs to
        :param stat_data: the parser _Gate object
        :param parent: the parent gate
    """
    def __init__(self, sample: Sample, stat_data: _StatGate, parent: AbstractGate=None) -> None:
        super().__init__(stat_data, parent)
        self._sample: Sample = sample
        self._stat: _StatGate = stat_data
        self.x: str = self._stat.id
        self.value: float = self._stat.value
        
        if self._stat.name == "Count":
            self.name = "Count"
        elif self._stat.name in ["CV", "Geometric Mean", "Mean", "Median", "Median Abs Dev", "Mode", "Robust CV", "Robust SD", "SD"]:
            try:
                label = self._sample._parameter_names[self.x]
            except KeyError:
                label = self.x
            self.name = f"{self._stat.name}({label})"
        elif self._stat.name == "Percentile":
            try:
                label = self._sample._parameter_names[self.x]
            except KeyError:
                label = self.x
            self.name = f"{self._stat.name}({self._stat.percent:.1f})({label})"
        elif self._stat.name == "fj.stat.freqofgrandparent":
            self.name = f"Freq of Grandparent"
        elif self._stat.name == "fj.stat.freqofparent":
            self.name = f"Freq of Parent"
        elif self._stat.name == "fj.stat.freqoftotal":
            self.name = f"Freq of Total"
        elif self._stat.name == "fj.stat.freqof":
            self.name = f"Freq of {self._stat.ancestor}"
        else:
            raise ValueError(f"unknown stat {self._stat.name}")

    @property
    def sample(self) -> Sample:
        """
        Returns the sample this gate belongs to
        """
        return self._sample

    def _repr_name(self, padding: int) -> str:
        """
        Generates a pretty representation of the current gate
            :param padding: the value to pad space to before placing count data
        """
        node_padding = padding - len(self.name)
        node_repr = f"{self.name}{' '*node_padding}[{self.value:.3f}]"

        return node_repr

class GroupStat(AbstractGate):
    """
    Convenience wrapper around a sample statistics gate node
    This is a pure read-only class. Most calculations cannot be redone due to requiring scale data (which is currently not implemented). 
        :param sample: the sample this gate belongs to
        :param stat_data: the parser _Gate object
        :param parent: the parent gate
    """
    def __init__(self, group: Group, stat_data: _StatGate, parent: AbstractGate=None) -> None:
        super().__init__(stat_data, parent)
        self._group: Group = group
        self._stat: _StatGate = stat_data
        self.x: str = self._stat.id
        
        if self._stat.name == "Count":
            self.name = "Count"
        elif self._stat.name in ["CV", "Geometric Mean", "Mean", "Median", "Median Abs Dev", "Mode", "Robust CV", "Robust SD", "SD"]:
            try:
                label = self._group._parameter_names[self.x]
            except KeyError:
                label = self.x
            self.name = f"{self._stat.name}({label})"
        elif self._stat.name == "Percentile":
            try:
                label = self._group._parameter_names[self.x]
            except KeyError:
                label = self.x
            self.name = f"{self._stat.name}({self._stat.percent:.1f})({label})"
        elif self._stat.name == "fj.stat.freqofgrandparent":
            self.name = f"Freq of Grandparent"
        elif self._stat.name == "fj.stat.freqofparent":
            self.name = f"Freq of Parent"
        elif self._stat.name == "fj.stat.freqoftotal":
            self.name = f"Freq of Total"
        elif self._stat.name == "fj.stat.freqof":
            self.name = f"Freq of {self._stat.ancestor}"
        else:
            raise ValueError(f"unknown stat {self._stat.name}")

    @property
    def group(self) -> Group:
        """
        Returns the group this stat belongs to
        """
        return self._group

    @property
    def value(self) -> pd.Series:
        """
        Returns a series of the stat values index on the sample name within the group
        """
        index = [x.name for x in self._group.samples]
        value = [x.gates[self.path].value for x in self._group.samples]
        return pd.Series(value, index=index)

    def _repr_name(self, padding: int) -> str:
        """
        Generates a pretty representation of the current gate
            :param padding: the value to pad space to before placing count data
        """
        node_padding = padding - len(self.name)
        node_repr = f"{self.name}{' '*node_padding}[{self.value:.3f}]"

        return node_repr

class Sample:
    """
    Convenience wrapper around a FlowJo workspace parser sample object.
    Wrapping this class allows for additional convenience functions and the hiding
    of implementation details.
        :param parser: the workspace parser to link the sample to the cytometer data
        :param sample_data: the parser _Sample object
    """
    def __init__(self, parser: _Parser, sample_data: _Sample) -> None:
        self._parser: _Parser = parser
        self._sample: _Sample = sample_data

        self.id: str = self._sample.id
        self.name: str = (self._sample.name).split(os.extsep, 1)[0]

        self.path_fcs: str = self._sample.path
        self.path_data: str = None

        # Make sure to keep track of the state of the data
        self.data_format: str = None
        self.is_compensated: bool = None
        self._data: pd.DataFrame = None

        self.cytometer: Cytometer = None
        try:
            cytometer_id = self.keywords["$CYT"]
        except KeyError:
            pass
        else:
            for cytometer in self._parser.cytometers:
                if cytometer.cyt == cytometer_id:
                    self.cytometer = Cytometer(self._parser, cytometer)
                    break
        
        self.compensation: MTX = self._sample.compensation
        self._transforms: Dict[str, _AbstractTransform] = self._sample.transforms

        self._parameter_names: Dict[str, str] = {}
        # Cannot use $PAR for iteration as Compensation adds additional $P entrees
        # Cannot just iterate as entrees can be missing
        # At most double then PAR will likely(?!) be created, so iterate using that....
        i_params = 2 * int(self.keywords["$PAR"])
        for i in range(0, i_params):
            try:
                param_id = self.keywords[f"$P{i}N"]
            except KeyError:
                continue
            try:
                param_name = self.keywords[f"$P{i}S"]
            except KeyError:
                param_name = ""
            
            self._parameter_names[param_id] = param_name

        # Parameter names must be known for some of the gate node statistics
        self._gates: _Gates = _Gates(self)

    @property
    def keywords(self) -> Dict[str, str]:
        """
        Getter for the keywords. Not very pythonic but cleans-up the __dict__ a lot.
        """
        return self._sample.keywords

    def load_data(self, path: str, format: str, compensated: bool) -> None:
        """
        Loads the data of the sample. Special care needs to be taken of the data type to be loaded.
        FlowJo can export 'scale' and 'channel' formatted data. These needs to be handled uniquely.
        Secondly FlowJo can export compensated and uncompensated data. This also needs a different data handling approach.
            :param path: path to the (with header) exported FlowJo data (in csv format)
            :param type: defines the data type found in path 'scale' or 'channel'. The export type of FlowJo export.
            :param compensated: whether the data in path is compensated
        """
        if not os.path.isfile(path):
            raise ValueError(f"path '{path}' doesnt point to a file")

        if os.path.basename(path).split(os.extsep, 1)[1] != "csv":
            raise ValueError(f"path '{path}' doesnt point to a csv file")

        if format == "scale":
            self.data_format = "scale"
        elif format == "channel":
            # Channel data is transformed (binned) data.
            self.data_format = "channel"
        else:
            raise ValueError(f"unknown format '{format}', format must be 'scale' or 'channel'")

        if compensated:
            self.is_compensated = True
        else:
            self.is_compensated = False

        # os.path.join fixes between OS path differences
        self._data = pd.read_csv(os.path.join(path), encoding="utf-8", on_bad_lines="error")
        self.path_data = path

        # Any gating expect compensated data, so apply compensation
        if not self.is_compensated:
            raise NotImplementedError("uncompensated data has to be compensated before further manipulation. This is currenlty not implemented")
            self.compensate_data()

        # Scale data is untransformed data. All gating assumes transformed data, so apply transform
        if self.data_format == "scale":
            raise NotImplementedError("scale data has to be transformed before further manipulation. This is currently not implemented.")
            self.transform_data()

        # Quick check of the data mainly to make sure the channel data adheres to the expected range
        for column in self._data.columns:
            if min(self._data[column]) < CHANNEL_MIN:
                raise ValueError(f"column '{column}' contains unexpected negative value(s). Is this really 'channel' formatted data?")
            if max(self._data[column]) > CHANNEL_MAX:
                raise ValueError(f"column '{column}' contains unexpected >=1024 value(s). Is this really 'channel' formatted data?")

        # Calculate the gate structure / which-cells are in which-gate
        self._gates._apply_gates()

    def transform_data(self) -> None:
        """
        Applies the data transformations
        """
        # Check if transforms are available and for all entrees

        raise NotImplementedError("data transformations are currently not implemented")

        self.data_format = "channel"
    
    def compensate_data(self) -> None:
        """
        Applies the compensation matrix to the data
        """
        # Check if compensation matrix is available
        
        raise NotImplementedError("data compensation is currently not implemented")

        # Change the compensated column names into "Comp-" + column name

        # Add the compensated parameter names to ._paramater_names
        
        self.is_compensated = True

    def data(self, start_node: str=None, translate: bool=True) -> pd.DataFrame:
        """
        Getter for the data. Transform the dataframe column names from the internal identifiers
        to the correct parameter names. Makes a deepcopy of the data.
            :param start_node: the gate node to retreive the data from
            :param translate: whether to change the column identifiers into the column names
        """
        if self._data is None:
            raise ValueError("sample does not contain any data. Make sure to load_data()")

        if start_node is None:
            # Sample needs to handle all data handling

            # I need to deep copy to not overwrite the original column id's
            data = copy.deepcopy(self._data)

            # Translate column names from identifier to names
            if translate:
                column_names = []
                for column in data.columns:
                    try:
                        name = self._parameter_names[column]
                    except KeyError:
                        name = column
                    if name == "":
                        name = column
                    column_names.append(name)

                data.columns = column_names

        else:
            # Return from a gate node -> gate node takes care of data handling
            data = self.gates[start_node].data(translate=translate)

        # Add sample identifyer
        data["__sample"] = self.name

        return data

    def gate_data(self, start_node: str=None, factor: Dict[str, Dict[str, str]]=None, translate: bool=True) -> pd.DataFrame:
        """
        Getter for the data with gate annotations. Makes a deepcopy of the data
            :param start_node: the gate node to retreive the data from
            :param factor: specifyer for factorization of gates membership. Dict[factor_column_name, Dict[gate_id, factor_level_name]]
            :param translate: whether to change the column identifiers into the column names
        """
        if start_node is None:
            # Sample need to handle the data
            data = self.data(translate=translate)

            # Mitigation for 'PerformanceWarning: DataFrame is highly fragmented'
            # Acquire references to the _in_gates within the gating tree then concat
            in_gate: List[pd.Series] = []
            self.gates._attach_gate(in_gate)
            data = pd.concat([data, *in_gate], axis=1)

            # factorize gate columns
            if factor is not None:
                # Mitigation for 'PerformanceWarning: DataFrame is highly fragmented'
                # First collect all new factors, then concat at once
                new_factors: List[pd.Series] = []

                for factor_name in factor:
                    factor_levels = factor[factor_name]
                    
                    # Check if all factor_levels are available if not, generate warning
                    if sum(data.columns.isin(factor_levels)) != len(factor_levels):
                        missing_levels = []
                        for factor_level in factor_levels:
                            if factor_level not in data.columns:
                                missing_levels.append(factor_level)

                        print(f"while factorizing '{factor_name}' not all levels were found {missing_levels}")

                    factor_slice = pd.Series(np.nan, index=data.index, name=factor_name, dtype="object")

                    for factor_level in factor_levels:
                        try:
                            factor_slice.loc[data[factor_level]] = factor_levels[factor_level]
                        except KeyError:
                            pass
                    
                    new_factors.append(factor_slice)
                
                data = pd.concat([data, *new_factors], axis=1)

        else:
            # Return from a gate node -> gate node takes care of data handling
            data = self.gates[start_node].gate_data(factor=factor, translate=translate)

        return data

    def has_data(self) -> bool:
        """
        Returns whether the sample's data has been loaded
        """
        if self._data is None:
            return False
        return True

    @property
    def gates(self) -> _Gates:
        """
        Returns the gate structure
        """
        return self._gates

    def transforms(self) -> Dict[str, _AbstractTransform]:
        """
        Getter for the transforms data, returns the transforms data adjusted for parameter name. The return
        dictionary is effectively a deepcopy
        """
        transforms = {}
        for key in list(self._transforms.keys()):
            try:
                name = self._parameter_names[key]
            except KeyError:
                name = key

            if name == "":
                name = key

            transforms[name] = self._transforms[key]
        return transforms

    @property
    def count(self) -> int:
        """
        Returns the amount of cells within this gate
        """
        if self._data is None:
            raise ValueError("No data has been loaded. Make sure to call load_data() first")

        return len(self._data.index)

    def __len__(self) -> int:
        raise AttributeError("__len__ is ambiguous for Sample object, for amount of gates use .gates attribute, for data size use .count attribute")

    def __getitem__(self, gate: str) -> None:
        raise AttributeError(".__getitem__ is ambiguous for Sample object, to get a specific gate use the .gates attribute, for data column use .data()")

    def __contains__(self, sample: str) -> None:
        """
        Checks whether the sample id/name exists within the data
        """
        raise AttributeError(".__contains__ is ambiguous for Sample object, for gates check using .gates attribute, for data check using .data()")

    def __repr__(self) -> str:
        output = f"name:   {self.name}\nid:     {self.id}"

        if self.is_compensated:
            output += f"\nformat: comp-{self.data_format}"
        else:
            output += f"\nformat: {self.data_format}"

        if self.compensation:
            output += f"\ncompensation: {self.compensation.name}"

        if self._transforms:
            transform = []
            if self._data is not None:
                for column in self._data.columns:
                    try:
                        name = self._parameter_names[column]
                    except KeyError:
                        transform.append(column)
                    if name == "":
                        transform.append(column)
                    else:
                        transform.append(name)
            else:
                # Fallback for when no data is loaded
                # Does show the name for both the uncompensated data and compensated data
                for key in self._transforms:
                    try:
                        name = self._parameter_names[key]
                    except KeyError:
                        transform.append(key)
                    if name == "":
                        transform.append(key)
                    else:
                        transform.append(name)
            output += f"\nparameters: [{', '.join(transform)}]"

        return output

    def subsample(self, n: int, seed:int=None) -> None:
        """
        Subsamples the dataset to the specified amount of cells
            :param n: amount of events to subsample to.
            :param seed: the seed used for sampling
        """
        if self._data is None:
            raise ValueError("No data has been loaded. Make sure to call load_data() first")

        # If data already contains less events then do no subsampling
        if len(self._data.index) <= n:
            return

        self._data = self._data.sample(n, replace=False, random_state=seed)
        self._data = self._data.sort_index()

        # because the indexes are still valid, no need to reapply gates
        # Yes, but the counts are no longer correct after subsampling; so reapply
        self._gates._apply_gates()

class _Gates:
    """
    Hook into a Sample's or Group's gate data
        :param parent: the Sample/Group object the gates belong to
    """
    def __init__(self, parent: Union[Sample, Group]) -> None:
        self._sample: Union[Sample, Group] = parent

        self._gates: Dict[str, Union[SampleGate, GroupGate]] = {}

        if isinstance(self._sample, Group):
            # Check if this group is based on a parsed group
            if self._sample._group is not None:
                for gate in self._sample._group.gates:
                    gate = self._sample._group.gates[gate]
                    if isinstance(gate, _AbstractGate):
                        self._gates[gate.name] = GroupGate(self._sample, gate, None)
                    elif isinstance(gate, _StatGate):
                        stat = GroupStat(self._sample, gate, None)
                        self._gates[stat.name] = stat
                    else:
                        raise NotImplementedError(f"Unknown groupgate node '{gate}'. Please contact the author.")
            else:
                pass

        elif isinstance(self._sample, Sample):
            for gate in self._sample._sample.gates:
                gate = self._sample._sample.gates[gate]
                if isinstance(gate, _AbstractGate):
                    self._gates[gate.name] = SampleGate(self._sample, gate, None)
                elif isinstance(gate, _StatGate):
                    stat = SampleStat(self._sample, gate, None)
                    self._gates[stat.name] = stat
                else:
                    raise NotImplementedError(f"Unknown gate node '{gate}'. Please contact the author.")

        else:
            raise NotImplementedError("unknown gate parent class")

        self.__iter: int = None

    @property
    def gates(self) -> str:
        return list(self._gates.keys())

    def _apply_gates(self) -> None:
        """
        Applies the gating structure to the dataset. Build boolean masks for each gate defining which cells fall in within the gate
        """
        for gate in self._gates:
            gate = self._gates[gate]
            if isinstance(gate, SampleGate):
                gate._apply_gates()

    def _attach_gate(self, in_gate: List[pd.Series]) -> None:
        """
        Adds the True/False annotation of self._in_gate to the data. Makes sure all indexes are available.
        Recurses into the child-gates
            :param data: a list of boolean gating masks
        """
        for gate in self._gates:
            self._gates[gate]._attach_gate(in_gate, remove=None)

    def __len__(self) -> int:
        """
        Returns the amount of root gate nodes
        """
        return len(self._gates)

    def __getitem__(self, gate: str) -> AbstractGate:
        """
        Returns the specified Gate. Accepts chained gates (gate chains separated by '/')
            :param gate: the sample id or name
        """
        if not isinstance(gate, str):
            raise KeyError(f"gate index should inherit str not '{gate.__class__.__name__}'")

        gates = gate.split("/", 1)

        if gates[0] == "":
            raise KeyError(f"gate node '{gates[0]}' is empty")

        try:
            gate = self._gates[gates[0]]
        except KeyError:
            raise KeyError(f"gate node '{gates[0]}' cannot be found") from None

        if len(gates) > 1:
            return gate[gates[1]]
        else:
            return gate

    def __iter__(self):
        self.__iter = 0
        return self

    def __next__(self):
        keys = list(self._gates.keys())
        
        if len(keys) <= self.__iter:
            raise StopIteration
        
        key = keys[self.__iter]
        self.__iter += 1

        return self._gates[key]

    def __contains__(self, gate: str) -> bool:
        """
        Checks if this gate contains a subgate with the specified name
        """
        gates = gate.split("/", 1)

        gate = gates[0]

        if gate not in self._gates:
            return False
        elif len(gates) > 1:
            return self._gates[gate].__contains__(gates[1])
        else:
            return True

    def __repr__(self) -> str:
        output: List[str] = []

        # Calculate padding
        subnode_pad = 0
        for gate_id in self._gates.keys():
            if len(gate_id) > subnode_pad:
                subnode_pad = len(gate_id)
        subnode_pad += 2

        for gate in self._gates:
            output.append(self._gates[gate]._repr_tree(prefix="", padding=subnode_pad))

        return "\n".join(output)

class Group:
    """
    Convenience wrapper around a FlowJo workspace parser group object.
    This class allows for additional convenience functions and the hiding
    of implementation details. Use the classmethods for proper instantiation
        :param parser: the workspace parser to link identifiers to sample data
        :param group_data: (optional) the parser _Group object; if None, group will be treated as a custom group not related to flowjo.
    """
    def __init__(self, parser: _Parser, group_data: _Group=None) -> None:
        self._parser: _Parser = parser
        self._group: _Group = group_data

        self.name: str = None                   # group name
        self._gates: _Gates = None              # group gates hook

        self._data: Dict[str, Sample] = {}      # sample data
        self._names: List[str] = []             # sample names

        self._parameter_names: Dict[str, str] = None    # group-wide parameter names for the samples, this helps handling samples with deviating parameter names

        self.__iter: int = None

    @classmethod
    def from_wsp(cls, parser: _Parser, group_data: _Group, samples: _Samples):
        """
        Instantiates a Group from a workspace parser
            :param parser: the workspace parser to link identifiers to sample data
            :param group_data: the parser _Group object
            :param samples: sample data hook
        """
        cls = cls(parser, group_data)
        cls.name = cls._group.name

        for sample_id in cls._group.samples:
            cls._data[sample_id] = samples[sample_id]

        for sample_id in cls._data:
            cls._names.append(cls._data[sample_id].name)

        if cls.name != "All Samples":
            cls._check_transform()

        # Gates must be parsed after name lookup for proper parsing of Stat nodes
        cls._name_lookup()
        cls._gates = _Gates(cls)

        return cls

    @classmethod
    def from_samples(cls, parser: _Parser, name: str, samples: List[Sample]):
        """
        Instantiates a Group from a workspace parser
            :param parser: the workspace parser to link identifiers to sample data
            :param name: the group name
            :param samples: a list of Sample's to add to the group
        """
        cls = cls(parser, None)
        cls.name = name

        for sample in samples:
            cls._data[sample.id] = sample

        for sample_id in cls._data:
            cls._names.append(cls._data[sample_id].name)

        cls._check_transform()

        # Gates must be parsed after name lookup for proper parsing of Stat nodes
        cls._name_lookup()
        cls._gates = _Gates(cls)

        return cls

    @property
    def id(self) -> str:
        """
        The group id is identical to the group name
        """
        return self.name

    @property
    def ids(self) -> List[str]:
        """
        Returns the sample unique identifiers
        """
        return list(self._data.keys())

    @property
    def names(self) -> List[str]:
        """
        Getter for sample names
        """
        return self._names

    @property
    def samples(self) -> List[Sample]:
        """
        Getter for sample data
        """
        return [self._data[key] for key in self._data]

    def data(self, start_node: str=None, translate: bool=True) -> pd.DataFrame:
        """
        Returns a combined DataFrame of all samples in this group.
        The returned DataFrame has a new index!
            :param start_node: the gate node to retreive the data from
            :param translate: whether to change the column identifiers into the column names
        """
        data: List[pd.DataFrame] = []
        for sample in self._data:
            try:
                data.append(self._data[sample].data(start_node, translate=False))
            except KeyError as error:
                error_message = error.__str__().strip('"')
                raise KeyError(f"in sample '{self._data[sample].name}' {error_message}").with_traceback(sys.exc_info()[2]) from None

        data = pd.concat(data, ignore_index=True)

        if translate:
            data = self._name_parameters(data)

        return data

    def gate_data(self, start_node: str=None, factor: Dict[str, Dict[str, str]]=None, translate: bool=True) -> pd.DataFrame:
        """
        Returns a combined DataFrame of all samples in this group with gate annotation.
        The returned DataFrame has a new index!
            :param start_node: the gate node to retreive the data from
            :param factor: specifyer for factorization of gates membership. Dict[factor_column_name, Dict[gate_id, factor_level_name]]
            :param translate: whether to change the column identifiers into the column names
        """
        data: List[pd.DataFrame] = []
        for sample in self._data:
            try:
                data.append(self._data[sample].gate_data(start_node, factor=factor, translate=False))
            except KeyError as error:
                error_message = error.__str__().strip('"')
                raise KeyError(f"in sample '{self._data[sample].name}' {error_message}").with_traceback(sys.exc_info()[2]) from None

        data = pd.concat(data, ignore_index=True)

        if translate:
            data = self._name_parameters(data)

        return data

    @property
    def gates(self) -> _Gates:
        """
        Returns the gate structure
        """
        return self._gates

    def keywords(self, keywords: Union[str, List[str]]) -> pd.DataFrame:
        """
        Gets the specified keyword(s) from all samples in the group.
        If keyword doesnt exists returns a np.nan instead.
            :param keywords: the keyword(s) to lookup
        """
        if isinstance(keywords, str):
            keywords = [keywords]

        data: List[pd.Series] = []
        for keyword in keywords:
            data_keyword: List[str] = []
            for sample_id in self._data:
                try:
                    data_keyword.append(self._data[sample_id].keywords[keyword])
                except KeyError:
                    data_keyword.append(np.nan)
            data.append(pd.Series(data_keyword))
        data = pd.concat(data, axis=1)

        # Set correct column and index specifyers
        data.columns = keywords
        # do not use self._names, as te dictionary doesnt have a constant order
        data.index = [self._data[sample_id].name for sample_id in self._data]

        return data

    def transforms(self) -> Dict[str, _AbstractTransform]:
        """
        Returns a general list of transforms. Transforms shouldnt be different between samples
        and the data is not corrected to 'fix' this. This should be fixed inside FlowJo itself. 
        The 'Time' transform is by definition not corrected between samples. Generates a shallow copy.
        """
        transform: Dict[str, _AbstractTransform] = {}

        # Update the dictionary to make sure all parameters are covered
        for sample_id in self._data:
            transform.update(self._data[sample_id]._transforms)

        # Check for transform disparities, this is likely unnecessary. Nonetheless good to double check.
        self._check_transform()

        # 'Fix' parameter names
        names = self._name_lookup()

        output = {}
        for key in list(transform.keys()):
            try:
                name = names[key]
            except KeyError:
                name = key

            if name == "":
                name = key

            output[name] = transform[key]
        
        return output

    def _check_transform(self) -> None:
        """
        Checks if all the transforms (except Time) are identical for all samples 
        """
        samples = list(self._data.keys())

        if not samples:
            return
        
        comparison = self._data[samples[0]]._transforms
        for i in range(1, len(samples), 1):
            for key in comparison:
                # Only compare parameters that are in both datasets
                try:
                    transform_b = self._data[samples[i]]._transforms[key]
                except KeyError:
                    continue
                
                transform_a = comparison[key]
    
                # Time is scaled to fit in the data range, so always uncomparable
                if key == "Time":
                    if type(transform_a) != type(transform_b):
                        print(f"WARNING: 'Time' transform should be of identical type not '{self._data[samples[0]].name}:{transform_a.__class__.__name__}' and '{self._data[samples[i]].name}:{transform_b.__class__.__name__}'")
                    continue

                if transform_a != transform_b:
                    print(f"WARNING: in group '{self.name}' sample '{self._data[samples[0]].name}' & '{self._data[samples[i]].name}' differ in transform of '{key}'")

    def _name_lookup(self) -> Dict[str, str]:
        """
        Resolves column naming conflicts by generating a lookup table.
        Everytime a new sample is added to the group this will have to be regenerated.
            :returns: a Dict[column name : name]
        """
        if self._parameter_names is None:
            names = []
            for sample in self._data:
                names.append(pd.DataFrame(self._data[sample]._parameter_names, index=[sample]))

            # Escape is no samples in the group
            if not names:
                self._parameter_names = {}
                return

            names: pd.DataFrame = pd.concat(names)

            name_dict = {}
            warnings = []
            for column in names.columns:
                unique_names = names[column].dropna().unique()
                unique_names = unique_names[unique_names != ""]
                if len(unique_names) == 0:
                    name_dict[column] = column
                elif len(unique_names) == 1:
                    name_dict[column] = unique_names[0]
                else:
                    warnings.append(f"In group '{self.name}' column '{column}' has multiple [{', '.join(unique_names)}] names. '{unique_names[0]}' is used to name the '{column}' parameter")
                    name_dict[column] = unique_names[0]

            if warnings:
                print("\n".join(warnings))

            self._parameter_names = name_dict

        return self._parameter_names

    def _name_parameters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Renames the column of data from the parameter identifiers to the parameter names
            :param data: the data to rename
        """
        # As some samples parameter identifiers can be named differently (or missing)
        # handle the identifiers uniquely for group wide data exports
        name_dict = self._name_lookup()

        column_names = []
        for column in data.columns:
            try:
                name = name_dict[column]
            except KeyError:
                name = column
            if name == "":
                # Shouldnt happen, but leave it in just in case
                name = column
            column_names.append(name)
        data.columns = column_names

        return data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, sample: Union[str, int]) -> Sample:
        """
        Returns the sample data. Lookup is special. First tries lookup by index.
        If that fails tries lookup by name. Names do not have to be unique.
        So will raise an error if the key would match multiple names
            :param sample: the sample id or name
        """
        if isinstance(sample, int):
            sample = str(sample)

        if not isinstance(sample, str):
            raise KeyError(f"sample index should inherit str or int, not '{sample.__class__.__name__}'")

        try:
            data = self._data[sample]
        except KeyError:
            pass
        else:
            return data

        # Now lookup by name
        count = self.names.count(sample)
        if count == 0:
            raise KeyError(f"unknown sample name/id '{sample}'")
        elif count >= 2:
            raise KeyError(f"sample name '{sample}' is not unique, please use the id")

        for sample_id in self._data:
            if self._data[sample_id].name == sample:
                break

        return self._data[sample_id]

    def __iter__(self):
        self.__iter = 0
        return self

    def __next__(self):
        keys = list(self._data.keys())
        
        if len(keys) <= self.__iter:
            raise StopIteration
        
        key = keys[self.__iter]
        self.__iter += 1
        
        return self._data[key]

    def __contains__(self, sample: str) -> bool:
        """
        Checks whether the sample id/name exists within the data
        """
        if sample in self._data.keys():
            return True

        # Not a matrix id check if name
        for sample_id in self._data:
            if self._data[sample_id].name == sample:
                return True
        
        return False

    def __repr__(self) -> str:
        output = f"name: {self.name}"
        combiner = ',\n  '
        output += f"\nsamples: [\n  {combiner.join(self._names)}\n]"
        return output

    def load_data(self, path: str, format: str, compensated: bool) -> None:
        """
        Loads the data of a group. The algorithm will look in the specified folder for files identical to the sample name.
        Special care needs to be taken of the data type to be loaded.
        FlowJo can export 'scale' and 'channel' formatted data. These needs to be handled uniquely.
        Secondly FlowJo can export compensated and uncompensated data. This also needs a different data handling approach.
            :param path: path to a directory containing the (with header) exported FlowJo data (in csv format)
            :param type: defines the data type found in path 'scale' or 'channel'. The export type of FlowJo export.
            :param compensated: whether the data in path is compensated
        """
        if not os.path.isdir(path):
            raise ValueError(f"path '{path}' doesnt point to a directory")

        # Get a list of all csv files in the directory
        csv_files: Dict[str, str] = {}
        for item in os.listdir(path):
            if not os.path.isfile(os.path.join(path,item)):
                continue

            file_name = item.split(os.path.extsep, 1)
            if file_name[1] != "csv":
                continue

            csv_files[file_name[0]] = os.path.join(path, item)

        if not csv_files:
            raise ValueError(f"path '{path}' doesnt contain any csv files")

        warnings: List[str] = []
        for sample_id in self.ids:
            sample_name = self[sample_id].name
            try:
                csv_path = csv_files[sample_name]
            except KeyError:
                warnings.append(f"no data file found for sample '{sample_name}'")
            else:
                self[sample_id].load_data(csv_path, format, compensated)
        
        if warnings:
            print("\n".join(warnings))

    def subsample(self, n: int=None, seed: int=None) -> None:
        """
        Subsamples all samples in this group to the specified n. 
            :param n: the amount of cells to subsample to. If no n is given, subsamples to the lowest sample.count
            :param seed: the seed used for sampling
        """
        if n is None:
            counts = [self._data[x].count for x in self._data]
            n = min(counts)
        
        for sample_id in self._data:
            self._data[sample_id].subsample(n=n, seed=seed)

class Cytometer:
    """
    Convenience wrapper around a FlowJo workspace parser cytometer object.
    Wrapping this class allows for additional convenience functions and the hiding
    of implementation details.
        :param parser: the workspace parser to link identifiers to matrix data
        :param cytometer_data: the parser _Cytometer object
    """
    def __init__(self, parser: _Parser, cytometer_data: _Group) -> None:
        self._parser: _Parser = parser
        self._cytometer: _Group = cytometer_data

        self.name: str = self._cytometer.cyt
        
        self.compensation: Dict[str, MTX] = {}
        for matrix_id in self._cytometer.transforms:
            try:
                self.compensation[matrix_id] = self._parser.matrices[matrix_id]
            except KeyError:
                self.compensation[matrix_id] = None
        
        self.transforms: Dict[Dict[str, _AbstractTransform]] = {}
        for matrix_id in self._cytometer.transforms:
            self.transforms[matrix_id] = self._cytometer.transforms[matrix_id]

    @property
    def id(self) -> str:
        """
        The cytometer id is identical to the cytometer name
        """
        return self.name

    def __repr__(self) -> str:
        output = f"name: {self.name}\n"
        output += f"compensation: [{', '.join([x for x in self.compensation])}]"

        return output

class Workspace:
    """
    Convenience wrapper around the FlowJo workspace parser.
    Provides a consistant interface to read and interact with the data.
        :param path to the flowjo workspace file
    """
    def __init__(self, path: str=None):
        self.parser = _Parser(path)
        
        self._cytometers = _Cytometers(self)
        self._samples = _Samples(self)
        self._groups = _Groups(self)
        self._compensation = _Compensation(self)

    @property
    def path(self) -> str:
        """
        Getter for the workspace path
        """
        return self.parser.path

    @path.setter
    def path(self, path: str) -> None:
        """
        Setter for the workspace path
            :raises ValueError: if the path is invalid
        """
        self.parser.path = path

        if path is not None:
            self._cytometers._load()
            self._compensation._load()
            self._samples._load()
            self._groups._load()
        else:
            self._cytometers._unload()
            self._compensation._unload()
            self._samples._unload()
            self._groups._unload()

    @property
    def cytometers(self) -> _Cytometers:
        if self.parser.path is None:
            raise ValueError("no data loaded, please set .path to workspace file")

        return self._cytometers

    @property
    def samples(self) -> _Samples:
        if self.parser.path is None:
            raise ValueError("no data loaded, please set .path to workspace file")

        return self._samples

    @property
    def groups(self) -> _Groups:
        if self.parser.path is None:
            raise ValueError("no data loaded, please set .path to workspace file")

        return self._groups

    @property
    def compensation(self) -> _Compensation:
        if self.parser.path is None:
            raise ValueError("no data loaded, please set .path to workspace file")

        return self._compensation

    def load_data(self, path: str, format: str, compensated: bool) -> None:
        """
        Loads the data of the entire workspace. The algorithm will look in the specified folder for files identical to the sample name.
        Special care needs to be taken of the data type to be loaded.
        FlowJo can export 'scale' and 'channel' formatted data. These needs to be handled uniquely.
        Secondly FlowJo can export compensated and uncompensated data. This also needs a different data handling approach.
            :param path: path to a directory containing the (with header) exported FlowJo data (in csv format)
            :param type: defines the data type found in path 'scale' or 'channel'. The export type of FlowJo export.
            :param compensated: whether the data in path is compensated
        """
        if not os.path.isdir(path):
            raise ValueError(f"path '{path}' doesnt point to a directory")

        # Get a list of all csv files in the directory
        csv_files: Dict[str, str] = {}
        for item in os.listdir(path):
            if not os.path.isfile(os.path.join(path,item)):
                continue

            file_name = item.split(os.path.extsep, 1)
            if file_name[1] != "csv":
                continue

            csv_files[file_name[0]] = os.path.join(path, item)

        if not csv_files:
            raise ValueError(f"path '{path}' doesnt contain any csv files")

        warnings: List[str] = []
        for sample_id in self.samples.ids:
            sample_name = self.samples[sample_id].name
            try:
                self.samples[sample_id].load_data(csv_files[sample_name], format, compensated)
            except KeyError:
                warnings.append(f"no data file found for sample '{sample_name}'")

        if warnings:
            print("\n".join(warnings))

    def __repr__(self) -> str:
        output = f"workspace: {self.parser.name}"
        output += f"\n samples[{self.samples.__len__()}]"
        output += f"\n groups[{self.groups.__len__()}]"
        output += f"\n cytometers[{self.cytometers.__len__()}]"
        output += f"\n compensation[{self.compensation.__len__()}]"

        return output

class _Samples:
    """
    Hook into the cytometer sample data
        :param workspace: the parent workspace
    """
    def __init__(self, parent: Workspace) -> None:
        self._workspace: Workspace = parent

        self._data: Dict[str, Sample] = {}
        self._names: List[str] = []

        if self._workspace.parser.path:
            self._load()

        self.__iter: int = None

    def _load(self) -> None:
        """
        Loads the data from parser into the representation
        """
        for sample_id in self._workspace.parser.samples:
            self._data[sample_id] = Sample(self._workspace.parser, self._workspace.parser.samples[sample_id])

        self._names = []
        for sample in self._data:
            self._names.append(self._data[sample].name)

    def _unload(self) -> None:
        """
        Unloads the data
        """
        self._data = {}
        self._names = []

    @property
    def names(self) -> List[str]:
        """
        Getter for sample names
        """
        return self._names

    @property
    def ids(self) -> List[str]:
        """
        Returns the sample unique identifiers
        """
        return list(self._data.keys())

    @property
    def data(self) -> Dict[str, Sample]:
        """
        Getter for all sample data, indexed by unique sample id
        """
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, sample: Union[str, int]) -> Sample:
        """
        Returns the sample data. Lookup is special. First tries lookup by index.
        If that fails tries lookup by name. Names do not have to be unique.
        So will raise an error if the key would match multiple names
            :param sample: the sample id or name
        """
        if isinstance(sample, int):
            sample = str(sample)

        if not isinstance(sample, str):
            raise KeyError(f"sample index should inherit str not '{sample.__class__.__name__}'")

        try:
            data = self._data[sample]
        except KeyError:
            pass
        else:
            return data

        # Now lookup by name
        count = self.names.count(sample)
        if count == 0:
            raise KeyError(f"unknown key '{sample}'")
        elif count >= 2:
            raise KeyError(f"key '{sample}' is not unique, please use the id")

        for sample_id in self._data:
            if self._data[sample_id].name == sample:
                break

        return self._data[sample_id]

    def __iter__(self):
        self.__iter = 0
        return self

    def __next__(self):
        keys = list(self._data.keys())
        
        if len(keys) <= self.__iter:
            raise StopIteration
        
        key = keys[self.__iter]
        self.__iter += 1
        
        return self._data[key]

    def __contains__(self, sample: str) -> bool:
        """
        Checks whether the sample id/name exists within the data
        """
        if sample in self._data.keys():
            return True

        # Not a matrix id check if name
        for sample_id in self._data:
            if self._data[sample_id].name == sample:
                return True
        
        return False

    def __repr__(self) -> str:
        output: List[str] = []
        for sample_id in self.ids:
            output.append(f"{sample_id}: {self[sample_id].name}")

        return "\n".join(output)

class _Groups:
    """
    Hook into the group data of the wsp_parser
        :param workspace: the parent workspace
    """
    def __init__(self, parent: Workspace) -> None:
        self._workspace: Workspace = parent

        self._data: Dict[str, Group] = {}
        self._names: List[str] = []
        
        if self._workspace.parser.path:
            self._load()

        self.__iter: int = None

    def _load(self) -> None:
        """
        Loads the data from parser into the representation
        """
        for group in self._workspace.parser.groups:
            self._data[group] = Group.from_wsp(self._workspace.parser, self._workspace.parser.groups[group], self._workspace.samples)

        self._names: List[str] = list(self._data)

    def _unload(self) -> None:
        """
        Unloads the data
        """
        self._data = {}
        self._names = []

    @property
    def names(self) -> List[str]:
        """
        Getter for group names
        """
        return self._names

    @property
    def ids(self) -> List[str]:
        """
        Getter for the group unique identifiers. This is equal to the group name!
        """
        return self.names

    @property
    def data(self) -> Dict[str, Group]:
        """
        Getter for all group data
        """
        return self._data

    def add(self, name: str, samples: List[str]) -> None:
        """
        Adds a group with identifier name and containing the samples.
            :param name: the name of the group (must be unique)
            :param samples: the samples to add can be in sample name or sample id
        """
        if name in self._data:
            raise ValueError(f"group name '{name}' already exists")

        sample_list = []
        for sample in samples:
            sample_list.append(self._workspace.samples[sample])

        group = Group.from_samples(self._workspace.parser, name, sample_list)

        self._data[group.name] = group
        self._names.append(group.name)

    def remove(self, name: str) -> None:
        """
        Remove the specified group. Only works for user-added Groups
            :param name: group name
        """
        if name not in self._data:
            raise ValueError(f"name '{name}' doesnt point to a group")

        if self._data[name]._group is not None:
            raise ValueError(f"can only remove user-added groups, not '{name}'")

        del self._data[name]
        self._names.remove(name)

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, group: str) -> Group:
        """
        Returns the group
            :param group: the group to return
        """
        if not isinstance(group, str):
            raise KeyError(f"group index should inherit str not '{group.__class__.__name__}'")

        return self._data[group]

    def __iter__(self):
        self.__iter = 0
        return self

    def __next__(self):
        keys = list(self._data.keys())
        
        if len(keys) <= self.__iter:
            raise StopIteration
        
        key = keys[self.__iter]
        self.__iter += 1
        
        return self._data[key]

    def __contains__(self, group: str) -> bool:
        """
        Checks whether the group name exists within the data
        """
        return group in self._names

    def __repr__(self) -> str:
        output: List[str] = []
        for name in self.names:
            output.append(f"{name} [{self._data[name].__len__()}]")

        return "\n".join(output)

class _Cytometers:
    """
    Hook into the cytometer data of the wsp_parser
        :param workspace: the parent workspace
    """
    def __init__(self, parent: Workspace) -> None:
        self._workspace: Workspace = parent

        self._data: Dict[str, Cytometer] = {}
        self._names: List[str] = []

        if self._workspace.parser.path:
            self._load()

        self.__iter: int = None

    def _load(self) -> None:
        """
        Loads the data from parser into the representation
        """
        for cytometer in self._workspace.parser.cytometers:
            self._data[cytometer.cyt] = Cytometer(self._workspace.parser, cytometer)

        self._names: List[str] = list(self._data.keys())

    def _unload(self) -> None:
        """
        Unloads the data
        """
        self._data = {}
        self._names = []

    @property
    def names(self) -> List[str]:
        """
        Getter for cytometer names
        """
        return self._names

    @property
    def ids(self) -> List[str]:
        """
        Getter for the cytometer unique identifiers. This is equal to the cytometer name!
        """
        return self.names

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, cytometer: str) -> Cytometer:
        """
        Returns the specified cytometer
        """
        if not isinstance(cytometer, str):
            raise KeyError(f"cytometer index should inherit str not '{cytometer.__class__.__name__}'")

        return self._data[cytometer]

    def __iter__(self):
        self.__iter = 0
        return self

    def __next__(self):
        keys = list(self._data.keys())
        
        if len(keys) <= self.__iter:
            raise StopIteration
        
        key = keys[self.__iter]
        self.__iter += 1
        
        return self._data[key]

    def __contains__(self, cytometer: str) -> bool:
        """
        Checks whether the cytometer name exists within the data
        """
        return cytometer in self._names

    def __repr__(self) -> str:
        output: List[str] = []
        for name in self.names:
            output.append(name)

        return "\n".join(output)

class _Compensation:
    """
    Hook into the compensation data of the wsp_parser
        :param workspace: the parent workspace
    """
    def __init__(self, parent: Workspace) -> None:
        self._workspace: Workspace = parent

        self._data: Dict[str, MTX] = {}
        self._names: List[str] = []
        
        if self._workspace.parser.path:
            self._load()

        self.__iter: int = None

    def _load(self) -> None:
        """
        Loads the data from parser into the representation
        """
        self._data = self._workspace.parser.matrices

        self._names = []
        for matrix in self._data:
            self._names.append(self._data[matrix].name)

    def _unload(self) -> None:
        """
        Unloads the data
        """
        self._data = {}
        self._names = []

    @property
    def names(self) -> List[str]:
        """
        Getter for compensation matrix names
        """
        return self._names

    @property
    def ids(self) -> List[str]:
        """
        Returns the matrix unique identifiers
        """
        return list(self._data.keys())

    @property
    def data(self) -> Dict[str, MTX]:
        """
        Getter for all compensation matrix data, indexed by unique matrix id
        """
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, matrix: str) -> MTX:
        """
        Returns the compensation matrix. Lookup is special. First tries lookup
        by index. If that fails tries lookup by name. Names do not have to be unique.
        So will raise an error if the key would match multiple names
            :param matrix: the compensation matrix id or name
        """
        if not isinstance(matrix, str):
            raise KeyError(f"compensation index should inherit str not '{matrix.__class__.__name__}'")

        try:
            data = self._data[matrix]
        except KeyError:
            pass
        else:
            return data

        # Now lookup by name
        count = self.names.count(matrix)
        if count == 0:
            raise KeyError(f"unknown key '{matrix}'")
        elif count >= 2:
            raise KeyError(f"key '{matrix}' is not unique, please use the id")

        for matrix_id in self._data:
            if self._data[matrix_id].name == matrix:
                break

        return self._data[matrix_id]

    def __iter__(self):
        self.__iter = 0
        return self

    def __next__(self):
        keys = list(self._data.keys())
        
        if len(keys) <= self.__iter:
            raise StopIteration
        
        key = keys[self.__iter]
        self.__iter += 1
        
        return self._data[key]

    def __contains__(self, matrix: str) -> bool:
        """
        Checks whether the matrix name exists within the data
        """
        if matrix in self._data.keys():
            return True

        # Not a matrix id check if name
        for matrix_id in self._data:
            if self._data[matrix_id].name == matrix:
                return True
        
        return False

    def __repr__(self) -> str:
        output: List[str] = []
        for matrix_id in self.ids:
            output.append(f"{matrix_id}: {self[matrix_id].name}")

        return "\n".join(output)
