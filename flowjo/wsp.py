##############################################################################     ##    ######
#    A.J. Zwijnenburg                   2020-09-30           v1.0                 #  #      ##
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
Provides a convenience interface over the flowjo workspace parser

:class: Gate
A gate node representing the data included in the gate
.sample     - returns the Sample object this gate belongs to
.parent     - (if applicable) the parent gate
.id         - the gate's unique identifier
.name       - the gate's name
.x          - the gate's x dimension
.y          - the gate's y dimension
.data()     - returns the data of all cells included in the gate (deepcopy)
.gate_data()- returns the data of all cells included in the gate, with annotated gate membership (deepcopy)
.gates      - returns a list of all direct subgates
.count      - returns the amount of cells included in this gate
.path       - returns the full subgate structure of the current gate node
.transforms - returns the dictionary of parameter transforms
.polygon    - returns a polygon representation of the gate
.[]         - returns the specified subgate
.__len__    - returns the amount of direct subgates
.__contains__ - checks whether the specified gate exists
.__str__    - returns a pretty print of the gate structure

:class: Sample
A class representing a single sample and all its components
.id         - the sample id
.name       - the sample name
.path_fcs   - the path to the source fcs file
.path_data  - the path to the loaded data
.data()     - the data of this sample (deepcopy)
.gate_data()- the data of this sample, with annotated gate membership (deepcopy)
.data_format - whether the internal data is in 'scale' or 'channel' units
.is_compensated - whether the internal data is compensated
.cytometer  - the cytometer this data is acquired on
.compensation - the compensation matrix applied to this sample
.transforms - the data parameters transformation
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
.__len__    - returns the amount of samples in this group
.[]         - returns the specified sample (first lookup by id, then by name)
.__contains__ - whether the group contains the specified sample (first lookup by id, then by name)

:class: Cytometer
A class representing a cytometer
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

from ._parser_wsp import Parser as _Parser, Gate, _AbstractGating, CHANNEL_MIN, CHANNEL_MAX
from ._parser_wsp import Cytometer as _Cytometer, Sample as _Sample, Group as _Group, Gate as _Gate
from .matrix import MTX
from .transform import _Abstract as _AbstractTransform

import pandas as pd
import numpy as np
import os
import copy

class Gate:
    """
    Convenience wrapper around a FlowJo workspace parser gate object.
    Wrapping this class allows for additional convenience functions and the hiding
    of implementation details.
        :param sample: the sample this gate belongs to
        :param data: the parser _Gate object
        :param parent: the parent gate
    """
    def __init__(self, sample: Sample, gate_data: _Gate, parent_gate: Gate=None) -> None:
        self.sample: Sample = sample
        self.parent: Gate = parent_gate
        self._gate: _Gate = gate_data
        self._gating: _AbstractGating = self._gate._gating

        self.id = self._gate.id
        self.name = self._gate.name
        self.x = self._gate.x
        self.y = self._gate.y

        self.boolean = self._gate.boolean

        self._gates: Dict[str, Gate] = {}
        self._in_gate: pd.Series = None

        for gate in self._gate.gates:
            gate = self._gate.gates[gate]
            self._gates[gate.name] = Gate(self.sample, gate, self)

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

    def data(self) -> pd.DataFrame:
        """
        Returns the data for all events contained in this gate (this takes the entire gate structure into account)
        The data is deepcopied.
        """
        if self.sample._data is None:
            raise ValueError("sample does not contain any data. Make sure to load_data()")

        # Make use of _data attribute to not make unnecessary deepcopies
        data = copy.deepcopy(self.sample._data.loc[self._in_gate])

        # Translate column names from identifier to names
        column_names = []
        for column in data.columns:
            try:
                name = self.sample._parameter_names[column]
            except KeyError:
                name = column
            if name == "":
                name = column
            column_names.append(name)

        data.columns = column_names

        return data

    def gate_data(self, factor: Dict[str, Dict[str, str]]=None) -> pd.DataFrame:
        """
        Getter for the data with gate annotations. Makes a deepcopy of the data
            :param factor: specifyer for factorization of gates membership. Dict[factor_column_name, Dict[gate_id, factor_level_name]]
        """
        data = self.data()

        remove = self.path + "/"

        for gate in self._gates:
            self._gates[gate]._annotate_gate(data, remove)

        # factorize gate columns
        if factor is not None:
            #redundant = []
            for factor_name in factor:
                data[factor_name] = np.nan

                factor_levels = factor[factor_name]
                for factor_level in factor_levels:
                    data.loc[data[factor_level], factor_name] = factor_levels[factor_level]

                    #redundant.append(factor_level)

            # Remove now redundant columns
            #data.drop(columns=redundant, inplace=True)

        return data

    @property
    def gates(self) -> List[str]:
        return list(self._gates.keys())

    @property
    def count(self) -> int:
        """
        Returns the amount of cells within this gate
        """
        if self._in_gate is None:
            raise ValueError("Gate has not been applied to the sample data. Make sure to load_data() or apply_gates()")

        return sum(self._in_gate)

    @property
    def transforms(self) -> Dict[str, _AbstractTransform]:
        """
        Returns the sample's parameter transforms
        """
        return self.sample.transforms

    @property
    def polygon(self) -> mpl_path.Path:
        """
        Returns a polygon representation of the _gating. Useful for plotting the gate.
        """
        transform_y = self.sample.transforms[self.y]
        transform_x = self.sample.transforms[self.x]

        return self._gating.polygon(transform_x, transform_y)

    def _apply_gates(self) -> None:
        """
        Applies the gating structure to the dataset. Build boolean masks for each gate defining which cells fall in within the gate
        """
        # For the gate be to applied properly the gate might need to be transformed to the correct dimensions
        transform_x = self.sample._transforms[self.x]
        transform_y = self.sample._transforms[self.y]

        in_gating = self._gating.contains(self.sample._data, transform_x, transform_y)

        # Apply boolean gating here
        if self.boolean:
            if self.boolean == "not":
                in_gating = ~in_gating 
            else:
                raise NotImplementedError(f"boolean transforms of type '{self.boolean}' are not yet implemented")

        # Append to parent gates (if available) to adhere to gate heirarchy
        if self.parent:
            self._in_gate = self.parent._in_gate & in_gating
        else:
            self._in_gate = in_gating

        # Forward signal
        for gate in self._gates:
            self._gates[gate]._apply_gates()

    def _annotate_gate(self, data: pd.DataFrame, remove: str=None) -> None:
        """
        Adds the True/False annotation of self._in_gate to the data. Makes sure all indexes are available.
        Recurses into the child-gates
            :param data: the dataframe to annotate
            :param remove: the prefixed gatenodes to remove in the output column headers
        """
        gate = self.path.split(remove, 1)
        if len(gate) < 2:
            gate = gate[0]
        else:
            gate = gate[1]

        data[gate] = self._in_gate
        
        for gate in self._gates:
            self._gates[gate]._annotate_gate(data, remove)

    def __len__(self) -> int:
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

    def __getitem__(self, gate: str) -> Gate:
        """
        Returns the specified Gate. Accepts chained gates (gate chains separated by '/')
            :param gate: the sample id or name
        """
        if not isinstance(gate, str):
            raise KeyError(f"gate index should inherit str not '{gate.__class__.__name__}'")

        gates = gate.split("/", 1)

        gate = self._gates[gates[0]]

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

    def _gate_repr(self, prefix: str, padding: int) -> str:
        """
        Generate a pretty representation of the gate stack for __str__. 
        Iterative as it needs to scan the entire gate structure.
        The root gate's prefix is always default ("").
            :param prefix: the values to be added to the beginning of the gate representation.
            :param padding: the value to pad space to before placing count data
        """
        if self._in_gate is None:
            node_repr = self.name
        else:
            node_padding = padding - len(self.name)
            node_repr = f"{self.name}{' '*node_padding}[{self.count}]"

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
                    subnode_repr = f" '{self._gates[gate_id]._gate_repr(subnode_prefix, subnode_pad)}"
                else:
                    subnode_prefix = " |"
                    subnode_repr = f" |{self._gates[gate_id]._gate_repr(subnode_prefix, subnode_pad)}"

                node_repr += "\n" + subnode_repr

        else:
            node_repr = f"-{node_repr}"

            for i, gate_id in enumerate(list(self._gates.keys())):
                # The last entree needs a fancy cap
                if len(self._gates) == i + 1:
                    subnode_prefix = f"{prefix}   "
                    subnode_repr = f"{prefix}  '{self._gates[gate_id]._gate_repr(subnode_prefix, subnode_pad)}"
                else:
                    subnode_prefix = f"{prefix}  |"
                    subnode_repr = f"{prefix}  |{self._gates[gate_id]._gate_repr(subnode_prefix, subnode_pad)}"

                node_repr += "\n" + subnode_repr

        return node_repr

    def __repr__(self) -> str:
        # Append parent gates
        if self.parent is not None:
            output = self.parent.path + "/"
        else:
            output = ""
        
        output += self._gate_repr(prefix="", padding=len(self.name) + 2)
        
        return output

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
        cytometer_id = self.keywords["$CYT"]
        for cytometer in self._parser.cytometers:
            if cytometer.cyt == cytometer_id:
                self.cytometer = Cytometer(self._parser, cytometer)
                break
        
        self.compensation: MTX = self._sample.compensation
        self._transforms: Dict[str, _AbstractTransform] = self._sample.transforms
        
        self._gates: _Gates = _Gates(self)

        self._parameter_names: Dict[str, str] = {}
        # Cannot use $PAR for iteration as Compensation adds additional $P entrees
        i = 1
        while True:
            try:
                param_id = self.keywords[f"$P{i}N"]
            except KeyError:
                break
            try:
                param_name = self.keywords[f"$P{i}S"]
            except KeyError:
                param_name = ""
            
            self._parameter_names[param_id] = param_name
            i += 1

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

        self._data = pd.read_csv(path, encoding="utf-8", error_bad_lines=True)
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

        # Add sample identifyer
        self._data["__sample"] = self.name

        # Calculate the gate structure / which-cells are in which-gate
        self._gates._apply_gates()

    def transform_data(self) -> None:
        """
        Applies the data transformations
        """
        # Check if transforms are available and for all entrees

        raise NotImplementedError("data transformations are currently not implemented")

        self.data_format = "channel"
    
    def compensate_date(self) -> None:
        """
        Applies the compensation matrix to the data
        """
        # Check if compensation matrix is available
        
        raise NotImplementedError("data compensation is currently not implemented")

        # Change the compensated column names into "Comp-" + column name

        # Add the compensated parameter names to ._paramater_names
        
        self.is_compensated = True

    def data(self, start_node: str=None) -> pd.DataFrame:
        """
        Getter for the data. Transform the dataframe column names from the internal identifiers
        to the correct parameter names. Makes a deepcopy of the data.
            :param start_node: the gate node to retreive the data from
        """
        if self._data is None:
            raise ValueError("sample does not contain any data. Make sure to load_data()")

        if start_node is None:
            # Sample needs to handle all data handling

            # I need to deep copy to not overwrite the original column id's
            data = copy.deepcopy(self._data)

            # Translate column names from identifier to names
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
            data = self.gates[start_node].data()

        return data

    def gate_data(self, start_node: str=None, factor: Dict[str, Dict[str, str]]=None) -> pd.DataFrame:
        """
        Getter for the data with gate annotations. Makes a deepcopy of the data
            :param start_node: the gate node to retreive the data from
            :param factor: specifyer for factorization of gates membership. Dict[factor_column_name, Dict[gate_id, factor_level_name]]
        """
        if start_node is None:
            # Sample need to handle the data
            data = self.data()
            self.gates._annotate_gate(data)

            # factorize gate columns
            if factor is not None:
                for factor_name in factor:
                    data[factor_name] = np.nan

                    factor_levels = factor[factor_name]
                    for factor_level in factor_levels:
                        data.loc[data[factor_level], factor_name] = factor_levels[factor_level]

        else:
            # Return from a gate node -> gate node takes care of data handling
            data = self.gates[start_node].gate_data(factor=factor)

        return data

    @property
    def gates(self) -> _Gates:
        return self._gates

    @property
    def transforms(self) -> Dict[str, _AbstractTransform]:
        """
        Getter for the transforms data, returns the transforms data adjusted for parameter name. This return
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

    def __getitem__(self, gate: str) -> Gate:
        raise AttributeError(".__getitem__ is ambiguous for Sample object, to get a specific gate use the .gates attribute, for data column use .data()")

    def __contains__(self, sample: str) -> bool:
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
            output += f"\ntransforms: [{', '.join(list(self.transforms.keys()))}]"

        return output

    def subsample(self, n: int) -> None:
        """
        Subsamples the dataset to the specified amount of cells
        """
        if self._data is None:
            raise ValueError("No data has been loaded. Make sure to call load_data() first")

        # If data already contains less events then do no subsampling
        if len(self._data.index) <= n:
            return

        self._data = self._data.sample(n, replace=False)

        # because the indexes are still valid, no need to reapply gates
        # self._apply_gates()

class _Gates:
    """
    Hook into a Sample's gate data
        :param parent: the Sample object the gates belong to
    """
    def __init__(self, parent: Sample) -> None:
        self._sample: Sample = parent

        self._gates: Dict[str, Gate] = {}

        for gate in self._sample._sample.gates:
            gate = self._sample._sample.gates[gate]
            self._gates[gate.name] = Gate(self._sample, gate, None)

        self.__iter: int = None

    @property
    def gates(self) -> str:
        return list(self._gates.keys())

    def _apply_gates(self) -> None:
        """
        Applies the gating structure to the dataset. Build boolean masks for each gate defining which cells fall in within the gate
        """
        for gate in self._gates:
            self._gates[gate]._apply_gates()

    def _annotate_gate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the True/False annotation of self._in_gate to the data. Makes sure all indexes are available.
        Recurses into the child-gates
            :param data: the dataframe to annotate
        """      
        for gate in self._gates:
            self._gates[gate]._annotate_gate(data, remove=None)

    def __len__(self) -> int:
        """
        Returns the amount of root gate nodes
        """
        return len(self._gates)

    def __getitem__(self, gate: str) -> Gate:
        """
        Returns the specified Gate. Accepts chained gates (gate chains separated by '/')
            :param gate: the sample id or name
        """
        if not isinstance(gate, str):
            raise KeyError(f"gate index should inherit str not '{gate.__class__.__name__}'")

        gates = gate.split("/", 1)

        gate = self._gates[gates[0]]

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
            output.append(self._gates[gate]._gate_repr(prefix="", padding=subnode_pad))

        return "\n".join(output)

class Group:
    """
    Convenience wrapper around a FlowJo workspace parser group object.
    This class allows for additional convenience functions and the hiding
    of implementation details. Use the classmethods for proper instantiation
        :param parser: the workspace parser to link identifiers to sample data
    """
    def __init__(self, parser: _Parser) -> None:
        self._parser: _Parser = parser
        self._group: _Group = None

        self.name: str = None
        self.gates: Dict[str, Gate] = {}

        self._data: Dict[str, Sample] = {}
        self._names: List[str] = []

        self.__iter: int = None

    @classmethod
    def from_wsp(cls, parser: _Parser, group_data: _Group):
        """
        Instantiates a Group from a workspace parser
            :param parser: the workspace parser to link identifiers to sample data
            :param group_data: the parser _Group object
        """
        cls = cls(parser)
        cls._group = group_data
        cls.name = cls._group.name
        cls.gates = cls._group.name

        for sample_id in cls._group.samples:
            cls._data[sample_id] = Sample(cls._parser, cls._parser.samples[sample_id])

        for sample_id in cls._data:
            cls._names.append(cls._data[sample_id].name)

        return cls

    @classmethod
    def from_samples(cls, parser: _Parser, name: str, samples: List[Sample]):
        """
        Instantiates a Group from a workspace parser
            :param parser: the workspace parser to link identifiers to sample data
            :param name: the group name
            :param samples: a list of Sample's to add to the group
        """
        cls = cls(parser)
        cls.name = name

        # Check transform equality of samples
        if len(samples) >= 2:
            comparison = samples[0]._transforms
            for i in range(1, len(samples), 1):
                for key in comparison:
                    # Only compare parameters that are in both datasets
                    try:
                        transform_b = samples[i]._transforms[key]
                    except KeyError:
                        continue
                    
                    transform_a = comparison[key]
        
                    # Time is scaled to fit in the data range, so always uncomparable
                    if key == "Time":
                        if type(transform_a) != type(transform_b):
                            raise ValueError(f"'Time' transform should be of identical type not '{samples[0].name}:{sample_transform[key].__class__.__name__}' and '{samples[i].name}:{transform[key].__class__.__name__}'")
                        continue

                    if transform_a != transform_b:
                        raise ValueError(f"sample '{samples[0].name}:{key}'-'{samples[i].name}:{key}' differ in parameter transforms")

        for sample in samples:
            cls._data[sample.id] = sample

        for sample_id in cls._data:
            cls._names.append(cls._data[sample_id].name)

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

    def data(self, start_node: str=None) -> pd.DataFrame:
        """
        Returns a combined DataFrame of all samples in this group.
        The returned DataFrame has a new index!
            :param start_node: the gate node to retreive the data from
        """
        data: List[pd.DataFrame] = []
        for sample in self._data:
            data.append(self._data[sample].data(start_node))
        return pd.concat(data, ignore_index=True)

    def gate_data(self, start_node: str=None, factor: Dict[str, Dict[str, str]]=None) -> pd.DataFrame:
        """
        Returns a combined DataFrame of all samples in this group with gate annotation.
        The returned DataFrame has a new index!
            :param start_node: the gate node to retreive the data from
            :param factor: specifyer for factorization of gates membership. Dict[factor_column_name, Dict[gate_id, factor_level_name]]
        """
        data: List[pd.DataFrame] = []
        for sample in self._data:
            data.append(self._data[sample].gate_data(start_node, factor))
        return pd.concat(data, ignore_index=True)

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
        data.index = self._names

        return data

    def transforms(self) -> Dict[str, _AbstractTransform]:
        """
        Returns a general list of transforms. Transforms shouldnt be different between samples
        and the data is not corrected to 'fix' this. This should be fixed inside FlowJo itself. 
        The 'Time' transform is by definition not corrected between samples. 
        """
        transform: Dict[str, _AbstractTransform] = {}

        # Update the dictionary to make sure all parameters are covered
        for sample_id in self._data:
            transform.update(self._data[sample_id]._transforms)

        # Check for transform disparities
        for sample_id in self._data:
            sample_transform = self._data[sample_id]._transforms
            for key in sample_transform:
                # Time is scaled to fit in the data range, so always uncomparable
                if key == "Time":
                    if type(sample_transform[key]) != type(transform[key]):
                        raise ValueError(f"'Time' transform should be of identical type not '{sample_transform[key].__class__.__name__}' and '{transform[key].__class__.__name__}'")
                    
                    continue
                
                if sample_transform[key] != transform[key]:
                    raise ValueError(f"transform '{key}' differs in transforms")

        # 'Fix' parameter names
        names = self._data[sample_id]._parameter_names

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

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, sample: str) -> List[Sample]:
        """
        Returns the sample data. Lookup is special. First tries lookup by index.
        If that fails tries lookup by name. Names do not have to be unique.
        So will raise an error if the key would match multiple names
            :param sample: the sample id or name
        """
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
                self[sample_id].load_data(csv_files[sample_name], format, compensated)
            except KeyError:
                warnings.append(f"no data file found for sample with name '{sample_name}'")
        
        if warnings:
            print("\n".join(warnings))

    def subsample(self, n: int=None) -> None:
        """
        Subsamples all samples in this group to the specified n. If no n is given, subsamples
        to the lowest sample.count
        """
        if n is None:
            counts = [self._data[x].count for x in self._data]
            n = min(counts)
        
        for sample_id in self._data:
            self._data[sample_id].subsample(n=n)

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
        
        self.compensation: List[MTX] = []
        for matrix_id in self._cytometer.transforms:
            self.compensation.append(self._parser.matrices[matrix_id])
        
        self.transforms: List[Dict[str, AbstractTransform]] = []
        for matrix_id in self._cytometer.transforms:
            self.transforms.append(self._cytometer.transforms[matrix_id])

    @property
    def id(self) -> str:
        """
        The cytometer id is identical to the cytometer name
        """
        return self.name

    def __repr__(self) -> str:
        output = f"name: {self.name}\n"
        output += f"compensation: [{', '.join([x.name for x in self.compensation])}]"

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
                warnings.append(f"no data file found for sample with name '{sample_name}'")

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

    def __getitem__(self, sample: str) -> List[Sample]:
        """
        Returns the sample data. Lookup is special. First tries lookup by index.
        If that fails tries lookup by name. Names do not have to be unique.
        So will raise an error if the key would match multiple names
            :param matrix: the sample id or name
        """
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
            self._data[group] = Group.from_wsp(self._workspace.parser, self._workspace.parser.groups[group])

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

    def __getitem__(self, group: str) -> List[Sample]:
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

    def __getitem__(self, matrix: str) -> List[Sample]:
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
