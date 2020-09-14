##############################################################################     ##    ######
#    A.J. Zwijnenburg                   2020-09-15           v1.0                 #  #      ##
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
.data       - returns the data of all cells included in the gate (deepcopy)
.gates      - returns a list of all direct subgates
.count      - returns the amount of cells included in this gate
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
.data       - the data of this sample (deepcopy)
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

:class: Group
A class representing a group of samples
.id         - the unique id of the group (identical to .name)
.name       - the name of the group
.gates      - the group gates (does not have to be identical to the gates of each individual sample)
.ids        - the identifiers of the samples included in this group
.names      - the names of the samples included in this group
.data       - returns a dictionary of the data of each included sample
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

from ._parser_wsp import Parser, Gate, _AbstractGating, CHANNEL_MIN, CHANNEL_MAX
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
        :param data: the parser _Sample object
        :param parent: the parent gate
    """
    def __init__(self, sample: Sample, gate_data: _Gate, parent_gate: Gate=None) -> None:
        self.sample: Sample = sample
        self.parent: Gate = parent_gate
        self._wsp: _Gate = gate_data
        self._gating: _AbstractGating = self._wsp._gating

        self.id = self._wsp.id
        self.name = self._wsp.name
        self.x = self._wsp.x
        self.y = self._wsp.y

        self.boolean = self._wsp.boolean

        self._gates: Dict[str, Gate] = {}
        self._in_gate: pd.Series = None

        for gate in self._wsp.gates:
            gate = self._wsp.gates[gate]
            self._gates[gate.name] = Gate(self.sample, gate, self)

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the data for all events contained in this gate (this takes the entire gate structure into account)
        The data is deepcopied.
        """
        if self.sample._data is None:
            raise ValueError("sample does not contain any data. Make sure to load_data()")

        # Make use of _data attribute to not make unnecessary deepcopies
        data = copy.deepcopy(self.sample._data.loc[self._in_gate])

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
        gates = gate.split("/", 1)

        gate = self._gates[gates[0]]

        if len(gates) > 1:
            return gate[gates[1]]
        else:
            return gate

    def __repr__(self) -> str:
        # Gate stack
        gates = self.name

        parent = self.parent
        while True:
            if parent is not None:
                gates = parent.name + "/" + gates
                parent = parent.parent
            else:
                break

        # If data has not been loaded, no counts can be given
        if self._in_gate is None:
            return f"(Gate[{gates}][x])"

        return f"(Gate[{gates}][{self.count}])"

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

    def __str__(self) -> str:
        return self._gate_repr(prefix="")

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
        self._wsp: _Sample = sample_data

        self.id: str = self._wsp.id
        self.name: str = (self._wsp.name).split(os.extsep, 1)[0]

        self.path_fcs: str = self._wsp.path
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
        
        self.compensation: MTX = self._wsp.compensation
        self._transforms: Dict[str, _AbstractTransform] = self._wsp.transforms
        
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
        return self._wsp.keywords

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

    @property
    def data(self) -> pd.DataFrame:
        """
        Getter for the data. Transform the dataframe column names from the internal identifiers
        to the correct parameter names. Makes a deepcopy of the data.
        """
        if self._data is None:
            raise ValueError("sample does not contain any data. Make sure to load_data()")

        # I need to deep copy to not overwrite the original column id's
        data = copy.deepcopy(self._data)

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
        raise AttributeError("len attribute is ambiguous for Sample object, for amount of gates use .gates attribute, for data size use .count attribute")

    def __getitem__(self, gate: str) -> Gate:
        raise AttributeError(".[] is ambiguous for Sample object, to get a specific gate use the .gates attribute, for data column use .data attribute")

    def __contains__(self, sample: str) -> bool:
        """
        Checks whether the sample id/name exists within the data
        """
        raise AttributeError("contains attribute is ambiguous for Sample object, for gates check using .gates attribute, for data check using .data attribute")

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

        for gate in self._sample._wsp.gates:
            gate = self._sample._wsp.gates[gate]
            self._gates[gate.name] = Gate(self._sample, gate, None)

    def _apply_gates(self) -> None:
        """
        Applies the gating structure to the dataset. Build boolean masks for each gate defining which cells fall in within the gate
        """
        for gate in self._gates:
            self._gates[gate]._apply_gates()

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
        gates = gate.split("/", 1)

        gate = self._gates[gates[0]]

        if len(gates) > 1:
            return gate[gates[1]]
        else:
            return gate

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
    Wrapping this class allows for additional convenience functions and the hiding
    of implementation details.
        :param parser: the workspace parser to link identifiers to sample data
        :param group_data: the parser _Group object
    """
    def __init__(self, parser: _Parser, group_data: _Group) -> None:
        self._parser: _Parser = parser
        self._wsp: _Group = group_data

        self.name: str = self._wsp.name
        self.gates: Dict[str, Gate] = self._wsp.gates

        self._data: Dict[str, Sample] = {}
        self._names: List[str] = []

        for sample_id in self._wsp.samples:
            self._data[sample_id] = Sample(self._parser, self._parser.samples[sample_id])

        for sample_id in self._data:
            self._names.append(self._data[sample_id].name)

    @property
    def id(self) -> str:
        """
        The group id is identical to the group name
        """
        return self.name

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
            :param sample: the sample id or name
        """
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

        for sample_id in self.ids:
            sample_name = self[sample_id].name
            try:
                self[sample_id].load_data(csv_files[sample_name], format, compensated)
            except KeyError:
                print(f"no data file found for sample with name '{sample_name}'")

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
        self._wsp: _Group = cytometer_data

        self.name: str = self._wsp.cyt
        
        self.compensation: List[MTX] = []
        for matrix_id in self._wsp.transforms:
            self.compensation.append(self._parser.matrices[matrix_id])
        
        self.transforms: List[Dict[str, AbstractTransform]] = []
        for matrix_id in self._wsp.transforms:
            self.transforms.append(self._wsp.transforms[matrix_id])

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
        self.parser = Parser(path)
        
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

        for sample_id in self.samples.ids:
            sample_name = self.samples[sample_id].name
            try:
                self.samples[sample_id].load_data(csv_files[sample_name], format, compensated)
            except KeyError:
                print(f"no data file found for sample with name '{sample_name}'")

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

    def _load(self) -> None:
        """
        Loads the data from parser into the representation
        """
        for group in self._workspace.parser.groups:
            self._data[group] = Group(self._workspace.parser, self._workspace.parser.groups[group])

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
        Returns the group data
        """
        return self._data

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, group: str) -> List[Sample]:
        """
        Returns the group
            :param group: the group to return
        """
        return self._data[group]

    def __contains__(self, group: str) -> bool:
        """
        Checks whether the group name exists within the data
        """
        return group in self._names

    def __repr__(self) -> str:
        output: List[str] = []
        for name in self.names:
            output.append(f"{name} [{self.data[name].__len__()}]")

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
        return self._data[cytometer]

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
