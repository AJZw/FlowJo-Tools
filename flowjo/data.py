##############################################################################     ##    ######
#    A.J. Zwijnenburg                   2020-09-09           v1.0                 #  #      ##
#    Copyright (C) 2020 - AJ Zwijnenburg          GPLv3 license                  ######   ##
##############################################################################  ##    ## ######

## Copyright notice ##########################################################
# FlowJo Tools provides a python API into FlowJo's .wsp files.
# Copyright (C) 2020 AJ Zwijnenburg
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
This is the OLD method of moving data from FlowJo to pandas. It is way more convenient
to make use of the Workspace() method (see README.md)

Allows for the exporting of FlowJo data to a gate-annotated pd.DataFrame

:class: _Abstract
Abstract FlowJo Data file, use for further subclassing
Implement here functions that do not depends on how-to-parse the data

:class: CSV
Data class for basic flowjo fcs export in csv format

:class: CSVData
Data class for parsing a FlowJo fcs data with sample/gate annotation.

For this to work follow the follow protocol:

This class auto-loads and annotates exported flowjo (as csv) data. The scripts assumes all necessary files are 
located in one folder (here named source_path). It needs additional txt files for sample, and gate identification

Step-by-step walkthrough
> The name/sample metadata identification assumes that the export flowjo files are named in a specific way:
    The name consists of a part with name/sample metadata, followed with the gate node of that data
    The metadata and gate are separated by an underscore.
    Example: [meta1]_[meta2]_[node].csv     ->  mouse_day8_0#1.csv
> For the name/sample metadata do the following:
	1 - Add a new keyword to the samples (Workspace > Add keyword) - for example SampleID
	2 - Define a underscore delimited name of all sample specific metadata. 
	    Example: SampleID 'AZ27_Blood_d35_1'    (4 metadata parts here - AZ27, Blood, d35, 1)
	3 - Write the specifications in name_setup.txt as a comma-separated single line. 
        The specification has to be identical for ALL samples
        Example: exp_nr,tissue,day,sample_nr    (make sure to not end with an comma, but do add a whiteline afterwards)

> Gate/Node identification
    Use the following system to define a name for the gate:
	Each gate gets an integer number, which is defined in gate_setup.txt (read further for the specs)
	Every nested step adds a '#' in the gates name.
	The gates must include every (grand)parent gate.
	    Example: source gate is named: '1'. Then a children gates can be named '1#2'.
		
> The gate_setup.txt is a comma-separated list of gate id, gate label, gate group (1#2, CD8+, CD3+)
    The gate group is optionally (3rd column). All gate labels with the same gate group are grouped together in a single data column 
	The gate can be fully written out (1#2), or you can use a wildcard (*#1)
	Then the wildcards functions as a template and fills in every possible gate node in the place of the wildcard
    A fully written out gate always has priority over a wildcarded gate. If two wildcarded gates could expand
    into the same gate, the last defined one has priority
> Do the following:
    1 - Give each (exported) node in de gating strategy an unique name according to the above rules, eg:

    Gating Strategy:        Node naming:
    CD3                     - 0
     |- CD4+                - 0#1
     |   |--- CD27-         - 0#1#3
     |   '--- CD27+         - 0#1#4
     |
     '- CD8+                - 0#2
         |--- CD27-         - 0#2#5
         '--- CD27+         - 0#2#6

    2 - Write the specifications in the gate_setup.txt as a comma-separated file, eg:
    0,CD3+,CD3
    0#1,CD4+,Tcells
    0#2,CD8+,Tcells
    0#1#3,CD27-,CD27
    0#1#4,CD27+,CD27
    0#1#5,CD27-,CD27
    0#1#6,CD27+,CD27

> Now mass export the files 
  > Select all gates of interest in one sample
      Make sure that (except for toplevel gates) all childgates have their parentgate included (so from the starting point the entire tree should be exported) 
  > Use Edit > Select Equivalent Nodes to select all nodes
  > Export -> make sure to export csv-channel (scaled binned data) / scale (unscaled data) of compensated data
  > Set 'Include Header' to 'Stain'
  > Change the name to:
  	- no prefix
  	- Body: custom -> edit:
  		- SampleID + FJ_LAST_UNIQUE_POP_NAME
  		- Note: FJ_LAST_UNIQUE_POP_NAME sometimes includes the parent gate, which will be automatically removed by the algorithm
  	- no suffix

> Export and place all these files in one folder, without any other non-relevant files
> Now import this library into a python environment and instantiate CSVData('here the path to the export directort')
> After instantiation or setting .directory, use the build() function to construct the data file
> The data can be found using the .data attribute

"""

from __future__ import annotations
from typing import List

import pandas as pd
import numpy as np
import os
import itertools
import warnings

class _Abstract():
    """
    Abstract data class provides the interface for a flowjo.data type.
    For new data types inherit this class
        :param path: path to the data to import
    """
    def __init__(self, path: str=None) -> None:
        self._path: str = None
        self.data: pd.DataFrame = None

        if path:
            self.path = path

    @property
    def path(self) -> str:
        """
        Getter for path to data file
            :returns: path to data file
        """
        return self._path

    @path.setter
    def path(self, path: str) -> None:
        """
        Setter for the data file path. Will parse the data.
            :param path: path to data file
            :raises ValueError: is file doesnt exist
        """
        if not os.path.isfile(path):
            raise ValueError(f"path '{path}' doesnt point to a valid file")

        self._path = path

        self.parse()

    def parse(self) -> None:
        """
        Parses the file in .path into .data
            :raises NotImplementedError:
        """
        raise NotImplementedError("implement in inherited class")

    def save(self, path) -> None:
        """
        Saves the data file in csv format (encoded 'utf-8') to path
            :raises ValueError: if path cannot be used for saving
        """
        if self.data is None:
            raise ValueError(".data attribute doesnt contain any data")

        if os.path.isdir(path):
            raise ValueError(f"path '{path}' points to a directory. Please use a path to a (non-existing) file")

        if os.path.isfile(path):
            value = input(f"Path '{path}' already exists.\n Overwrite? Y(es), N(o): ")
            while True:
                if value == "Y" or value == "Yes":
                    break
                elif value == "N" or value == "No":
                    return
                else:
                    value = input("Unknown input. Overwrite? Y(es), N(o): ")
        
        self.data.to_csv(path, sep=",", index=False, encoding="utf-8")

    def load(self, path) -> None:
        """
        Loads a previously saved data file from csv format into the class.
            :param path: path to file to load
            :raises ValueError: if file cannot be loaded
        """
        if not os.path.isfile(path):
            raise ValueError(f"path '{path}' does not point to a file")

        self._path = path

        self.data = pd.read_csv(path, encoding="utf-8", error_bad_lines=True)

class CSV(_Abstract):
    """
    Class representing a basic csv exported FlowJo file
    """
    def __init__(self, path: str = None):
        super().__init__(path)
        
        if self.path:
            self.parse()

    def parse(self) -> None:
        """
        Parses the file in .path into .data
        """
        self.data = pd.read_csv(self.path, encoding="utf-8", error_bad_lines=True)

class CSVGated(_Abstract):
    """
    Class representing a gate annotated exported FlowJo (csv) files
        :param directory: path to the directory containing the exported FlowJo Files
    """
    def __init__(self, directory: str = None) -> None:
        super().__init__(None)
        self._directory: str = directory

        self._name_setup: str = None
        self._gate_setup: str = None
        self._data_files: List[str] = []

        self.name_map: pd.Series = None
        self.gate_map: List[pd.DataFrame] = None

        # If directory is given, try to find all source data files
        if directory:
            self._parse_directory()
        
        # If all source data has been found, we can parse the data
        if self.directory is not None and self.name_map is not None and self.gate_map is not None:
            self.parse()

    @property
    def path(self) -> str:
        """
        This class is special and doesnt have .path. It has a .directory.
        .path is only available when this class is used with .load()
            :raises NotImplementedError:
        """
        if not self.path:
            raise NotImplementedError("CSVGated object uses .directory not .path")
        else:
            return self.path

    @path.setter
    def path(self, path: str) -> None:
        """
        This class is special and doesnt have .path. It has a .directory.
            :raises NotImplementedError:
        """
        raise NotImplementedError("CSVGated object uses .directory not .path")

    @property
    def directory(self) -> str:
        """
        Getter for directory
        """
        return self._directory
    
    @directory.setter
    def directory(self, directory: str) -> None:
        """
        Setter for directory. If new directory is invalid; reverts to old directory
            :param directory: path to the flowjo exported csv files
            :raises ValueError: if directory is invalid
        """
        old_directory = self._directory

        self._directory = directory
        try:
            self._parse_directory()
        except ValueError as error:
            self._directory = old_directory
            raise error

    @property
    def name_setup(self) -> str:
        """
        Getter for name_setup filepath
        """
        return self._name_setup

    @name_setup.setter
    def name_setup(self, path: str) -> None:
        """
        Setter for name_setup filepath. Exception safe
            :param path: the path to name_setup.txt
            :raises ValueError: if path doesnt point to a name_setup file
        """
        if not os.path.isfile(path):
            raise ValueError(f"file '{path}' does not exist")

        old_name_setup = self._name_setup

        self._name_setup = path

        try:
            self._load_name_setup()
        except ValueError as error:
            self._name_setup = old_name_setup
            raise error
    
    @property
    def gate_setup(self) -> str:
        """
        Getter for gate_setup filepath
        """
        return self._gate_setup

    @gate_setup.setter
    def gate_setup(self, path: str) -> None:
        """
        Setter for gate_setup filepath. Exception safe
            :param path: the path to gate_setup.txt
            :raises ValueError: if path doesnt point to a gate_setup file
        """
        if not os.path.isfile(path):
            raise ValueError(f"file '{path}' does not exist")

        old_gate_setup = self._gate_setup

        self._gate_setup = path

        try:
            self._load_gate_setup()
        except ValueError as error:
            self._gate_setup = old_gate_setup
            raise error

    def parse(self) -> None:
        """
        Parses the directory data into .data
            :raises ValueError: if parsing/collapsing/annotating failes at any point
        """
        data = self._load_data()
        data = self._collapse_data(data)

        if "group" not in pd.concat(self.gate_map).columns:
            data = self._annotate_single_gates(data)
        else:
            data = self._collapse_gates(data)
            data = self._annotate_group_gates(data)

        data.index = list(range(0, len(data.index)))

        self.data = data

    def _check_data(self) -> None:
        """
        Not implemented, if you feel like it yourself, go ahead
        Checks the validity of the data.csv fil"]e
            :raises ValueError: if data file is invalid
        """
        # Check if all data files have the same amount of columns
        # Check if data file name holds enough section for all nodes & name_meta
        data_ncol: int = 0
        sample_ncol: int = 0

        # Additional data verification possibilities:

        # gate cannot have "gaps" or only "wildcard" options
        # final child node cannot be wildcard
        # "__index" name is reserved
        # "__node" name is reserved
        # "__file_id" name is reserved

    def _parse_directory(self) -> None:
        """
        Checks the flowjo directory for the necessary files. 
        Assumes all csv / txt files in this directory are to be used.
        Exception safe
        """
        if not os.path.isdir(self.directory):
            raise ValueError(f"path '{self.directory}' does not point to a valid directory")

        items: List[str] = os.listdir(self.directory)

        # For exception safety parse into local attributes first
        data_files: List[str] = []
        name_setup: str = None
        gate_setup: str = None

        # Extract files
        for item in items:
            extension: str = item.split(os.extsep, 1)[1]
            if extension == "csv":
                data_files.append(os.path.join(self.directory, item))

            elif extension == "txt":
                if item == "name_setup.txt":
                    name_setup = os.path.join(self.directory, item)
                if item == "gate_setup.txt":
                    gate_setup = os.path.join(self.directory, item)

        # Check files
        if not data_files:
            raise ValueError(f"path '{self.directory}' does not contain any csv files")

        # Passed the error check, now set the files paths
        self._data_files = data_files

        if name_setup:
            self.name_setup = name_setup
        if gate_setup:
            self.gate_setup = gate_setup

    def _load_gate_setup(self) -> None:
        """
        Loads and parses the gate setup from the gate_setup path.
        Constructs a node to metadata lookup table. Exception safe
            :raises ValueError: if parsing failes
        """
        if not self._gate_setup:
            raise ValueError("no gate_setup.txt defined, please add path using the .gate_setup attribute")

        csv = pd.read_csv(self._gate_setup, header=None)

        if len(csv.columns) == 2:
            csv.columns = ["node", "label"]
        elif len(csv.columns) == 3:
            csv.columns = ["node", "label", "group"]
        else:
            raise ValueError("gate_setup should contain 2 (node, label) or 3 (node, label, group) columns")

        self.gate_map = self._resolve_gates(csv)

    def _load_name_setup(self) -> None:
        """
        Loads the name_setup and parses it to map the name specifiers
            :raises ValueError: if parsing failes
        """
        if not self._name_setup:
            raise ValueError("no name_setup.txt defined, please add path using the .name_setup attribute")

        csv = pd.read_csv(self._name_setup, header=None)

        if len(csv.columns) < 1 and len(csv.index) != 1:
            raise ValueError("name_setup should atleast contain one column, and exactly one row")

        self.name_map = csv.iloc[0]

    def _load_data(self) -> pd.DataFrame:
        """
        Loads all samples and binds them into one big data.frame. At the same time adds the sample meta data information
        Also tries to add an unique identifyer to cells
            :returns: a dataframe of all files bound together. This will cotnain duplicates
            :raises ValueError: if loading failes
        """
        if not self._data_files:
            raise ValueError("no data to load, please use the .directory attribute to set the directory to get the data from")

        if self.name_map is None:
            self._load_name_setup()

        if self.gate_map is None:
            self._load_gate_setup()

        # Check for overlap between name and gate setups
        for value in pd.concat(self.gate_map)["label"]:
            if value in self.name_setup:
                raise ValueError(f"name_setup contains column header '{value}' which is also defined as label in gate_setup. All labels & sample headers should be unique.")

        if "group" in self.gate_map[0].columns:
            for value in pd.concat(self.gate_map)["group"]:
                if value in self.name_setup:
                    raise ValueError(f"name_setup contains column header '{value}' which is also defined as group in gate_setup. All group & sample headers should be unique.")

        temp = []
        for path in self._data_files:
            # Load data
            try:
                data = pd.read_csv(path, encoding="utf-8", error_bad_lines=True)
            except pd.errors.ParserError as error:
                raise ValueError(f"unable to read data from {path}. Is it a valid csv file?") from error
            
            # Check for overlapping header / sample_map / gate_map names
            # Duplicates will cause overwriting otherwise
            for value in self.name_map:
                if value in data.columns:
                    raise ValueError(f"data file '{path}' contains column header '{value}' which is also defined in name_setup. All column headers should be unique.")

            for value in pd.concat(self.gate_map)["label"]:
                if value in data.columns:
                    raise ValueError(f"data file '{path}' contains column header '{value}'' which is also defined as label in gate_setup. All labels & data headers should be unique.")

            if "group" in self.gate_map[0].columns:
                for value in pd.concat(self.gate_map)["group"]:
                    if value in data.columns:
                        raise ValueError(f"data file '{path}' contains column header '{value}'' which is also defined as group in gate_setup. All group & data headers should be unique.")

            # No data, so immediatly continue
            if len(data.index) == 0:
                continue

            # Index the file
            data = self._index_data(data, path)

            # Retrieve sample meta info from sample name
            file_name = os.path.basename(path).split(os.extsep, 1)[0]
            properties = file_name.split("_")
            if len(properties) < (len(self.name_map) + 1):
                raise ValueError(f"data file '{file_name}' does not contain enough sections for sample and node metadata")
            gate_node = properties[-1]
            properties = properties[:len(self.name_map)]

            # Add sample meta info
            data[self.name_map] = properties
            data["__file_id"] = path
            data["__node"] = gate_node
            
            temp.append(data)
        
        return pd.concat(temp, ignore_index=True)

    def _collapse_data(self, data: pd.DataFrame, sort: bool=False) -> pd.DataFrame:
        """
        Adds gate node metadata and collapses duplicates into one event using the index parameter
           :param data: all samples rbind together (see _load_data)
           :param sort: whether to perform a sort before collapsing, probably not necessary and very slow, turn it on when the function requests it
        """
        # gate identifyer
        gate_names = pd.concat(self.gate_map)["node"]

        for name in gate_names:
            if name in data.columns:
                raise(f"data headers should be unique from gate identifiers, {name} cannot be part of the data headers")

        data[gate_names] = False

        # Split samples
        data_split: List[(str, pd.DataFrame)] = [(x, y) for x, y in data.groupby("__node", as_index=True)]

        # Make sure the samples are sorted based on index, otherwise collapsing goes wrong
        if sort:
            for i in range(0,len(data_split)):
                data_split[i][1].sort_values("__index", inplace=True)
        
        # Add gate meta_data
        for data in data_split:
            data[1][data[0]] = True

        # Now collapse by walking backwards through the node tree
        # you have to walk, because there is no guarantee that the level 1 nodes have non-duplicated indexes
        for node_level in reversed(list(range(0, len(self.gate_map)))):
            for node_nr in range(0, len(self.gate_map[node_level].index)):
                # get node 
                node = self.gate_map[node_level].iloc[node_nr]["node"]

                # Find the to-be-collapsed-into-node, aka the parent-node
                parent_node = node.split("#")
                if len(parent_node) == 1:
                    if node_level != 0:
                        raise ValueError(f"(this is possible?) parentless non-top level node '{node}', please fully export and define the gate tree")

                    # top-level nodes have no parents, so no need to collapse
                    continue
                else:
                    parent_node = "#".join(parent_node[:-1])

                # Merge into parent node
                # True values have been annotated
                # False values have not been annotated, so can be overwritten if the other is True

                # First get relevant parent node data
                try:
                    is_parent = next(i for i, x in enumerate(data_split) if x[0] == parent_node)
                except StopIteration:
                    raise ValueError(f"undefined parent node '{parent_node}', please make sure you have defined and exported the entire gate tree")
                # Get child node data
                is_node = next(i for i, x in enumerate(data_split) if x[0] == node)

                # Get overlap index
                is_index = data_split[is_parent][1]["__index"].isin(data_split[is_node][1]["__index"])
                
                if sum(is_index) != len(data_split[is_node][1].index):
                    raise ValueError(f"not all event of node {node} are in parent node {parent_node}. Please reexport.")

                # We use custom indexes so ignore the pandas indexes. Therefor use numpy for index-agnostic functions
                if not np.array_equal(data_split[is_parent][1].loc[is_index]["__index"], data_split[is_node][1]["__index"]):
                    raise ValueError(f"the indexes of node {node} are unequal to parent node {parent_node}. Please rerun with sort enabled.")
                
                # Merge gate metadata
                data_split[is_parent][1].loc[is_index, gate_names] = np.logical_or(data_split[is_parent][1].loc[is_index, gate_names], data_split[is_node][1][gate_names])

        # Now that we have collapsed the data, we can remove the non source entrees
        source_nodes = self.gate_map[0]

        output = []
        for data in data_split:
            if (source_nodes["node"] == data[0]).any() >= 1:
                output.append(data[1])

        output = pd.concat(output)

        # remove now useless columns
        output = output.drop(labels=["__index", "__node", "__file_id"], axis="columns")

        return output

    def _collapse_gates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        collapse_samples just gives a 'has-this-annotation' TRUE/FALSE output, often gates represent a factorized input
        this function collapses those gates together as a factor of one property. 
        If events are part of multiple factors, it randomly distributes them to one or the other
            :param data: data collapsed DataFrame
        """
        # All gates/nodes
        meta_gate = pd.concat(self.gate_map)

        if "group" not in meta_gate.columns:
            raise ValueError("cannot collapse gates if group names havent been added to gate_setup")

        # Find groups
        groups: List[(str, pd.DataFrame)] = [(x, y) for x, y in meta_gate.groupby("group", as_index=True)]
        group_names = [x[0] for x in groups]

        # Add necessary columns
        data[group_names] = None

        # COllapse
        for group in groups:
            group_name = group[0]
            group_data = group[1]

            # Transform to np array to subvert indexing error
            group_labels = np.array(group_data["node"])
            group_splice = data[group_labels]

            # This function has a side-effect! It also manipulates duplicate to keep track of duplicate/random sampling!
            # keep track of number of group resolves
            duplicate = 0
            def collapse_group(row, labels, duplicate):
                group_name = labels[row]

                if len(group_name) == 0:
                    return None
                elif len(group_name) == 1:
                    group_name = group_name
                else:
                    group_name = np.random.choice(group_name, 1)
                    duplicate += 1

                return group_name[0]
            
            data[group_name] = group_splice.apply(lambda x: collapse_group(x, group_labels, duplicate), axis="columns")

            if duplicate > 0:
                warnings.warn(f"{duplicate} events which belonged to multiple entrees of {group_name}. These events were randomly distributed to one of these labels")

        # Clean data
        data = data.drop(labels=meta_gate["node"], axis="columns")
        
        return data

    def _resolve_gates(self, gates: pd.DataFrame) -> pd.DataFrame:
        """
        Resolves all possible gate names. Generates a lookup DataFrame for all possible gate names. Gate nodes will be read from 
        the input gates DataFrame. (Grand)parent nodes can be written as *. In that case all possible combinations will be made.
        Fully declared (no wildcard) gates have priority. Any wildcard drops that priority.
            :param gates: node + label, wildcard is allowed (*), any wildcard cancels priority
            :raises ValueError: if gate resolving/parsing failes
        """
        # Get max nest_level
        gates["node_level"] = gates["node"].apply(lambda x: x.count("#"))
        # Sort by nest_level to make sure you start with parent nodes and finish with child nodes
        gates.sort_values("node_level", axis="index", ascending=True, inplace=True)

        node_levels: List[pd.DataFrame] = [y for x, y in gates.groupby("node_level", as_index=False)]

        resolved_nodes: List = []
        # Iterate through nest levels
        for node_level in node_levels:
            # Per nest_level check the nodes
            resolved_node_level = []

            for node in node_level.index:
                resolved = self._parse_gates(node_level.loc[node], resolved_nodes)
                resolved_node_level.append(resolved)
            
            # All nodes for this level have been resolved, now parse output to main
            resolved_nodes.append(pd.concat(resolved_node_level))

        # All gates have been expanded, now it is time to clean up the list
        # remove duplicates and prioritize unexpanded entrees
        for i, node_level in enumerate(resolved_nodes):
            # Sort entrees to allow duplicate detection
            node_level.sort_values("node", axis="index", ascending=True, inplace=True)

            # Manually walk over the DataFrame to get rid of double cases
            if len(node_level.index) == 1:
                continue
            
            # Setup temporary
            temp: List[pd.Series] = []

            previous_node = 0
            temp.append(node_level.iloc[previous_node])

            for node in range(1, len(node_level.index)):
                # Check if node duplicate
                if temp[previous_node]["node"] == node_level.iloc[node]["node"]:
                    if not temp[previous_node]["expanded"] and not node_level.iloc[node]["expanded"]:
                        # Check if noth are not expanded
                        raise ValueError("two identical non-expanded gates. Did you define a gate twice?")
                    elif not temp[previous_node]["expanded"]:
                        # Previous one is not expanded, so keep it
                        pass
                    elif not node_level.iloc[node]["expanded"]:
                        # Current one is not expanded so that it
                        temp[previous_node] = node_level.iloc[node]
                    else:
                        # Both expanded, last defined overwrites, so take that one
                        temp[previous_node] = node_level.iloc[node]
                
                else:
                    previous_node += 1
                    temp.append(node_level.iloc[node])

            resolved_nodes[i] = pd.concat(temp, axis="columns").T

        return resolved_nodes

    def _annotate_single_gates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Changes the single node information into name labels
            :param data: the gate (non grouped) annotated data
        """
        gate_labels = pd.concat(self.gate_map)
        gate_labels.index = gate_labels["node"]

        new_columns = list(data.columns)

        for i in range(0, len(new_columns)):
            if (new_columns[i] == gate_labels["node"]).any():
                new_columns[i] = gate_labels.loc[new_columns[i], "label"]
        
        data.columns = new_columns

        return data

    def _annotate_group_gates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Changes the grouped node information into the name labels
            :param data: the gate (grouped) annotated data
        """
        gate_labels = pd.concat(self.gate_map)
        gate_labels.index = gate_labels["node"]

        groups = gate_labels["group"].unique()

        for group_name in groups:
            data[group_name] = data[group_name].apply(lambda x: gate_labels.loc[x, "label"] if x is not None else None)

        return data

    @staticmethod
    def _parse_gates(gate: pd.Series, tree_nodes: List[pd.DataFrame]=[]) -> pd.DataFrame:
        """
        Takes one gate and generates all possible gates
            :param gate: the gate to parse
            :param tree_nodes: the lower nodes ((grand)parent etc)
            :raises ValueError: if gate parsing failed
        """
        gate_nodes = gate["node"].split("#")
        
        # keep track of expansion for priority selection
        expanded = False

        # storage list for all nested gates
        temp = [[] for i in range(len(gate_nodes))]

        # go through the nesting one by one
        for i, node_level in enumerate(gate_nodes):
            # Now it can be given or wildcard
            if node_level == "":
                raise ValueError("gate_setup contains an empty node. The entire node tree must be defined")
            elif node_level == "*":
                expanded = True
                # All nodes allowed, so lookup in tree_nodes and get the relevant node out
                try:
                    parent_nodes = tree_nodes[i]["node"].apply(lambda x: x.split("#")[i])
                except IndexError:
                    raise ValueError(f"cannot lookup nodes on level {i}. Did you define a bottommost node as wildcard?")
                # Because of the nesting some duplicates can now be in the nodes, so remove them
                temp[i] = pd.unique(parent_nodes)
            else:
                # Gate is given, directly parse to temp
                temp[i] = [node_level]

        # Now we have all values possible per node, so time to expand them in all possible combinations
        temp = list(itertools.product(*temp))
        # Merge them back into the gate structure
        temp = ["#".join(x) for x in temp]

        # Wrap the output as DataFrame and add the necessary metadat
        temp = pd.DataFrame(temp, columns=["node"])
        temp["label"] = gate["label"]
        temp["expanded"] = expanded
        if "group" in gate:
            temp["group"] = gate["group"]

        return temp

    @staticmethod
    def _index_data(dataframe: pd.DataFrame, data_file: str) -> pd.DataFrame:
        """
        Tries to generate a unique identifyer for a cell. This is done by combining different
        parameters into one. This effectively creates an (almost) unique hash
            :param dataframe: input data
            :param data_file: file name, used for more precise error messages
        """
        dataframe["__index"] = dataframe.apply(lambda x: "_".join(x.astype(str)), axis="columns")
        
        # Make sure indexes are UNIQUE! Otherwise eventually merging will merge duplicates into one row, which will cause plotting errors
        duplicates = dataframe.duplicated(subset="__index", keep="first")
        
        if sum(duplicates) != 0:
            warnings.warn(f"sample '{data_file}' contains {sum(duplicates)} duplicated indexes - duplicates are removed")
            dataframe = dataframe.loc[duplicates]

        return dataframe
