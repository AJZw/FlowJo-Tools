# FlowJo Tools

A library with (handy) tools for the handling of FlowJo related data.  
• The 'data' module allows for the loading and saving of CSV data.  
• The 'wsp' module parses and exposes a FlowJo wsp file and allows for the annotation and export of gate annotated data.  
• The 'plot' module provides FlowJo-like (and more) plotting functions for the data retreived by the data or wsp modules. Also provides an interface for dimensional reduction methods.  
• The 'transform' module provides basic transformations for plotting.  
• The 'matrix' module allows the manipulation (and import, export) of FlowJo mtx compensation matrices.

## Authors

AJ Zwijnenburg

## Requirements

python >= 3.8.1  
pandas >= 1.3.0  
lxml >= 4.5.2 (for matrix & wsp module)  
plotnine >= 0.7.1 (for plot module)  
scikit-learn >= 0.23.2 (for plot module Plotter.add_pca() & Plotter.density_overlay())  
umap-learn >=0.4.6 (for plot module Plotter.add_umap())

## Installation

Copy the folder 'flowjo' with its contents into the project folder of the project and import directly.  
Please, make sure the requirements are met.

## Quickstart

A quick start for quick results!  
For a full list of attributes and function see 'Usage' or the documentation inside the modules.  
  
This 'flowjo' module provides an interface with FlowJo's workspace (.wsp) files.  
It extends FlowJo's capabilities and presents FlowJo's data as a Panda matrix.  
In the workflow FlowJo still has to be used for compensation and gating.  
  
Workflow:  
• Use FlowJo to setup the compensation and gating. Note: All parameters starting with a double underscore "__" are reserved by the module implementation.  
• Export the fcs data in csv format:  
•• Press 'right mouse button' on the (selection of) sample(s)  
•• Press 'Export / Concatenate Populations'  
•• Set the output format to 'CSV - Channel values'  
•• Set 'Include header' to 'Parameter'  
•• Set 'Parameters' to 'All compensated parameters'  
•• Enable 'Advanced Options'  
•• Clear the 'Prefix'  
•• Set 'Body' to 'Custom' and use 'Edit' to remove 'FJ_LAST_UNIQUE_POP_NAME'  
•• Keep 'Suffix' empty  
•• Press 'Export'
• Save the workspace as a .wsp file

• Now that we have used FlowJo to setup up the data for us, we will continue using this 'flowjo' module:

```python
# The flowjo.wsp module contains components for the reading of .wsp files
from flowjo.wsp import Workspace

# Lets assume we have stored all data in the local 'mydata' directory
# First we have to read the workspace file 
workspace = Workspace("mydata/workspace.wsp")

# We need to add the csv data to this workspace
# The format and compensation status have to be filled-in
workspace.load_data("mydata", format="channel", compensated=True)

# Now we can have a look at the data within the .wsp file
# Check the samples using:
print(workspace.samples)
# Check the groups using:
print(workspace.groups)

# All data can be identically manipulated as a group or single sample, let's use a single sample as example:
# Request a sample using the sample's name
sample = workspace.samples["sample_1"]
# You can check the gates like this:
print(sample.gates)
# You can select all events in a gate as follows:
gate = sample.gates["gate_name"]

# You can grab all events in this gate:
# This returns a Panda's DataFrame
gate_data = gate.data()


# You might want to plot this data
# Therefore a useful plotting module is available
from flowjo.plot import Plotter

# The plotter works with workspace/group/sample data
plot = Plotter(sample)

# The plotter needs to know how to transform the data
plot.transforms.update(sample.transforms())

# Plot a scatter plot with
plot = plotter.scatter(x="CX3CR1", y="CD27", c="CCR7")
# Or use the raster plot
plot = plotter.raster(x="CX3CR1", y="CD27")
# Additional statical operations can be performed on the rasterized data
plot = plotter.raster_special(x="CX3CR1", y="CD27", c="CCR7", c_stat="mean")

# Finally lets show the graph to the world
print(plot)

# For further details check the 'usage' or the function documentation inside the modules
```

## Usage

These code-examples provide an overview of the commonly used functions and attributes.  
The examples are split up based on the to-be-explained module.  
  
FlowJo workspace example:

```python
from flowjo.wsp import Workspace

# Load the wsp file
workspace = Workspace("path/to/workspace.wsp")

# You can retreive the components of the workspace like this:
samples = workspace.samples
groups = workspace.groups
cytometers = workspace.cytometers
compensation = workspace.compensation

# All the data components have unique identifiers, (example uses the .samples component)
sample_identifiers = workspace.samples.ids
# For ease of use each component also has a name allotted to it
sample_names = workspace.samples.names
# Both can be used to retreive the data of a specific component
sample = workspace.samples["sample id/name"]

# Each sample contains many useful attributes
sample.id           # sample id
sample.name         # sample name
sample.data()       # sample data (deepcopy)
sample.gate_data()  # sample data with gate membership information (deepcopy)
sample.count        # the amount of events in this sample
sample.cytometer    # the cytometer this data is acquired on
sample.compensation # this sample's compensation matrix
sample.transforms() # the data parameter scale transformations (shallow copy)
sample.keywords     # the FCS keywords
sample.gates        # the sample gate structure

# The measured data is not stored in the .wsp file and should be added manually
# Make sure the export format (channel or scale) and compensation state are correct. (And you export with parameter names)
sample.load_data("path/to/exported_compensated_channel_data.csv", format="channel", compensated=True)

# The events contained in a (sub)gate can be retreive:
gate_node = samples.gates["gate_name"]
gate_node = samples.gates["or/chain/multiple/gates/using/forward_slashes"]

# Each gate node also contains many useful attributes
gate_node.id        # the gate's unique identifier
gate_node.name      # the gate's name
gate_node.parent    # the parent gate (if applicable)
gate_node.sample    # the sample this gate belong to
gate_node.x         # the gate's x dimension
gate_node.y         # the gate's y dimension
gate_node.data()    # returns the data of all cells included in the gate (deepcopy)
gate_node.gate_data() # returns the data of all cells included in the gate with gate membership information (deepcopy)
gate_node.transforms() # the gate_node's sample data parameter scale transformation (shallow copy)
gate_node.count     # returns the amount of cells included in this gate
gate_node.gates     # the subgate structure
gate_node.polygon() # returns a polygon representation of the gate (handy for plotting)

# Each group contains the following data:
group = workspace.groups["all samples"]
group.id            # the group unique identifier (identical to .name)
group.name          # the group name
group.ids           # a list of the identifiers of all samples contained in this group
group.names         # a list of the names of all samples contained in this group
group.gates         # the group gate structure, this doesnt have to be identical to the sample gate structure!
group.data()        # concatenated sample data (deepcopy)
group.gate_data()   # concatenated sample data with gate membership information (deepcopy)
group.keywords("$CYT") # the keyword(s) data of all samples in the group
group.transforms()  # the transforms used on the samples in the group
group["sample id/name"] # retreive a specific sample contained in the group

# Each cytometer contains the following data:
cyto = workspace.cytometers["cytometer id/name"]
cyto.id             # the unique id of the cytometer (identical to .name)
cyto.name           # the name of the cytometer
cyto.compensation   # all compensation matrixes defined in the fcs-files for this cytometer ('Acquisition-defined')
cyto.transforms     # the cytometer's DEFAULT transformations

## An example to retreive the data of a specific group of samples ##
# First create a new group containing the samples of interest
workspace.groups.add(
    "new_group_name",
    ["sample id/name 1", "sample id/name2", "sample id/name3"]
)
group = workspace.groups["new_group_name"]

# You can downsample the samples to equal numbers
group.subsample(n=3000)

# Export the data without or with gate membership annotation
data_without_gate = group.data()
data_with_gate = group.gate_data()

# You might want to only export data/gates from a specific gate node (also works for .data())
data = group.gate_data(start_node="this/is/the/starting/node")

# And maybe you want to factorize multiple gates into a single factor
factor = {"factor_name":{
    "level_a":"name a",
    "level_b":"name b"
}}
data = group.gate_data(factor=factor)

# The exported data can be plotted with correct scales as follows:
# First assign the data to a plotter
from flowjo.plot import Plotter
plot = Plotter(data)
# The scale transformations has to be set manually; here the transforms are updated from the wsp information. Without transformation data, the plotter cannot rasterize or generate the proper axis ticks & labels.
plot.transforms.update(group.transforms())
plot.scatter("x", "y", "color")
```

FlowJo data export with gate annotation example:

```python
# This is an example of the python code to run
# For the export instructions see the flowjo/data.py file
# It's a lot less work to use the flowjo.wsp interface

from flowjo.data import CSVGated

data = CSVGated("export_directory")
facs_data = data.data

# Parsed data can be saved and loaded
data.save("annotated_data.csv")
data.load("annotated_data.csv")
```

FlowJo like plotting of the data:

```python
from flowjo.data import CSV
from flowjo.plot import Plotter, LinearScale, BiexScale, save_raster

# First load the data into the plotter
plotter = Plotter(CSV("annotated_data.csv"))

# Make sure to set the proper scale for each parameter
plotter.transforms["FSC-A"] = LinearScale(l_begin=0, l_end=1023, g_begin=0, g_end=262144)
plotter.transforms["CX3CR1"] = BiexScale(l_begin=0, l_end=1023, g_end=262144, neg_decade=0, width=-100, pos_decade=4.42)

# Plot a scatter plot with
plot = plotter.scatter(x="CX3CR1", y="FSC-A", c="CCR7")
# Or plot a (density) raster plot
plot = plotter.raster(x="CX3CR1", y="FSC-A")

# Finally lets show the graph to the world
print(plot)

# Or directly save the plot
save_raster(plot, name="plot", path="export")
```

Matrix tools example:  

```python
from flowjo.matrix import MTX
import os

# Get list of files .mtx files representing single stain compensation matrixes
path = "singles"
files = os.listdir(path)

# Construct the 'combined' matrix
combined = MTX(os.path.join(path, files[0]))
combined.name = "combined"
# Add transform identifyer; if the transform is already known, the mtx will be (silently) rejected by FlowJo
combined.id = "9999aaaa-aaaa-aaaa-aaaa-aaaaaaaa9999"

# Add the remaining single stain matrixes
for single in files[1:]:
    combined.matrix += single.matrix

# Export combined matrix
combined.save("combined.mtx")
```

## Version Info

v1.0 - Implemented the compensation matrix tools and flowjo data annotated gate export protocol  
v1.1 - Implemented the basic plotting functionalities  
v1.2 - Implemented scatter_3d and convenience saving functions  
v1.3 - Implemented show_3d  
v1.4 - Implemented FlowJo wsp parser  
v1.5 - Implemented factorized gate membership export  
v1.6 - Implemented PCA dimensional reduction  
v1.7 - Implemented Correlation line graph  
v1.8 - Implemented range gates  
v1.9 - Implemented basic histogram graph  
v1.10 - Bugfix: not all parameter names were correctly extracted / group gates now show properly  
v1.11 - Scale ticks/labels now generator based, better error messages  
v1.12 - GroupGates now properly implemented  
v1.13 - Implemented proper histogram graph and (much) faster density raster plots  
v1.14 - Improved implementation of transformations and rasterization  
v1.15 - Implemented masking  
v1.16 - Implemented flowjo gate and density overlays  
v1.17 - Implemented statistics gate nodes  
v1.18 - Implemented reverse hyperbolic sin & logicle transforms. Updated to pandas 1.3  
v1.19 - Implemented contribution and improved correlation graphs

## License

[GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
