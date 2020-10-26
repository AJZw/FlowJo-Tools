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

Python >= 3.8.1  
pandas >= 1.1.1  
lxml >= 4.5.2 (for matrix & wsp module)  
plotnine >= 0.7.1 (for plot module)  
scikit-learn >= 0.23.2 (for plot module Plotter.add_pca())  
umap-learn >=0.4.6 (for plot module Plotter.add_umap())

## Installation

Copy the folder 'flowjo' with its contents into the project folder of the project and import directly.

## Usage

FlowJo workspace example:

```python
from flowjo.wsp import Workspace
from flowjo.plot import Plotter

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
gate_node = samples.gates["or/chain/multiple/gates/using/backslashes"]

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
group.ids           # A list of the identifiers of all samples contained in this group
group.names         # A list of the names of all samples contained in this group
group.gates         # The group gate structure, this doesnt have to be identical to the sample gate structure!
group.data()        # Concatenated sample data (deepcopy)
group.gate_data()   # Concatenated sample data with gate membership information (deepcopy)
group.keywords("$CYT") # The keyword(s) data of all samples in the group
group.transforms()  # The transforms used on the samples in the group
group["sample id/name"] # Retreive a specific sample contained in the group

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
plot = Plotter(data)
plot.scale.update(group.transforms())
plot.scatter("x", "y", "color")
```

FlowJo data export with gata annotation example:

```python
# This is an example of the python code to run
# For the export instructions see the flowjo/data.py file

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
from flowjo.plot import Plotter, LinearScale, BiexScale

# First load the data into the plotter
plotter = Plotter(CSV("annotated_data.csv"))

# Make sure to set the proper scale for each parameter
plotter.scale["FSC-A"] = LinearScale(begin=0, end=262144)
plotter.scale["CX3CR1"] = BiexScale(end=262144, neg_decade=0, width=-100, pos_decade=4.42)

# Plot a scatter plot with
plot = plotter.scatter(x="CX3CR1", y="FSC-A", c="CCR7")
# Or use the raster plot for more functionalities (but slow)
plot = plotter.raster(x="CX3CR1", y="FSC-A", c="CCR7", c_stat="density")

# Finally lets show the graph to the world
print(plot)

# Or directly save the plot (filename is autogenerated)
plotter.save_jpg(path="export", x="CX3CR1", y="FSC-A", c="CCR7")
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
# Add transform identifyer; if transform is already known the mtx will be rejected by FlowJo
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

## License

[GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
