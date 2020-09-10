# FlowJo Tools

A library with (handy) tools for the handling of FlowJo related data

## Authors

AJ Zwijnenburg

## Requirements

Python >= 3.8.1  
lxml >= 4.5.2 (for matrix module)  
pandas >= 1.1.1 (for export module)

## Installation

Copy the folder 'flowjo' with its contents into the project folder of the project and import directly.

## Usage

Matrix tools example:  

```python
from flowjo.matrix import MTX
import os

# Get list of files .mtx files representing single stain compensation matrixes
path = "singles"
files = os.listdir(path)

# Combine the single stain compensations
matrix = None
for single in files:
    single = MTX(os.path.join(path, single))

    if matrix:
        matrix += single
    else:
        matrix = single

# Export combined matrix
matrix.name = "combined"
matrix.save("combined.mtx")
```

FlowJo data export with gata annotation example:

```python
# This is an example of the python code to run
# For the export instructions see the flowjo/data.py file

from flowjo.data import CSVGated

parser = CSVGated("export_directory")
parser = parser.build()
facs_data = parser.data

```

## Version Info

v1.0 - Implemented the compensation matrix tools and flowjo data annotated gate export protocol

## License

[GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
