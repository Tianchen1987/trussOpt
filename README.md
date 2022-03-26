# trussOpt

### Installation
Clone the entire repository
Install cvxpy (https://www.cvxpy.org) with pip
Install Jupyter https://jupyter.org with pip

Alternatively place the files on Google Drive and run it on Google Colab (you still need to modify the path to the folder) 

### Functionality

This code performs a linear elastic analysis on 3D truss frames. It also formulates two optimization problems, the first (linear programming) minimizes the volume by optimizing the cross sectional area, while adhering to yield stress constraints. The second (second order cone programming) minimizes the global compliances by optimizing the cross sectional area, while adhering to yield stress constraints and volume constraints.
