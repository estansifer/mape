# Computation of Moist Available Potential Energy

Hosted at `https://github.com/estansifer/mape/`.

Code can be found in `src/`.

Input data is found in `data/`.

Figures will be saved in `output/`.

Writeup is located in `paper.pdf`.

To run the code, enter the src directory and issue the command:

    python run.py

This will reproduce the figures in the paper and the values in the table.

## About the code

`thermo.py` contains all thermodynamic constants and the functions to perform thermodynamic computations.

`problem.py` contains classes to represent a domain of air and an arrangement of parcels within such a domain.

`examples.py` contains code to read in the description of a domain of air from input and produce various artificial cases, and to compare the performance of various algorithms.

`solvers.py` conatins implementations of a variety of algorithms for finding a low enthalpy configuration.

`run.py` conatins code for producing the figures and other output.
