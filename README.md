# Beam Analysis using Direct Stiffness Method

This is a Python GUI-based application for analyzing beam structures using the **Direct Stiffness Method**. Built using `Tkinter` and `Matplotlib`, it provides a visual and interactive platform for structural analysis.

## Features

- Input Young's Modulus, number of elements, lengths, and moments of inertia
- Apply nodal and element loads (UDL or point)
- Generate:
  - Element stiffness matrices
  - Global stiffness matrix
  - Reduced system matrix
  - Displacement vector
  - Member force vectors
- Visualizations:
  - Beam diagram with loads
  - Resultant plots
- Save results and figures to file

## Technologies Used

- Python 3
- Tkinter for GUI
- Matplotlib for visualization
- NumPy for matrix operations

## How to Run

1. Ensure Python 3 and required libraries (`numpy`, `tkinter`, `matplotlib`) are installed.
2. Run the program:

```bash
python beam_stiffness.py
