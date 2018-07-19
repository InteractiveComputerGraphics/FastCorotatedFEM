<img src="teaser.png" width="1000">
<br>

This project presents a minimalistic implementation of our [paper](https://www.animation.rwth-aachen.de/publication/0561/) about fast corotated FEM using operator splitting.

**Author**: [Tassilo Kugelstadt](https://www.animation.rwth-aachen.de/person/37/), **License**: MIT

## Usage
The demo simulates one deformable armadillo mesh that is fixed at one hand and swings due to gravity.
<br> 
Controls:
- Alt + Mouse: rotate the camera
- Ctrl + Mouse: zoom
- Shift + Mouse: move the camera
- Space: start / pause the simulation
- R: reset the simulation and print timings
- W: toggle wireframe rendering

## Build Instructions

This project is based on [CMake](https://cmake.org/). All external dependencies are included. Simply generate project, Makefiles, etc. using [CMake](https://cmake.org/) and compile the project with the compiler of your choice. The code was tested with the following configurations:
- Windows 10 64-bit, CMake 3.10.1, Visual Studio 2015 and Visual Studio 2017
- Debian 9.4 64-bit, CMake 3.8.1, GCC 6.3.0

## Video

[![Video](https://img.youtube.com/vi/Q5W0x40QRJE/0.jpg)](https://www.youtube.com/watch?v=Q5W0x40QRJE)

## References

* T. Kugelstadt, D. Koschier, J. Bender, "Fast Corotated FEM using Operator Splitting", In Computer Graphics Forum (SCA), 2018