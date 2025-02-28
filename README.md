# Navier-Stokes and Heat Equation Finite Volume Solver

This repository contains a series of Python modules implementing the finite volume method for solving the Navier-Stokes equations (including advection/convection, diffusion, and pressure terms) and a coupled heat equation. While the code and math work here is original, the algorithms were taken from **Prof. Tony Saad’s** YouTube lectures: https://www.youtube.com/@ProfessorSaadExplains

---

## Contents

1. [Overview](#overview)  
2. [Mathematical Documentation](#mathematical-documentation)  
3. [Modules](#modules)  
   - [Module_1.py](#module_1py)  
   - [Module_2.py](#module_2py)  
   - [Module_2_Couette_Validation.py](#module_2_couette_validationpy)  
   - [Module_2_Poiseuille_Validation.py](#module_2_poiseuille_validationpy)  
   - [Module_3.py](#module_3py)  
4. [Acknowledgments](#acknowledgments)  
5. [License](#license)

---

## Overview

This project explores numerical methods for fluid and thermal simulations using the finite volume method. The Navier-Stokes equations for incompressible flow are solved with discrete approximations of the advection/convection, diffusion, and pressure terms. A heat equation is also coupled to the solver to model thermal transport within the fluid.

---

## Mathematical Documentation

- **Math.pdf**  
  This document outlines the detailed derivations and methodologies used to discretize the Navier-Stokes equations using the finite volume method. It includes:
  - Transient term discretization
  - Diffusion term discretization
  - Advection/convection term discretization
  - Body force (source) term handling
  - Pressure term derivation (algorithm inspired by Prof. Tony Saad’s approach)

---

## Modules

### Module_1.py
A simple 1D Cartesian model focusing on advection, convection, and diffusion. This serves as a foundational learning tool for understanding the numerical implementation of the advection/convection and diffusion terms in the momentum equations.

### Module_2.py
A complete and validated 2D Cartesian fluid and thermodynamics model incorporating:
- Navier-Stokes equations for incompressible flow
- Coupled heat equation for thermal transport

### Module_2_Couette_Validation.py
This script uses **Module_2.py** to model Couette flow and compares the results with the known analytical solution. The error decreases with increasing spatial resolution, demonstrating the accuracy of the finite volume discretization.

### Module_2_Poiseuille_Validation.py
This script uses **Module_2.py** to model Poiseuille flow and compares the results with the known analytical solution. Similarly, error diminishes as the mesh is refined.

### Module_3.py
A 2D model similar to **Module_2.py**, but with two separate meshes sharing a boundary. This configuration allows for thermal coupling between regions without fluid transfer. The long-term objective is to apply this approach to geophysical simulations, such as modeling the core-mantle boundaries of planetary bodies.

---

## Acknowledgments

- **Prof. Tony Saad (YouTube)**: Provided the general algorithmic framework for handling the pressure terms and other concepts in finite volume formulations.  

---

## License

Open-source
