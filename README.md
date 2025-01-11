NOTE: 
While this repository contains original work, I have Prof. Tony Saad from YouTube to thank for teaching me the general algorithms, 
of which I rewrote with numpy and my own comments. I also coupled a heat equation to the final model. 

Math.pdf:
This file serves to document my organized mathematical work for solving the terms found in the Navier-Stokes equations using the
finite volume method. My work for the transient, diffusion, advection/convection, and body force terms was done indepentently, 
whereas the work for pressure was reverse-engineered from Prof. Tony Saad's algorithm.

Module_1.py:
This file is a simple 1D Cartesian advection/convection and diffusion model. The purpose of Module_1.py was to rigorously teach 
myself the foundations of numerically solving and computationally modeling the advection/convection and diffusion terms found in the 
momentum Navier-Stokes equation.

Module_2.py:
This file is a complete and validated 2D Cartesian fluid/thermo-dynamics model using the Navier-Stokes equations and a heat equation 
for convection/advection and diffusion. 

Module_2_Couette_Validation.py:
This file is Module_2.py's model tested against the Couette analytical solution to the Navier-Stokes equations. 
For higher spatial resolutions, the error decreases as expected. 

Module_2_Poiseuille_Validation.py:
This file is Module_2.py's model tested against the Poiseuille analytical solution to the Navier-Stokes equations. 
For higher spatial resolutions, the error decreases as expected. 

Module_3.py:
This file is a model similar to Module_2.py, but with a two meshes that share a boundary. There is heat transfer, but not fluid transfer. The idea 
is to ultimately model the core-mantle boundaries of planetary bodies.
