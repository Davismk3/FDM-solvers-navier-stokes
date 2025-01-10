Module_1.py:
Within Module_1.py is a simple 1D Cartesian advection/convection and diffusion model. The purpose of Module_1.py was to rigorously teach 
myself the foundations of numerically solving and computationally modeling the advection/convection and diffusion terms found in the 
momentum Navier-Stokes equation.

Module_2.py:
Within Module_2.py is a complete and validated 2D Cartesian fluid/thermo-dynamics model using the Navier-Stokes equation and a heat equation 
for convection/advection and diffusion. 

Module_2_Couette_Validation.py:
Within Module_2_Couette_Validation.py is Module_2.py's model tested against the Couette analytical solution to the Navier-Stokes equations. 
For higher spatial resolutions, the error decreases as expected. 

Module_2_Poiseuille_Validation.py:
Within Module_2_Poiseuille_Validation.py is Module_2.py's model tested against the Poiseuille analytical solution to the Navier-Stokes equations. 
For higher spatial resolutions, the error decreases as expected. 

Module_3.py:
Within Module_3.py is a model similar to Module_3.py, but with a two bounded meshes. There is heat transfer, but not fluid transfer. The idea 
is to ultimately model the core-mantle boundaries of planetary bodies. 
