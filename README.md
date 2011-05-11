Description
-----------
This is code associated with the work done to measure the physical parameters
of a bicycle and the rider of a bicycle. Physical parameters include but are
not limited to the geometry, mass, mass location and mass distribution of the
bicycle rider system. More detail can be found in our papers and the
[website](http://biosport.ucdavis.edu/research-projects/bicycle/bicycle-parameter-measurement/).

Papers
-------------------
These papers detail the methods associated with the experiments and
calculations.

1. Moore, J. K., Hubbard, M., Peterson, D. L., Schwab, A. L., and Kooijman, J.
   D. G. (2010). An accurate method of measuring and comparing a bicycle's
   physical parameters. In Bicycle and Motorcycle Dynamics: Symposium on the
   Dynamics and Control of Single Track Vehicles, Delft, Netherlands.
2. Moore, J. K., Kooijman, J. D. G., Hubbard, M., and Schwab, A. L. (2009). A
   Method for Estimating Physical Properties of a Combined Bicycle and Rider.
   In Proceedings of the ASME 2009 International Design Engineering Technical
   Conferences & Computers and Information in Engineering Conference,
   IDETC/CIE 2009, San Diego, CA, USA. ASME.

References
----------
The methods associated with this software were built on these previous works,
among others. See the previous papers for more details.

1. Kooijman, J. D. G., Schwab, A. L., and Meijaard, J. P. (2008). Experimental
   validation of a model of an uncontrolled bicycle. Multibody System Dynamics,
   19:115–132.
2. Kooijman, J. D. G. (2006). Experimental validation of a model for the motion
   of an uncontrolled bicycle. MSc thesis, Delft University of Technology.
3. Roland J R ., R. D., and Massing , D. E. A digital computer simulation of
   bicycle dynamics. Calspan Report YA-3063-K-1, Cornell Aeronautical
   Laboratory, Inc., Buffalo, NY, 14221, Jun 1971. Prepared for Schwinn Bicycle
   Company, Chicago, IL 60639.

Modules
-------

- BicycleParameters : A module for calculating the physical parameters of a
  bicycle.
- HumanParameters : A yet to be module for calculating the physical parameters
  of a person (while seated on a bicycle). The functionality in paper [2] can
  be found in the matlab code in the PhysicalParameters directory.
- PhysicalParameters : Code used to write paper [1]. It works but is
  depreciated. Most of it is now in BicycleParameters and is much more user
  friendly.
