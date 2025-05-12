######################################
Paraxial Optics
######################################
DNOIS models paraxial optics (a.k.a. Gaussian optics or linear optics)
by :class:`~dnois.optics.ParaxialSystem`, which represents a paraxial optical
system with properties like principal points. According to whether the focal
lengths are finite or infinite, :class:`~dnois.optics.ParaxialSystem` is
subclassed by :class:`~dnois.optics.FiniteParaxialSystem` and
:class:`~dnois.optics.InfiniteParaxialSystem`.

*****************************************
Rules for coordinates
*****************************************
The positions of principal points (and focal points or nodal points) are
represented by their coordinates on the optical axis, where smaller
coordinate implies object space. There is no prior
assumption about its zero point, so only their relative values matter.
Here a sign rule is needed for quantities like focal lengths, which
is described as follows:

#.  The object focal length is positive if the object focal point
    is to the left of the object principal point and negative otherwise.
#.  The image focal length is positive if the image focal point
    is to the right of the image principal point and negative otherwise.
#.  When constructing a :class:`~dnois.optics.ParaxialSystem` by calling
    :meth:`~dnois.optics.ParaxialSystem.from_interface`, the radius of curvature
    is positive when it is curved towards image space and negative when
    curved towards object space.

****************************************
Composition
****************************************
One can compose two :class:`~dnois.optics.ParaxialSystem` objects by
calling :meth:`~dnois.optics.ParaxialSystem.compose`, which takes as arguments
another :class:`~dnois.optics.ParaxialSystem` object and their distance and
returns a new :class:`~dnois.optics.ParaxialSystem` object.
A :class:`~dnois.optics.FiniteParaxialSystem` object can be composed with
both finite or infinite system, but a :class:`~dnois.optics.InfiniteParaxialSystem`
object can only be composed with finite one.
