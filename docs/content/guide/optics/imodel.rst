################################
DNOIS Imaging Model
################################
This chapter describes the imaging model of DNOIS.

**********************************
Pinhole Camera
**********************************
The most cameras can be modeled as **pinhole cameras**. A pinhole camera is specified
by a **optical center** :math:`O` and an **image plane**. The distance between them is called
**focal length** :math:`f` (not equivalent to a lens' focal length). When an object point
emits a bundle of rays, only the one passing :math:`O` can be seen by the camera,
whose intersection with the image plane is called its **perspective projection** or **image point**.
In this way, the image points of all object points form an image of the scene.

Pinhole optical system is implemented as :class:`~dnois.optics.PinholeOptics`.

.. _guide_imodel_cameras_coordinate_system:

Camera's coordinate system
======================================

:numref:`pinhole_camera` illustrates a pinhole camera. The coordinate system herein is
called **camera's coordinate system (CCS)**. The camera's optical center is the origin of this
coordinate system and the image plane is perpendicular to its z-axis. In this way,
the image point of an object point :math:`P(x,y,z)` given :math:`z>0`
is :math:`P'(-\frac{f}{z}x, -\frac{f}{z}y, -f)`. The z-coordinate of an object point
is also called its **depth**.

.. figure:: /_static/img/pinhole_camera.*
    :align: center
    :alt: Pinhole camera and camera's coordinate system (CCS)
    :name: pinhole_camera
    :scale: 50 %

    Pinhole camera and camera's coordinate system (CCS)

Field of view angle
============================================
Perspective projection inherent in pinhole camera implies that it is the "direction
of an object points" that dictates the position of its image point.
Therefore, **field of view (FoV) angle** is defined to describe its direction:

.. math::
    \varphi_x=-\arctan\frac{x}{z},\\
    \varphi_y=-\arctan\frac{y}{z}.

Both of them are in range :math:`(-\pi/2, \pi/2)`. The negative sign is introduced
to ensure the object points with positive FoV angles have positive coordinate of
corresponding image points.

.. _guide_imodel_ccs_inf:

Convention for coordinates of infinite points
=================================================
In optics we often consider object points infinitely far away. In this case,
only their FoV angles matter. Hence the coordinates of them in DNOIS are defined as

.. math::
    (x,y,z)=(-\tan\varphi_x, -\tan\varphi_y, \infty).

where :math:`\varphi_x` and :math:`\varphi_y` are the FoV angles. This convention is applicable
for any CCS coordinate in DNOIS unless otherwise specified.

*******************************************************
Imaging simulation based on point spread function
*******************************************************
Pinhole camera is only an ideal model of realistic cameras, which suffer from
various imperfections and aberrations. Specifically, the light wave emitted by
an object point, through a realistic optical system, typically forms an extended irradiance
distribution, a.k.a. **Point Spread Function (PSF)** rather than a single point
on the image plane. Representing the coordinate on image plane as :math:`(x',y')`,
the PSF of an object point :math:`P(x,y,z)` can be expressed as :math:`p(x',y';x,y,z)`.
Given some object points :math:`\{P_i(x_i,y_i,z_i)\}_{i=1}^N` whose intensities are given as
:math:`I_i`, the image of the scene can be represented as

.. math::
    I(x',y')=\sum_{i=1}^N I_i p(x',y';x_i,y_i,z_i).

Pinhole camera can be regarded as a special case whose PSF is Dirac function.

.. _guide_imodel_ref_model:

Reference Model
===================================
In fact, most optical system can be modeled as a pinhole camera
plus its PSF (aberrations). This pinhole camera is called its **reference model**.
Reference model is required to map image points to object points given depth.

Imaging simulation from images
==========================================
The most common form of imaging simulation is to render an image virtually captured
by a sensor with optical components (e.g. lens) given a *clear* image and PSF.
By "clear" we mean that it is the perspective projection of a scene free from
PSF-blurring. Such a scene is represented by :class:`~dnois.scene.ImageScene`.
In this case, a mapping from pixel locations to object points is
required to find the sources of corresponding PSFs. As mentioned above,
this process is well defined for a camera similar to a pinhole camera
(that is, with a reference model) given depth. Depth can either be a single
value (as an :ref:`external parameter <guide_exparam_external_parameters>`
of :class:`~dnois.optics.PsfImagingOptics`) or a pixel-wise depth map.
Clearly, possible region for object points is a frustum pointed at the
optical center of reference model, which is symmetric w.r.t. to x and y-axis.

Some optical systems may not behavior like a pinhole camera, whose
possible region for object points is not a frustum symmetric w.r.t.
x and y-axis (for example, put a prism before a lens). Therefore
:class:`~dnois.optics.PsfImagingOptics` defines four properties to represent
lower and upper limit of x and y FoV angle. These properties can be computed
in other ways rather than pinhole model in subclasses (for example, by inverse
ray tracing in :class:`~dnois.optics.rt.CoaxialRayTracing`).

