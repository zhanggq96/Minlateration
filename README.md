# Minlateration
An optimization-based approach to multilateration with an unknown number of targets

## Overview
[Multilateration](https://en.wikipedia.org/wiki/Multilateration) is the technique of using only distance measurements from stations to determine the location of a target, with no information about the direction from which this distance was measured. Given a coordinate and a measured distance, the set of all points for which the target could be located forms a circle; the coordinate is the circle center and the distance is the circle radius. Multilateration is a classic problem in navigation and target tracking, and also applicable to  problems such as using time-domain reflectometry to determine locations of faults in modem connections.

![][multilateration intro]

This library focuses on the case where the targets and stations are stationary in two dimensions (i.e. only 2D spatial information is required). It is designed to handle cases where there is a small amount of noise in the data, there are multiple stations and multiple targets, and the number of targets is unknown.

## Requirements and Documentation

Language: Python 3  
Libraries: scipy, matplotlib

Usage:

Define circles ref as a list of [[x, y], radius], as done in multilateration.py. The only function to call is locate_intersections.

function locate_intersections

| Field                | Type    | Default | Description                                                                                                                                                                                       |
|----------------------|---------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| circles_ref          | list    | -       | list of [(x,y),r,label] where label is optionally a reference back to an object wrapper for each circle.                                                                                          |
| xlim                 | tuple   | None    | If not None, x-limits of all the circles. If None, they are determined automatically.                                                                                                             |
| ylim                 | tuple   | None    | If not None, y-limits of all the circles. If None, they are determined automatically.                                                                                                             |
| num_lat_clusters     | int     | None    | If not None, the number of targets to locate. If None, it is determined automatically                                                                                                             |
| clustering_threshold | numeric | None    | If not None, the clustering threshold for guessing the number of faults in multilateration. Decreasing this threshold increases the amount of targets guessed. If None, determined automatically. |
| plot_circles_on_iter | boolean | True    | Whether or not to generate plots visualizing estimated target locations on each iteration                                                                                                         |
| verbose              | boolean | True    | Verbosity    

(Note, in the case where a target has two significantly overlapping circles with two intersection points - both intersections can be returned. See the example below.)

## Multilateration of a single target

With only one station, there are infinitely many points on the circle where the targets could be located. With two stations whose distance circles intersect, the possible target location is reduced down to two possible points. With three or more stations that intersect at a single point, the target location can be isolated to that point.

<p float="left">
  <img src="https://i.imgur.com/xluV3kJ.png" width="300" /> 
  <img src="https://i.imgur.com/g3DI1Yg.png" width="300" />
</p>

However, even with accurate sensors, it is unlikely that three circles would intersect at exactly the same point at which the target is located. This means that attempts to tackle this problem analytically will almost always fail. Instead, an optimization-based approach is used [0]; it infers a target's location by finding the point which minimizes the squared euclidean distances between this point and all the circle circumferences. Let **p** denote the target's location, **x**<sub>i</sub> the ith station's location and *r* the radius of the circle. Note that **p** and **x** are 2-dimensional vectors. The function *L* which calculates the euclidean distance between the target and all *k* circles is then

![][sum loss]

The optimal value of **p**, **p**<sup>*</sup> is then found as

![][argmin p]

This problem is quadratic but not convex. It is also worth noting that the optimization method used (scipy.optimize: SQLSP) requires a Jacobian (gradient) function. It can be shown that that gradient for *L* is

![][gradient]

where we note that this is in vectorized notation (to get the gradient for a single component j of **p**, replace *only* the final **p**-**x**<sub>i</sub> with **p**<sub>j</sub>-**x**<sub>ij</sub>).

## Multi-target multilateration of a fixed number of targets > 1

Now the method above is extended to the case where multiple stations can be tracking a known fixed number of targets > 1.

![Five targets][five circles]

In the above diagram, we can reasonably assume that there are five targets; it is known that the five targets are somewhere in there, and their locations must all be found. A subproblem of this is to determine which circles (stations) are sharing the same target (this sounds similar to a clustering problem!) - each circle is associated with one target, and each target can have multiple circles associated with it.

In multi-target multilateration, the locations of the five targets are randomly initialized within the plot limits and then each circle is assigned to the target which is closest to it. Then, each of each of these groups has one target and multiple circles, so it becomes possible to perform single-target multilateration on them. From this, the assignments of the target locations change. This should place the guesses of where the faults are in slightly better locations than the random initialization. At this point, we can again reassign each circle to the target location closest to it.

The idea is to keep repeating this procedure many times until it converges to a local minimum. This cluster determining mechanism is essentially identical in principle to k-means clustering [1].

<p float="left">
  <img src="https://i.imgur.com/ALQQ6Py.png" width="280" /> 
  <img src="https://i.imgur.com/kMlPIqZ.png" width="280" />
<img src="https://i.imgur.com/1EOikTp.png" width="280" />
</p>

## Multi-target multilateration of an unknown number of targets > 1

The final challenge is to be able to perform multilateration without knowing how many targets there are beforehand. The number of targets can be estimated by finding all the centers of the circles and intersections on pairs of circles, then performing hierarchical clustering (scipy.cluster.heirarchical) on these points and counting the number of clusters hierarchical returns. Reasonable thresholds for heirarchical were experimentally determined based off the average radii of circles in the input.

In fact, if the target of each circle is initialized based off of the result from hierarchical clustering as well, this can make for a better initial guess than initializing them randomly as is what is done in Multi-target multilateration of a fixed number of targets.

## References

[0] A. Mathias, M. Leonardi and G. Galati, "An efficient multilateration algorithm," 2008 Tyrrhenian International Workshop on Digital Communications - Enhanced Surveillance of Aircraft and Vehicles, Capri, 2008, pp. 1-6.
doi: 10.1109/TIWDC.2008.4649038\
[1] Aristidis Likas, Nikos Vlassis, Jakob Verbeek. The global k-means clustering algorithm. [Technical
Report] IAS-UVA-01-02, 2001, pp.12. <inria-00321515>

[multilateration intro]: https://i.imgur.com/hrwqui1.png
[two circles]: https://i.imgur.com/xluV3kJ.png
[three circles]: https://i.imgur.com/g3DI1Yg.png
[sum loss]: https://i.imgur.com/jDchRYw.png
[argmin p]: https://i.imgur.com/UljfXuK.png
[gradient]: https://i.imgur.com/GW6VCww.png
<!---
[five circles old (this breaks for some reason)]: https://i.imgur.com/QOBOKZo.png
-->
[five circles]: https://i.imgur.com/2YjaS73.png
[abc]: https://i.imgur.com/GW6VCww.png
