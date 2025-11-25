GD-1 and Beordo2024 joint qz proxy

Data sets

GD-1 stream: proper motions along GD-1 from data/gd1/gd1_lite.csv, processed into an orbit based proxy for the vertical halo flattening qz.

Milky Way rotation curve: Beordo et al. 2024 all stars sample from data/rotation_curve/beordo2024_allstars_rc_ready.csv, giving circular velocity as a function of radius around the Galactic centre.

Stream only proxy

From the orbit based GD-1 fit we summarize the constraint on qz as a Gaussian in qz. The best fit is about 1.093 and the one sigma width is about 0.0127. This is represented as a quadratic profile in qz.

Rotation curve proxy

The contribution of the Beordo2024 rotation curve is approximated as a quadratic penalty in qz around values close to 1, using a single scale factor to match the typical chi squared level of the data. This is a simplified proxy and not a full rotation curve likelihood.

Joint toy model

The joint constraint is defined as the sum of the stream and rotation curve quadratic profiles in qz and is evaluated on a grid using scripts/joint_rc_stream_qz_grid.py.

The main outputs are:

results/joint_qz_profile.pdf and png: global joint delta chi squared profile as a function of qz with confidence bands.

results/joint_qz.txt: qz fit and confidence intervals.

results/joint_q_components_plot.pdf and png: decomposition into stream only and rotation curve only curves.

results/joint_qz_components.txt: numerical summary of the components.

For the current setup the joint best fit is about qz equal to 1.004 with a one sigma width of about 0.0027. The 68 percent interval is roughly 1.002 to 1.007 and the 95 percent interval is roughly 0.999 to 1.010.

Interpretation

GD-1 alone prefers qz around 1.09 with a relatively broad constraint. The rotation curve prefers values close to 1. The joint toy model yields an almost spherical halo, with the rotation curve dominating the final uncertainty.

Limitations and next steps

Both components are represented as simple Gaussians in qz. The true likelihood shapes may deviate from pure quadratics.

The rotation curve term currently uses a single scaling coefficient rather than a full prediction of the circular velocity as a function of radius in the disk bulge halo model.

Other potential parameters, such as the halo normalization, are held fixed.

Planned improvements include replacing the quadratic proxies with the actual delta chi squared curves from the orbit fits and from the rotation curve in the full potential, allowing more parameters to vary and marginalizing over them, and performing robustness tests on GD-1 selection cuts and the rotation curve radial range.
