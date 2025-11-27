# GD-1 + Beordo2024 joint q_z constraints (toy project)

Using a simple axisymmetric potential with a vertical flattening parameter q_z:

- GD-1 proper-motion proxy alone prefers q_z ≈ 1.09 with σ ≈ 0.013.
- A physical Milky Way rotation curve based on Beordo et al. (2024) was implemented by comparing the model circular velocity v_c(R; q_z) to the observed Vphi_kms vs R_kpc.
- In this potential family, the rotation curve is almost insensitive to q_z over 0.8–1.2, so the RC contribution to Δχ²(q_z) is nearly flat.
- The joint fit therefore stays dominated by the stream and yields q_z ≈ 1.095 with σ ≈ 0.01 (68% CI ≈ [1.085, 1.105]).

Plotted outputs:
- `results/joint_qz_profile.pdf`: toy quadratic proxy with a simple RC scaling.
- `results/joint_q_components_plot.pdf`: stream vs RC contributions in the toy model.
- `results/joint_qz_profile_physicalrc.pdf`: joint Δχ²(q_z) using the physical Beordo2024 rotation curve (acceleration-based).
