# This work is an extension of MAD-ICP

This repository extends <a href="https://github.com/rvp-group/mad-icp#">MAD-ICP</a>, the original LiDAR odometry pipeline. Please refer to the original repository for the base project and its publication.

## What's new

The main addition of this fork is a **pluggable velocity estimator**: instead of the single Gauss-Newton estimator used by the original MAD-ICP, you can now choose among three solvers via the `solver` key in [`mad_params.py`](mad_icp/configurations/mad_params.py):

| `solver` value | Description |
| -------------- | ----------- |
| `"classic"` | Original Gauss-Newton velocity estimator — always available, no extra dependencies. |
| `"gsolver"` | Factor-graph estimator based on Gaussian Processes, solved with a distributed Gaussian Belief Propagation (GBP) solver ([gsolver](https://github.com/rvp-group/gsolver)). |
| `"gtsam"` | Same GP factor-graph formulation, but solved with a centralized Gauss-Newton optimizer using [GTSAM](https://github.com/borglab/gtsam). |

The `"gsolver"` and `"gtsam"` solvers are only available if the package is built with `-DUSE_GP_VEL_ESTIMATOR=ON` (see [Install using `pip`](#install-using-pip)); with the base install only `"classic"` is available.

Alongside `solver`, a few more entries in `mad_params.py` control the factor-graph solvers:
- `qc_default`: intensity of the White-Noise-On-Acceleration (WNOA) process noise (used when no ground truth is provided to calibrate it).
- `prior_cov`: covariance of the prior factor on each pose; leave empty/`None` to use the real per-pose ICP covariance instead.
- `gn_iterations`: number of GN/GBP iterations run on the factor graph.
- `gt_path`: path to a KITTI ground-truth file used to calibrate `Qc`; leave empty to use the default.

# Install using `pip`

You can install this package directly with `pip` from the repository root. The build compiles the C++/pybind11 extension via CMake, so no extra manual build step is needed.

There are two installation modes, depending on which velocity estimator you want to use:

- **Base install** — uses the classic Gauss-Newton velocity estimator, no extra dependencies:
  ```bash
  pip install -e .
  ```
- **GP velocity estimator (gsolver) install** — enables the factor-graph-based velocity estimator, which fetches and builds [gsolver](https://github.com/rvp-group/gsolver) automatically:
  ```bash
  pip install -e . -Ccmake.args="-DUSE_GP_VEL_ESTIMATOR=ON"
  ```

The `-e` flag installs the package in editable mode, so changes to the Python sources are picked up without reinstalling. Omit it for a regular (non-editable) install.

# Usage

We provide a Python launcher for Rosbag1, Rosbag2, and KITTI binary formats. The dataset configuration is important for the sensor characteristics and extrinsic information (typically, ground truths are not expressed in the LiDAR frame). In `configurations/datasets/dataset_configurations.py` we provide configurations for many datasets.

The internal parameters (used by default) are stored in `configurations/mad_params.py`. All the experiments have been run with this same set.
You can specify a new set in `configurations/mad_params.py` and use it with the option `--mad-icp-params`.

Both the options `--dataset-config` and `--mad-icp-params` also accept `.cfg` files like those in `configurations`.

To run the pipeline, choose the appropriate dataset configuration (`kitti` for this example) and type:
```bash
mad_icp --data-path /input_dir/ \
        --estimate-path /output_dir/ \
        --dataset-config kitti
```
Our runner directly saves the odometry estimate file in KITTI format (homogenous matrix row-major 12 scalars); soon, we will provide more available formats like TUM.

Our pipeline is `anytime realtime`! You can play with parameters `num_keyframes` and `num_cores` and, if you have enough _computation capacity_, we suggest increasing these (we run demo/experiments with `num_keyframes=16` and `num_cores=16`).
