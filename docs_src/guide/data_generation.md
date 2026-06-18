# Data Generation

CAMPD does not include a data generation pipeline — it is designed to work with trajectory datasets you generate using your own motion planning or simulation tools.

## Expected HDF5 format

Each sample in the HDF5 file must contain:

- A **trajectory array** — joint positions, or positions + velocities/accelerations depending on the `trajectory_state` setting (`"pos"`, `"pos+vel"`, or `"pos+vel+acc"`).
- **Context arrays** — environment descriptors such as obstacle center positions, dimensions, and orientations.

The mapping from HDF5 keys to internal fields is configured via `field_config` in your experiment YAML:

```yaml
dataset:
  trajectory_state: "pos"
  field_config:
    trajectory_field: "solutions"   # HDF5 key for trajectory data
    q_dim: 7                        # configuration-space dimension
    context_fields:
      spheres: ["sphere_centers", "sphere_radii"]
      cuboids: ["cuboid_centers", "cuboid_dims", "cuboid_quaternions"]
```

See `src/campd/data/trajectory_dataset.py` for the full field configuration schema.

## Franka example data

The Franka example datasets were generated using [CuRobo](https://github.com/NVlabs/curobo) as a reference motion planner. Refer to the paper ([arXiv:2510.14615](https://arxiv.org/abs/2510.14615)) for details on the data generation procedure.
