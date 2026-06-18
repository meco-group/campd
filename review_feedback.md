**Documentation** 
- Add a clear "Getting Started" section with step-by-step instructions (download data → training → inference) 
- Add simple, complete "toy examples" showing a typical YAML config structure, relevant parameters, and how components connect -
 Clarify in Requirements that Windows is not supported 
- Add instructions for verifying that training data downloaded correctly 
- Expand documentation on trajectory generation (consider adding a figure) 
- Document how to update model_dir and hdf5_file in the inference YAML 
- Fix the broken link to the experiment-launcher package 
- Add a section on data generation 
- Reduce the focus on software implementation details in Core Concepts; emphasize actual trajectory generation instead 
- Add explicit documentation of the framework's limitations and boundaries 

**User Feedback / Output** 
- Show results/metrics to the user after training completes 
- Show results/plots after inference completes (currently exits with a warning and no output) **Open Questions to Investigate** 
- Why does inference take ~7h vs ~1h for training? Is this expected? 
- Add validation tools for visually or qualitatively assessing generated trajectories (currently only losses are logged)

For the user feedback/output, make clear that this is an example and that metrics / visualizations need to be implemented according to their own wishes / project / use case … Also training metrics & summaries are currently implemented in the example but you need to view them in W&B