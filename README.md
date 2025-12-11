# Install

1. Navigate to cs224w_learning_heuristic
2. Optionally make a conda environment (e.g., cs224w_learning_heuristic)
3. Run: pip install -e . (uses setup.py to install python package)

# Code Organization

**To accomodate making this repository public, we have removed a small number of files (as noted in the below comments) which predate the project code and were developed with Clippership. Please email aofeldma@stanford.edu to request access to these files if needed.**

## Overview

1. Data: Used to store any data files we generate e.g., created dataset, trained models, and sailboat polar (removed).
2. Results: Used to store any results we generate e.g., the figures and videos in the blog post.
3. Scripts: Used to store the python code we write.

## Scripts

To run a python script, navigate to cs224w_learning_heuristic, then provide relative path e.g., python3 Scripts/Testing/train_model.py 

**For the purposes of the CS224W course, the most relevant folders are Learning, Testing, and DatasetCreation.**

A typical workflow would be:
1. Testing/create_dataset.py to generate a pytorch-geometric dataset (or use premade dataset in Data/)
2. Testing/train_model.py to train the learned heuristic graph ML model
3. Testing/vis_heuristic.py to visualize the learned heuristic heatmap
4. Testing/multi_scenario_test.py to use the learned heuristic in planning

### Configs

This holds the configuration files specifying the parameters for the graph planner, the machine learning pipeline (e.g., the GNN architecture parameters), and for scenario generation (e.g., wind speed range, number of other obstacles).

1. load_from_config.py: Tools to load the relevant objects using .yaml config files.

2. ml_config.yaml: The ML pipeline and architecture parameters

3. (removed) planner_config.yaml: The graph planner parameters

4. scenario_config.yaml: The scenario generation parameters

### DatasetCreation

This will hold files for creating random path planning scenarios and generating the ML dataset for our project.

1. scenario_creator.py: Provides tools for generating random planning scenarios e.g., random start/goal generation (EndpointSampler), wind generation (WindSampler), creation of other moving boats (TargetListSampler), and full random trial/scenario (TrialSampler)

2. data_generator.py: Generate random scenarios, call graph planner with non-learned heuristic, and process to generate a dataset of GraphDatum objects.

3. data_to_torch.py: Given a list of GraphDatum objects, convert/process into pytorch-geometric (PyG) Data objects for ML.

### GraphPlanner

This holds files for implementing the graph planner search.

1. (removed) sailboat_polar.py: Provides the SailboatPolar class and associated tools for capturing the sailboat dynamics e.g., how fast you can travel as a function of the wind speed and relative angle between your sailboat heading and the wind.

2. state_node.py: Provides Pose2dState to represent the state of sailboat at given instant in time 2D position (x,y) and orientation/heading psi. StateNode class used in the graph (A*) planner extends this to also provide information about the parent, cost, cost-to-go estimate for a given node during the graph search. 

3. (removed) graph_planner.py: Provides the GraphPlanner class which is the main object for executing the graph search to find a path from start to goal given a radar costmap of static obstacles (empty for now, see Obstacles/costmap.py) and dynamic obstacles (boats, see Obstacles/target.py). Operates by expanding a tree of possible nodes/states to travel to, prioritizing which nodes to expand/grow based on cost and estimated cost-to-go.

4. (removed) pruner.py: Since GraphPlanner tree search grows exponentially, can use a Pruner class internally to eliminate near-duplicate nodes. Implements logic so that if your path would reach nearby nodes at similar time, only keep the lower cost node.

5. heuristic.py: Provides PlanningHeuristic object and non-learned child heuristic class for use within the graph planner. (MaxSpeedHeuristic removed)

### Math

This holds general math tools, currently used by GraphPlanner files.

1. geometry.py: Helper functions for basic geometry e.g., distance to a rectangle.

2. math_utils.py: Helper functions for fitting Fourier series and wrapping angles.

### Learning

This holds the classes for learning a graph-planning heuristic using a GNN.

1. heuristic_nn.py: The overall ML model (HeuristicNN) for implementing the learned heuristic.

2. learned_heuristic.py: A wrapper for the HeuristicNN class for use in the graph planner as a PlanningHeuristic.

3. target_graph_encoder.py: The GNN-specific component of the ML model to encode the other targets/boats in the scenario.

4. training_tools.py: Training/evaluation pipeline for the ML model.

### Obstacles

This holds classes to represent the static and dynamic obstacles the sailboat can encounter and that the path planner must avoid.

1. obstacle_helpers.py: Contains base class ObstacleInterface representing a generic obstacle and ObstacleCostCalculator class for computing the cost/penalty during path planning for travelling near the obstacle. (ObstacleCostCalculator removed)

2. costmap.py: Contains Costmap class which represents static obstacles using a (radar) image, then converted into a binary occupancy map. For now, this will be empty i.e., no land, see empty_costmap function.

3. target.py: Contains Target, TargetList classes for representing other boats/dynamic objects moving in lines with constant velocity.

### Plotting

Provides various helpers to visualize and animate the results of the graph planner paths.

1. graph_animation.py: Provides helpers for animating the graph planner search (see by setting animate=True in graph_planner.py solve)

2. plotting_helpers.py: Provides general plotting helpers to plot triangles/rectangles/ellipses.

3. waypoint_plotting.py: Provides helpers for plotting and animating the path/waypoint information alongside the static/dynamic obstacles.

4. heuristic_plotting.py: Use for plotting a heatmap visualization of the (learned) graph planner heuristic.

### Testing

Contains files for testing/deployment of the graph planner in specific/random test scenarios.

1. graph_test.py: Runs and visualizes the graph planner in one randomly generated scenario.

2. multi_scenario_test.py: Runs the graph planner in several randomly generated scenarios, saving associated videos, and plotting metrics like success rate, runtime, and path cost.

3. train_model.py: Trains the ML model using a loaded dataset and generates a learning/loss curve.

4. vis_heuristic.py: Visualizes the ML model learned heuristic for different scenarios.

5. vis_polar.py: Visualize the sailboat polar governing ego-boat speed based on wind.

6. vis_scenario.py: Visualize an example scenario e.g., targets, wind, start, and goal that would have to plan in.

7. create_dataset.py: Creates a dataset by generating random scenarios and running planner.
