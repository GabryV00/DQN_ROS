# Robot Navigation with Reinforcement Learning with ROS and Gazebo

This repository contains the files for the execution of a Reinforcement Learning algorithm, i.e. the Deep Q-Learning algorithm, for the navigation of a robot, the TurtleBot3, in a simulated environment in Gazebo, using ROS.

## Architecture

The project consists of the following files:
- `src/my_turtlebot3_openai_example/scripts/start_deepqlearning.py` which is the main file that contains the details of the DQN architecture, all the calls to the OpenAI ROS library and the function for saving the data
- `src/my_turtlebot3_openai_example/config/my_turtlebot3_openai_deepqlearn_params.yaml` which contains all the parameters for the DQN architecture
- `src/openai_ros/src/openai_ros/task_envs/turtlebot3/config/turtlebot3_world.yaml` which contains parameters related to the TurtleBot3 enviroment
- `training_results/` contains the folder related to all the experiments conducted. Each folder contains:
  * `plot.png` image containing the three plots relating to cumulative reward and total time, with the data not normalized
  * `plot-std.png` image containing the three plots relating to cumulative reward and total time, with the normalized data
  * `results-date.json` file containing all the parameters used in the experiment, plus the results obtained
