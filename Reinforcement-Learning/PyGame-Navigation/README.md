# PyGame Navigation Mini-Project

## Introduction

The idea of this project is to build a reinforcement learning project where an agent (interactive object) moves on a track rendered in PyGame, designed using Matplotlib. The goal is to train the agent to navigate the track efficiently, avoiding obstacles and reaching a predefined endpoint. 

#### **Markov Decision Process (MDP)**
1. **State Space**: 
 - The position of the object (x, y coordinates).
 - The velocity (if dynamic movement is included).
 - Track features (e.g., proximity to edges, distance to obstacles, or checkpoints).

2. **Action Space**: 
 - Actions: Move left, right, up, down.

3. **Reward Function**:
 - Positive rewards for:
 - Progressing toward the endpoint.
 - Reaching checkpoints.
 - Negative rewards for:
 - Colliding with track boundaries.
 - Moving away from the endpoint.

4. **Environment Dynamics**:
 - Updates to the agent's state based on its actions.
 - Detect collisions or out-of-bounds movements.

---

### Proposed Project Structure

#### **1. Environment Setup**
- Create a custom environment class using PyGame and Matplotlib for rendering:
 - **Track design**: A procedurally generated track using Matplotlib.
 - **Rendering**: Export the Matplotlib track as an image and load it into PyGame.

#### **2. Code Structure**
Here's a modular project structure:

```
project/
│
├── main.py # Main script to run the project
├── environment/
│ ├── __init__.py # Initialization of the environment package
│ ├── track_generator.py # Generate and render tracks using Matplotlib
│ ├── game_env.py # PyGame-based environment with RL interface
│ └── collision.py # Handle collision detection and boundary checks
│
├── agent/
│ ├── __init__.py # Initialization of the agent package
│ ├── dqn_agent.py # Implementation of the Deep Q-Network agent
│ └── replay_buffer.py # Replay buffer for experience replay
│
├── models/
│ ├── dqn_model.py # Define the neural network for Q-value approximation
│ └── __init__.py # Initialization of the models package
│
├── config/
│ ├── settings.py # Configuration for hyperparameters and environment
│
├── utils/
│ ├── logger.py # Logging and visualization
│ └── plot_results.py # Plot training results (e.g., reward curves)
│
└── README.md # Project documentation
```

---

### Key Modules and Their Responsibilities

#### **1. `track_generator.py`**
- Generate the track with curves, obstacles, or boundaries using Matplotlib.
- Save the track as an image to be loaded into PyGame.

#### **2. `game_env.py`**
- Define the RL-compatible environment with:
 - `reset()`: Initialize the environment.
 - `step(action)`: Update the agent's state and return `(next_state, reward, done)`.
 - `render()`: Render the environment using PyGame.

#### **3. `dqn_agent.py`**
- Implement the Deep Q-Network (DQN) algorithm with:
 - Neural network architecture.
 - Epsilon-greedy policy for exploration.
 - Experience replay and target network updates.

#### **4. `collision.py`**
- Implement collision detection with:
 - Track boundaries.
 - Obstacles (if any).

#### **5. `settings.py`**
- Store configurable parameters like:
 - Learning rate, gamma, epsilon.
 - Environment dimensions, track difficulty.

---

### Suggested Workflow (ChatGPT)
1. **Track Creation**: Start by generating a simple track using Matplotlib.
2. **Environment Integration**: Build a PyGame-based environment that uses the Matplotlib-generated track.
3. **Agent Development**: Implement a basic DQN to move the agent around the track.
4. **Reward Shaping**: Fine-tune rewards for smooth learning.
5. **Visualization**: Track agent performance and visualize learning progress.

---

## Features of each Class

### **1. `track_generator.py`**
This module is responsible for generating and exporting the track.

#### **Features**:
- **`generate_track()`**: 
 - Generate a track using Matplotlib? (e.g., curves, boundaries).
 - Output: 2D array representing the track or an image file.
- **`visualize_track()`** : 
 - Display the track for debugging.

---

### **2. `game_env.py`**
This module wraps the simulation environment in a class compatible with reinforcement learning frameworks.

#### **Features**:
- **Initialization (`__init__`)**:
 - Load the track image into PyGame.
 - Define agent properties (size, initial position).
 - Set environment dimensions and state space.

- **`reset()`**:
 - Reset the environment to its initial state.
 - Initialize the agent's position.
 - Return the initial state.

- **`step(action)`**:
 - Update the agent’s position based on the action.
 - Check for collisions or if the agent is out of bounds.
 - Calculate rewards and determine if the episode is done.
 - Return `(next_state, reward, done, info)`.

- **`render()`**:
 - Render the environment in PyGame.
 - Display the agent and the track.

- **`get_state()`**:
 - Return the current state (e.g., agent's position, track boundaries).

- **`check_collision()`**:
 - Detect collisions using pixel-based boundary checks or track masks.

---

### **3. `collision.py`**
Handle all collision detection logic.

#### **Features**:
- **`is_collision(x, y)`**:
 - Check if the agent is at a collidable position on the track.
 - Input: Agent's current position.
 - Output: Boolean (True if collision, False otherwise).

- **`check_track_boundary(x, y)`**:
 - Verify if the agent is outside track boundaries.
 - Input: Agent's position.
 - Output: Boolean.

- **`proximity_to_edge()`** (optional):
 - Calculate the agent's distance from track edges (useful for rewards).

---

### **4. `dqn_agent.py`**
The reinforcement learning agent, implementing the DQN algorithm.

#### **Features**:
- **Initialization (`__init__`)**:
 - Initialize the neural network for Q-value estimation.
 - Define hyperparameters (e.g., epsilon, gamma, learning rate).

- **`select_action(state)`**:
 - Choose an action based on the epsilon-greedy policy.
 - Input: Current state.
 - Output: Chosen action.

- **`update_network()`**:
 - Perform a gradient descent step to minimize the Q-loss.
 - Input: Batch of experiences from the replay buffer.

- **`update_target_network()`**:
 - Synchronize weights of the target network with the main network.

- **`load_model()`**:
 - Load a pre-trained model (optional).

- **`save_model()`**:
 - Save the trained model.

---

### **5. `replay_buffer.py`**
Manage experience replay for training stability.

#### **Features**:
- **Initialization (`__init__`)**:
 - Define the replay buffer size and structure.

- **`store_experience(state, action, reward, next_state, done)`**:
 - Add an experience tuple to the buffer.

- **`sample_batch(batch_size)`**:
 - Sample a random batch of experiences for training.
 - Input: Batch size.
 - Output: Batch of experiences.

---

### **6. `dqn_model.py`**
Define the neural network architecture for approximating Q-values.

#### **Features**:
- **Initialization (`__init__`)**:
 - Define the input size (state space) and output size (action space).
 - Create hidden layers (e.g., fully connected layers).

- **`forward(x)`**:
 - Perform a forward pass through the network.
 - Input: State tensor.
 - Output: Q-values for each action.

---

### **7. `settings.py`**
Store all configuration parameters.

#### **Features**:
- Environment Settings:
 - `TRACK_WIDTH`, `TRACK_HEIGHT`, `AGENT_SIZE`, etc.
- Agent Hyperparameters:
 - `LEARNING_RATE`, `GAMMA`, `EPSILON`, etc.
- Training Settings:
 - `BATCH_SIZE`, `MAX_EPISODES`, etc.

---

### **8. `logger.py`**
Log training progress and debugging information.

#### **Features**:
- **`log_episode_rewards(episode, reward)`**:
 - Log cumulative reward per episode.

- **`save_logs()`**:
 - Save logs to a file for analysis.

---

### **9. `plot_results.py`**
Generate visualizations of training progress.

#### **Features**:
- **`plot_rewards(rewards)`**:
 - Plot cumulative rewards over episodes.
 - Input: List of rewards.

- **`save_plot()`**:
 - Save plots as images for reporting.

