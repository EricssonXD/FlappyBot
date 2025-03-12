# System Patterns: Flappy Bird with DQN

## Architecture:

*   **Game Environment:** `src/flappy_bird.py` - Handles game logic, rendering, and state.
*   **DQN Agent:** `src/dqn.py` - Implements the DQN algorithm, including the neural network, experience replay, and action selection.
*   **Main Script:** `src/main.py` - Orchestrates the game, agent training, and interaction.

## Key Technical Decisions:

*   **DQN Implementation:** Using a Deep Q-Network for reinforcement learning.
*   **Pygame:** Using Pygame for game development.
*   **GPU Acceleration:** Utilizing PyTorch for GPU acceleration of the neural network.

## Design Patterns:

*   **Observer Pattern:** Potentially used for game events (e.g., score updates, game over).
*   **Singleton Pattern:** Potentially used for game configuration or other global resources.
