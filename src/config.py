# GAME CONSTANTS
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.08
FLAP_STRENGTH = 7
PIPE_SPEED = 6
PIPE_GAP = 150
PIPE_SPACING = 200  # Distance between pipes
STATE_SIZE = 7
ACTION_SIZE = 2

GAME_FPS = 30
TRAINING_FPS = 999999999  # Unlimited FPS for training

# REWARD CONSTANTS
SURVIVAL_REWARD = 1
FLAP_REWARD = 0
PIPE_PASSED_REWARD = 10
COLLISION_REWARD = -100
OUT_OF_BOUNDS_REWARD = -1000

# TRAINING CONSTANTS
USE_GPU = False

BATCH_SIZE = 32
EPISODES = 1000
EPISODE_STOP_REWARD = 50000
USE_DOUBLE_DQN = True
USE_DUELING_DQN = True

# MODEL HYPERPARAMETERS
EPSILON_START = 1.0
EPSILON_MIN = 0.0001
EPSILON_DECAY = 0.9995

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.0004
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10

# LOGGING
LOG_DIR = "logs"
MODEL_DIR = "models"
