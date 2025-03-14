import sys
from agent import Agent
from config import ACTION_SIZE, STATE_SIZE
import flappy_bird


def agent(train: bool = False):
    env = flappy_bird.Game(training_mode=train, pump=True)
    agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    if train:
        agent.train(env=env)
    else:
        agent.test(env=env)


def play():
    flappy_bird.run()


if __name__ == "__main__":
    # Read arguments from the command line
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            agent(True)
        elif sys.argv[1] == "test":
            agent()
        elif sys.argv[1] == "play":
            play()
        else:
            print("Invalid argument")
        exit()

    # Ask the user if they want to train or test the model or play the game

    while True:
        print("1. Train the model")
        print("2. Test the model")
        print("3. Play the game")

        choice = input("Enter your choice: ")
        if choice == "1":
            agent(True)
        elif choice == "2":
            agent()
        elif choice == "3":
            play()
        else:
            print("Invalid choice")
