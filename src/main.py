import sys
from agent import Agent
from config import ACTION_SIZE, STATE_SIZE
import flappy_bird


def agent(train: bool = False, load_checkpoint: bool = False):
    env = flappy_bird.Game(training_mode=train, pump=True)
    agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    if train:
        agent.train(env=env, load_checkpoint=load_checkpoint)
    else:
        agent.test(env=env)


def play():
    flappy_bird.run()


if __name__ == "__main__":
    # Read arguments from the command line
    if len(sys.argv) > 1:
        match sys.argv[1]:
            case "train":
                agent(True)
            case "train_continue":
                agent(True, True)
            case "test":
                agent()
            case "play":
                play()
            case _:
                print("Invalid argument")
        exit()

    # Ask the user if they want to train or test the model or play the game

    while True:
        print("1. Train the model")
        print("2. Test the model")
        print("3. Play the game")
        print("4. Train the model with checkpoint")
        print("5. Exit")

        choice = input("Enter your choice: ")
        match choice:
            case "1":
                agent(True)
            case "2":
                agent()
            case "3":
                play()
            case "4":
                agent(True, True)
            case "5":
                break
            case _:
                print("Invalid choice")
