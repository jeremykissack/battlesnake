# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import os

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# DQN settings
input_size = 11 * 11  # Assuming an 11x11 board
hidden_size = 4096
output_size = 4  # Number of possible moves
learning_rate = 0.001

# Initialize the DQN
dqn = DQN(input_size, hidden_size, output_size)

# Load the saved model, if available
model_path = "model.pth"
if os.path.exists(model_path):
    dqn.load_state_dict(torch.load(model_path))


optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Other Deep Q-Learning settings
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
memory = []
max_memory_size = 1000000
batch_size = 64

# Global variable to store the previous state and action
previous_state = None
previous_action = None
previous_game_state = {}

high_score = 0


import numpy as np

def preprocess_game_state(game_state):
    # Preprocess game_state and convert it into a tensor
    board_width = game_state["board"]["width"] 
    board_height = game_state["board"]["height"]
    board = np.zeros((board_height, board_width), dtype=int)

    # Fill in the game board with information from the game state (e.g., snake positions, food)
    # Mark food with 1
    for food in game_state["board"]["food"]:
        board[food["y"] - 1][food["x"] - 1] = 1

    # Mark snake body segments with -1
    for snake in game_state["board"]["snakes"]:
        for segment in snake["body"]:
            board[segment["y"] - 1][segment["x"] - 1] = -1

    # Mark our snake's head with 2
    our_head = game_state["you"]["head"]
    board[our_head["y"] - 1][our_head["x"] - 1] = 2

    # Flatten the board and convert it to a tensor
    input_tensor = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0)
    return input_tensor



def update_q_values(batch):
    states, rewards, next_states, actions = zip(*batch)

    states = torch.cat(states)
    non_terminal_next_states = torch.cat([s for s in next_states if s is not None])

    q_values = dqn(states)
    target_q_values = q_values.clone().detach()  # Create a tensor with the same shape as q_values

    non_terminal_next_q_values = dqn(non_terminal_next_states).max(1)[0].detach()

    non_terminal_idx = 0
    for i, (reward, next_state, action) in enumerate(zip(rewards, next_states, actions)):
        if next_state is not None:
            target_q_values[i, action] = reward + gamma * non_terminal_next_q_values[non_terminal_idx]
            non_terminal_idx += 1
        else:
            target_q_values[i, action] = reward

    loss = criterion(q_values, target_q_values)
    print("Loss:", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def experience_replay():
    # Perform experience replay and update Q-values
    if len(memory) >= batch_size:
        batch = random.sample(memory, batch_size)
        update_q_values(batch)

def is_in_bounds(x: int, y: int, board_width: int, board_height: int) -> bool:
    return 0 <= x < board_width and 0 <= y < board_height

def is_occupied(x: int, y: int, snakes: typing.List[typing.Dict]) -> bool:
    for snake in snakes:
        for segment in snake["body"]:
            if segment["x"] == x and segment["y"] == y:
                return True
    return False

def get_safe_moves(game_state: typing.Dict) -> typing.List[str]:
    head = game_state["you"]["head"]
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    snakes = game_state["board"]["snakes"]

    possible_moves = {
        "up": {"x": head["x"], "y": head["y"] + 1},
        "down": {"x": head["x"], "y": head["y"] - 1},
        "left": {"x": head["x"] - 1, "y": head["y"]},
        "right": {"x": head["x"] + 1, "y": head["y"]}
    }

    safe_moves = []
    for move, new_position in possible_moves.items():
        if is_in_bounds(new_position["x"], new_position["y"], board_width, board_height) and \
                not is_occupied(new_position["x"], new_position["y"], snakes):
            safe_moves.append(move)

    # If safe_moves list is empty, add a default move (e.g., "up")
    if not safe_moves:
        safe_moves.append("up")

    return safe_moves



# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }

# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    global epsilon
    epsilon = max(epsilon * epsilon_decay, min_epsilon)
    print("GAME START")
    print(f"Epsilon: {epsilon}")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    global high_score
    # Add the final state to the memory
    memory.append((preprocess_game_state(game_state), -1, None, previous_action))
    # Perform experience replay
    experience_replay()

    # Update the high score
    current_score = game_state["you"]["length"]
    high_score = max(high_score, current_score)
    print(f"Score: {current_score}, High Score: {high_score}")

    print("GAME OVER\n")

    # After N games or in the end() function
    torch.save(dqn.state_dict(), "model.pth")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    global previous_state, previous_action, previous_game_state

    safe_moves = get_safe_moves(game_state)

    # Use the DQN to select the next action
    input_tensor = preprocess_game_state(game_state)
    q_values = dqn(input_tensor)
    action = torch.argmax(q_values).item()

    # Exploration: choose a random action with probability epsilon
    if random.random() < epsilon:
        action = random.randint(0, 3)
        exploration_status = "Exploration"
    else:
        exploration_status = "Exploitation"

    # If we have a previous state, store the experience in memory
    if previous_state is not None:
        # Compute the reward based on game state
        current_health = game_state["you"]["health"]
        previous_health = previous_game_state["you"]["health"]
        current_length = game_state["you"]["length"]
        previous_length = previous_game_state["you"]["length"]
        if current_length > 5 and current_length < 10:
            reward = 5
        if not len(safe_moves):
            reward = -100
        elif current_health == 0:
            reward = -50
        elif current_length <= 1:
            reward = -50
        else:
            reward = -1

        memory.append((previous_state, reward, input_tensor, action))  # Store the action index

        # Make sure memory doesn't exceed max_memory_size
        if len(memory) > max_memory_size:
            memory.pop(0)

    # Update Q-values during training
    experience_replay()

    # Convert the selected action to a move
    move_mapping = {0: "up", 1: "down", 2: "left", 3: "right"}
    selected_move = move_mapping[action]
    print(f"Selected move: {selected_move} ({exploration_status})")

    # Check if the selected move is safe. If not, choose a random safe move.
    if selected_move not in safe_moves:
        selected_move = random.choice(safe_moves)

    # Update the previous state and action
    previous_state = input_tensor
    previous_action = action

    previous_game_state = game_state

    return {"move": selected_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
