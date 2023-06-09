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
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    print(game_state)

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # Map border hard stops
    board_width = game_state['board']['width'] -1
    board_height = game_state['board']['height'] -1

    if my_head["x"] <= 0:               # Left Border Hard Stop
        is_move_safe["left"] = False

    elif my_head["x"] >= board_width:   # Right Border Hard Stop
        is_move_safe["right"] = False

    if my_head["y"] <= 0:               # Bottom Border Hard Stop
        is_move_safe["down"] = False

    elif my_head["y"] >= board_height:  # Top Border Hard Stop
        is_move_safe["up"] = False
    

    # TODO: Step 2 - Prevent your Battlesnake from colliding with itself
    my_body = game_state['you']['body']

    if {'x':my_head["x"] -1, 'y':my_head["y"]} in my_body:  
        is_move_safe["left"] = False

    if {'x':my_head["x"] +1, 'y':my_head["y"]} in my_body:  
        is_move_safe["right"] = False

    if {'y':my_head["y"] -1, 'x':my_head["x"]} in my_body:  
        is_move_safe["down"] = False

    if {'y':my_head["y"] +1, 'x':my_head["x"]} in my_body:  
        is_move_safe["up"] = False

    # TODO: Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    # opponents = game_state['board']['snakes']

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # TODO: Step 4 - Move towards food instead of random, to regain health and survive longer

    # Find all the food on the board
    food_locations = game_state['board']['food']

    # Find the closest food to the head of your snake
    closest_food = None
    closest_distance = float('inf')
    for food in food_locations:
        distance = abs(my_head['x'] - food['x']) + abs(my_head['y'] - food['y'])
        if distance < closest_distance:
            closest_distance = distance
            closest_food = food

    # Decide which direction to move based on the closest food
    if closest_food['x'] < my_head['x'] and is_move_safe['left']:
        return {"move": "left"}
    elif closest_food['x'] > my_head['x'] and is_move_safe['right']:
        return {"move": "right"}
    elif closest_food['y'] < my_head['y'] and is_move_safe['down']:
        return {"move": "down"}
    elif closest_food['y'] > my_head['y'] and is_move_safe['up']:
        return {"move": "up"}
    else:
        return {"move": random.choice(safe_moves)}



# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
