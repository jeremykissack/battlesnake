import subprocess

num_games = 1000

for i in range(num_games):
    print(f"Starting game {i+1} of {num_games}")

    command = "./battlesnake play -W 11 -H 11 --name 'Python Starter Project' --url http://localhost:8000 -g solo"
    process = subprocess.run(command, shell=True, cwd='/home/jeremykissack/projects/neural-networks/other/rules', check=True, text=True)

    print(f"Finished game {i+1} of {num_games}")
