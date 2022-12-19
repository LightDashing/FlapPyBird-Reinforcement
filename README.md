FlapPyBird-Reinforcement
===============

A Flappy Bird Clone made using [python-pygame][pygame]

I converted original project to OpenAI Gym environment for usage with stable-baselines3. If you want to teach your own agent, execute
`flappygym.py`, otherwise for agent evaluation run `flappygym_evaluate.py` <br>
For this project I trained PPO algorithm for 1 million timesteps, average amount of passed pipes is 1698, agent saved as model.zip. 

![Alt Text](Final.gif)

Setup (as tested on Manjaro on python 3.9)
---------------------------

1. Clone the repository:

   ```bash
   $ git clone https://github.com/LightDashing/FlapPyBird-Reinforcement.git
   ```

   or download as zip and extract.

1. In the root directory run

   ```bash
   $ python install -r requirements.txt
   ```

1. Use <kbd>&uarr;</kbd> or <kbd>Space</kbd> key to play and <kbd>Esc</kbd> to close the game.

(For x64 windows, get exe [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygame))

