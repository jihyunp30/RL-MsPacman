This project is a deep learning model using reinforcement learning to maximize score in Atari 2600 game MsPacman.
Reinforcement learning is a field of machine learning and AI directed at understanding and solving general tasks that follow a particular model.
As an agent takes actions and steps through an environment, it is trained to map the observation of the environment to an action.
Based on expected reward from the Q function, an agent chooses an action in a given state.
It learns to perform its task with recommended actions to maximize the potential rewards.

The project uses the Open AI Gym environment.
According to Open AI Gym, in the environment ‘MsPacman-ram-v0’, the observation is the RAM of the Atari machine, consisting of 128 bytes.
For a duration of k frames, each action is repeatedly performed, and k is uniformly sampled from {2, 3, 4}.
In the project, reinforcement learning is implemented, so an agent can learn to maximize the game score.
A CNN is built using Keras for the agent’s neural network. The model has 2 hidden layers with 128 neurons for each. 

From the Open AI gym, the program receives the MsPacman and the observation is it’s RAM.
The program takes a random action or it inputs the current state into the CNN model to get an action.
It takes an action, receives a reward, and the state is updated.
The state, action, reward, and new state values are appended to the memory for training.
After each episode, the agent is trained from its memory with deep reinforcement learning. 

For the evaluation, rewards for each episode are appended to a list.
Also, an average of 10 episodes are calculated for evaluations and also stored in a list.
Looking at the generated graphs from the lists, the overall score is slowly increasing, however, there are few outliers.
Scores seem to be inconsistent, and a possible solution is tuning hyperparameters.
Increasing the number of episodes would help to determine the overall trend of the scores.
After running the code several times, there are cases that the average score increases, decreases, and increases again.
This can be caused by large learning rate or small batch size, so adjusting those values can possibly maximize the rewards.
Moreover, in the code, the replay function is called after each episode.
It can be called after every action taken, so the model can be trained more often.

For the project, Deep Q-Learning is implemented to maximize the score of MsPacman from the Atari 2600 game.
An agent is created to successfully play the game using the RAM of the Atari machine.
The project showed the scores increase as the model learned how to play the game.
The average score of the first 10 episodes was 232.
The average score increases, and the average score of the last 10 episodes becomes 457, which is almost double of the beginning average score.
