# FIFA_PLAY_WHOLE_GAME
In this project, I use DDQN algorithm in reinforcement learning to train AI play whole soccer game. It is similar to free kick, one subgame of full soccer game. 

However, there are also much difference like we will need more actions, more difficult to judge the reward, which will hugely
affect the result of algorithm. I'm still trying to make it a better code, even though it can now train AI in a relatively foundamental way.

If you want to see the demo of free kick(a simpler one), you can refer to my another repository
https://github.com/CoderNoMercy/FIFA_Free_Kick_Pytorch_DDQN.

On that page, I will illustrate the algorithm in detail, which is also the method I use in this repository

# To do list

- [ ] Find a more reliable way to return reward from environment.
- [ ] Try other algorithms like PPO and compare their performance.
- [ ] Apply auto-training. Since we don't have the source code of game, we just have to use some method to judge whether the game is end and we can automatically restart game.

- [ ] Control the action of AI players more smoothly, which will affect the performance of AI
