# Snake_Neural_Network
A small AI that learns to play snake with reinforcement learning.  

  
![snek.png](snek.png)  

The nn used is a small, 3 layer, 291 x 1024 x 1 model.

There are 2 different scoring functions used:  
-the first one is a naive scoring that rewards moving towards food, used to kickstart training  
-the second one is standard reward with delay

In training the delay went up to 0.998 as model plateaued repeatedly for lower values  

Learning rate started at 1e-2 and finished at 3e-5
