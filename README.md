# SBI-Tutorial
Material for the AIMS AI for Science summer school tutorial on neural simulation based inference.


#### Intended Learning Outcomes
At the end of the session, students will be able to:

1. __Define__ the core concept of simulation-based inference (SBI) and __distinguish__ implicit-likelihood models from classical likelihood-based approaches.
2. __Explain__ how naive Approximate Bayesian Computation (ABC) generates posterior approximations and discuss why it can be inefficient in high dimensions.
3. __Implement__ a simple ABC approach in Python to demonstrate how changing the acceptance threshold impacts posterior accuracy and computational cost.
4. __Describe__ the change-of-variables formula used in normalizing flows and interpret how it enables us to .
5. __Construct__ and train an affine coupling layer from scratch in PyTorch, computing both the forward pass and "inverse pass".
6. Assemble a normalizing flow by stacking multiple coupling layers and permutation layers, visualize its performance on the “two moons” problem, and __compare__ it to naive ABC in terms of accuracy and efficiency.
7. Use the sbi Python library to approximate posteriors in a neuromuscular simulation problem, evaluating convergence and the impact of neural network hyperparameters.
8. __Reflect__ on the limitations of normalizing flows (e.g., invertibility constraints, architectural choices) and identify potential improvements or alternative SBI techniques for complex simulators.
