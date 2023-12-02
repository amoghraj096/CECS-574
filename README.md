## CECS-574

**All reduce**

This code snippet sets up a deep neural network model for federated learning on the MNIST dataset using TensorFlow. It defines a model architecture, loss function, optimizer with a learning rate schedule, and splits the data into a specified number of clients. The training loop iterates through the clients and simulates local training, applying gradient updates to each client's model. It then averages the client weights with the global model's weights to update the global model. Finally, it evaluates the global model's performance on the test dataset.

**Federated learning**

Similar to Code 1, this snippet sets up a deep neural network model and splits the MNIST data into multiple clients. However, it differs in the training loop by conducting local training for a fixed number of iterations per client while recording and storing training loss and accuracy for visualization. After each client's local training, it updates the central server model with averaged weights and evaluates the global model on the test dataset. Additionally, it includes code to plot and visualize the training loss and accuracy for each client throughout the federated learning process using Matplotlib.
