# basic-neural-network
A basic Neural Network machine learning application written in C# on the .NET 9 Platform.

A neural network is an AI algorithm frequently used in prediction and classification problems (such as recognizing images).  Its structure is loosely based on the workings of biological neurons, particularly the way that they interact with and activate each other via synaptic connections and activation thresholds.  Essentially, it generates a function that attempts to describe the input data and iteratively updates that function until the input data matches the expected output data.

This application accepts a set of numeric inputs along with their expected outputs.  The weights and biases, which loosely correspond to parameters related to a biological neuron's action potential, are forward-fed through the network's layers of nodes.  The inputs to each "neuron" (node) are aggregated and fed into an activation function, which determines the degree to which a neuron should "fire". Unlike biological neurons, whose thresholds are binary (they either fire or don't fire), the activation function of an artificial neural network returns a decimal value between 0.0 and 1.0 indicating the degree to which it is activated by the inputs.

When the output nodes are reached, their weights and biases are aggregated and checked by computing the mean squared error vis-a-vis the expected output.  The data is then backpropagated through the network and the connection weights are adjusted by applying the chain rule for multivariable calculus (computing the derivative one variable at a time).  The data is then forward-fed again through the network, utilizing the updated weights, as many times as specified by the variable `epochs`.  If successful, the actual output values will converge on the expected values after each epoch.
