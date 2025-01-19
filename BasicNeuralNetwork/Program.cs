using BasicNeuralNetwork;

var nNet = new NeuralNetwork(3, 2);
double[] inputs = { 1.0, 2.0, 3.0 };
double[] expectedOutputs = { 0.5, 1.5 };

for (int epoch = 0; epoch < 1000; epoch++)
{
    nNet.Backpropagate(inputs, expectedOutputs);
    var outputs = nNet.Forward(inputs);
    var loss = Maths.MeanSquaredError(outputs, expectedOutputs);
    if (epoch % 100 == 0)
    {
        Console.WriteLine($"Epoch {epoch}, Loss: {loss}");
    }
}

Console.WriteLine("Final Outputs:");
var finalOutputs = nNet.Forward(inputs);
foreach (var output in finalOutputs)
{
    Console.WriteLine(output);
}
