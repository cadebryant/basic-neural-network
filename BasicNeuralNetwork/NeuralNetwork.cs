using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicNeuralNetwork
{
    public class NeuralNetwork
    {
        private double[][] weights;
        private double[] biases;
        private double learningRate;

        public NeuralNetwork(int inputSize, int outputSize, double learningRate = 0.01)
        {
            this.learningRate = learningRate;
            weights = new double[outputSize][];
            biases = new double[outputSize];
            Random rand = new Random();

            for (int i = 0; i < outputSize; i++)
            {
                weights[i] = new double[inputSize];
                for (int j = 0; j < inputSize; j++)
                {
                    weights[i][j] = rand.NextDouble();
                }
                biases[i] = rand.NextDouble();
            }
        }

        public double[] Forward(double[] inputs)
        {
            int outputSize = weights.Length;
            double[] outputs = new double[outputSize];

            for (int i = 0; i < outputSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputs.Length; j++)
                {
                    sum += inputs[j] * weights[i][j];
                }
                sum += biases[i];
                outputs[i] = Maths.ReLU(sum);
            }

            return outputs;
        }

        public void Backpropagate(double[] inputs, double[] expectedOutputs)
        {
            double[] outputs = Forward(inputs);
            double[] errors = new double[outputs.Length];
            double[] deltas = new double[outputs.Length];

            for (int i = 0; i < outputs.Length; i++)
            {
                errors[i] = outputs[i] - expectedOutputs[i];
                deltas[i] = errors[i] * Maths.ReLUDerivative(outputs[i]);
            }

            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    weights[i][j] -= learningRate * deltas[i] * inputs[j];
                }
                biases[i] -= learningRate * deltas[i];
            }
        }
    }
}
