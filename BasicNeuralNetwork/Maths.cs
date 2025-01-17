using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicNeuralNetwork
{
    public static class Maths
    {
        public static double ReLU(double x)
        {
            return Math.Max(0, x);
        }

        public static double ReLUDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }

        public static double MeanSquaredError(double[] predicted, double[] actual)
        {
            double sum = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                sum += Math.Pow(predicted[i] - actual[i], 2);
            }
            return sum / predicted.Length;
        }
    }
}
