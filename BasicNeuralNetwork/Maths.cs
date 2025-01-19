using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicNeuralNetwork
{
    public static class Maths
    {
        public static double ReLU(double x) => Math.Max(0, x);

        public static double ReLUDerivative(double x) => x > 0 ? 1 : 0;

        public static double MeanSquaredError(double[] predicted, double[] actual) 
            => predicted.Zip(actual, (p, a) => Math.Pow(p - a, 2)).Average();
    }
}
