using NeuralNetwork.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public Layer Inputs { get; private set; }
        public Layer[] Hiddens { get; private set; }
        public Layer Outputs { get; private set; }

        private float[,] WeightsIH;
        private List<float[,]> WeightsH;
        private float[,] WeightsHO;

        private float[,] Bias;

        private delegate float ActivateFunction(float number);

        public NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, int hiddenLayerDepth = 1)
        {
            Inputs = new Layer(inputLayerSize);
            var hiddenSize = new Layer(hiddenLayerSize);
            var hiddenDepth = new Layer(hiddenLayerDepth);
            Hiddens = new Layer[] { hiddenSize, hiddenDepth };
            Outputs = new Layer(outputLayerSize);
        }

        public void DefineAndPopulateWeights()
        {
            int inputLayerSize = Inputs.size;
            int hiddenLayerSize = Hiddens[0].size;
            int outputLayerSize = Outputs.size;
            int hiddenLayerDepth = Hiddens[1].size;

            WeightsIH = new float[hiddenLayerSize, inputLayerSize];
            Bias = new float[hiddenLayerDepth + 1, 1];

            if(hiddenLayerDepth == 1)
            {
                WeightsH = null;
            }
            else
            {
                WeightsH = new List<float[,]>();
                for (int x = 0; x < hiddenLayerDepth; x++)
                {
                    WeightsH.Add(new float[hiddenLayerSize, hiddenLayerSize]);
                }
            }
            WeightsHO = new float[outputLayerSize, hiddenLayerSize];
            Task.Run(async () => await InitializeWeights()).Wait();
        }

        public void FeedForward(float[] inputs)
        {
            ActivateFunction activateFunction = ActivateFunctions.Sigmoid;

            Hiddens[0] = ApplyActivationFunction(Task.Run(() => Feed(WeightsH[0], Matrix.ArrayToMatrix(inputs), Bias[0,0])).Result, activateFunction);
            for(int i = 1; i < Hiddens.Count(); i++)
            {
                Hiddens[i] = ApplyActivationFunction(Task.Run(() => Feed(WeightsH[i], Matrix.ArrayToMatrix(Hiddens[i-1].Items), Bias[i, 0])).Result, activateFunction);
            }
            var lastBias = Bias.GetLength(0);
            Outputs = ApplyActivationFunction(Task.Run(() => Feed(WeightsHO, Matrix.ArrayToMatrix(Hiddens.LastOrDefault().Items), Bias[lastBias, 0])).Result, activateFunction);
        }

        private Layer Feed(float[,] weightsA, float[,] weightsB, float bias)
        {
            var product = Matrix.Sum(Matrix.DotProduct(weightsA, weightsB), bias);
            return new Layer(product.GetLength(0)) { Items = Matrix.MatrixToArray(product) };
        }

        private async Task InitializeWeights()
        {
            // Multithreading for populating the matrices
            List<Task> tasks = new List<Task>
            {
                new Task(() => PoulateMatrixRandomly(ref WeightsIH)),
                new Task(() => PoulateMatrixRandomly(ref WeightsHO)),
                new Task(() => PoulateMatrixRandomly(ref Bias))
            };

            for(int x = 0; x < WeightsH.Count(); x++)
            {
                var z = WeightsH[x];
                tasks.Add(new Task(() => PoulateMatrixRandomly(ref z)));
            }

            foreach(var task in tasks)
            {
                task.Start();
            }

            await Task.WhenAll(tasks);
        }

        private float GenerateRandomNumber(Random random, float minimum, float maximum)
        {
            return Convert.ToSingle(random.NextDouble()) * (maximum - minimum) + minimum;
        }

        private void PoulateMatrixRandomly(ref float[,] item)
        {
            for(int x = 0; x < item.GetLength(0); x++)
            {
                for (int y = 0; y < item.GetLength(1); y++)
                {
                    var random = new Random();
                    item[x, y] = GenerateRandomNumber(random, 1, -1);
                }
            }
        }
        private Layer ApplyActivationFunction(Layer layer, ActivateFunction activateFunction)
        {
            var result = new Layer(layer.size);
            for(int i= 0; i < layer.size; i++) {
                result.Items[i] = activateFunction(layer.Items[i]);
            }
            return result;
        }

        public class Layer
        {
            public float[] Items { get; set; }
            public int size { get; private set; }
            public Layer(int size)
            {
                Items = new float[size];
                this.size = size;
            }

        }
    }
}
