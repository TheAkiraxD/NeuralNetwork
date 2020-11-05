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
        private const float WEIGHT_THRESHOLD_MIN = -0.8f;
        private const float WEIGHT_THRESHOLD_MAX = 0.8f;
        public Layer Inputs { get; private set; }
        public List<Layer> Hiddens { get; private set; }
        public Layer Outputs { get; private set; }

        public int MutationRate { get; set; }
        private float[,] WeightsIH;
        private List<float[,]> WeightsH;
        private float[,] WeightsHO;
        private float[,] Bias;

        public int Mutations { get; private set; }

        private delegate float ActivateFunction(float number);

        public NeuralNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize, int hiddenLayerDepth = 1)
        {
            Inputs = new Layer(inputLayerSize);
            Hiddens = new List<Layer>();
            for(int i = 0; i < hiddenLayerDepth; i++)
            {
                Hiddens.Add(new Layer(hiddenLayerSize));
            }
            Outputs = new Layer(outputLayerSize);
            Mutations = 0;
        }

        public NeuralNetwork(NeuralNetwork neuralNetwork)
        { 
            Inputs = new Layer(neuralNetwork.Inputs.size);
            Hiddens = new List<Layer>(neuralNetwork.Hiddens);
            Outputs = new Layer(neuralNetwork.Outputs.size);

            MutationRate = neuralNetwork.MutationRate;
            WeightsIH = neuralNetwork.WeightsIH.Clone() as float[,];
            WeightsHO = neuralNetwork.WeightsHO.Clone() as float[,];
            WeightsH = new List<float[,]>(neuralNetwork.WeightsH);

            Mutations = neuralNetwork.Mutations;

            Bias = neuralNetwork.Bias.Clone() as float[,];
            if (Mutate())
            {
                Mutations++;
            }
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
                for (int x = 0; x < hiddenLayerDepth-1; x++)
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
            Hiddens[0] = ApplyActivationFunction(Task.Run(() => Feed(WeightsIH, Matrix.ArrayToMatrix(inputs), Bias[0,0])).Result, activateFunction);
            for(int i = 0; i < Hiddens.Count()-1; i++)
            {
                Hiddens[i+1] = ApplyActivationFunction(Task.Run(() => Feed(WeightsH[i], Matrix.ArrayToMatrix(Hiddens[i].Items), Bias[i+1, 0])).Result, activateFunction);
            }
            var lastBias = Bias.GetLength(0)-1;
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
                    item[x, y] = GenerateRandomNumber(random, WEIGHT_THRESHOLD_MIN, WEIGHT_THRESHOLD_MAX);
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

        private bool Mutate()
        {
            if(Task.Run(() => ApplyMutation(WeightsIH)).Result)
            {
                return true;
            }
            if (Task.Run(() => ApplyMutation(WeightsHO)).Result)
            {
                return true;
            }

            foreach (var weight in WeightsH)
            {
                if (Task.Run(() => ApplyMutation(weight)).Result)
                {
                    return true;
                }
            }
            return false;
        }

        private bool ApplyMutation(float[,] weights)
        {
            Random random = new Random();
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    if(random.Next(100) <= MutationRate)
                    {
                        random = new Random();
                        weights[i,j] = Convert.ToSingle(random.NextDouble()) * (WEIGHT_THRESHOLD_MIN - WEIGHT_THRESHOLD_MAX);
                        return true;
                    }
                }
            }
            return false;
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
