using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public Layer Inputs { get; private set; }
        public Layer[] Hiddens { get; private set; }
        public Layer Outputs { get; private set; }

        private double[,] WeightsIH;
        private List<double[,]> WeightsH;
        private double[,] WeightsHO;

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

            WeightsIH = new double[hiddenLayerSize, inputLayerSize];
            if(hiddenLayerDepth == 1)
            {
                WeightsH = null;
            }
            else
            {
                WeightsH = new List<double[,]>();
                for (int x = 0; x < hiddenLayerDepth; x++)
                {
                    WeightsH.Add(new double[hiddenLayerSize, hiddenLayerSize]);
                }
            }
            WeightsHO = new double[outputLayerSize, hiddenLayerSize];
            Task.Run(async () => await InitializeWeights()).Wait();
        }

        private async Task InitializeWeights()
        {
            // Multithreading for populating the matrices
            List<Task> tasks = new List<Task>
            {
                new Task(() => PoulateArrayRandomly(ref WeightsIH)),
                new Task(() => PoulateArrayRandomly(ref WeightsHO))
            };

            for(int x = 0; x < WeightsH.Count(); x++)
            {
                var z = WeightsH[x];
                tasks.Add(new Task(() => PoulateArrayRandomly(ref z)));
            }

            foreach(var task in tasks)
            {
                task.Start();
            }

            await Task.WhenAll(tasks);
        }

        private double GenerateRandomNumber(Random random, double minimum, double maximum)
        {
            return random.NextDouble() * (maximum - minimum) + minimum;
        }

        private void PoulateArrayRandomly(ref double[,] item)
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

        public class Layer
        {
            public double[] Items { get; private set; }
            public int size { get; private set; }
            public Layer(int size)
            {
                Items = new double[size];
                this.size = size;
            }

        }
    }
}
