using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Utilities
{
    internal static class Matrix
    {
        public static float[,] DotProduct(float[,] matrixA, float[,] matrixB)
        {
            var product = new float[matrixA.GetLength(0), matrixB.GetLength(1)];
            if(IsGeoEquivalent(matrixA, matrixB))
            {
                for(int i = 0; i < matrixA.GetLength(0); i++)
                {
                    for (int j = 0; j < matrixB.GetLength(1); j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < matrixA.GetLength(1); k++)
                        {
                            sum += matrixA[i, k] * matrixB[k,j];
                        }
                        product[i, j] = sum;
                    }
                }
            }
            else
            {
                throw new Exception("The matrices are not geo-equivalent.");
            }

            return product;
        }

        public static float[,] Sum(float[,] matrix, float value)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    matrix[i, j] += value;
                }
            }
            return matrix; //TODO: Check this is necessary or its being done by reference already ;)
        }

        public static float[,] ArrayToMatrix(float[] array)
        {
            float[,] product = new float[array.Length, 1];
            for(int i = 0; i < array.Length; i++)
            {
                product[i, 0] = array[i];
            }
            return product;
        }

        public static float[] MatrixToArray(float[,] matrix)
        {
            if(matrix.GetLength(1) != 1)
            {
                throw new Exception("The matrix cannot be converted because it has more than one column.");
            }

            float[] product = new float[matrix.GetLength(0)];
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                product[i] = matrix[i, 0];
            }
            return product;
        }

        private static bool IsGeoEquivalent(float[,] matrixA, float[,] matrixB)
        {
            return matrixA.GetLength(1) == matrixB.GetLength(0) ? true : false;
        }
    }
}