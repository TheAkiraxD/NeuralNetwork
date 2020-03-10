using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Utilities
{
    internal static class Matrix
    {
        public static double[,] DotProduct(double[,] matrixA, double[,] matrixB)
        {
            var product = new double[matrixA.GetLength(0), matrixB.GetLength(1)];
            if(IsGeoEquivalent(matrixA, matrixB))
            {
                for(int i = 0; i < matrixA.GetLength(0); i++)
                {
                    for (int j = 0; j < matrixB.GetLength(1); j++)
                    {
                        double sum = 0;
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

        private static bool IsGeoEquivalent(double[,] matrixA, double[,] matrixB)
        {
            return matrixA.GetLength(1) == matrixB.GetLength(0) ? true : false;
        }
    }
}