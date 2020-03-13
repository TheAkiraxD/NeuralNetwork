using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace NeuralNetwork.Utilities
{
    internal static class ActivateFunctions
    {
        [DllImport("activationFunctions.dll", CallingConvention = CallingConvention.Cdecl)]
        private extern static float sigmoid(float x);

        public static float Sigmoid(float number)
        {
            return sigmoid(number);
        }
    }
}
 