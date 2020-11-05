using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace NeuralNetwork.Utilities
{
    internal static class ActivateFunctions
    {
        //Currently this DLL only works for x64 bits CPU platform applications
        //TODO: Compile other C++ DLLs for better compatibility
        [DllImport("activationFunctions.dll", CallingConvention = CallingConvention.Cdecl)]
        private extern static float sigmoid(float x);

        public static float Sigmoid(float number)
        {
            return sigmoid(number);
        }
    }
}
 