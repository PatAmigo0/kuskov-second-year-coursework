using System;
using System.Collections.Generic;
using SequenceClassificationModel.Core.Interface;

namespace SequenceClassificationModel.Core.Models
{
    public class CustomSequenceClassifier : IImageSequenceClassifier
    {
        public string AlgorithmName => "Мой алгоритм (DTW + 1-NN)";

        private List<double[][]> _trainSequences;
        private int[] _trainLabels;

        public void Train(List<double[][]> inputSequences, int[] labels)
        {
            _trainSequences = inputSequences;
            _trainLabels = labels;
        }

        public int Predict(double[][] sequence)
        {
            if (_trainSequences == null || _trainSequences.Count == 0)
                throw new InvalidOperationException("Модель еще не обучена!");

            double minDistance = double.MaxValue;
            int bestLabel = -1;

            for (int i = 0; i < _trainSequences.Count; i++)
            {
                double dist = CalculateDTWDistance(sequence, _trainSequences[i]);

                if (dist < minDistance)
                {
                    minDistance = dist;
                    bestLabel = _trainLabels[i];
                }
            }

            return bestLabel;
        }

        // DTW
        private double CalculateDTWDistance(double[][] seq1, double[][] seq2)
        {
            int n = seq1.Length;
            int m = seq2.Length;
            double[,] dtw = new double[n + 1, m + 1];

            for (int i = 0; i <= n; i++)
                for (int j = 0; j <= m; j++)
                    dtw[i, j] = double.MaxValue;

            dtw[0, 0] = 0;
            for (int i = 1; i <= n; i++)
            {
                for (int j = 1; j <= m; j++)
                {
                    double cost = EuclideanDistance(seq1[i - 1], seq2[j - 1]);
                    dtw[i, j] = cost + Math.Min(dtw[i - 1, j], Math.Min(dtw[i, j - 1], dtw[i - 1, j - 1]));
                }
            }

            return dtw[n, m];
        }

        private double EuclideanDistance(double[] vec1, double[] vec2)
        {
            double sum = 0;
            for (int i = 0; i < vec1.Length; i++)
            {
                double diff = vec1[i] - vec2[i];
                sum += diff * diff;
            }
            return Math.Sqrt(sum);
        }
    }
}