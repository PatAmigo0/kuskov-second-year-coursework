using SequenceClassificationModel.Core.Interface;
using System;
using System.Collections.Generic;
using System.Text.Json;
using System.IO;

namespace SequenceClassificationModel.Core.Models
{
    public class CustomSequenceClassifier : IImageSequenceClassifier
    {
        public string AlgorithmName => "Мой алгоритм (DTW + 1-NN)";

        private List<double[][]> _trainSequences;
        private int[] _trainLabels;

        public List<double[][]> TrainSequences => _trainSequences;
        public int[] TrainLabels => _trainLabels;

        public void Train(List<double[][]> inputSequences, int[] labels)
        {
            _trainSequences = inputSequences;
            _trainLabels = labels;
        }

        public int Predict(double[][] sequence)
        {
            if (_trainSequences == null || _trainSequences.Count == 0)
                throw new InvalidOperationException("Модель еще не обучена");

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

        private class CustomModelData
        {
            public List<double[][]> Sequences { get; set; }
            public int[] Labels { get; set; }
        }

        public void Save(string path)
        {
            if (_trainSequences == null || _trainLabels == null) throw new InvalidOperationException("Модель не обучена");

            var data = new CustomModelData { Sequences = _trainSequences, Labels = _trainLabels };
            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 65536))
                JsonSerializer.Serialize(stream, data);
        }

        public void Load(string path)
        {
            string json = File.ReadAllText(path);
            var data = JsonSerializer.Deserialize<CustomModelData>(json);
            _trainSequences = data.Sequences;
            _trainLabels = data.Labels;
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