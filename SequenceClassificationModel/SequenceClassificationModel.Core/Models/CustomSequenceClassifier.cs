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
            if (_trainSequences == null || _trainLabels == null)
                throw new InvalidOperationException("Модель не обучена");

            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 65536))
            using (var writer = new BinaryWriter(fs))
            {
                writer.Write(_trainSequences.Count);

                foreach (var seq in _trainSequences)
                {
                    writer.Write(seq.Length);
                    foreach (var frame in seq)
                    {
                        writer.Write(frame.Length);
                        foreach (var pixel in frame)
                            writer.Write(pixel);
                    }
                }

                writer.Write(_trainLabels.Length);
                foreach (var label in _trainLabels)
                    writer.Write(label);
            }
        }

        public void Load(string path)
        {
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536))
            using (var reader = new BinaryReader(fs))
            {
                int seqCount = reader.ReadInt32();
                _trainSequences = new List<double[][]>(seqCount);

                for (int i = 0; i < seqCount; i++)
                {
                    int frameCount = reader.ReadInt32();
                    var seq = new double[frameCount][];

                    for (int f = 0; f < frameCount; f++)
                    {
                        int pixelCount = reader.ReadInt32();
                        var frame = new double[pixelCount];

                        for (int p = 0; p < pixelCount; p++)
                            frame[p] = reader.ReadDouble();
                        seq[f] = frame;
                    }
                    _trainSequences.Add(seq);
                }

                int labelCount = reader.ReadInt32();
                _trainLabels = new int[labelCount];
                for (int i = 0; i < labelCount; i++)
                    _trainLabels[i] = reader.ReadInt32();
            }
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