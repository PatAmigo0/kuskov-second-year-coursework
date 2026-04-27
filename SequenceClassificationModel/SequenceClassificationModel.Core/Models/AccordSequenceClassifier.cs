using Accord.MachineLearning;
using Accord.Math.Distances;
using SequenceClassificationModel.Core.Interface;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SequenceClassificationModel.Core.Models
{
    public class AccordSequenceClassifier : IImageSequenceClassifier
    {
        public string AlgorithmName => "Алгоритм из Accord.NET (k-NN)";

        private KNearestNeighbors _knn;

        public void Train(List<double[][]> inputSequences, int[] labels)
        {
            double[][] flattenedInputs = inputSequences.Select(seq => FlattenSequence(seq)).ToArray();

            this._knn = new KNearestNeighbors(k: 3, distance: new Euclidean());
            this._knn.Learn(flattenedInputs, labels);
        }

        public int Predict(double[][] sequence)
        {
            if (_knn == null)
                throw new InvalidOperationException("Модель Accord.NET еще не обучена!");

            double[] flattenedSeq = FlattenSequence(sequence);
            return this._knn.Decide(flattenedSeq);
        }

        public KNearestNeighbors GetInternalModel()
        {
            return this._knn;
        }

        private double[] FlattenSequence(double[][] sequence)
        {
            var result = new List<double>();
            foreach (var frame in sequence)
                result.AddRange(frame);
            return result.ToArray();
        }
    }
}