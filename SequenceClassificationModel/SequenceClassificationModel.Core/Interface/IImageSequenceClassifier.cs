using System.Collections.Generic;

namespace SequenceClassificationModel.Core.Interface
{
    public interface IImageSequenceClassifier
    {
        void Train(List<double[][]> sequences, int[] labels);
        int Predict(double[][] sequence);
    }
}