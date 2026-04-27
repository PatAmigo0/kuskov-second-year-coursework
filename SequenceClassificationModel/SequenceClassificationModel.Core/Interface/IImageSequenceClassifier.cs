using System.Collections.Generic;

namespace SequenceClassificationModel.Core.Interface
{
    public interface IImageSequenceClassifier
    {
        string AlgorithmName { get; }

        /// <summary>
        /// Trains the classifier using the provided feature sequences and corresponding labels.Метод для обучения классификатора на последовательностях признаков
        /// </summary>
        /// <remarks>The method associates each feature sequence in the input with its corresponding label
        /// to fit the classifier. The input sequences and labels must be aligned such that the label at each index
        /// corresponds to the sequence at the same index.</remarks>
        /// <param name="inputSequences">A list of feature sequences, where each sequence is represented as a two-dimensional array of doubles. Each
        /// sequence corresponds to a single training instance.</param>
        /// <param name="labels">An array of integer labels representing the correct class for each input sequence. The length of this array
        void Train(List<double[][]> inputSequences, int[] labels);

        /// <summary>
        /// Predicts the most likely class label for the specified input sequence.
        /// </summary>
        /// <param name="sequence">An array of double arrays representing the input sequence to classify. Each inner array typically
        /// corresponds to a feature vector at a time step. Cannot be null.</param>
        /// <returns>An integer representing the predicted class label for the input sequence.</returns>
        int Predict(double[][] sequence);
    }
}