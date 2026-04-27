using System.IO;
using System.Text.Json;
using Accord.IO;
using Accord.MachineLearning;
using SequenceClassificationModel.Core.Models;
using SequenceClassificationModel.Core.Interface;

namespace SequenceClassificationModel.Core.Utils
{
    public static class ModelStorage
    {
        // 1. Сохранение и загрузка для ACCORD.NET
        public static void SaveAccordModel(AccordSequenceClassifier model, string path)
        {
            Serializer.Save(model.GetInternalModel(), path);
        }

        public static KNearestNeighbors LoadAccordModel(string path)
        {
            return Serializer.Load<KNearestNeighbors>(path);
        }

        private class CustomModelData
        {
            public double[][][] Sequences { get; set; }
            public int[] Labels { get; set; }
        }

        //public static void SaveCustomModel(CustomSequenceClassifier model, string path)
        //{
        //    var data = new CustomModelData
        //    {
        //        Sequences = model.GetTrainSequences().ToArray(),
        //        Labels = model.GetTrainLabels()
        //    };

        //    string json = JsonSerializer.Serialize(data);
        //    File.WriteAllText(path, json);
        //}
    }
}