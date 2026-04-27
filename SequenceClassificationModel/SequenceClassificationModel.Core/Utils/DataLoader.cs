using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace SequenceClassificationModel.Core.Utils
{
    public static class DataLoader
    {
        public static (List<double[][]> Sequences, int[] Labels) LoadData(string basePath)
        {
            var sequences = new ConcurrentBag<double[][]>();
            var labels = new ConcurrentBag<int>();

            var classDirs = Directory.GetDirectories(basePath);

            Parallel.ForEach(classDirs, classDir =>
            {
                string dirName = new DirectoryInfo(classDir).Name;
                if (!dirName.StartsWith("Class_")) return;

                if (int.TryParse(dirName.Substring(6), out int label))
                {
                    var seqDirs = Directory.GetDirectories(classDir);

                    Parallel.ForEach(seqDirs, seqDir =>
                    {
                        var imageFiles = Directory.GetFiles(seqDir, "*.jpg").OrderBy(f => f).ToList();

                        if (imageFiles.Count > 0)
                        {
                            var seqFeatures = new double[imageFiles.Count][];
                            for (int i = 0; i < imageFiles.Count; i++)
                                seqFeatures[i] = ImagePreprocessor.ProcessImage(imageFiles[i]);

                            sequences.Add(seqFeatures);
                            labels.Add(label);
                        }
                    });
                }
            });

            return (sequences.ToList(), labels.ToArray());
        }
    }
}