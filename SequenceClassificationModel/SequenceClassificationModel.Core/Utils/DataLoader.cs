using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SequenceClassificationModel.Core.Utils
{
    public static class DataLoader
    {
        public static (List<double[][]> Sequences, int[] Labels, List<string> ClassNames) LoadData(string rootFolderPath)
        {
            var sequences = new List<double[][]>();
            var labels = new List<int>();
            var classNames = new List<string>();

            var classDirs = Directory.GetDirectories(rootFolderPath);
            for (int classIndex = 0; classIndex < classDirs.Length; classIndex++)
            {
                string classDir = classDirs[classIndex];
                classNames.Add(Path.GetFileName(classDir));

                var sequenceDirs = Directory.GetDirectories(classDir);
                foreach (var seqDir in sequenceDirs)
                {
                    var imageFiles = Directory.GetFiles(seqDir, "*.*")
                                              .Where(f => f.EndsWith(".jpg") || f.EndsWith(".png") || f.EndsWith(".bmp"))
                                              .OrderBy(f => f)
                                              .ToArray();

                    if (imageFiles.Length == 0) continue;

                    var sequenceFrames = new List<double[]>();
                    foreach (var imgPath in imageFiles)
                    {
                        double[] features = ImagePreprocessor.ProcessImage(imgPath);
                        sequenceFrames.Add(features);
                    }

                    sequences.Add(sequenceFrames.ToArray());
                    labels.Add(classIndex);
                }
            }

            return (sequences, labels.ToArray(), classNames);
        }
    }
}