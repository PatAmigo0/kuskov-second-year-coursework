using System.Drawing;

namespace SequenceClassificationModel.Core.Utils
{
    public static class ImagePreprocessor
    {
        private const int TargetWidth = 32;
        private const int TargetHeight = 32;

        public static double[] ProcessImage(string imagePath)
        {
            using (var originalImage = new Bitmap(imagePath))
            {
                using (var resizedImage = new Bitmap(originalImage, new Size(TargetWidth, TargetHeight)))
                {
                    double[] features = new double[TargetWidth * TargetHeight];
                    int index = 0;

                    for (int y = 0; y < TargetHeight; y++)
                    {
                        for (int x = 0; x < TargetWidth; x++)
                        {
                            var pixel = resizedImage.GetPixel(x, y);
                            double grayscale = (pixel.R * 0.3 + pixel.G * 0.59 + pixel.B * 0.11) / 255.0;

                            features[index] = grayscale;
                            index++;
                        }
                    }

                    return features;
                }
            }
        }
    }
}