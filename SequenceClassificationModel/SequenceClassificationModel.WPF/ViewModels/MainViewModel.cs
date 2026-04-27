using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Forms;
using SequenceClassificationModel.Core.Interface;
using SequenceClassificationModel.Core.Models;
using SequenceClassificationModel.Core.Utils;

namespace SequenceClassificationModel.WPF.ViewModelsы
{
    public class MainViewModel : ViewModelBase
    {
        private string _statusText = "Система готова к работе";
        public string StatusText { get => _statusText; set { _statusText = value; OnPropertyChanged(); } }

        private string _selectedFolderPath = "Папка не выбрана";
        public string SelectedFolderPath { get => _selectedFolderPath; set { _selectedFolderPath = value; OnPropertyChanged(); } }

        private int _testSizePercentage = 20;
        public int TestSizePercentage { get => _testSizePercentage; set { _testSizePercentage = value; OnPropertyChanged(); } }

        private bool _runTesting = true;
        public bool RunTesting { get => _runTesting; set { _runTesting = value; OnPropertyChanged(); } }



        public List<AlgorithmType> AvailableAlgorithms { get; }
        private AlgorithmType _selectedAlgorithm;
        public AlgorithmType SelectedAlgorithm { get => _selectedAlgorithm; set { _selectedAlgorithm = value; OnPropertyChanged(); } }

        private bool _isBusy;
        public bool IsBusy
        {
            get => _isBusy;
            set { _isBusy = value; OnPropertyChanged(); System.Windows.Input.CommandManager.InvalidateRequerySuggested(); }
        }

        private bool _isModelTrained;
        public bool IsModelTrained { get => _isModelTrained; set { _isModelTrained = value; OnPropertyChanged(); } }

        private IImageSequenceClassifier _trainedClassifier;

        private string _interactiveSequencePath;
        public string InteractiveSequencePath { get => _interactiveSequencePath; set { _interactiveSequencePath = value; OnPropertyChanged(); } }

        private string _previewImagePath;
        public string PreviewImagePath { get => _previewImagePath; set { _previewImagePath = value; OnPropertyChanged(); } }

        private string _singlePredictionResult = "Ожидание последовательности...";
        public string SinglePredictionResult { get => _singlePredictionResult; set { _singlePredictionResult = value; OnPropertyChanged(); } }

        public ICommand SelectFolderCommand { get; }
        public ICommand TrainTestCommand { get; }
        public ICommand SelectSingleImageCommand { get; }
        public ICommand PredictSingleCommand { get; }
        public ICommand SaveModelCommand { get; }
        public ICommand LoadModelCommand { get; }

        public MainViewModel()
        {
            AvailableAlgorithms = Enum.GetValues(typeof(AlgorithmType)).Cast<AlgorithmType>().ToList();
            SelectedAlgorithm = AvailableAlgorithms[0];

            SelectFolderCommand = new RelayCommand(ExecuteSelectFolder, _ => !IsBusy);
            TrainTestCommand = new RelayCommand(ExecuteTrainTest, CanExecuteTrainTest);
            SelectSingleImageCommand = new RelayCommand(ExecuteSelectSingleImage, _ => !IsBusy);
            PredictSingleCommand = new RelayCommand(ExecutePredictSingle, CanExecutePredictSingle);

            SaveModelCommand = new RelayCommand(ExecuteSaveModel, _ => IsModelTrained && !IsBusy);
            LoadModelCommand = new RelayCommand(ExecuteLoadModel, _ => !IsBusy);
        }

        private bool CanExecuteTrainTest(object p) => !string.IsNullOrEmpty(SelectedFolderPath) && Directory.Exists(SelectedFolderPath) && !IsBusy;
        private bool CanExecutePredictSingle(object p) => IsModelTrained && !string.IsNullOrEmpty(InteractiveSequencePath) && !IsBusy;

        private async void ExecuteTrainTest(object parameter)
        {
            IsBusy = true;
            IsModelTrained = false;
            var progress = new Progress<string>(msg => StatusText = msg);
            var reporter = (IProgress<string>)progress;

            try
            {
                await Task.Run(() =>
                {
                    reporter.Report("Загрузка и предобработка данных...");
                    var data = DataLoader.LoadData(SelectedFolderPath);
                    if (data.Sequences.Count == 0) throw new Exception("Данные не найдены :(");

                    reporter.Report("Перемешивание выборки...");
                    Random rnd = new Random();
                    var indices = Enumerable.Range(0, data.Sequences.Count).ToList();
                    var shuffledIndices = indices.OrderBy(x => rnd.Next()).ToList();

                    var allSeqs = shuffledIndices.Select(i => data.Sequences[i]).ToList();
                    var allLabels = shuffledIndices.Select(i => data.Labels[i]).ToArray();

                    double testRatio = TestSizePercentage / 100.0;
                    int testCount = (int)(allSeqs.Count * testRatio);
                    int trainCount = allSeqs.Count - testCount;

                    var trainSeqs = allSeqs.Take(trainCount).ToList();
                    var trainLabels = allLabels.Take(trainCount).ToArray();

                    var testSeqs = allSeqs.Skip(trainCount).Take(testCount).ToList();
                    var testLabels = allLabels.Skip(trainCount).Take(testCount).ToArray();

                    reporter.Report($"Обучение на {trainCount} примерах...");

                    IImageSequenceClassifier classifier = SelectedAlgorithm == AlgorithmType.DTW_1NN
                        ? (IImageSequenceClassifier)new CustomSequenceClassifier()
                        : new AccordSequenceClassifier();

                    classifier.Train(trainSeqs, trainLabels);
                    _trainedClassifier = classifier;

                    if (!RunTesting || testSeqs.Count == 0)
                    {
                        reporter.Report("Обучение завершено, тестирование пропущено");
                        return;
                    }

                    reporter.Report($"Тестирование на {testSeqs.Count} новых примерах...");
                    int correct = 0;
                    int processed = 0;
                    object lockObj = new object();

                    Parallel.For(0, testSeqs.Count, i =>
                    {
                        int predicted = classifier.Predict(testSeqs[i]);
                        lock (lockObj)
                        {
                            if (predicted == testLabels[i]) correct++;
                            processed++;
                            if (processed % 10 == 0) reporter.Report($"Тест: {processed}/{testSeqs.Count}...");
                        }
                    });

                    double accuracy = (double)correct / testSeqs.Count * 100.0;
                    reporter.Report($"Готово! Точность на НОВЫХ данных: {accuracy:F2}%");
                });
                IsModelTrained = true;
            }
            catch (Exception ex) { StatusText = $"Ошибка: {ex.Message}"; }
            finally { IsBusy = false; }
        }

        private async void ExecuteSaveModel(object p)
        {
            var dialog = new Microsoft.Win32.SaveFileDialog { Filter = "Model Files|*.model", Title = "Сохранить модель" };
            if (dialog.ShowDialog() == true)
            {
                IsBusy = true; 
                StatusText = "Сохранение модели на диск...";

                try
                {
                    string filePath = dialog.FileName;

                    await Task.Run(() =>
                        _trainedClassifier.Save(filePath)
                    );

                    StatusText = "Модель успешно сохранена";
                }
                catch (Exception ex)
                {
                    StatusText = $"Ошибка сохранения: {ex.Message}";
                }
                finally
                {
                    IsBusy = false; 
                }
            }
        }

        private async void ExecuteLoadModel(object p)
        {
            var dialog = new Microsoft.Win32.OpenFileDialog { Filter = "Model Files|*.model", Title = "Загрузить модель" };
            if (dialog.ShowDialog() == true)
            {
                IsBusy = true;
                StatusText = "Загрузка модели в память...";

                try
                {
                    string filePath = dialog.FileName;
                    AlgorithmType selectedAlg = SelectedAlgorithm;

                    await Task.Run(() =>
                    {
                        IImageSequenceClassifier classifier = selectedAlg == AlgorithmType.DTW_1NN
                            ? (IImageSequenceClassifier)new CustomSequenceClassifier()
                            : new AccordSequenceClassifier();

                        classifier.Load(filePath);
                        _trainedClassifier = classifier;
                    });

                    IsModelTrained = true; 
                    StatusText = $"Модель загружена";
                }
                catch (Exception ex)
                {
                    StatusText = $"Ошибка загрузки: совпадает ли выбранный алгоритм с файлом? ({ex.Message})";
                    IsModelTrained = false;
                }
                finally
                {
                    IsBusy = false;
                }
            }
        }

        private void ExecuteSelectFolder(object p)
        {
            using (var dialog = new FolderBrowserDialog())
            {
                if (dialog.ShowDialog() == DialogResult.OK) SelectedFolderPath = dialog.SelectedPath;
            }
        }

        private void ExecuteSelectSingleImage(object p)
        {
            using (var dialog = new FolderBrowserDialog())
            {
                dialog.Description = "Выберите папку с кадрами (например, Seq_0)";
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    InteractiveSequencePath = dialog.SelectedPath;

                    var firstImage = Directory.GetFiles(InteractiveSequencePath, "*.jpg").OrderBy(f => f).FirstOrDefault();
                    if (firstImage != null)
                    {
                        PreviewImagePath = firstImage;
                        SinglePredictionResult = "Последовательность загружена, нажмите 'Распознать'";
                    }
                    else
                    {
                        PreviewImagePath = null;
                        SinglePredictionResult = "В папке нет .jpg файлов";
                    }
                }
            }
        }

        private async void ExecutePredictSingle(object p)
        {
            IsBusy = true;
            SinglePredictionResult = "Анализирую последовательность...";

            var imageProgress = new Progress<string>(path => PreviewImagePath = path);
            var reporter = (IProgress<string>)imageProgress;

            try
            {
                await Task.Run(async () => 
                {
                    var imageFiles = Directory.GetFiles(InteractiveSequencePath, "*.jpg").OrderBy(f => f).ToList();
                    if (imageFiles.Count == 0) throw new Exception("Нет кадров для анализа!");

                    var sequence = new double[imageFiles.Count][];
                    for (int i = 0; i < imageFiles.Count; i++)
                    {
                        reporter.Report(imageFiles[i]);
                        sequence[i] = ImagePreprocessor.ProcessImage(imageFiles[i]);
                        await Task.Delay(30);
                    }

                    int res = _trainedClassifier.Predict(sequence);
                    SinglePredictionResult = $"Результат: Класс {res}. Анализ завершен!";
                });
            }
            catch (Exception ex) { SinglePredictionResult = $"Ошибка: {ex.Message}"; }
            finally { IsBusy = false; }
        }
    }
}