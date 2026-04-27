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

namespace SequenceClassificationModel.WPF.ViewModels
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

        private string _singleImagePath;
        public string SingleImagePath { get => _singleImagePath; set { _singleImagePath = value; OnPropertyChanged(); } }

        private string _singlePredictionResult = "Ожидание изображения...";
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
        private bool CanExecutePredictSingle(object p) => IsModelTrained && !string.IsNullOrEmpty(SingleImagePath) && !IsBusy;

        private async void ExecuteTrainTest(object parameter)
        {
            IsBusy = true;
            IsModelTrained = false;
            var progress = new Progress<string>(msg => StatusText = msg);
            var reporter = (IProgress<string>)progress;

            try
            {
                await Task.Run(() => {
                    reporter.Report("Загрузка данных...");
                    var (Sequences, Labels) = DataLoader.LoadData(SelectedFolderPath);
                    if (Sequences.Count == 0) throw new Exception("Нет данных для обучения!");

                    Random rnd = new Random();
                    var shuffledIndices = Enumerable.Range(0, Sequences.Count).OrderBy(x => rnd.Next()).ToList();
                    var shuffledSeqs = shuffledIndices.Select(i => Sequences[i]).ToList();
                    var shuffledLabels = shuffledIndices.Select(i => Labels[i]).ToArray();

                    double testRatio = TestSizePercentage / 100.0;
                    int testCount = (int)(shuffledSeqs.Count * testRatio);
                    int trainCount = shuffledSeqs.Count - testCount;

                    var trainSeqs = shuffledSeqs.Take(trainCount).ToList();
                    var trainLabels = shuffledLabels.Take(trainCount).ToArray();
                    var testSeqs = shuffledSeqs.Skip(trainCount).ToList();
                    var testLabels = shuffledLabels.Skip(trainCount).ToArray();

                    reporter.Report($"Обучение {SelectedAlgorithm}...");

                    IImageSequenceClassifier classifier = SelectedAlgorithm == AlgorithmType.DTW_1NN
                        ? (IImageSequenceClassifier)new CustomSequenceClassifier()
                        : new AccordSequenceClassifier();

                    classifier.Train(trainSeqs, trainLabels);
                    _trainedClassifier = classifier;

                    if (!RunTesting)
                    {
                        reporter.Report($"Обучение завершено (тестирование пропущено)");
                        return;
                    }

                    reporter.Report("Тестирование...");

                    int correctPredictions = 0;
                    int processedCount = 0;
                    object lockObj = new object();

                    Parallel.For(0, testSeqs.Count, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, i =>
                    {
                        int predicted = classifier.Predict(testSeqs[i]);
                        lock (lockObj)
                        {
                            if (predicted == testLabels[i]) correctPredictions++;
                            processedCount++;

                            if (processedCount % 10 == 0 || processedCount == testSeqs.Count)
                                reporter.Report($"Тестирование: обработано {processedCount} из {testSeqs.Count}");
                        }
                    });

                    double accuracy = (double)correctPredictions / testSeqs.Count * 100.0;
                    reporter.Report($"Готово, точность: {accuracy:F2}% (угадано {correctPredictions} из {testSeqs.Count})");
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
                    {
                        _trainedClassifier.Save(filePath);
                    });

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
            var dialog = new Microsoft.Win32.OpenFileDialog { Filter = "Images|*.jpg;*.png;*.bmp" };
            if (dialog.ShowDialog() == true) SingleImagePath = dialog.FileName;
        }

        private async void ExecutePredictSingle(object p)
        {
            IsBusy = true;
            try
            {
                await Task.Run(() => {
                    var features = ImagePreprocessor.ProcessImage(SingleImagePath);
                    int res = _trainedClassifier.Predict(new double[][] { features });
                    SinglePredictionResult = $"Класс: {res}";
                });
            }
            finally { IsBusy = false; }
        }
    }
}