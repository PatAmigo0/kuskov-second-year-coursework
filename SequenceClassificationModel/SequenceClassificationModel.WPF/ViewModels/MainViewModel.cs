using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Forms; // Для FolderBrowserDialog
using Microsoft.Win32; // НОВОЕ: Для OpenFileDialog (выбор одного файла)
using SequenceClassificationModel.Core.Interface;
using SequenceClassificationModel.Core.Models;
using SequenceClassificationModel.Core.Utils;

namespace SequenceClassificationModel.WPF.ViewModels
{
    public class MainViewModel : ViewModelBase
    {
        private string _statusText = "Программа готова к работе";
        public string StatusText
        {
            get => _statusText;
            set { _statusText = value; OnPropertyChanged(); }
        }

        private string _selectedFolderPath = "Папка не выбрана";
        public string SelectedFolderPath
        {
            get => _selectedFolderPath;
            set { _selectedFolderPath = value; OnPropertyChanged(); }
        }

        private int _testSizePercentage = 20;
        public int TestSizePercentage
        {
            get => _testSizePercentage;
            set { _testSizePercentage = value; OnPropertyChanged(); }
        }

        public List<string> AvailableAlgorithms { get; }
        private string _selectedAlgorithm;
        public string SelectedAlgorithm
        {
            get { return _selectedAlgorithm; }
            set { _selectedAlgorithm = value; OnPropertyChanged(); }
        }

        public ICommand SelectFolderCommand { get; }
        public ICommand TrainTestCommand { get; }

        // UI
        private bool _isBusy;
        public bool IsBusy
        {
            get { return _isBusy; }
            set
            {
                _isBusy = value;
                OnPropertyChanged();
                System.Windows.Input.CommandManager.InvalidateRequerySuggested();
            }
        }

        private IImageSequenceClassifier _trainedClassifier;

        private bool _isModelTrained;
        public bool IsModelTrained 
        {
            get { return _isModelTrained; }
            set { _isModelTrained = value; OnPropertyChanged(); }
        }

        private string _singleImagePath;
        public string SingleImagePath
        {
            get { return _singleImagePath; }
            set { _singleImagePath = value; OnPropertyChanged(); }
        }

        private string _singlePredictionResult = "Ожидание картинки...";
        public string SinglePredictionResult
        {
            get { return _singlePredictionResult; }
            set { _singlePredictionResult = value; OnPropertyChanged(); }
        }

        public ICommand SelectSingleImageCommand { get; }
        public ICommand PredictSingleCommand { get; }

        public MainViewModel()
        {
            AvailableAlgorithms = new List<string> { "Мой алгоритм (Custom)", "Алгоритм из Accord.NET" };
            SelectedAlgorithm = AvailableAlgorithms[0];

            SelectFolderCommand = new RelayCommand(ExecuteSelectFolder);
            TrainTestCommand = new RelayCommand(ExecuteTrainTest, CanExecuteTrainTest);
            SelectSingleImageCommand = new RelayCommand(ExecuteSelectSingleImage);
            PredictSingleCommand = new RelayCommand(ExecutePredictSingle, CanExecutePredictSingle);
        }

        private void ExecuteSelectFolder(object parameter)
        {
            using (var dialog = new FolderBrowserDialog())
            {
                dialog.Description = "Выберите папку с обучающими изображениями";
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    SelectedFolderPath = dialog.SelectedPath;
                    StatusText = "Папка выбрана. Готов к обучению.";
                }
            }
        }

        private bool CanExecuteTrainTest(object parameter)
        {
            return SelectedFolderPath != "Папка не выбрана" && Directory.Exists(SelectedFolderPath) && !IsBusy;
        }

        private async void ExecuteTrainTest(object parameter)
        {
            IsBusy = true;
            IsModelTrained = false;

            try
            {
                var progress = new Progress<string>(message => StatusText = message);
                var progressReporter = (IProgress<string>)progress;

                await Task.Run(() =>
                {
                    progressReporter.Report("Шаг 1/3: Загрузка данных с диска...");
                    var data = DataLoader.LoadData(SelectedFolderPath);
                    var sequences = data.Sequences;
                    var labels = data.Labels;

                    if (sequences.Count == 0) throw new Exception("В папке нет картинок!");

                    Random rnd = new Random();
                    var shuffledIndices = Enumerable.Range(0, sequences.Count).OrderBy(x => rnd.Next()).ToList();
                    var shuffledSeqs = shuffledIndices.Select(i => sequences[i]).ToList();
                    var shuffledLabels = shuffledIndices.Select(i => labels[i]).ToArray();

                    double testRatio = TestSizePercentage / 100.0;
                    int testSize = (int)(shuffledSeqs.Count * testRatio);
                    int trainSize = shuffledSeqs.Count - testSize;

                    var trainSeqs = shuffledSeqs.Take(trainSize).ToList();
                    var trainLabels = shuffledLabels.Take(trainSize).ToArray();
                    var testSeqs = shuffledSeqs.Skip(trainSize).ToList();
                    var testLabels = shuffledLabels.Skip(trainSize).ToArray();

                    progressReporter.Report($"Шаг 2/3: Обучение {SelectedAlgorithm} (может занять несколько минут)...");

                    IImageSequenceClassifier classifier;
                    if (SelectedAlgorithm == "Собственный алгоритм (DTW)")
                        classifier = new CustomSequenceClassifier();
                    else
                        classifier = new AccordSequenceClassifier();

                    classifier.Train(trainSeqs, trainLabels);
                    _trainedClassifier = classifier;

                    int correctPredictions = 0;
                    for (int i = 0; i < testSeqs.Count; i++)
                    {
                        if (i % 5 == 0)
                            progressReporter.Report($"Шаг 3/3: Тестирование... Проверено {i} из {testSeqs.Count}");

                        if (classifier.Predict(testSeqs[i]) == testLabels[i])
                            correctPredictions++;
                    }

                    double accuracy = (double)correctPredictions / testSeqs.Count * 100.0;
                    progressReporter.Report($"Готово! Точность: {accuracy:F2}% (Угадано {correctPredictions} из {testSeqs.Count})");
                });

                IsModelTrained = true;
            }
            catch (Exception ex)
            {
                StatusText = $"Ошибка: {ex.Message}";
            }
            finally
            {
                IsBusy = false;
                System.Windows.Input.CommandManager.InvalidateRequerySuggested();
            }
        }

        private void ExecuteSelectSingleImage(object parameter)
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Изображения|*.jpg;*.jpeg;*.png;*.bmp",
                Title = "Выберите картинку для проверки"
            };

            if (dialog.ShowDialog() == true)
            {
                SingleImagePath = dialog.FileName;
                SinglePredictionResult = "Картинка загружена. Нажмите 'Распознать'.";
            }
        }

        private bool CanExecutePredictSingle(object parameter)
        {
            return IsModelTrained && !string.IsNullOrEmpty(SingleImagePath) && !IsBusy;
        }

        private async void ExecutePredictSingle(object parameter)
        {
            IsBusy = true;
            SinglePredictionResult = "Анализирую...";

            try
            {
                await Task.Run(() =>
                {
                    double[] features = ImagePreprocessor.ProcessImage(SingleImagePath);
                    double[][] sequence = new double[][] { features };
                    int predictedClass = _trainedClassifier.Predict(sequence);

                    SinglePredictionResult = $"Модель считает, что это класс: {predictedClass}";
                });
            }
            catch (Exception ex)
            {
                SinglePredictionResult = $"Ошибка распознавания: {ex.Message}";
            }
            finally
            {
                IsBusy = false;
            }
        }
    }
}