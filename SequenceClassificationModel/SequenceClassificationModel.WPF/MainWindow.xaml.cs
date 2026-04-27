using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using SequenceClassificationModel.WPF.ViewModels;

namespace SequenceClassificationModel.WPF
{
    public partial class MainWindow : Window
    {
        [DllImport("dwmapi.dll", PreserveSig = true)]
        public static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int attrValue, int attrSize);

        public MainWindow()
        {
            InitializeComponent();
            //this.DataContext = new MainViewModel();

            this.SourceInitialized += (s, e) =>
            {
                IntPtr hwnd = new WindowInteropHelper(this).Handle;
                int trueValue = 1;

                DwmSetWindowAttribute(hwnd, 20, ref trueValue, Marshal.SizeOf(typeof(int)));
            };
        }
    }
}