using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SOM
{
    public class Map
    {
        public Neuron[,] outputs;  // выход нейрона
        private int iteration;      // итерации
        private int length;        // длина таблицы
        private int dimensions;    // число входных параметров.
        public Random rnd = new Random();
        public List<string> labels = new List<string>();
        public List<double[]> patterns = new List<double[]>();
        //static void Main(string[] args)
        //{

        //    try
        //    {
        //        new Map(3, 8, "*****.csv"); // Адрес файла в формате (.csv)
        //                                    // Формат (число входных данных таблицы – 1 из сроки
        //                                    //так как первая должна быть текстовая 
        //                                    //размер самоорганизующейся карты
        //                                    //адрес файла в формате (.csv)
        //        Console.ReadLine();
        //    }
        //    catch
        //    {
        //        Console.WriteLine("Неправильный выбор выходных данных");
        //    }
        //}
        public Map(int dimensions, int length, string file)
        {
            this.length = length;
            this.dimensions = dimensions;
            Initialise();
            LoadData(file);
            NormalisePatterns();
            Train(0.0000001);
            DumpCoordinates();
        }
        public void Initialise()  // подбор весов
        {
            outputs = new Neuron[length, length];
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    outputs[i, j] = new Neuron(i, j, length);
                    outputs[i, j].Weights = new double[dimensions];
                    for (int k = 0; k < dimensions; k++)
                    {
                        outputs[i, j].Weights[k] = rnd.NextDouble();
                    }
                }
            }
        }

        public void LoadData(string file) // загрузка файла
        {
            try
            {
                StreamReader reader = File.OpenText(file);
                reader.ReadLine(); // не использовать первую линию
                while (!reader.EndOfStream)
                {

                    string[] line = reader.ReadLine().Split(';'); // знак разделения между значениями в таблице формата (.csv) 
                    labels.Add(line[0]);


                    double[] inputs = new double[dimensions];
                    for (int i = 0; i < dimensions; i++)
                    {
                        inputs[i] = double.Parse(line[i + 1]);
                    }
                    patterns.Add(inputs);
                }
                reader.Close();
            }
            catch
            {
                MessageBox.Show(
           "Неправильность входных данных",
           "Ошибка",
           MessageBoxButtons.OK,
           MessageBoxIcon.Error,
           MessageBoxDefaultButton.Button1);
            }
        }


        public void NormalisePatterns()
        {
            for (int j = 0; j < dimensions; j++)
            {
                double sum = 0;
                for (int i = 0; i < patterns.Count; i++)
                {
                    sum += patterns[i][j];
                }
                double average = sum / patterns.Count;
                for (int i = 0; i < patterns.Count; i++)
                {
                    patterns[i][j] = patterns[i][j] / average;
                }
            }
        }
        public void Train(double maxError) // обучение
        {
            double currentError = double.MaxValue;
            while (currentError > maxError)
            {
                currentError = 0;
                List<double[]> TrainingSet = new List<double[]>();
                foreach (double[] pattern in patterns)
                {
                    TrainingSet.Add(pattern);
                }
                for (int i = 0; i < patterns.Count; i++)
                {
                    double[] pattern = TrainingSet[rnd.Next(patterns.Count - i)];
                    currentError += TrainPattern(pattern);
                    TrainingSet.Remove(pattern);
                }
                Console.WriteLine(currentError.ToString("0.0000000"));
            }
        }
        public double TrainPattern(double[] pattern) // выбор победителей 
        {
            double error = 0;
            Neuron winner = Winner(pattern);
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    error += outputs[i, j].UpdateWeights(pattern, winner, iteration);
                }
            }
            iteration++;
            return Math.Abs(error / (length * length));
        }
        public void DumpCoordinates() //вывод результов
        {
            try
            {
                for (int i = 0; i < patterns.Count; i++)
                {
                    Neuron n = Winner(patterns[i]);
                    //Console.WriteLine($"{ labels[i]},{ n.X},{ n.Y}");

                }
            }
            catch
            {
                Console.WriteLine("Ошибка вывода данных");
            }
        }
        public Neuron Winner(double[] pattern)
        {
            Neuron winner = null;
            double min = double.MaxValue;
            for (int i = 0; i < length; i++)
                for (int j = 0; j < length; j++)
                {
                    double d = Distance(pattern, outputs[i, j].Weights);
                    if (d < min)
                    {
                        min = d;
                        winner = outputs[i, j];
                    }
                }
            return winner;
        }
        private double Distance(double[] vector1, double[] vector2) // расстояние между векторами
        {
            double value = 0;
            for (int i = 0; i < vector1.Length; i++)
            {
                value += Math.Pow((vector1[i] - vector2[i]), 2);
            }
            return Math.Sqrt(value);
        }
    }
}
