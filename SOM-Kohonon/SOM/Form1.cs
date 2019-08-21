using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SOM
{
    public partial class Form1 : Form
    {

        public Form1()
        {
            InitializeComponent();
        }

        public void Button1_Click(object sender, EventArgs e)
        {
            int col;
            int size;
            string adrees;
           
            try
            {
                adrees = textBox1.Text;             
                col = int.Parse(textBox2.Text);
                size = int.Parse(textBox3.Text);
             
                int Xmin = 0;

                int Xmax = size;

                double Step = 1;
                int count = (int)Math.Ceiling((Xmax - Xmin) / Step) + 1;
                chart1.ChartAreas[0].AxisX.Minimum = 0;
                chart1.ChartAreas[0].AxisX.Maximum = Xmax;
                chart1.ChartAreas[0].AxisX.MajorGrid.Interval = Step;
                double[] x = new double[count];

                Map b = new Map(col, size, adrees);

                for (int i = 0; i < b.patterns.Count; i++)
                {
                    Neuron n = b.Winner(b.patterns[i]);
                    listBox1.Items.Add($"{ b.labels[i]},{ n.X},{ n.Y}");
                    chart1.Series[0].Points.AddXY(n.X, n.Y);
                    //chart1.Series[0].Points.AddXY($"{ n.X},{ n.Y}");

                }
                
                MessageBox.Show(
           "Данные успешно получены",
           "Успех",
           MessageBoxButtons.OK,
           MessageBoxIcon.Information,
           MessageBoxDefaultButton.Button1);
            
            }
            catch
            {
              
           MessageBox.Show(
           "Неверный ввод входных данных",
           "Ошибка",
           MessageBoxButtons.OK,
           MessageBoxIcon.Error,
           MessageBoxDefaultButton.Button1);
            }

        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void Button4_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void ListBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            

        }

        private void ListBox2_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void Button2_Click(object sender, EventArgs e)
        {
            textBox1.Clear();
            textBox2.Clear();
            textBox3.Clear();
            listBox1.Items.Clear();
            chart1.Series[0].Points.Clear();
            MessageBox.Show(
           "Программа готова к использованию",
           "Поля очищены.",
           MessageBoxButtons.OK,
           MessageBoxIcon.Information,
           MessageBoxDefaultButton.Button1);
        }

        private void Chart1_Click(object sender, EventArgs e)
        {

        }
    }

}
