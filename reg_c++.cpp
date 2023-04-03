#include<iostream>
#include<vector>
#include<tuple>
#include<numeric>
#include<cmath>
#include<limits>

using namespace std;

class LinearReg
{
public:
	LinearReg(){}
	~LinearReg(){}
	LinearReg(vector<double> & m_x_vals_, vector<double> m_y_vals_) : m_x_vals(m_x_vals_),
	m_y_vals(m_y_vals_), m_num_elems(m_y_vals.size()) {}

	void trainAlg(int num_iters, double a_init,double b_init)
	{	
		int iter = 0;
		m_a = a_init;
		m_b = b_init;
		while (iter < num_iters)
		{
			double step = 0.02;
			double a_grad = 0;
			double b_grad = 0;

			for (int i = 0; i < m_num_elems; i++)
			{
				a_grad += m_x_vals[i] * ((m_a * m_x_vals[i] + m_b) - m_y_vals[i]);

			}
			a_grad = (2 * a_grad) / m_num_elems;

			for (int i = 0; i < m_num_elems; i++)
			{
				b_grad += ((m_a * m_x_vals[i] + m_b) - m_y_vals[i]);
			}
			b_grad = (2 * b_grad) / m_num_elems;
		
			m_a = m_a - (step * a_grad);
			m_b = m_b - (step * b_grad);

			cout << "a: \t" << m_a << ", b: \t" << m_b << "\r" << endl;
			cout << "grad_a:\t" << a_grad << ", grad_b:\t" << b_grad << "\r" << endl;
			iter++;
		}

	}
	double regress(double x)
	{
		double res = m_a * x + m_b;
		return res;
	}

private:
	vector<double> m_x_vals;
	vector<double> m_y_vals;
	double m_num_elems;
	double m_a;
	double m_b;

};

int main(int argc, char** argv)
{
	setlocale(LC_ALL, "");

	vector<double> y({10,20,30,40,50});
	vector<double> x({1,2,3,4,5});
	LinearReg lr(x, y);
	lr.trainAlg(1000, 3, -10);
	cout << lr.regress(3)<<endl;
	system("pause");
	return 0;
}
