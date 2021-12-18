#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
using namespace std;

void InitialWeightAndBias(double* weight, int weightCount, double& bias);
double Sigmoid(double z);
double LossFunction(double expResult, double actResult);
void ReadDataFromFile(string fileName, double* data);
void VectorMatrixMultiplication(int w, int h, double* i_matrix, double* i_vector, double* o_vector, double bias = 0);
double ForwardBackwardPropagation(double* x, double* y, double* w, double& bias);
void MatrixTranpose(int w, int h, double* i_matrix, double* o_matrix);
void Test(double* x, double* y, double* w, double bias);

const double LRate = 0.01;
const int FeatureCount = 30;			//FEATURES IN DATASET
const int TrainCount = 455;				
const int TestCount = 114;
const int MaxIteration = 1000;
// TRAIN DATA= 80% TEST DATA=20%

int main()
{
	string trainXPath = "x_train.txt", trainYPath = "y_train.txt", testXPath = "x_test.txt", testYPath = "y_test.txt";


	double x_Train[FeatureCount * TrainCount], x_Test[FeatureCount * TestCount], y_Train[TrainCount], y_Test[TestCount];
	double weight[FeatureCount];
	double bias;
	double cost = 0;

	// READING DATA FROM FILES
	ReadDataFromFile(trainXPath, x_Train);
	ReadDataFromFile(testXPath, x_Test);
	ReadDataFromFile(trainYPath, y_Train);
	ReadDataFromFile(testYPath, y_Test);

	InitialWeightAndBias(weight, FeatureCount, bias);

	// TRAINING THE MODEL
	//FINDING COST AFTER EACH ITERATION
	for (int i = 0; i < MaxIteration; i++)
	{
		cost = ForwardBackwardPropagation(x_Train, y_Train, weight, bias);

		if (i % 10 == 0)
		{
			cout << "Cost after iteration " << i << " is " << cost << endl;
		}
	}
	//END

	//TESTING THE MODEL
	Test(x_Test, y_Test, weight, bias);


	return 0;
}
//CALCULATING THE LOSS FUNCTION (BINARY CROSS ENTROPY)(LOG LOSS)
double LossFunction(double expResult, double actResult)
{
	double result = 0;

	if (expResult == 0)
	{
		result = -log(1 - actResult);
	}
	else if (expResult == 1)
	{
		result = -log(actResult);
	}

	return result;
}

void InitialWeightAndBias(double* weight, int weightCount, double& bias)
{
	bias = 0;

	for (int i = 0; i < weightCount; i++)
	{
		weight[i] = LRate;      
	}
	// END
}
//USING THE LOGISTIC FUNCTION
double Sigmoid(double z)
{
	double y = 0;

	y = 1 / (1 + exp(-z));

	return y;
}
//FUNCTION TO READ THE FILES
void ReadDataFromFile(string fileName, double* data)
{
	ifstream infile;
	infile.open(fileName);

	if (infile.is_open() == false)
	{
		cout << fileName << " could not be opened" << endl;
		return;
	}

	long long index = 0;

	while (!infile.eof())
	{
		infile >> data[index];
		index++;
	}
	// END
	infile.close();

	cout << fileName << " read succesfully " << endl;
}
//CALCULATING z=w.x+b
void VectorMatrixMultiplication(int w, int h, double* i_matrix, double* i_vector, double* o_vector, double bias)
{
	for (int i = 0; i < h; i++)
	{
		o_vector[i] = 0.0;

		for (int j = 0; j < w; j++)
		{
			o_vector[i] += i_vector[j] * i_matrix[(i * w) + j] + bias;
		}
	}
	//END
}
//CALCULATING THE TRANSPOSE MATRIX
void MatrixTranpose(int w, int h, double* i_matrix, double* o_matrix)
{
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			o_matrix[(j * h) + i] = i_matrix[(i * w) + j];
		}
	}
	// END
}
//REDUCING THE COST FUNCTION USING DERIVATIVE CHAIN RULE TO GET LOCAL MINIMA
double ForwardBackwardPropagation(double* x, double* y, double* w, double& bias)
{
	double z[TrainCount];
	double yResult[TrainCount];
	double lossResult = 0;
	double costResult = 0;
	double ActExpectedDifferance[TrainCount];
	double sumDifferance = 0;
	double derivativeWeight[FeatureCount];
	// Z =X.W + b;
	VectorMatrixMultiplication(FeatureCount, TrainCount, x, w, z, bias);

	// CALCULATING THE LOSS FUNCTION AND THE COST USING SIGMOID FUNCTION
	// yResult = 1 / (1 + e^(-x))
	for (int i = 0; i < TrainCount; i++)
	{
		yResult[i] = Sigmoid(z[i]);

		ActExpectedDifferance[i] = yResult[i] - y[i];

		sumDifferance = sumDifferance + ActExpectedDifferance[i];

		// EXPECTED RESULT - ACTUAL RESULT
		lossResult = LossFunction(y[i], yResult[i]);

		costResult += lossResult;
	}

	costResult = costResult / TrainCount;		// FOR SCALING   (COST FUNCTION IS FOR ONE ITERATION SO IT IS AVERAGE)

	// DERIVATIVE WEIGHT
	// (ACTUAL - EXPECTED) * X^t (PROBABLILITY IF WE REPEAT THE EXPERIMENT MANY TIMES)
	double transposeX[TrainCount * FeatureCount];

	// TRANSPOSE
	MatrixTranpose(FeatureCount, TrainCount, x, transposeX);

	// GET DERIVATIVE WEIGHT
	VectorMatrixMultiplication(TrainCount, FeatureCount, transposeX, ActExpectedDifferance, derivativeWeight, 0);

	// UPDATE WEIGHT
	for (int i = 0; i < FeatureCount; i++)
	{
		w[i] = w[i] - LRate * derivativeWeight[i];
	}
	// UPDATE BIAS
	bias = bias - LRate * (sumDifferance / TrainCount);

	return costResult;
}
void Test(double* x, double* y, double* w, double bias)
{
	double z[TestCount];
	double yResult[TestCount];
	double accuracy = 0;
	// DOT PRODUCT OF w WITH t
	VectorMatrixMultiplication(FeatureCount, TestCount, x, w, z, bias);
	for (int i = 0; i < TestCount; i++)
	{
		yResult[i] = Sigmoid(z[i]);

		if (yResult[i] < 0.5)
		{
			yResult[i] = 0;
		}
		else
		{
			yResult[i] = 1;
		}

		accuracy = accuracy + abs(y[i] - yResult[i]);
	}

	accuracy = 100 - (accuracy / TestCount) * 100;

	cout << "Test accuracy: " << accuracy << endl;
}