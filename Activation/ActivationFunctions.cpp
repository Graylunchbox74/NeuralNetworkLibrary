#include "ActivationFunctions.h"

auto Activation::Identity(float x) -> float 
{
	return x;
}

auto Activation::Identity_Derivative(float x) -> float 
{
	return 1;
}

auto Activation::Fast_Sigmoid(float x) -> float
{
	return (x / (1 + fabs(x)));
}

auto Activation::Fast_Sigmoid_Derivative(float x) -> float
{
	return 1 / ((1.f + fabs(x)) * (1.f + fabs(x)));
}

auto Activation::Sigmoid(float x) -> float
{
	return (float)(1 / 1 + exp(-x));
}

auto Activation::Sigmoid_Derivative(float x) -> float 
{
	return 0;
}

auto Activation::Relu(float x) -> float
{
	return fmax(0,x);
}

auto Activation::Relu_Derivative(float x) -> float 
{
	return x > 0 ? 1 : 0;
}

auto Activation::Hyperbolic_Tangent(float x) -> float
{
	return (exp(x)-exp(-1))/(exp(x)+exp(-x));
}

auto Activation::Hyperbolic_Tangent_Derivative(float x) -> float 
{
	return 0;
}