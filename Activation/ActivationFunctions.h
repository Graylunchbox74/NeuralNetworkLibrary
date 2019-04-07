#include <cmath>

namespace Activation{
	enum Activation_Function{IDENTITY, RELU, SIGMOID, FAST_SIGMOID, HYPERBOLIC_TANGENT};
	auto Identity(float x) -> float;
	auto Identity_Derivative(float x) -> float;
	auto Fast_Sigmoid(float x) -> float;
	auto Fast_Sigmoid_Derivative(float x) -> float;
	auto Sigmoid(float x) -> float;
	auto Sigmoid_Derivative(float x) -> float;
	auto Relu(float x) -> float;
	auto Relu_Derivative(float x) -> float;
	auto Hyperbolic_Tangent(float x) -> float;
	auto Hyperbolic_Tangent_Derivative(float x) -> float;
};
