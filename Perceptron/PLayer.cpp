#include "PLayer.h"

#include <functional>

PLayer::PLayer(unsigned int previous_layers_node_count, unsigned int node_count, Activation::Activation_Function activation_function)
{
	for (int i = 0; i < node_count; ++i)
	{
		this->neurons.push_back(PNeuron(previous_layers_node_count + 1, node_count));
	}

	switch(activation_function)
	{
		case Activation::IDENTITY:
			this->activation_function = Activation::Identity;
			this->activation_function_derivative = Activation::Identity_Derivative;
			break;
		case Activation::SIGMOID:
			this->activation_function = Activation::Sigmoid;
			this->activation_function_derivative = Activation::Sigmoid_Derivative;
			break;
		case Activation::RELU:
			this->activation_function = Activation::Relu;
			this->activation_function_derivative = Activation::Relu_Derivative;
			break;
		case Activation::HYPERBOLIC_TANGENT:
			this->activation_function = Activation::Hyperbolic_Tangent;
			this->activation_function_derivative = Activation::Hyperbolic_Tangent_Derivative;
			break;
		case Activation::FAST_SIGMOID:
			this->activation_function = Activation::Fast_Sigmoid;
			this->activation_function_derivative = Activation::Fast_Sigmoid_Derivative;
			break;
	}
}

auto PLayer::Activate_Layer(std::vector<float> input) -> std::vector<float>
{
	std::vector<float> output;
	for (int i = 0; i < this->neurons.size(); ++i)
	{
		output.push_back(this->neurons[i].Activate(input));
		this->neurons[i].value = this->activation_function(output[i]);
		output[i] = this->neurons[i].value;
	}
	return output;
}

auto PLayer::set_weights(std::vector<std::vector<float> > new_weights) -> void
{
	for (int i = 0; i < this->neurons.size(); ++i)
	{
		this->neurons[i].change_weights(new_weights[i]);
	}
}

