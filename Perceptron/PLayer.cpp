#include "PLayer.h"

#include <functional>

PLayer::PLayer(unsigned int previous_layers_node_count, unsigned int node_count, Activation::Activation_Function activation_function)
{
	std::vector<float> neuron_w(previous_layers_node_count + 1);

	for (int i = 0; i < node_count; ++i)
	{
		for (int x = 0; x < previous_layers_node_count + 1; ++x)
		{
			neuron_w[x] = ((float)(rand() % 2000) - 1000) / 1000;
		}
		this->neuron_weights.push_back(neuron_w);
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
	for (int i = 0; i < this->neuron_weights.size(); ++i)
	{
		this->pre_activation_function_values[i] = 0;
		for (int x = 0; x < this->neuron_weights[i].size(); ++x)
		{
			this->pre_activation_function_values[i] += input[x] * this->neuron_weights[i][x];
		}
		this->neuron_values[i] = this->activation_function(this->pre_activation_function_values[i]);
		output.push_back(this->neuron_values[i]);
	}
	return output;
}

auto PLayer::set_weights(std::vector<std::vector<float> > new_weights) -> void
{
	for (int i = 0; i < this->neuron_weights.size(); ++i)
	{
		for (int x = 0; x < this->neuron_weights[i].size(); ++x)
		{
			this->neuron_weights[i][x] += new_weights[i][x];
		}
	}
}

