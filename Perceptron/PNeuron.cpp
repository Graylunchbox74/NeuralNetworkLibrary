#include "PNeuron.h"

PNeuron::PNeuron(unsigned int previous_layer_node_count, unsigned int node_count){
	for (int i = 0; i < previous_layer_node_count; ++i)
	{
		this->weights.push_back(((float)(rand() % 2000) - 1000) / 10000);
	}
	this->pre_activation_function_value = 0;
	this->delta = 0;
	this->value = 0;
}

auto PNeuron::Activate(std::vector<float> input) -> float
{
	if (input.size() != this->weights.size())
	{
		return 0;
	}

	this->pre_activation_function_value = 0;
	for (int i = 0; i < this->weights.size(); ++i)
	{
		this->pre_activation_function_value += this->weights[i] * input[i];
	}
	return this->pre_activation_function_value;
}

auto PNeuron::change_weights(std::vector<float> new_weights) -> void
{
	for (int i = 0; i < weights.size(); ++i)
	{
		this->weights[i] += new_weights[i];
	}
}