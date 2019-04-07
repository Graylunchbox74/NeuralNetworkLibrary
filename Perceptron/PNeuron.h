#include <cstdio>
#include <vector>

class PNeuron{
public:
	float pre_activation_function_value;
	float delta;
	float value;
	std::vector<float> weights;

	PNeuron(unsigned int previous_layer_node_count, unsigned int node_count);


	auto Activate(std::vector<float> input) -> float;
	auto change_weights(std::vector<float> new_weights) -> void;
};