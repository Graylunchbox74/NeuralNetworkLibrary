#include <functional>
#include <vector>

#include "../Activation/ActivationFunctions.h"
#include "PNeuron.h"

namespace NN {
	class Perceptron;
}

class PLayer{
	friend NN::Perceptron;
public:
	PLayer(unsigned int previous_layers_node_count, unsigned int node_count, Activation::Activation_Function activation_function);

	auto Activate_Layer(std::vector<float> input) -> std::vector<float>;
private:
	std::function<float(float)> activation_function;
	std::vector<PNeuron> neurons;
	std::function<float(float)> activation_function_derivative;

	auto set_weights(std::vector<std::vector<float> > new_weights) -> void;
};