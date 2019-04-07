#include <functional>
#include <vector>
#include <string>

//#include "Activation/ActivationFunctions.h"
#include "Perceptron/PLayer.h"

namespace NN{

	class Perceptron
	{
	public:
		Perceptron(unsigned int input_node_count, unsigned int output_node_count, std::vector<unsigned int> hidden_layer_node_count, Activation::Activation_Function activation_function, float starting_learning_rate);
		Perceptron(unsigned int input_node_count, unsigned int output_node_count, std::vector<unsigned int> hidden_layer_node_count, std::vector<Activation::Activation_Function> layer_activation_functions, float starting_learning_rate);
		Perceptron(std::string file_name);

		auto Save(std::string file_name) -> void;
		auto Activate(std::vector<float> input) -> std::vector<float>;

		auto Train(std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& expectedOutput) -> void;
		float cost;

	private:
		std::vector<PLayer> layers;
		unsigned int num_layers;
		unsigned int input_node_count;
		std::vector<unsigned int> num_nodes_per_layer;
		float learning_rate;

		auto back_propagate(std::vector<std::vector<std::vector<float>>>& changeWeights,std::vector<float> input) -> void;
		auto back_propagate_delta(std::vector<float>& expectedOutput) -> void;
	};


	class Convolutional
	{
	public:
	private:
	};

};