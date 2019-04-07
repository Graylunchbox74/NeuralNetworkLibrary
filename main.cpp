#include "NN.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <string>

inline std::vector<float> getNextImage(std::ifstream& imageFile){
	//784
	std::vector<float> image;
	int buff;
	for (int i = 0; i < 784; ++i)
	{
		imageFile >> buff;
		//image.push_back(float(c)/255);
		image.push_back(float(buff) / 255);
	}
	return image;
}

inline int getNextLabel(std::ifstream& labelFile){
	int c;
	labelFile >> c;
	return int(c);
}

int main(){

	//init

	NN::Perceptron n(784, 10, {10, 10}, Activation::RELU, 1);

	std::vector<std::vector<float> > image_batch;
	std::vector<std::vector<float> > label_batch;
	std::vector<float> current_label = {0,0,0,0,0,0,0,0,0,0};

	for (int i = 0; i < 600; ++i)
	{
		for (int x = 0; x < 25; ++x)
		{
			int p = rand() % 30000;
			std::string filename = "./mnist/training_data/" + std::to_string(p);
			std::ifstream imageFile(filename);

			auto label = getNextLabel(imageFile);
			auto image = getNextImage(imageFile);

			// for (int z = 0; z < 28; ++z)
			// {
			// 	for (int o = 0; o < 28; ++o)
			// 	{
			// 		std::cout << image[z * 28 + o] << " ";
			// 	}
			// 	std::cout << std::endl;
			// }
			// 	std::cout << std::endl;
			// 	std::cout << std::endl;

			current_label[label] = 1;
			label_batch.push_back(current_label);
			current_label[label] = 0;
			image_batch.push_back(image);
		}
		n.Train(image_batch, label_batch);
		image_batch.clear();
		label_batch.clear();
		std::cout << n.cost << std::endl;
	}

	float accuracy = 0;

	for (int i = 0; i < 2000; ++i)
	{
		int p = rand() % 30000;
		std::string filename = "./mnist/training_data/" + std::to_string(p);
		std::ifstream imageFile(filename);
		auto label = getNextLabel(imageFile);
		auto image = getNextImage(imageFile);

		std::vector<float> answer = n.Activate(image);
		int max = 0;
		for (int m = 0; m < 10; ++m)
		{
			if (answer[m] > answer[max])
			{
				max = m;
			}
		}
		if (max == label)
		{
			accuracy++;
		}
	}
	std::cout << "accuracy: ";
	std::cout << accuracy/2000 << std::endl;

	return 0;
}
