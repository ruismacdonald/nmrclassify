# nmrclassify

The model is a supervised single-channel source separation network inspired by Cov-TasNet https://ieeexplore.ieee.org/abstract/document/8707065. The model separates a mixture of 3 single compounds into the single compounds and classifies each.


## Motivation

The long-term objective is a network that classifies compounds in an NMR spectrum. Multiplets are 18 peak groups with specific area ratios and separation distances. Pure compounds have a unique multiplet pattern (multiplets at 65536 possible ppm positions). The network will be trained on database multiplet data for 877 pure compounds. The input to the network will be a mixture of multiplet sequences of 50-100 pure compounds, and the model will separate the mixture into the single compounds and classify them. 

A accurate automatic NMR classification method would be extremely valuable because manual compound identification is time-consuming, challenging and expert dependent. There are consistent weaknesses in existing automatic approaches that would be overcome using neural networks. Current approaches are slow, expensive (commercial), requires user input, and/or results in high false positives and negatives. Neural networks would improve on current methods with high computation speed, robustness and accuracy, and low user dependence.


## Datasets

The synethetic data is a simplified abstraction of the real NMR data. Multiplets are represented by (1, 6) arrays with elements representing area ratios. There are 6 different multiplets representing singlets, doublets, triplets, quartets, quintets and sextets. Single compounds (singles) are the addition of 2 sets of multiplets occupying 20% of 21 possible ppm positions. Addition is used to represent addition ratios (multiplets of 2 compounds occupying the same position). Mixed compounds (mixed) are the addition of 3 singles, which results in multiplets occupying 60% of 21 ppm positions.  The testing and training datasets each have 10 mixed compounds. 


## Model

The model uses a temporal convolutional network with stacked dilated 1D convolution blocks to estimate the mask for each compound. The model consists of an encoder, separation module and decoder. 

**Encoder:**

The encoder uses 1D convolution to transform the (1, 6, 21) (effectively one-dimensional because each multiplet is (1, 6)) input to a (1, 64, 21) (multi-dimensional) representation. The model then uses layer normalization to normalize each channel and pointwise convolution to reduce the number of channels (bottleneck layer). 

**Separation:**

The model estimates the masks using 3 stacks of 4 layers of convolution blocks with increasing dilation factors. Dilation factors are used to capture the long-range dependence of the pattern. Each convolution block uses depthwise separable convolution (depthwise then pointwise convolution, which reduces the number of parameters thus the model size). The skip connections are summed and the 3 masks are estimated using a pointwise convolution and a Sigmoid nonlinear activation function. The mixture is separated into the single compounds by element-wise multiplying the encoded input mixture and the masks. 

**Classification:**

1D convolution is used to transform the separated mixture from multi-dimensional to one-dimensional, a linear layer is used to transform the third dimension from 21 (ppm positions) to 10 (compound classes), and a Softmax layer is used to determine the probability distribution across the classes.


