# nmrclassify

The model is a supervised single-channel source separation network inspired by Cov-TasNet https://ieeexplore.ieee.org/abstract/document/8707065. The model separates a mixture of 3 single compounds into the single compounds and classifies each.


## Motivation

The long-term objective is a network that classifies compounds in an NMR spectrum. Multiplets are 18 peak groups with specific area ratios and separation distances. Pure compounds have a unique multiplet pattern (multiplets at 65536 possible ppm positions). The network will be trained on database multiplet data for 877 pure compounds. The input to the network will be a mixture of multiplet sequences of 50-100 pure compounds, and the model will separate the mixture into the single compounds and classify them. 

An accurate automatic NMR classification method would be extremely valuable because manual compound identification is time-consuming, challenging and expert dependent. There are consistent weaknesses in existing automatic approaches that would be overcome using neural networks. Current approaches are slow, expensive (commercial), requires user input, and/or result in high false positives and negatives. Neural networks would improve on current methods with high computation speed, robustness and accuracy, and low user dependence.


## Datasets

The synthetic data is a simplified abstraction of the real NMR data. Multiplets are represented by (1, 6) arrays with elements representing the area ratios between peaks. There are 6 different multiplets representing singlets, doublets, triplets, quartets, quintets and sextets. Singles are 10 single compounds. Masks are used so that multiplets only occupy a percentage of the 21 ppm positions. Two sets are added to represent multiplets of 2 different compounds occupying the same positions. Mixed are a mixture of 3 single compounds.  The testing and training datasets each have 10 mixed compounds.


## Model

The model uses a temporal convolutional network with stacked dilated 1D convolution blocks to estimate the mask for each compound. The model consists of an encoder, separation module and classification module. Depth conv is the convolution block in the separation module, TCN is the separation module, and Net is the full network. The parameters are based on Cov-TasNet but the values are lower since the data is smaller and is pattern data which is simpler than spectral data. These parameters will be scaled up when real data is used.

**Encoder:**

The encoder uses 1D convolution to transform the (1, 6, 21) (effectively one-dimensional because each multiplet is (1, 6)) input to a (1, 64, 21) (multi-dimensional) representation. The model then uses layer normalization to normalize each channel and pointwise convolution to reduce the number of channels (bottleneck layer). 

**Separation:**

The model estimates the masks using 3 stacks of 4 layers of convolution blocks with increasing dilation factors. Dilation factors are used to capture the long-range dependence of the pattern. Each convolution block uses depthwise separable convolution (depthwise then pointwise convolution, which reduces the number of parameters thus the model size). The skip connections are summed and the 3 masks are estimated using a pointwise convolution and a Sigmoid nonlinear activation function. The mixture is separated into the single compounds by element-wise multiplying the encoded input mixture and the masks. 

**Classification:**

1D convolution is used to transform the separated mixture from multi-dimensional to one-dimensional, a linear layer is used to transform the third dimension from 21 (ppm positions) to 10 (compound classes), and a Softmax layer is used to determine the probability distribution across the classes.

The output of the network is the probability distribution across the classes. The model is trained using cross entropy loss and 1000 epochs. The stochastic gradient descent, stochastic gradient descent with momentum of 0.99 and Adam optimizer were used, each with a weight decay of 1e-5. These parameters were chosen because they result in the lowest losses.

## Results

The optimizer that results in the lowest training loss and highest testing accuracy changes each time I run the model. I'm not sure why this happens because I used a random seed so that the data stays the same. Example results are 1.9537 training loss and 0.8286 testing accuracy for SGD, 1.707 training loss and 0.4416 testing accuracy for SGD with momentum, and 1.7201 training loss and 0.6298 testing accuracy for Adam. The results are always around these values. 

## Future work

The current network is a good first step for building the full network, however it requires the number of single compounds in the mixture to be known. In real data the number is unknown and will need to be determined by the network. One idea I had is to set the number to a value greater than the maximum possible and cluster the estimated masks to get the true single compounds. I will need to determine whether setting the number higher than the true number causes issues for mask estimation.

Other future work is to use both the raw spectral data and derived multiplet data as the inputs to the network. The raw and multiplet data will be combined into a single 2D array during the encoder step. This may improve performance because 1D convolution works better on continuous data, and the network will have 2 layers of paired information to work with.

Once that is done, the model will be scaled up and used with real data. This involves finishing preparing the database data and collecting enough experimental data.
