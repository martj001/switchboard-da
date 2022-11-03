## Dataset: Switchboard Dialog Act
- 10-class classification
    - Train: ~80k samples
    - Test: ~10k samples
    - Label: imbalanced
- Model
    - Baseline/engineered speech features accuracy: 40-50%
    - Self-supervised learning (CPC) accuracy: 42% on training, 44% on testing
    - Supervised learning accuracy: 69% on training, 69% on testing
    - Similar to the CPC paper's result

## Contrastive predictive coding (CPC)
- [Paper](https://arxiv.org/pdf/1807.03748.pdf)
- Model architecture
	- Input: 20 seconds of raw waveform, dimension: [1,20*sample_rate]
	- Encoder
		- 4 layer 1-d CNN with ReLU activation and batch normalization
		- Encode raw waveform into 256 dimensional feature vector
		- Each feature vector encode 28.125ms of data, stride is 10ms
	- Autoregressive: 1 layer GRU with 128 dimension hidden state
	- Decoder
		- 12-head decoder, each head predict 3*n step of future state (encoded waveform from the next 30, 60, 90, ... ms time window)
		- Implemented as 1-d CNN for higher efficiency (to avoid loop in python)
	- Loss: infoNCEloss loss computed on 150 random sampled RNN hidden states
	- Negative samples: 16, drawn once for each training batch
- Training
	- Dataset: Librispeech train-clean-100, train-other-500, totaling 38.4G training data
	- Batch size: 16
	- Optimizer: Adam with constant or cyclic exponential decaying learning rate
- Features extraction
	- For each speech turn, extract the 20s of waveform before the end_time, run Model 1 and output the last hidden state

## Supervised learning
- Based on Model 1, replace the Multi-head Decoder by a MLP classification head
- Model architecture
	- Input
		- Raw waveform + indicator sequence, dimension: [2,8*sample_rate]
		- Raw waveform contains 8 seconds of audio data, 5 seconds before the end_time and 3 seconds after the end_time. The reason to consider the speech after the end_time and possibly before start_time  is that neighborhood speech may contains information on current turn
		- Indicator sequence: sequence between start_time and end_time are labeled as 1, the rest is labeled as 0
	- Encoder
		- Encoder 1: Same as the Model 1
		- Encoder 2: Encode the indicator sequence. Weights on the indicator sequence dimension is fixed/not optimized, their weights are initialized as 1 and no bias
		- Output from Encoder 1 and 2 are concatenated and passed to Autoregressive
	- Autoregressive: 1 layer GRU with 128 dimension hidden state
	- Decoder: 2 layer 128-dimensional perceptron + 10-class softmax layer
- Training
	- Dataset: Switchboard-Dialog Act dataset, filtered by top 10-labels into 81929 rows
	- Top 10-labels: [('x', 29998), ('sd', 19364), ('b', 10028), ('sv', 7428), ('+', 4919), ('%', 4294), ('aa', 3014), ('ba', 1112), ('qy', 1043), ('ny', 729)]
	- Batch size: 32
	- Optimizer: Adam with constant learning rate
- Features extraction
	- For each speech turn, extract the 8s of waveform + indicator sequence as specified in the Model architecture/Input
	- Run the model and extract the output of the decoder's 2nd layer perceptron
	