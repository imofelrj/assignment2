# Report of Assignment 2: Sentimental Analysis
Qiuzhen College, Tsinghua University, imofelrj, li-rj22@mails.tsinghua.edu.cn, May 6th, 2025

## Abstract
This report presents the results of a sentiment analysis project using RNNs, CNNs, and MLPs, with the framework PyTorch. The project aims to classify text data into positive or negative sentiments.

The codes are available at [GitHub](https://github.com/imofelrj/assignment2), including the data preprocessing, model training, and evaluation. Make sure ```pytorch, scikit-learn``` are installed.

## Structures of the Models
The class ```TextDataset```, a subclass of ```torch.utils.data.Dataset```, is used to load the data. It returns the text and label of each sample, ```torch.tensor(input_ids), torch.tensor(label)```. And all these three models have an embedding layer, which is initialized with the pre-trained word vectors. The embedding layer is not trainable, so the model will not update the word vectors during training. The models are trained using the Adam optimizer and the binary cross-entropy loss function.

### CNN
There are in total 3 convolutional layers, with kernel sizes of 3, 4, and 5. The output of each convolutional layer is passed through a ReLU activation function and a max pooling layer. The output of the max pooling layer is then concatenated and passed through a fully connected layer. A dropout layer is added to prevent overfitting.

### RNN
LSTM is recurrent, which means it takes the output of the previous time step as input for the current time step. We take the last output of the LSTM as the output of the model. The output is passed through a fully connected layer and a dropout layer. The hidden size is set to 128, and the dropout rate is set to 0.5.

### MLP
The embedding layer maps the sentence to a tensor of size ```[1,10,128]```, and take the mean of the tensor along the first dimension. The output is passed through a fully connected layer with a ReLU activation function and a dropout layer. The output of the dropout layer is then passed through another fully connected layer. The hidden size is set to 128, and the dropout rate is set to 0.5.

## Results
The models are trained for 5 epochs, with a batch size of 32. The learning rate is set to 0.001. The training and validation loss are recorded for each epoch. The accuracy of the models is evaluated on the test set.
The results are shown in the following table:
| Model | Test Accuracy | Test Precision | Test Recall | Test F1 Score |
|-------|---------------|----------------|-------------|---------------|
| CNN   | 0.7507         | 0.7509          | 0.7507        | 0.7507          |
| RNN   | 0.7886          | 0.8019          | 0.7886       | 0.7867         |
| MLP   | 0.7615        | 0.7616           | 0.7615       | 0.7615       |

If we increase the number of epochs to 10, the accuracy of the models will increase. That's because the models are not overfitting. The training and validation loss are decreasing, and the accuracy is increasing. The models are able to learn the patterns in the data and generalize well to the test set.

## Difference between CNN, RNN, and MLP

CNN model focuses on the local features of the text (in our case, we use the kernel size of 3, 4, and 5), while RNN model focuses on the global features of the text. It is able to 'memorize' information from the previous time. MLP model is a simple feedforward neural network, which is not able to capture the sequential information in the text. Therefore, RNN model performs better than CNN and MLP models.

## Questions
1. The training stops when the validation loss does not decrease for 3 epochs. 
   If we fix the number of epochs, the code will be much simpler, however, the model may overfit the training data.
   Using the validation loss to early stop the training can prevent overfitting. However, we need to focus on the performance of the model on the validation set either, thus the training time may be longer.

2. In PyTorch, the default initialization of the weights, in 
```self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)```
is uniform distribution. The weights are initialized with a uniform distribution in the range of ```-1/sqrt(embed_dim)``` to ```1/sqrt(embed_dim)```. 
The initialization of ```self.fc = nn.Linear(~,~), nn.Conv2d(1, 100, (k, embed_dim))``` is He initialization.
The initialization of ```self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1, batch_first=True)``` is orthogonal initialization.

| Initialization Method     | Applicable Layer Types           | Recommended Activation | Principle / Reason                                         |
|---------------------------|----------------------------------|------------------------|------------------------------------------------------------|
| **Xavier (Glorot)**       | `nn.Linear`, `nn.Embedding`      | `tanh`, `sigmoid`      | Keeps variance consistent across layers to avoid vanishing/exploding gradients |
| **Kaiming (He)**          | `nn.Conv2d`, `nn.Linear`         | `ReLU`, `LeakyReLU`    | Designed for ReLU to compensate for zeroed-out activations |
| **Normal Distribution**   | `nn.Embedding`                   | Any                    | Controls initial vector scale, good for simple or pretrained embeddings |
| **Orthogonal**            | `nn.RNN`, `nn.LSTM`, `nn.GRU`    | Any                    | Helps maintain gradient flow through time steps            |
| **Constant Initialization** | Bias terms of any layer        | Any                    | Usually set to 0 for faster convergence                    |

3. While training the model, we focus on its performance on the validation set. This can effectively prevent overfitting on the training set.
4. Based on their principles, 
   CNN model is parallelizable, thus much faster in training. It is good at detecting key phrases regardless of their position in the text. However, it is not able to capture the sequential information in the text. For example, the phrase "不 喜欢" is negative, while "喜欢" is positive. CNN model may misclassify the phrase "不 喜欢" as positive. RNN model is able to capture the sequential information in the text, thus it performs better than CNN model.
   RNN model is sequential, so it remembers word order and sequential relationships.Thus when the text is long, and dependent on the order of words, RNN model has a better performance than CNN model. However, it can be biased by the first few words.
   MLP model is much simpler than CNN and RNN models, and is faster to train. But it ignores the sequential information in the text, such as the order of words.

## 心得体会
通过这次实验， 我对不同神经网络模型有了更加深入的了解，尤其是在模型的初始化和训练方面。模型训练、调参、验证评估等完整流程对我的coding能力也是一次挑战。不同模型之间的差异让我意识到，选择合适的模型和参数对于任务的成功至关重要。CNN、RNN和MLP各有优缺点，适用于不同类型的数据和任务。通过对比它们在情感分析任务中的表现，我更加理解了深度学习在自然语言处理中的应用。