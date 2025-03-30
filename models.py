import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        initialize a new perceptronmodel instance 
        args:
            dimensions: the number of input dimensions
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        returns the weights of the perceptron
        returns:
            nn.parameter object containing the weights (1 × dimensions)
        """
        return self.w

    def run(self, x_point):
        """
        calculates the dot product of weights and input
        args:
            x_point: input data point (nn.constant) with shape (1 × dimensions)
        returns:
            nn.dotproduct object representing the score
        """
        return nn.DotProduct(self.w, x_point)

    def get_prediction(self, x_point):
        """
        makes a prediction based on the dot product
        args:
            x_point: input data point (nn.constant) with shape (1 × dimensions)
        returns:
            1 if dot product is non-negative -1 otherwise
        """
        # get the scalar value of the dot product
        dot_product = nn.as_scalar(self.run(x_point))
        # return 1 if non-negative -1 if negative
        return 1 if dot_product >= 0 else -1

    def train_model(self, dataset):
        """
        trains the perceptron until 100% accuracy is achieved
        args:
            dataset: dataset object with iterate_once method
        """
        # continue training until no updates are needed
        while True:
            mistakes = False
            # iterate through the entire dataset once
            for x, y in dataset.iterate_once(1):
                # get prediction and true label
                prediction = self.get_prediction(x)
                true_label = nn.as_scalar(y)  # convert label to scalar
                
                # if prediction is wrong update weights
                if prediction != true_label:
                    mistakes = True
                    # update rule: w = w + y * x
                    # multiplier is 1 direction is y * x
                    # since y is either 1 or -1 we can multiply x by y
                    direction = x if true_label > 0 else nn.Constant(-x.data)
                    self.w.update(1, direction)
            
            # if no mistakes were made in this pass we're done
            if not mistakes:
                break


class RegressionModel(object):
    def __init__(self):
        # initialize a two-layer neural network
        # input (1) -> hidden (50) -> output (1)
        self.W1 = nn.Parameter(1, 50)  # weights from input to hidden layer
        self.b1 = nn.Parameter(1, 50)  # bias for hidden layer
        self.W2 = nn.Parameter(50, 1)  # weights from hidden to output layer
        self.b2 = nn.Parameter(1, 1)   # bias for output layer

    def run(self, x):
        """
        runs the model for a batch of examples
        inputs:
            x: a node with shape (batch_size x 1)
        returns:
            a node with shape (batch_size x 1) containing predicted y-values
        """
        # first layer: x * w1 + b1 with relu activation
        hidden = nn.Linear(x, self.W1)      # linear transformation
        hidden = nn.AddBias(hidden, self.b1) # add bias
        hidden = nn.ReLU(hidden)            # non-linearity
        
        # second layer: hidden * w2 + b2 (no relu for regression output)
        output = nn.Linear(hidden, self.W2)  # linear transformation
        output = nn.AddBias(output, self.b2) # add bias
        return output

    def get_loss(self, x, y):
        """
        computes the loss for a batch of examples
        inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1) containing true y-values
        returns: a loss node
        """
        # calculate predictions and return square loss
        predictions = self.run(x)
        return nn.SquareLoss(predictions, y)

    def train_model(self, dataset):
        """
        trains the model until loss < 0.02
        """
        learning_rate = 0.01    # moderate learning rate
        batch_size = 20        # reasonable batch size for regression
        parameters = [self.W1, self.b1, self.W2, self.b2]  # all trainable parameters
        
        while True:
            total_loss = 0
            num_batches = 0
            
            # process one epoch
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(parameters, loss)
                
                # update all parameters using gradient descent
                for param, grad in zip(parameters, gradients):
                    param.update(-learning_rate, grad)
                
                total_loss += nn.as_scalar(loss)
                num_batches += 1
            
            # calculate average loss for this epoch
            avg_loss = total_loss / num_batches
            
            # stop if loss requirement is met
            if avg_loss < 0.02:
                break

class DigitClassificationModel(object):
    def __init__(self):
        # initialize a two-layer neural network
        # input (784) -> hidden (200) -> output (10)
        self.W1 = nn.Parameter(784, 200)  # weights from 784 inputs to 200 hidden units
        self.b1 = nn.Parameter(1, 200)    # bias for hidden layer
        self.W2 = nn.Parameter(200, 10)   # weights from 200 hidden units to 10 outputs
        self.b2 = nn.Parameter(1, 10)     # bias for output layer

    def run(self, x):
        """
        runs the model for a batch of examples
        inputs:
            x: a node with shape (batch_size x 784)
        output:
            a node with shape (batch_size x 10) containing predicted scores
        """
        # first layer: x * w1 + b1 with relu activation
        hidden = nn.Linear(x, self.W1)      # linear transformation
        hidden = nn.AddBias(hidden, self.b1) # add bias
        hidden = nn.ReLU(hidden)            # non-linearity
        
        # second layer: hidden * w2 + b2 (no relu for classification logits)
        output = nn.Linear(hidden, self.W2)  # linear transformation
        output = nn.AddBias(output, self.b2) # add bias
        return output

    def get_loss(self, x, y):
        """
        computes the loss for a batch of examples
        inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        returns: a loss node
        """
        # calculate predictions and return softmax loss
        predictions = self.run(x)
        return nn.SoftmaxLoss(predictions, y)

    def train_model(self, dataset):
        """
        trains the model to achieve >= 97% test accuracy
        """
        learning_rate = 0.1      # higher learning rate for faster convergence
        batch_size = 100         # larger batch size for mnist
        parameters = [self.W1, self.b1, self.W2, self.b2]  # all trainable parameters
        
        while True:
            # process one epoch
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(parameters, loss)
                
                # update all parameters using gradient descent
                for param, grad in zip(parameters, gradients):
                    param.update(-learning_rate, grad)
            
            # check validation accuracy after each epoch
            val_accuracy = dataset.get_validation_accuracy()
            if val_accuracy >= 0.975:  # target 97.5% validation to ensure 97% test
                break