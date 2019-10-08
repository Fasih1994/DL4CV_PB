import  numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        #initialize weight and store network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha
        self.losses = []


        # strat looping from first layer but stop when reach 2nd last layer
        for i in np.arange(0, len(layers)-2):
            # randomly initialize a weight matrix connecting the
            # number of nodes in each respective layer together,
            # adding an extra node for the bias
            w = np.random.randn(layers[i]+1, layers[i+1]+1)
            self.W.append(w / np.sqrt(layers[i]))

        # the last two layer are the special case where
        # the second last layer needs a bia but the last layer does not
        w = np.random.randn(layers[-2]+1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network
        # architecture
        return "Neural Network = {}".format(
            '-'.join(str(l) for l in self.layers))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


    #sigmoid derivative
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # insert the column of ones
        #this way our biases are trainable parameters
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train
            # our network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            if epoch == 0 or (epoch+1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                self.losses.append(loss)
                print("[INFO] epoch = {}, loss = {:.7f}".format(epoch+1, loss))
    def get_losses(self):
        return self.losses
    def fit_partial(self, x, y):
        # construct our list of output activations for each layer
        # as our data point flows through the network; the first
        # activation is a special case -- it’s just the input
        # feature vector itself
        A = [np.atleast_2d(x)]

        #FEEDFORWARD
        #loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by
            # taking the dot product between the activation and
            # the weight matrix -- this is called the "net input"
            # to the current layer
            net = A[layer].dot(self.W[layer])

            # computing the "net output" is simply applying our
            # nonlinear activation function to the net input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of
            # activations
            A.append(out)


        #BACKWARDPASS

        error = A[-1] - y
        """
        from here, we need to apply the chain rule and build our
        list of deltas ‘D‘; the first entry in the deltas is
        simply the error of the output layer times the derivative
        of our activation function for the output value
        """
        D = [error * self.sigmoid_derivative(A[-1])]
        # once you understand the chain rule it becomes super easy
        # to implement with a ‘for‘ loop -- simply loop over the
        # layers in reverse order (ignoring the last two since we
        # already have taken them into account)
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta
            # of the *previous layer* dotted with the weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivative of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_derivative(A[layer])
            D.append(delta)

        # since we loop over deltas in reverse order
        # we need to reverse the deltas
        D = D[::-1]


        #WEIGHTUPDATEPHASE

        for layer in np.arange(0, len(self.W)):

            # update our weights by taking the dot product of the layer
            # activations with their respective deltas, then multiplying
            # this value by some small learning rate and adding to our
            # weight matrix -- this is where the actual "learning" takes
            # place
            self.W[layer] += -(self.alpha * A[layer].T.dot(D[layer]))

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p
    def calculate_loss(self, X, target):

        target = np.atleast_2d(target)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - target)**2)
        return loss
