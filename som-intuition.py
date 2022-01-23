# Raja SP. To illustrate the geometrical intuition behind the 
# Kohonen Self Organizing Maps.
import numpy as np
import matplotlib.pyplot as plt

def som(inputs, weights, learningRate, iterations, showTrace=True):
    numDataSets = len(inputs)
    features = len(inputs[0])

    #Plot the inputs
    for i in range(numDataSets):
        plt.text(inputs[i,0], inputs[i,1] + 0.04, 'input-' + str(i+1),horizontalalignment='center')
        plt.scatter(inputs[i,0], inputs[i,1])

    plt.text(weights[0, 0], weights[1, 0] + 0.04, 'start', horizontalalignment='center')
    plt.scatter(weights[0, 0], weights[1, 0], color="blue", s=100, alpha=0.5, linewidths=5)

    # Iterate and plot the final state
    for itr in range(iterations):
        for i in range(numDataSets):
            # adjustment function: weight(t+1) = weight(t) + learningRate( input - weight(t) )
            # simplified to: weight(t+1) = weight(t) * (1 - learningRate) + ( learningRate * input )
            weights[:, 0] = weights[:, 0] * (1-learningRate) + learningRate * inputs[i, 0:features]
        if showTrace or (not showTrace and itr == (iterations-1)):
            plt.text(weights[0, 0], weights[1, 0] + 0.04, 'itr - ' + str(itr + 1), horizontalalignment='center')
            plt.scatter(weights[0, 0], weights[1, 0], color="blue", s=100, alpha=0.5, linewidths=1) 
    plt.show()

def main():
    # learning rate = 0.2 and number of iterations = 5.
    som(np.array([[1.0, 1.0]]), np.array([[3.0], [3.0]]), 0.2, 5)
    
    # with 3 inputs and 100 iterations
    #som(np.array([[1.0,1.0], [2.0, 2.0], [5.0, 5.0] ]), np.array([[10.0], [10.0]]), 0.2, 100, False)

if __name__ == "__main__":
    main()