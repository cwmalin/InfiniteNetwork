#A library to define basic elements of an infinite neural network
#An infinite neural network can be thought of as taking the limit of a neural network as its nodes approach infinity
#However we add the constraint that nearby nodes must have close values
#This means we can also think of an infinite neural network as a finite neural network with each of the nodes taking on the value of a continuous function instead of a number
import numpy as np
import bernstein as b
#The building block of a network
class Node:
    #dimensions is the number of dimensions in the function your node outputs
    #0 dimensions gives a finite node
    #currently the functions it outputs is just the sum of bernstein polynomials of different variables
    def __init__(self, dimensions):
        self.dimensions=dimensions
    
    #matrix is a matrix of bernstein polynomials. It has shape (n,) or () if its dimension is 0
    def setValue(self, vector):
        self.value = matrix
#An aggregation of nodes
class Layer:
    #dimensionVector is a vector (n,) that represents the dimension of every node
    def __init__(self, dimensionMatrix, trainingSize):
        self.nodeMatrix = np.empty((trainingSize, len(dimensionMatrix), object))
        self.size = len(dimensionMatrix)
        self.nodeDimensions = dimensionMatrix
        for row in range(trainingSize):
            for col in range(len(dimensionMatrix)):
                nodeMatrix[row, col] = Node(dimensionMatrix[row, col])
        
    #def applySynapse
        
#a Synapse object defines the whole set of synapses between two layers
class Synapse:        
    #inputDimensionVector is a vector (n,) that represents the dimension of each node
    def __init__(self, inputDimensionVector, outputDimensionVector):
        self.inputDimensionVector = inputDimensionVector
        self.outputDimensionVector = outputDimensionVector        
        self.weightMatrix = np.empty((len(inputDimensionVector), len(outputDimensionVector)), object)
    
    def initializeWeights(degree, seed, randomness, startRange-1, endRange=1):
        for row in range(len(self.weightMatrix)):
            for col in range(len(self.weightMatrix[0])):
                bVector = np.empty((self.inputDimensionVector[row],),object)                
                for j in range():                
                    bVector[j]=b.Bernstein.createRandomBernstein(degree, randomness, seed, startRange, endRange)
                
    
            
        