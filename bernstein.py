#The point of this library is to provide an easy to use interface to build networks with infinite input, hidden, or output nodes
#The standard is to use numpy matrices for everything so we can avoid situations where a matrix is expected but a list is given

import scipy as sp
import numpy as np
import math
class Bernstein:
    #a bernstein polynomial is a np.array of shape (n,)
    #the input should be a np.array of shape (n,)
    def __init__(self, matrix):
        if(type(matrix)!=np.ndarray):
            print("Bernstein(matrix) takes a numpy matrix as input")
        self.poly = matrix.astype(float)
        self.degree = len(matrix)-1
        
    #This prints the polynomial in a form somewhat compatible for graphing
    #choose parameter decides wether or not to write out binomial coefficients
    def toString(self, choose=False):
        for i in range(len(self.poly)):
            if choose:
                print("{0:.1f}".format(self.poly[i])+"*("+repr(self.degree-1)+" choose "+repr(i)+")*x^"+repr(i)+"*(1-x)^"+repr(self.degree-1-i),end="+")
            else:
                print("{0:.1f}".format(self.poly[i]*sp.special.binom(self.degree-1,i))+"*x^"+repr(i)+"*(1-x)^"+repr(self.degree-1-i),end="+")
    #This integrates a polynomial from 0 to 1
    #It uses the fact that the integral of any basis of an n degree polynomia is 1/(n+1)    
    def integrate(self):
        sum=0
        for i in range(self.degree+1):
            sum+=self.poly[i]
        sum/=len(self.poly)
        return sum
    
    #This just multiplies two polynomials together
    #Not intuitive. I looked it up
    #Degree out is deg(p1)+deg(p2)  
    @staticmethod    
    def multiply(p1, p2):
        m=p1.degree
        n=p2.degree
        product = np.zeros((m+n+1,))
        for i in range(m+n+1):
            coefficient=0
            for j in range(max(0, i-n),min(m,i)+1):
                coefficient+= sp.special.binom(m,j)*sp.special.binom(n,i-j)\
                /sp.special.binom(m+n,i)*p1.poly[j]*p2.poly[i-j]
                product[i]=coefficient
        return Bernstein(product)
    @staticmethod
    #This method only works for adding Bernsteins of the same degree
    def add(p1, p2):
        return Bernstein(p1.poly+p2.poly)
    
    #This defines a piecewise linear function
    #points'  should be [n,2] numpy matrix and points[x,0] should be increasing
    @staticmethod
    def piecewise(points, x):
        l=0
        r=(np.shape(points))[0]-1
        while l<=r:
            m=math.floor((l+r)/2)
            if points[m,0]<x:
                l=m+1
            elif points[m,0]>x:
                r=m-1
            else:
                return points[m,1]
        if points[m][0]<x:
            return points[m,1]+(points[m+1,1]-points[m,1])\
            /(points[m+1,0]-points[m,0])*(x-points[m,0])
        else:
            return points[m,1]+(points[m-1,1]-points[m,1])\
            /(points[m-1,0]-points[m,0])*(x-points[m,0])
    
    #Returns a bernstein that approximates a piecewise linear function with those points
    #Points' shape should be (n,2)
    #The resulting polynomial models the points squished into [0,1]
    @staticmethod
    def createBernsteinFromPoints(points, degree):
        start=points[0,0]
        end=points[np.shape(points)[0]-1,0]
        squashed=np.empty_like(points,float)
        for row in range(np.shape(points)[0]):
            squashed[row, 0]=(points[row, 0]-start)/(end-start)
            squashed[row, 1]=points[row, 1]
        p=np.zeros((degree+1),)
        for i in range(degree+1):
            p[i]=Bernstein.piecewise(squashed,i/degree)
        return Bernstein(p)
    
    #Returns a bernstein that approximates a function
    #start and end are where you want to approximate it
    #The resulting polynomial models the points squished into [0,1]
    @staticmethod
    def createBernsteinFromFunction(f, degree, start, end):
        p=np.zeros((degree+1),)
        for i in range(degree+1):
            p[i]=f(i/degree*(end-start)+start)
        return Bernstein(p)
    
    @staticmethod
    def createRandomBernstein(degree, randomness, seed, startRange=-1, endRange=1):
        points = np.empty((randomness,1), float)
        np.random.seed(seed)
        for i in range(randomness):
            points[i]=i/(randomness-1)
        points=np.concatenate((points,(endRange-startRange)*np.random.random((randomness,2))+startRange),1)
        return Bernstein.createBernsteinFromPoints(points, degree)
        