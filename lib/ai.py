import numpy as np
import math

def sig(x):
    return 1/(1 + math.exp(-x))

def sigV(x):
    return np.vectorize(sig)(x)

def dsig(x):
    f = sig(x)
    return f * (1 - f)

def dsigV(x):
    return np.vectorize(dsig)(x)

# def getValue(input,w,b):
#     for i in range(0, len(w)):
#         input = w[i] * input
#         input = input.T
#         for col in input:
#             col += b[i]
#         input = input.T
#         input = sigV(input)

#     return input

# def getValues(input,w,b):
#     zs = [input]
#     az = [input]

#     for i in range(0, len(w)):
#         z = w[i] * az[len(az)-1]
#         z = z.T
#         for col in z:
#             col += b[i]
#         z = z.T

#         a = sigV(z)
        
#         zs.append(z)
#         az.append(a)

#     return (az,zs)

# def meanV(x):
#     return np.vectorize(np.mean)(x)

# def train(input, output,w,b):
#     (az, zs) = getValues(input,w,b)

#     dws = []
#     dbs = []


#     numData = input.shape[1]

#     res = az[len(az)-1]

#     error = np.multiply(dsigV(zs[len(zs)-1]), (res - output))

#     dbs.append(error)
 
#     a = az[len(az)-2]
    
#     dw = np.zeros((w[len(w)-1].shape[0], w[len(w)-1].shape[1], numData))

#     for j in range(0, dw.shape[0]):
#         for k in range(0, dw.shape[1]):
#             dw[j][k] = np.multiply(a[k], error[j])

#     dws.append(dw)


#     for i in range(len(w)-2, -1, -1):     
#         error = np.multiply(w[i+1].T * error , dsigV(zs[i+1]))        
        
#         a = az[i]

#         dw = np.zeros((w[i].shape[0], w[i].shape[1], numData))

#         for j in range(dw.shape[0]):
#             for k in range(dw.shape[1]):
#                 dw[j][k] = np.multiply(a[k], error[j])


#         dws.append(dw)
#         dbs.append(error)

#     dws.reverse()
#     dbs.reverse()


#     for i in range(len(dws)):
#         w[i] -= np.mean(dws[i], axis=2)


#     for i in range(len(dbs)):
#         for j in range(dbs[i].shape[0]):
#                 b[i][j] -= np.mean(dbs[i][j])

        







class NeuralNet:

    def __init__(self, layerSizes):
        self.w = []
        self.b = []
        for i in range(0, len(layerSizes)-1):
            self.w.append(2*np.random.rand(layerSizes[i+1], layerSizes[i])-1)
            self.b.append(2*np.random.rand(layerSizes[i+1])-1)




    def getValue(self, input):
        w = self.w
        b = self.b
        for i in range(0, len(w)):
            input = np.dot(w[i], input)
            input = input.T
            for col in input:
                col += b[i]
            input = input.T
            input = sigV(input)

        return input

    def getValues(self,input):
        w = self.w
        b = self.b
        zs = [input]
        az = [input]

        for i in range(0, len(w)):
            z = np.dot(w[i], az[len(az)-1])
            z = z.T
            for col in z:
                col += b[i]
            z = z.T

            a = sigV(z)
            
            zs.append(z)
            az.append(a)

        return (az,zs)

    def train(self, input, output):
        w = self.w
        b = self.b
        (az, zs) = self.getValues(input)

        dws = []
        dbs = []


        numData = input.shape[1]

        res = az[len(az)-1]

        error = np.multiply(dsigV(zs[len(zs)-1]), (res - output))

        dbs.append(error)
    
        a = az[len(az)-2]
        
        dw = np.zeros((w[len(w)-1].shape[0], w[len(w)-1].shape[1], numData))

        for j in range(0, dw.shape[0]):
            for k in range(0, dw.shape[1]):
                dw[j][k] = np.multiply(a[k], error[j])

        dws.append(dw)


        for i in range(len(w)-2, -1, -1):     
            error = np.multiply(np.dot(w[i+1].T, error), dsigV(zs[i+1]))        
            
            a = az[i]

            dw = np.zeros((w[i].shape[0], w[i].shape[1], numData))

            for j in range(dw.shape[0]):
                for k in range(dw.shape[1]):
                    dw[j][k] = np.multiply(a[k], error[j])


            dws.append(dw)
            dbs.append(error)

        dws.reverse()
        dbs.reverse()


        for i in range(len(dws)):
            w[i] -= np.mean(dws[i], axis=2)


        for i in range(len(dbs)):
            for j in range(dbs[i].shape[0]):
                    b[i][j] -= np.mean(dbs[i][j])


    def TrainEpoch(self, input, output, iterations):
        for i in range(iterations):
            self.train(input, output)
            r = self.getValue(input)
            print(np.sum(np.mean(r-output)**2))

        # print(output)
        # print(self.getValue(input))
    

    def oneHotHighestValue(self, input):
        results = self.getValue(input).T
        return np.argmax(results, axis=1)

    def accuracyOneHotHighestValue(self, input, output):
        results = self.oneHotHighestValue(input)
        numCorrect = 0
        for (r,o) in zip(results, output):
            if(r==o):
                numCorrect+=1
        
        return numCorrect/len(output)


             

    def save(self, name="ai"):
        np.save(name+".npy", (self.w,self.b))
    
    def load(self, name="ai"):
        (self.w, self.b) = np.load(name+".npy", allow_pickle=True)


def main():
    input = np.matrix([
        [1, 0],
        [0.2, 0.5],
        [1, 0.2],
        [0.3, 0.7]
    ])
    input = np.transpose(input)
 

    output = np.matrix([
        [0.2, 1.0], 
        [0.9, 0.5],
        [0.2, 0.9],
        [0.8, 0.4]
    ])
    output = output.T

    layerSizes = [2, 7, 7, 2]

    neural = NeuralNet(layerSizes)

    neural.TrainEpoch(input, output)

print("Hello World")