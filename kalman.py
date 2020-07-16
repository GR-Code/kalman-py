import numpy as np 
from numpy.linalg import inv

"""
Theory : http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html
"""
class KalmanFilterDiscrete:
    def __init__(self,X,P,A,H,Z,Q,R,B=np.array([0],np.float32),U=np.array([0],np.float32)):
        """    
        X: State estimates
        P: Estimate covariance
        A: State model
        Z: Measurement
        H: Observation model
        Q: Process noise covariance
        R: Measurement noise covariance
        """
        self.X, self.P, self.A, self.B, self.U, self.H, self.Z, self.Q, self.R =  X,P,A,B,U,H,Z,Q,R
        

    def project(self): # Time update equations, projects estimates
        self.X = self.A @ self.X + self.B @ self.U
        self.P = self.A @ self.P @ (self.A.T) + self.Q
        return self.X

    def update(self,Z): # Measurement update equations
        K = self.P @ (self.H.T) @ inv(self.H @ self.P @ (self.H.T) + self.R)
        self.X += K @ (Z - self.H @ self.X)
        self.P -= K @ self.H @ self.P
        return K