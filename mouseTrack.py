import numpy as np
import cv2
from kalman import KalmanFilterDiscrete
from collections import deque

WINDOW_NAME = 'Mouse Tracking with Kalman Filter'
BUFFER = 1024
COLOR_RED = (0,0,255)
COLOR_BLACK = (0,0,0)
COLOR_CYAN = (255,255,0)
COLOR_GREEN = (0,255,0)
COLOR_WHITE = (255,255,255)
pts = deque(maxlen=BUFFER)

N = 1 # Noise factor
trueDraw = 1 # Boolean to show true mouse position
D = 0.95 # Drag

# Tweak these factors to adjust covariance matrices
pFactor = 10
qFactor = 100
rFactor = 10000

# |v| Needs refactoring
class Points:
    def __init__(self, x, y, color=COLOR_RED, fadeTo = COLOR_BLACK, steps=BUFFER/10):
        self.coords, self.color, self.fadeTo = (x,y), color, fadeTo
        self.step = 1/steps
        self.currentColor = color
        self.f = 1

    def __call__(self):
        newColor = list()
        self.f -= self.step
        for (a,b) in zip(self.color, self.fadeTo):
            newColor.append((self.f*a + (1-self.f)*b)//2)
        self.currentColor = newColor
        return newColor
# |^| Needs refactoring


def addPoint(event, x,y, flags, param):
    if trueDraw:
        truePoint = Points(x,y,color = COLOR_GREEN)
        pts.appendleft(truePoint)

    theta = np.random.random_sample()*2*np.pi
    r = N*np.sqrt(np.random.random_sample())    # Cumulative distribution function for noise
    _X,_Y = (np.int32(x + r*np.cos(theta)), np.int32(y + r*np.sin(theta)))
    noisyPoint = Points(_X,_Y,color = COLOR_RED)
    pts.appendleft(noisyPoint)

    currentZ = np.array([[np.float32(_X)],[np.float32(_Y)]])
    output = kf.update(currentZ)
    return

def noise(x):
    global N
    N = x
    pts.clear()

def trueShow(x):
    global trueDraw
    trueDraw = x
    pts.clear()

def mouseTrackInit(p=pFactor, q=qFactor, r=rFactor):
    state = np.zeros((4,1), np.float32) # x, y, delta_x, delta_y
    estimateCovariance = np.eye(state.shape[0])*p

    stateEstimate = np.array([[1,0,1,0],[0,1,0,1],[0,0,D,0],[0,0,0,D]],np.float32)
    observation = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    measurement = np.zeros((2,1),np.float32)

    processNoiseCovariance = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)*q
    measurementNoiseCovariance = np.array([[1,0],[0,1]], np.float32)*r

    return KalmanFilterDiscrete(X=state,P=estimateCovariance,A=stateEstimate,H=observation,Z=measurement,Q=processNoiseCovariance,R=measurementNoiseCovariance)

if __name__ == "__main__":
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE|cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME,addPoint)
    cv2.createTrackbar("Noise factor", WINDOW_NAME, N, 50, noise)
    cv2.createTrackbar("True measurements\n0 : OFF \n1 : ON", WINDOW_NAME, trueDraw, 1, trueShow)

    kf = mouseTrackInit()

    while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
        img = np.zeros((800,800,3),np.uint8)        
        cv2.putText(img, f"(Press R to reset Kalman filter, Q/Esc to quit",(30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, COLOR_WHITE)
        
        cv2.putText(img, f"RED = Noisy Measurements",(30, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, COLOR_RED)
        cv2.putText(img, f"CYAN = Predictions",(30, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, COLOR_CYAN)
        if trueDraw:
            cv2.putText(img, f"GREEN = True Mouse Trail",(30, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, COLOR_GREEN)

        predicted = kf.project()
        prediction = Points(x=predicted[0],y=predicted[1], color=COLOR_CYAN)
        pts.appendleft(prediction)
        for i,point in enumerate(pts):
            cv2.circle(img, point.coords,radius=2, color = point(),thickness=-1)

        cv2.imshow(WINDOW_NAME,img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            pts.clear()
            kf = mouseTrackInit()
        elif key in (27, ord('q')):
            break

    cv2.destroyAllWindows()