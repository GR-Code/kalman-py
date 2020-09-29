let current;
let previous;
let drawing = false;
let T = 0;
let N;
let sample = 0;
let trajectories = [], noisyData = [];
let estimatedPositions;
let kf;
let p = .1;
let q;
let r;


function setup() 
{
    cv = createCanvas(windowWidth*.5, windowHeight*.7);
    cv.parent('sketch-holder');
    cv.mousePressed(canvasPressed);
    cv.mouseReleased(canvasReleased);

    sampling = createSlider(0,100,0,10)
    noiseFactor = createSlider(0,50,10,10);
    qFactor = createSlider(-5,5,1,1);
    rFactor = createSlider(-5,5,3,1);

    
    let sliders = [sampling,noiseFactor,qFactor,rFactor];
    let sliderNames = ["Sample time (0-100ms)","Noise factor (0,50)","Q = 10^(-5,5) * I", "R = 10^(-5,5) * I"]
    sliderGroup = createDiv('');
    sliderGroup.parent('sliders');
    sliders.forEach((e,i)=>{
        e.changed(factorChange);
        e.parent(sliderGroup);
        let label = createSpan(sliderNames[i]);
        label.parent(sliderGroup);
        createP().parent(sliderGroup);
    });

    current = createVector(0,0);
    previous = createVector(0,0);
    estimatedPositions = new Trajectory(color('green'));

    originalLine = createCheckbox('True data',true);
    noisyLine = createCheckbox('Noisy data', false);
    estimateLine = createCheckbox('Kalman Data', true);
    let checkboxes = [originalLine,noisyLine,estimateLine];
    checkboxes.forEach(e=>e.parent('checkboxes'))
    factorChange();
}

function factorChange()
{
    T = sampling.value();
    N = noiseFactor.value();
    q = Math.pow(10,qFactor.value());
    r = Math.pow(10,rFactor.value());
    kf = mousetrackInit(p,q,r);
    trajectories.forEach(e=> e.clear());
    noisyData.forEach(e=> e.clear());
    estimatedPositions.clear();
}

function matrixTable(id,m,precision)
{
    if(m instanceof Matrix)
    {
        var table = document.getElementById(id);
        for(let i=0; i< m.rows; i++)
        {   
            if(table.childNodes.length > m.rows-1)
            {
                table.removeChild(table.childNodes[0]);
            }
            var tr = document.createElement("tr");
            for(let j=0;j<m.cols;j++)
            {
                var td = document.createElement("td");
                td.innerHTML = (precision)?m.data[i][j].toFixed(precision):m.data[i][j];
                tr.appendChild(td);
            }
            table.appendChild(tr);
        }
    }
}

function mousetrackInit(pF, qF, rF)
{
    let state = new Matrix(4,1);                                                    // X
    let estimation = Matrix.eye(4).mul(pF);                                          // P

    let stateModel = Matrix.arr([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]);   // A
    let observationModel = Matrix.arr([[1,0,0,0],[0,1,0,0]]);                       // H

    let processNoiseCovariance = Matrix.arr([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).mul(qF);  // Q
    let measurementNoiseCovariance = Matrix.arr([[1,0],[0,1]]).mul(rF);                          // R

    // Dummy Control Input terms
    let B = new Matrix(state.rows,state.rows);  
    let U = new Matrix(state.rows, state.cols);
    matrixTable("P",estimation,3);
    matrixTable("Q",processNoiseCovariance);
    matrixTable("R",measurementNoiseCovariance);
    return new KalmanFilter(state,estimation,stateModel,observationModel,processNoiseCovariance,measurementNoiseCovariance, B, U);
}

function draw() 
{
    background(200);

    if(millis() > sample && drawing)
    {
        current.x = mouseX;
        current.y = mouseY;
        trajectories[trajectories.length-1].add(current);

        let theta = Math.random()*2*Math.PI;
        let radius = N*Math.sqrt(Math.random());    // Cumulative Distribution function for 2D Noise
        let noisyPoint = p5.Vector.add(current,createVector(radius*cos(theta),radius*sin(theta)));
        noisyData[noisyData.length-1].add(noisyPoint);
        let K = kf.update(Matrix.arr(noisyPoint.array().splice(0,2)));
        matrixTable("K",K,3);
        sample = millis() + T;
    }
    let projection = kf.project().data;
    estimatedPositions.add(createVector(projection[0][0],projection[1][0]));

    trajectories.forEach(e => {e.update(); e.display(originalLine.checked());})
    noisyData.forEach(e => {e.update(); e.display(noisyLine.checked());})
    estimatedPositions.update();
    estimatedPositions.display(estimateLine.checked());
    matrixTable("P",kf.P,3);
}

function canvasPressed()
{
    drawing = true;
    previous.x = mouseX;
    previous.y = mouseY;
    trajectories.push(new Trajectory(color('white')));
    noisyData.push(new Trajectory(color('red')));
}

function canvasReleased()
{
    drawing = false;
}

class Trajectory{
    constructor(pathColor)
    {
        this.points = [];
        this.pathColor = pathColor;
    }

    add(position)
    {
        this.points.push(new Point(position,this.pathColor));
    }

    update()
    {
        this.points.forEach(e=>e.update());
    }

    display(line)
    {
        for(let i=this.points.length-1;i>=0;i--)
        {
            if(this.points[i].lifespan<=0)
            {
                this.points.splice(i,1);
            }
            else
            {
                this.points[i].display((i!==this.points.length - 1 && line) ? this.points[i+1]:[]);   
                // For some reason the ternary expression isn't necessary here, but I'd rather play it safe
            }
        }
    }

    clear()
    {
        this.points = []
    }
}

class Point{
    constructor(coords, color)
    {
        this.position = createVector(coords.x,coords.y);
        this.color = color;
        this.lifespan = 255;
    }

    update(){
        this.lifespan--;
    }

    display(other)
    {
        let pointColor = this.color;
        pointColor.setAlpha(this.lifespan);
        stroke(pointColor);
        pointColor.setAlpha(this.lifespan/2);
        fill(pointColor);
        ellipse(this.position.x,this.position.y, 5,5);

        if(other && other instanceof Point)
        {
            line(this.position.x,this.position.y,other.position.x,other.position.y);
        }
    }
}

class KalmanFilter{
    constructor(X,P,A,H,Q,R,B,U)
    {
        this.X = X;
        this.P = P;
        this.A = A;
        this.H = H;
        this.Q = Q;
        this.R = R;
        this.B = B;
        this.U = U;
    }

    project()
    {
        this.X = Matrix.dot(this.A,this.X).add(Matrix.dot(this.B,this.U));
        this.P = Matrix.dot(this.A,Matrix.dot(this.P,this.A.T)).add(this.Q);
        return this.X;
    }

    update(z)
    {
        let K = Matrix.dot(this.P,Matrix.dot(this.H.T,Matrix.dot(this.H,Matrix.dot(this.P,this.H.T)).add(this.R).inv));
        this.X.add(Matrix.dot(K,z.subtract(Matrix.dot(this.H,this.X))));
        this.P.subtract(Matrix.dot(K,Matrix.dot(this.H,this.P)));
        return K;
    }
}