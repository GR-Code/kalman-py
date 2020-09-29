class Matrix{
    constructor(rows,cols)
    {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
    }

    map(f)
    {
        for(let i=0;i<this.rows;i++)
        {
            for(let j=0;j<this.cols;j++)
            {
                let e = this.data[i][j];
                this.data[i][j] = f(e,i,j);
            }
        }
        return this;
    }

    get copy()
    {
        return new Matrix(this.rows,this.cols).map((e,i,j)=>this.data[i][j])
    }

    get str()
    {
        console.table(this.data);
        return this;
    }

    get T()
    {
        return new Matrix(this.cols,this.rows).map((e,i,j)=> this.data[j][i]);
    }

    randInit()
    {
        return this.map(() => Math.random()*2-1);
    }

    static arr(a)
    {
        if(a[0][0] === undefined)   // Vector instead of 2D Array
            return new Matrix(a.length,1).map((e,i) => a[i]);
        else
            return new Matrix(a.length,a[0].length).map((e,i,j) => a[i][j]);
    }

    static eye(n)
    {
        return new Matrix(n,n).map((e,i,j) => (i===j)?1:0);
    }

    static dot(a,b)
    {
        if (a.cols !== b.rows)
        {
            console.log('Can\'t multiply this!');
            return;
        }      
        return new Matrix(a.rows,b.cols).map((e,i,j) => {
            let sum = 0;
            for(let k=0;k<a.cols;k++)
                sum += a.data[i][k]*b.data[k][j];   //Dot Product, or Matrix Multiplication

            return sum;
            })
    }

    // Destructive edits, create copies beforehand

    mul(m)
    {
        if (m instanceof Matrix)
        {
            if (this.rows !== m.rows || this.cols !== m.cols)
            {
                console.log('Can\'t do a Hadamard product on this!');
                return;
            }
            return this.map((e,i,j) => m.data[i][j]*e); //Hadamard Product
        }
        else
        {
            return this.map(e => m*e);
        }
    }

    add(m)
    {
        if (m instanceof Matrix)
        {
            if (this.rows !== m.rows || this.cols !== m.cols)
            {
                console.log('Can\'t do addition on this!');
                return;
            }
            return this.map((e,i,j) => m.data[i][j] + e);
        }
        else
        {
          return this.map(e => m + e);
        }
    }

    subtract(m)
    {
        if (m instanceof Matrix)
        {
            if (this.rows !== m.rows || this.cols !== m.cols)
            {
                console.log('Can\'t do subtraction on this!');
                return;
            }
            return this.map((e,i,j) => e - m.data[i][j]);
        }
        else
        {
            return this.map(e => e - m);
        }
    }

    get inv()   // Gaussian Elimination
    {
        if(this.rows !== this.cols)
            return;
        
        let n = this.rows;
        let M = this.copy;
        let I = Matrix.eye(n);

        //Check if diagonal has zeros
        for(let i=0;i<n;i++)
        {
            if(M.data[i][i] === 0)
            {
                //Swap rows with one after current row that has a non-zero element in the same column
                for(let k=i;k<n;k++)
                {
                    if(M.data[k][i] !== 0)
                    {
                        for(let j=0;j<n;j++)
                        {
                            let temp = M.data[i][j]
                            M.data[i][j] = M.data[k][j];
                            M.data[k][j] = temp;
                            temp = I.data[i][j];
                            I.data[i][j] = I.data[k][j];
                            I.data[k][j] = temp;
                        }
                    }
                    break;
                }
                if(M.data[i][i] === 0)
                return;         // Not invertible
            }
            let diag = M.data[i][i];
            for(let j=0;j<n;j++)
            {   
                M.data[i][j]/=diag;
                I.data[i][j]/=diag;
            }

            //Echelon form
            for(let k=0;k<n;k++)
            {
                if (k!==i)
                {
                    let temp = M.data[k][i];
                    for(let j=0;j<n;j++)
                    {
                        M.data[k][j] -= temp*M.data[i][j];
                        I.data[k][j] -= temp*I.data[i][j];
                    }
                }
            }
            
        }
        return I;
    }

}