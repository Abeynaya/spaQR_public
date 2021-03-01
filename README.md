### spaQR - Sparsified QR for sparse square matrices

(Code for sparse least squares matrices coming up soon)

This repositroy contains the spaQR code as described in the paper "Hierarchical Orthogonal Factorization : sparse square matrices" available at https://arxiv.org/abs/2010.06807 (preprint)

## Packages/Libraries
Necessary:

1. Download Eigen library: http://eigen.tuxfamily.org/index.php?title=Main_Page
2. Download PaToH library: https://www.cc.gatech.edu/~umit/software.html
3. Download, build and install Openblas: https://www.openblas.net/ or Intel MKL https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html

Optional:

1. To experiment with the row reordering strategy based on bipartite matching -- Get access to the mc64 routine from http://www.hsl.rl.ac.uk/catalogue/mc64.html (Not open source, free academic access)

## Build

1. Clone the repository
2. Create an ```obj/``` directory

  ``` mkdir obj ```

3. Copy one of the files in ```Makefile-confs/ ``` folder and save as ```Makefile.conf``` with the location of the necessary libraries in your system. In addition, you can set the following variables in the ```Makefile.conf```
   - ```USE_MKL``` to 1 if you want to use Intel MKL or 0 to use Openblas. Default is 0.
   - ```USE_METIS``` to 1 if you want to use Metis to partition the matrix or 0 to use PaToH to partition. Default is 0. (For some reason, I couldn't build the code including both Metis and PaToH)
   - ```HSL_AVAIL``` to 1 if you have HSL MC64 routine. Default is 0.
You can also set these options at compile time.
4. Compile it with ``` make ``` or ```make USE_MKL=1 HSL_AVAIL=1```

## Run

Run it as
```./spaQR -m mats/advdiff/2d/advdiff_2_128_1_p1_1000.mm  -n 128 -d 2 -t 1e-3 --scale 1```

You can get all the arguments you can pass in to the function by typing ```./spaQR --help```

## Example

```
DNa1c037c:spaQR_public abey$ ./spaQR -m mats/advdiff/2d/advdiff_2_128_1_p1_1000.mm  -n 128 -d 2 -t 1e-3 --scale 1
Matrix mats/advdiff/2d/advdiff_2_128_1_p1_1000.mm with 16384 rows,  16384 columns loaded
--Levels not provided
 Levels set to ceil(log2(ncols/64)) =  8
Pre-process time: 0.00138402
Tensor coordinate matrix of size 128^2 built
Time to partition: 0.0263851
nnzA: 81408
Time to assemble: 0.022332
lvl: 0    update fill-in: 0.001  elmn: 1.466  scale: 0.137  sprsfy: 0.222  merge: 0.068  s_top: 254, 254
lvl: 1    update fill-in: 0.016  elmn: 3.975  scale: 0.114  sprsfy: 0.363  merge: 0.039  s_top: 216, 216
lvl: 2    update fill-in: 0.006  elmn: 0.992  scale: 0.045  sprsfy: 0.127  merge: 0.008  s_top: 163, 163
lvl: 3    update fill-in: 0.001  elmn: 0.171  scale: 0.014  sprsfy: 0.032  merge: 0.002  s_top: 132, 132
lvl: 4    update fill-in: 0.000  elmn: 0.018  scale: 0.003  sprsfy: 0.006  merge: 0.000  s_top: 99, 99
lvl: 5    update fill-in: 0.000  elmn: 0.002  scale: 0.001  sprsfy: 0.001  merge: 0.000  s_top: 63, 63
lvl: 6    update fill-in: 0.000  elmn: 0.000  scale: 0.000  sprsfy: 0.000  merge: 0.000  s_top: 63, 63
lvl: 7    update fill-in: 0.000  elmn: 0.000  scale: 0.000  sprsfy: 0.000  merge: 0.000  s_top: 63, 63
Tolerance set: 1.000e-03
Time to factorize:  7.828e+00
Size of top separator: 63
nnzA: 81408 nnzR: 13666957
nnzH: 24920434 nnzQ: 201985
<<<<tsolv=1.060e-01
One-time solve (Random b):
<<<<|(Ax-b)|/|b| : 1.234e-03
initial residual 1.234e-03
GMRES converged!
GMRES: #iterations: 5, residual |Ax-b|/|b|: 3.008e-13
  GMRES: 6.615e-01 s.
<<<<GMRES=5
```
## Note

1. You can generate more sparse matrices corresponding to Poisson equation, (High constrast) Advection diffusion using the open source codes in https://github.com/leopoldcambier/MatrixGen
2. The MC64 routine (bipartite matching) rountine from HSL is needed for problems with high condition number, to ensure stability
