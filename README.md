### spaQR - Sparsified QR

This repositroy contains the spaQR code as described in the papers

1. "Hierarchical Orthogonal Factorization : sparse square matrices" available at https://arxiv.org/abs/2010.06807 (preprint)
2. "Hierarchical Orthogonal Factorization : sparse least squares problems" available at https://arxiv.org/abs/2102.09878 (preprint)

## Disclaimer

This is a very researchy code and can be optimized more. I'll try to update the repository as improvements are made. That being said, it still gives reasonable performance on test problems. I'll be happy to help or discuss more if you are interested in using the code for any of your applications. 

## Packages/Libraries
Necessary:

1. Download Eigen library: http://eigen.tuxfamily.org/index.php?title=Main_Page
2. Download PaToH library: https://www.cc.gatech.edu/~umit/software.html
3. Download, build and install Openblas: https://www.openblas.net/ or Intel MKL https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html

Optional:

1. To experiment with the row reordering strategy based on bipartite matching -- Get access to the mc64 routine from http://www.hsl.rl.ac.uk/catalogue/mc64.html (Not open source, free academic access) -- Necessary for rectangular matrices

## Build

1. Clone the repository
2. Create an ```obj/``` directory

  ``` mkdir obj ```

3. Copy one of the files in ```Makefile-confs/ ``` folder and save as ```Makefile.conf``` with the location of the necessary libraries in your system. In addition, you can set the following variables in the ```Makefile.conf```
   - ```USE_MKL``` to 1 if you want to use Intel MKL or 0 to use Openblas. Default is 0.
   - ```USE_METIS``` to 1 if you want to use Metis to partition the matrix or 0 to use PaToH to partition. Default is 0. (For some reason, I couldn't build the code including both Metis and PaToH)
   - ```HSL_AVAIL``` to 1 if you have HSL MC64 routine. Default is 0
You can also set these options at compile time
4. Compile it with ``` make ``` or ```make USE_MKL=1 HSL_AVAIL=1```

## Run

Run it as
```./spaQR -m mats/advdiff/2d/advdiff_2_128_1_p1_1000.mm  -n 128 -d 2 -t 1e-3 --scale 1```

```./spaQR -m mats/invpoi/2d/invpoi_2d_2_256.mm -t 1e-2 --skip 3  -n 256 -d 2 --hsl 1```

You can get all the arguments you can pass in to the function by typing ```./spaQR --help```

## Example

```
DNa1c037c:spaQR_public abey$ ./spaQR -m mats/advdiff/2d/advdiff_2_128_1_p1_1000.mm  -n 128 -d 2 -t 1e-3 --scale 1
Matrix mats/advdiff/2d/advdiff_2_128_1_p1_1000.mm with 16384 rows,  16384 columns loaded
--Levels not provided
 Levels set to ceil(log2(ncols/64)) =  8
Pre-process time: 0.00171
Tensor coordinate matrix of size 128^2 built
Using GeometricPartition... 
May need HSL bipartite matching routine to work correctly.
Proceeding by matching row i to col i ...
May need HSL bipartite matching routine to work correctly.
Proceeding by matching row i to col i ...
Time to partition: 0.030164
Time to assemble: 0.018473
Aspect ratio of top separator: 1
lvl: 0    update fill-in: 0.001  elmn: 0.695  shift: 0.000  scale: 0.044  sprsfy: 0.035  merge: 0.006  size top_sep: 254, 254   a.r top_sep: 1.000
lvl: 1    update fill-in: 0.001  elmn: 0.018  shift: 0.000  scale: 0.038  sprsfy: 0.034  merge: 0.003  size top_sep: 216, 216   a.r top_sep: 1.000
lvl: 2    update fill-in: 0.000  elmn: 0.011  shift: 0.000  scale: 0.016  sprsfy: 0.046  merge: 0.001  size top_sep: 163, 163   a.r top_sep: 1.000
lvl: 3    update fill-in: 0.000  elmn: 0.009  shift: 0.000  scale: 0.008  sprsfy: 0.014  merge: 0.000  size top_sep: 132, 132   a.r top_sep: 1.000
lvl: 4    update fill-in: 0.000  elmn: 0.003  shift: 0.000  scale: 0.003  sprsfy: 0.005  merge: 0.000  size top_sep: 99, 99   a.r top_sep: 1.000
lvl: 5    update fill-in: 0.000  elmn: 0.001  shift: 0.000  scale: 0.001  sprsfy: 0.001  merge: 0.000  size top_sep: 63, 63   a.r top_sep: 1.000
lvl: 6    update fill-in: 0.000  elmn: 0.000  shift: 0.000  scale: 0.000  sprsfy: 0.000  merge: 0.000  size top_sep: 63, 63   a.r top_sep: 1.000
lvl: 7    update fill-in: 0.000  elmn: 0.000  shift: 0.000  scale: 0.000  sprsfy: 0.000  merge: 0.000  size top_sep: 63, 63   a.r top_sep: 1.000
Tolerance set: 1.000e-03
Time to factorize:  9.976e-01
Size of top separator: 63
nnzA: 81408 nnzR: 1594464
nnzH: 2697960 nnzQ: 201985
<<<<tsolv=3.582e-02
One-time solve (Random b):
<<<<|(Ax-b)|/|b| : 1.338e-03
initial residual 1.338e-03
GMRES converged!
GMRES: #iterations: 5, residual |Ax-b|/|b|: 3.519e-13
  GMRES: 1.412e-01 s.
<<<<GMRES=5
```

## Benchmarks

You can reproduce the benchmarks in the spaQR papers by running the scripts available in ```benchmarks/``` folder


## Note

1. You can generate more sparse matrices corresponding to Poisson equation, (High constrast) Advection diffusion using the open source codes in https://github.com/leopoldcambier/MatrixGen
2. You can generate more least squares matrices corresponding to the inverse Poisson problem using the MATLAB codes available in ```matrix_gen/```
3. The MC64 routine (bipartite matching) rountine from HSL is needed for problems with high condition number and for rectangular matrices 
