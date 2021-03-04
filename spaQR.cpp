#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#include "mmio.hpp"
#include "cxxopts.hpp"
#include "tree.h"
#include "partition.h"
#include "util.h"
#include "is.h"


using namespace Eigen;
using namespace std;

typedef SparseMatrix<double, 0, int> SpMat; 

int main(int argc, char* argv[]){
  
     cxxopts::Options options("spaQR", "Sparsified QR for general sparse matrices");
     options.add_options()
        ("help", "Print help")
        ("m,matrix", "Matrix file in martrix market format", cxxopts::value<string>())
        ("l,lvl","Number of levels", cxxopts::value<int>())
        // Geometry
        ("coordinates", "Coordinates MM array file. If provided, will do a geometric partitioning.", cxxopts::value<string>())
        ("n,coordinates_n", "If provided with -n, will use a tensor n^d & geometric partitioning. Overwrites --coordinates", cxxopts::value<int>()->default_value("-1"))
        ("d,coordinates_d", "If provided with -d, will use a tensor n^d & geometric partitioning. Overwrites --coordinates", cxxopts::value<int>()->default_value("-1"))
        // Partition
        ("hsl","Use bipartite matching routine from HSL to perform row ordering. Use only if compiled with USE_HSL=1 flag. Default false.", cxxopts::value<int>()->default_value("0"))
        // Sparsification
        ("t,tol", "Tolerance", cxxopts::value<double>()->default_value("1e-1")) 
        ("skip", "Skip sparsification", cxxopts::value<int>()->default_value("0")) 
        ("scale", "Do scaling. Default true.", cxxopts::value<int>()->default_value("1"))
        // Iterative method
        // ("solver","Wether to use CG or GMRES or CGLS. Default GMRES for square matrices and CGLS for rectangular.", cxxopts::value<string>()->default_value("GMRES"))
        ("i,iterations","Iterative solver iterations", cxxopts::value<int>()->default_value("300"))
        ("rhs", "Provide RHS to solve in matrix market format", cxxopts::value<string>())
        ("res", "Desired relative residual for the iterative solver. Default 1e-12", cxxopts::value<double>()->default_value("1e-12")) 

        // Use solvers from Eigen library
        ("useEigenLSCG","If true, run CGLS scheme with standard diagonal preconditioner from Eigen library. Default false.", cxxopts::value<int>()->default_value("0"))
        ("useEigenQR","If true, run SparseQR with default ordering from Eigen library. Default false.", cxxopts::value<int>()->default_value("0"))
        ("useCholesky","If true, run SimplicialLDL^T with AMDOrdering from Eigen library. Default false.", cxxopts::value<int>()->default_value("0"));
    

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        cout << options.help({"", "Group"}) << endl;
        exit(0);
    }

    if ( (!result.count("matrix"))  ) {
        cout << "--matrix is mandatory" << endl;
        exit(0);
    }

    string matrix = result["matrix"].as<string>();
    int nlevels;


    // Geometry
    string coordinates;
    int cn = result["coordinates_n"].as<int>();
    int cd = result["coordinates_d"].as<int>();
    bool geo_file = (result.count("coordinates") > 0);
    if ( (cn == -1 && cd >= 0) || (cd == -1 && cn >= 0) ) {
        cout << "cn and cd should be both provided, or none should be provided" << endl;
        return 1;
    }
    bool geo_tensor = (cn >= 0 && cd >= 0);
    bool geo = geo_file || geo_tensor;
    if(geo_file) {
        coordinates = result["coordinates"].as<string>();
    }

    // Load matrix
    SpMat A = mmio::sp_mmread<double, int>(matrix);

    if (A.rows() < A.cols()){
        cout << " <<< Warning!!! nrows < ncols. Finding QR on A.transpose() instead. " << endl;
        SpMat T = A.transpose();
        A = T;
    }
    
    int nrows = A.rows();
    int ncols = A.cols();
    cout << "Matrix " << matrix << " with " << nrows << " rows,  " << ncols << " columns loaded" << endl;

    // Iterative method 
    bool useGMRES = (nrows == ncols) ? true : false;
    bool useCGLS = (nrows > ncols) ? true : false;
    int iterations = result["iterations"].as<int>();
    
    bool useEigenLSCG = result["useEigenLSCG"].as<int>();
    bool useEigenQR = result["useEigenQR"].as<int>();
    bool useCholesky = result["useCholesky"].as<int>();

    double residual = result["res"].as<double>();

    if ( (!result.count("lvl"))  ) {
        cout << "--Levels not provided" << endl;
        nlevels = ceil(log2(ncols/64));
        cout << " Levels set to ceil(log2(ncols/64)) =  " << nlevels << endl;
    }
    else{
        nlevels = result["lvl"].as<int>();
    }

    // Partition
    int hsl = result["hsl"].as<int>();

    // Sparsification parameters
    int scale = result["scale"].as<int>();
    if (nrows != ncols && scale == 0){
        cout << "Scaling necessary for rectangular matrices" << endl;
        cout << "Setting scale to 1" << endl;
        scale = 1;
    }
    double tol = result["tol"].as<double>();

    // Pre-process matrix to have columns of unit norm (diagonal scaling)
    VectorXd Dentries(A.cols());
    DiagonalMatrix<double, Eigen::Dynamic> D(A.cols());

    auto pstart = wctime();
    for (int i=0; i < A.cols(); ++i) {
        double sum = 0;
       for (SpMat::InnerIterator it(A,i); it; ++it){
            sum += it.value()*it.value();
       }
       Dentries[i] = (double)(1.0/sqrt(sum));
    }

    D = Dentries.asDiagonal();
    A = A*D*10;
    auto pend = wctime();
    cout << "Pre-process time: " << elapsed(pstart, pend) << endl;
    

    int skip = (tol == 0 ? nlevels-1 : result["skip"].as<int>());
    
    // Load coordinates ?
    MatrixXd X;
    if(geo_tensor) {
        if(pow(cn, cd) != ncols) {
            cout << "Error: cn and cd where both provided, but cn^cd != N where A is NxN" << endl;
            return 1;
        }
        X = linspace_nd(cn, cd);
        cout << "Tensor coordinate matrix of size " << cn << "^" << cd << " built" << endl;
    } else if(geo_file) {
        X = mmio::dense_mmread<double>(coordinates);
        cout << "Coordinate file " << X.rows() << "x" << X.cols() << " loaded from " << coordinates << endl;
        if(X.cols() != ncols) {
            cout << "Error: coordinate file should hold a matrix of size d x N" << endl;
        }
    }


    // Tree
    Tree t(nlevels, skip);
    t.set_scale(scale);
    t.set_tol(tol);
    t.set_hsl(hsl);


    if (nrows == ncols) t.set_square(1); // default 0
    if(geo) t.set_Xcoo(&X);

    if (useEigenLSCG) {
        VectorXd x = VectorXd::Zero(ncols);
        VectorXd b = random(nrows,2021);
        
        LeastSquaresConjugateGradient<SpMat, LeastSquareDiagonalPreconditioner<double>> lscg;
        lscg.compute(A);

        LeastSquareDiagonalPreconditioner<double> diag_precond = lscg.preconditioner();

        timer cgls0 = wctime();
        auto iter = lscg_eigen(A, b, x, diag_precond, iterations, residual, true);
        timer cgls1 = wctime();

        cout << "CGLS error: " << scientific <<  (A.transpose()*(A*x-b)).norm() / (A.transpose()*b).norm() << endl;
        cout << "  CGLS: " << elapsed(cgls0, cgls1) << " s." << endl;
        cout << "<<<<CGLS=" << iter << endl;
        return 0;
    }

    if (useEigenQR){
        VectorXd x = VectorXd::Zero(ncols);
        VectorXd b = random(nrows,2021);
        
        A.makeCompressed();  
        SparseQR<SpMat, COLAMDOrdering<int>> eigenQR;

        eigenQR.setPivotThreshold(1e-14);
        cout << "\n <<<< Using Eigen SparseQR routine..." << endl;
        timer qr0 = wctime();
        eigenQR.compute(A);
        timer qr1 = wctime();
        cout << "Time to factorize: " << elapsed(qr0, qr1) << " s." << endl;

        timer qrs0 = wctime();
        x = eigenQR.solve(b);
        timer qrs1 = wctime();
        cout << "Time to solve: " << elapsed(qrs0, qrs1) << " s." << endl;
        cout << "Error: " << scientific << (A.transpose()*(A*x-b)).norm() / (A.transpose()*b).norm() << endl; 
        
        return 0;
    }

    if (useCholesky){
        VectorXd x = VectorXd::Zero(ncols);
        VectorXd b = random(nrows,2021);

        VectorXd Atb = A.transpose()*b;
        SpMat AtA = A.transpose()*A;
        
        AtA.makeCompressed();
        SimplicialLLT<SpMat, Lower, AMDOrdering<int> > eigenCholesky;
        cout << "\n <<<< Using Eigen's SimplicalLL^T routine..." << endl;

        timer qr0 = wctime();
        eigenCholesky.compute(AtA);
        timer qr1 = wctime();
        cout << "Time to factorize: " << elapsed(qr0, qr1) << " s." << endl;

        timer qrs0 = wctime();
        x = eigenCholesky.solve(Atb);
        timer qrs1 = wctime();

        cout << "Time to solve: " << elapsed(qrs0, qrs1) << " s." << endl;
        cout << "Error: " << scientific << (A.transpose()*(A*x-b)).norm() / (A.transpose()*b).norm() << endl; 
        
        return 0;
    }

    // Partition
    t.partition(A);
    // Setup
    t.assemble(A);

    // Factorize
    int err = t.factorize();


    if (!err)
    // Run one solve
    {
         // Random b
        {
            VectorXd b = random(nrows, 2021);
            VectorXd bcopy = b;
            VectorXd x(ncols, 1);
            x.setZero();
            timer tsolv_0 = wctime();

            if (nrows == ncols){
                t.solve(bcopy, x);
                timer tsolv = wctime();
                cout << "<<<<tsolv=" << elapsed(tsolv_0, tsolv) << endl;
                cout << "One-time solve (Random b):" << endl;             
                cout << "<<<<|(Ax-b)|/|b| : " << scientific <<  ((A*x-b)).norm() / (b).norm() << endl;
            }
            else {
                t.solve_nrml(A.transpose()*bcopy, x);
                timer tsolv = wctime();
                cout << "<<<<tsolv=" << elapsed(tsolv_0, tsolv) << endl;
                cout << "One-time solve (Random b):" << endl;             
                cout << "<<<<|A'(Ax-b)|/|A'b| : " << scientific <<  (A.transpose()*(A*x-b)).norm() / (A.transpose()*b).norm() << endl;
            }
        }
    }

    bool verb = false; 
    int iter = 0;
    if (!err)
    {
        VectorXd x = VectorXd::Zero(ncols);
        VectorXd b;
        if ((!result.count("rhs"))){
            b = random(nrows,2021);
        }
        else {
            string rhs_file = result["rhs"].as<string>();
            b = mmio::vector_mmread<double>(rhs_file);
        }
        VectorXd bcopy = b;

    
        if(useGMRES) {
            timer gmres0 = wctime();
            iter = gmres(A, b, x, t, iterations, iterations, residual, verb);
            timer gmres1 = wctime();
            cout << "GMRES: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A*x-b).norm() / b.norm() << endl;
            cout << "  GMRES: " << elapsed(gmres0, gmres1) << " s." << endl;
            cout << "<<<<GMRES=" << iter << endl;
        }
        else if(useCGLS){
            timer cg0 = wctime();

            Index max_iters = (long)iterations;
            iter = cgls(A, b, x, t, max_iters, residual, verb);
            cout << "CGLS: #iterations: " << iter << ", residual |A'(Ax-b)|/|A'(b)|: " << (A.transpose()*(A*x-b)).norm() / (A.transpose()*b).norm() << endl;
            timer cg1 = wctime();
            cout << "  CGLS: " << elapsed(cg0, cg1) << " s." << endl;
            cout << "<<<<CGLS=" << iter << endl;
        }
    }

  return 0;  
}
