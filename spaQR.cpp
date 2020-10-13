#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include <Eigen/IterativeLinearSolvers>
#include "mmio.hpp"
#include "cxxopts.hpp"
#include "tree.h"
#include "partition.h"
#include "util.h"
#include "stats.h"
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
        // Statistics
        ("stats", "Output statistics to file. Default false.", cxxopts::value<string>())
        ("edges", "Output the matrix structure to file. Default false.", cxxopts::value<string>())
         // Iterative method
        ("solver","Wether to use GMRES or CG", cxxopts::value<string>()->default_value("GMRES"))
        ("i,iterations","Iterative solver iterations", cxxopts::value<int>()->default_value("300"))
        ("rhs", "Provide RHS to solve in matrix market format", cxxopts::value<string>());
 
        

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

    // Iterative method 
    bool useCG = (result["solver"].as<string>() == "CG");
    bool useGMRES = (result["solver"].as<string>() == "GMRES");
    int iterations = result["iterations"].as<int>();
    if(!useCG && !useGMRES) {
        cout << "Wrong solver picked. Should be CG or GMRES" << endl;
        return 1;
    }

    // Load matrix
    SpMat A = mmio::sp_mmread<double, int>(matrix);
    
    int nrows = A.rows();
    int ncols = A.cols();
    cout << "Matrix " << matrix << " with " << nrows << " rows,  " << ncols << " columns loaded" << endl;

    if (nrows != ncols){
        cout << "Error: Matrix is not square!" << endl;
        return 1;
    }

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

    if(geo) t.set_Xcoo(&X);

    auto partstart = wctime();
    t.partition(A);
    auto partend = wctime();
    cout << "Time to partition: " << elapsed(partstart, partend)  << endl;

    auto astart = wctime();
    t.assemble(A);
    auto aend = wctime();
    cout << "Time to assemble: " << elapsed(astart, aend)  << endl;

    // Output edges
    if (result.count("edges")){
        string edges_fn = result["edges"].as<string>();
        t.set_output(1, edges_fn);
        write_basic_info(t, edges_fn+"_merge.txt");
        write_basic_info(t, edges_fn+"_elmn.txt");
    }

    int err = t.factorize();

    // Output stats
    if (result.count("stats")){
        string stats_fn = result["stats"].as<string>();
        std::cout << "Writing cluster stats to " << stats_fn << endl;        
        write_stats(t, stats_fn);
    }


    if (!err)
    // Run one solve
    {
         // Random b
        {
            VectorXd b = random(nrows, 2019);
            VectorXd bcopy = b;
            VectorXd x(ncols, 1);
            x.setZero();
            timer tsolv_0 = wctime();

            
            t.solve(bcopy, x);
            timer tsolv = wctime();
            cout << "<<<<tsolv=" << elapsed(tsolv_0, tsolv) << endl;
            cout << "One-time solve (Random b):" << endl;             
            cout << "<<<<|(Ax-b)|/|b| : " << scientific <<  ((A*x-b)).norm() / (b).norm() << endl;
            
        }
    }

    bool verb = false; 
    int iter = 0;
    if (!err)
    {
        VectorXd x = VectorXd::Zero(ncols);
        VectorXd b;
        if ((!result.count("rhs"))){
            b = random(nrows,2019);
        }
        else {
            string rhs_file = result["rhs"].as<string>();
            b = mmio::vector_mmread<double>(rhs_file);
        }
        VectorXd bcopy = b;

        if(useCG) {            
            timer cg0 = wctime();
            iter = cg(A, b, x, t, iterations, 1e-12, verb);   
            timer cg1 = wctime();

            cout << "CG: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A*x-b).norm() / b.norm() << endl;
            cout << "  CG: " << elapsed(cg0, cg1) << " s." << endl;
            cout << "<<<<CG=" << iter << endl;
        } else if(useGMRES) {
            timer gmres0 = wctime();
            iter = gmres(A, b, x, t, iterations, iterations, 1e-12, verb);
            timer gmres1 = wctime();
            cout << "GMRES: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A*x-b).norm() / b.norm() << endl;
            cout << "  GMRES: " << elapsed(gmres0, gmres1) << " s." << endl;
            cout << "<<<<GMRES=" << iter << endl;
        }
    }

  return 0;  
}
