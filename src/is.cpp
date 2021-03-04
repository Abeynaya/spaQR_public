// Code partially from Eigen http://eigen.tuxfamily.org
// Originally published under the MPLv2 License https://www.mozilla.org/en-US/MPL/2.0/
// See http://eigen.tuxfamily.org/index.php?title=Main_Page#License

#include "is.h"

using namespace Eigen;

template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
int cg(const MatrixType& mat, const Rhs& rhs, Dest& x, const Preconditioner& precond, int iters, double tol, bool verb)
{
    using std::sqrt;
    using std::abs;
    typedef Eigen::Matrix<double,Eigen::Dynamic,1> VectorType;
    
    double t_matvec = 0.0;
    double t_preco  = 0.0;
    timer start     = wctime();
    
    int maxIters = iters;
    
    int n = mat.cols();

    timer t00 = wctime();
    VectorType residual = rhs - mat * x; // r_0 = b - A x_0
    timer t01 = wctime();
    t_matvec += elapsed(t00, t01);

    double rhsNorm2 = rhs.squaredNorm();
    if(rhsNorm2 == 0) 
    {
        x.setZero();
        iters = 0;
        return iters;
    }
    double threshold = tol*tol*rhsNorm2;
    double residualNorm2 = residual.squaredNorm();
    if (residualNorm2 < threshold)
    {
        iters = 0;
        return iters;
    }
   
    VectorType p(n);
    p = residual;
    timer t02 = wctime();
    precond.solve(p,p);      // p_0 = M^-1 r_0
    timer t03 = wctime();
    t_preco += elapsed(t02, t03);

    VectorType z(n), tmp(n);
    double absNew = residual.dot(p);  // the square of the absolute value of r scaled by invM
    int i = 0;
    while(i < maxIters)
    {
        timer t0 = wctime();
        tmp.noalias() = mat * p;                    // the bottleneck of the algorithm
        timer t1 = wctime();
        t_matvec += elapsed(t0, t1);

        double alpha = absNew / p.dot(tmp);         // the amount we travel on dir
        x += alpha * p;                             // update solution
        residual -= alpha * tmp;                    // update residual
        
        residualNorm2 = residual.squaredNorm();
        if(verb) printf("%d: |Ax-b|/|b| = %3.2e <? %3.2e\n", i, sqrt(residualNorm2 / rhsNorm2), tol);
        if(residualNorm2 < threshold) {
            if(verb) printf("Converged!\n");
            break;
        }
     
        z = residual; 
        timer t2 = wctime();
        precond.solve(z,z);                           // approximately solve for "A z = residual"
        timer t3 = wctime();
        t_preco += elapsed(t2, t3);
        double absOld = absNew;
        absNew = residual.dot(z);                   // update the absolute value of r
        double beta = absNew / absOld;              // calculate the Gram-Schmidt value used to create the new search direction
        p = z + beta * p;                           // update search direction
        i++;
    }
    iters = i+1;
    if(verb) {
        timer stop = wctime();
        printf("# of iter:  %d\n", iters);
        printf("Total time: %3.2e s.\n", elapsed(start, stop));
        printf("  Matvec:   %3.2e s.\n", t_matvec);
        printf("  Precond:  %3.2e s.\n", t_preco);
    }
    return iters;
};

template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
int gmres(const MatrixType& mat, const Rhs& rhs, Dest& x, const Preconditioner& precond, int iters, int restart, double tol_error, bool verb) {
    timer start     = wctime();
    double t_matvec = 0.0;
    double t_preco  = 0.0;

    using std::sqrt;
    using std::abs;

    typedef typename Dest::RealScalar RealScalar;
    typedef typename Dest::Scalar Scalar;
    typedef Matrix < Scalar, Dynamic, 1 > VectorType;
    typedef Matrix < Scalar, Dynamic, Dynamic, ColMajor> FMatrixType;

    const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();

    if(rhs.norm() <= considerAsZero) {
        x.setZero();
        tol_error = 0;
        return true;
    }


    RealScalar tol = tol_error;
    const int maxIters = iters;
    iters = 0;

    const int m = mat.rows();

    // residual and preconditioned residual
    timer t0 = wctime();    
    VectorType p0 = rhs - mat*x;
    timer t1 = wctime();
    t_matvec += elapsed(t0, t1);
    VectorType r0 = p0;
    precond.solve(r0, r0);
    timer t2 = wctime();
    t_preco += elapsed(t1, t2);

    const RealScalar r0Norm = r0.norm();
    std::cout << "initial residual " << (mat*r0-p0).norm()/ p0.norm() << std::endl;
    // is initial guess already good enough?
    if(r0Norm == 0) {
        printf("Converged!");
        return true;
    }

    // storage for Hessenberg matrix and Householder data
    FMatrixType H   = FMatrixType::Zero(m, restart + 1);
    VectorType w    = VectorType::Zero(restart + 1);
    VectorType tau  = VectorType::Zero(restart + 1);

    // storage for Jacobi rotations
    std::vector < JacobiRotation < Scalar > > G(restart);

    // storage for temporaries
    VectorType t(m), v(m), workspace(m), x_new(m);

    // generate first Householder vector
    Ref<VectorType> H0_tail = H.col(0).tail(m - 1);
    RealScalar beta;
    r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
    w(0) = Scalar(beta);

    for (int k = 1; k <= restart; ++k)
    {
        ++iters;

        v = VectorType::Unit(m, k - 1);

        // apply Householder reflections H_{1} ... H_{k-1} to v
        // TODO: use a HouseholderSequence
        for (int i = k - 1; i >= 0; --i) {
            v.tail(m - i).applyHouseholderOnTheLeft(H.col(i).tail(m - i - 1), tau.coeffRef(i), workspace.data());
        }

        // apply matrix M to v:  v = mat * v;
        timer t0 = wctime();
        t.noalias() = mat * v;
        timer t1 = wctime();
        t_matvec += elapsed(t0, t1);
        v = t;
        precond.solve(v, v);
        timer t2 = wctime();
        t_preco += elapsed(t1, t2);

        // apply Householder reflections H_{k-1} ... H_{1} to v
        // TODO: use a HouseholderSequence
        for (int i = 0; i < k; ++i) {
            v.tail(m - i).applyHouseholderOnTheLeft(H.col(i).tail(m - i - 1), tau.coeffRef(i), workspace.data());
        }

        if (v.tail(m - k).norm() != 0.0) {
            if (k <= restart)
            {
            // generate new Householder vector
            Ref<VectorType> Hk_tail = H.col(k).tail(m - k - 1);
            v.tail(m - k).makeHouseholder(Hk_tail, tau.coeffRef(k), beta);

            // apply Householder reflection H_{k} to v
            v.tail(m - k).applyHouseholderOnTheLeft(Hk_tail, tau.coeffRef(k), workspace.data());
            }
        }

        if (k > 1) {
            for (int i = 0; i < k - 1; ++i) {
                // apply old Givens rotations to v
                v.applyOnTheLeft(i, i + 1, G[i].adjoint());
            }
        }

        if (k<m && v(k) != (Scalar) 0) {
            // determine next Givens rotation
            G[k - 1].makeGivens(v(k - 1), v(k));

            // apply Givens rotation to v and w
            v.applyOnTheLeft(k - 1, k, G[k - 1].adjoint());
            w.applyOnTheLeft(k - 1, k, G[k - 1].adjoint());
        }

        // insert coefficients into upper matrix triangle
        H.col(k-1).head(k) = v.head(k);

        tol_error = abs(w(k)) / r0Norm;
        bool stop = (k==m || tol_error < tol || iters == maxIters);
        if(verb) printf("%d: |Ax-b|/|b| = %3.2e <? %3.2e\n", iters, tol_error, tol);

        if (stop || k == restart) {
            // solve upper triangular system
            Ref<VectorType> y = w.head(k);
            H.topLeftCorner(k, k).template triangularView <Upper>().solveInPlace(y);

            // use Horner-like scheme to calculate solution vector
            x_new.setZero();
            for (int i = k - 1; i >= 0; --i) {
                x_new(i) += y(i);
                // apply Householder reflection H_{i} to x_new
                x_new.tail(m - i).applyHouseholderOnTheLeft(H.col(i).tail(m - i - 1), tau.coeffRef(i), workspace.data());
            }

            x += x_new;

            if(stop) {
                printf("GMRES converged!\n");
                if(verb) {
                    timer stop = wctime();
                    printf("# of iter:  %d\n", iters);
                    printf("Total time: %3.2e s.\n", elapsed(start, stop));
                    printf("  Matvec:   %3.2e s.\n", t_matvec);
                    printf("  Precond:  %3.2e s.\n", t_preco);
                }
                return iters;
            } else {
                k=0;

                // reset data for restart
                timer t0 = wctime();
                p0.noalias() = rhs - mat*x;
                timer t1 = wctime();
                t_matvec += elapsed(t0, t1);                
                r0 = p0;
                precond.solve(r0, r0);
                timer t2 = wctime();
                t_preco += elapsed(t1, t2);

                // clear Hessenberg matrix and Householder data
                H.setZero();
                w.setZero();
                tau.setZero();

                // generate first Householder vector
                r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
                w(0) = Scalar(beta);
            }
        }
    }
    printf("GMRES failed to converge in %d iterations\n", iters);
    return iters;
}

template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
int cgls(const MatrixType& mat, const Rhs& rhs, Dest& x, const Preconditioner& precond, Index& iters, typename Dest::RealScalar& tol_error, bool verb)
{
    using std::sqrt;
    using std::abs;
    typedef typename Dest::RealScalar RealScalar;
    typedef typename Dest::Scalar Scalar;
    typedef Matrix<Scalar,Dynamic,1> VectorType;
    
    RealScalar tol = tol_error;
    Index maxIters = iters;
    
    Index m = mat.rows(), n = mat.cols();

    VectorType residual        = rhs - mat * x;
    VectorType normal_residual = mat.adjoint() * residual;

    RealScalar rhsNorm2 = (mat.adjoint()*rhs).squaredNorm();
    if(rhsNorm2 == 0) 
    {
      x.setZero();
      iters = 0;
      tol_error = 0;
      return iters;
    }
    RealScalar threshold = tol*tol*rhsNorm2;
    RealScalar residualNorm2 = normal_residual.squaredNorm();
    if (residualNorm2 < threshold)
    {
      iters = 0;
      tol_error = sqrt(residualNorm2 / rhsNorm2);
      return iters;
    }
    
    VectorType p(n);
    precond.solve_nrml(normal_residual,p);                         // initial search direction

    VectorType z(n), tmp(m);
    RealScalar absNew = numext::real(normal_residual.dot(p));  // the square of the absolute value of r scaled by invM
    Index i = 0;
    while(i < maxIters)
    {
      tmp.noalias() = mat * p;

      Scalar alpha = absNew / tmp.squaredNorm();      // the amount we travel on dir
      x += alpha * p;                                 // update solution
      residual -= alpha * tmp;                        // update residual
      normal_residual = mat.adjoint() * residual;     // update residual of the normal equation
      
      residualNorm2 = normal_residual.squaredNorm();
      // if(verb  && (i % 5 == 0)) printf("%ld: |A'(Ax-b)|/|A'b| = %3.2e <? %3.2e\n", i, sqrt(residualNorm2 / rhsNorm2), tol);
      if(verb) printf("%ld %3.2e \n", i, sqrt(residualNorm2 / rhsNorm2));

      if(residualNorm2 < threshold) {
          if(verb) printf("Converged!\n");
          break;
      }
      
      
      precond.solve_nrml(normal_residual,z);             // approximately solve for "A'A z = normal_residual"

      RealScalar absOld = absNew;
      absNew = numext::real(normal_residual.dot(z));  // update the absolute value of r
      RealScalar beta = absNew / absOld;              // calculate the Gram-Schmidt value used to create the new search direction
      p = z + beta * p;                               // update search direction
      i++;
    }
    tol_error = sqrt(residualNorm2 / rhsNorm2);
    iters = i;
    return iters;

}


// Use solve method to use diagonal precondtionders from eigen
template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
int lscg_eigen(const MatrixType& mat, const Rhs& rhs, Dest& x, const Preconditioner& precond, int iters, double tol, bool verb)
{
    using std::sqrt;
    using std::abs;
    typedef Eigen::Matrix<double,Eigen::Dynamic,1> VectorType;
    
    double t_matvec = 0.0;
    double t_preco  = 0.0;
    timer start     = wctime();
    
    int maxIters = iters;
    
    int n = mat.cols();
    int m = mat.rows();


    timer t00 = wctime();
    VectorType residual = rhs - mat * x; // r_0 = b - A x_0
    VectorType normal_residual = mat.adjoint()*residual;
    timer t01 = wctime();
    t_matvec += elapsed(t00, t01);

    double rhsNorm2 = (mat.adjoint()*rhs).squaredNorm(); 

    if(rhsNorm2 == 0) 
    {
        x.setZero();
        iters = 0;
        return iters;
    }
    double threshold = tol*tol*rhsNorm2;
    double residualNorm2 = normal_residual.squaredNorm();

    if (residualNorm2 < threshold)
    {
        iters = 0;
        return iters;
    }
   
    VectorType p(n), ptemp(n);
    p.setZero(); ptemp.setZero();

    timer t02 = wctime();
    p = precond.solve(normal_residual);      // p_0 = M^-1 r_0 

    timer t03 = wctime();
    t_preco += elapsed(t02, t03);
    // std::cout << "p: " << (mat.adjoint()*p)/ << std::endl; 

    VectorType z(n), tmp(m) , ztemp(n);
    z.setZero(); ztemp.setZero();
    double absNew = normal_residual.dot(p);  // the square of the absolute value of r scaled by invM
    int i = 0;
    while(i < maxIters)
    {
        timer t0 = wctime();
        tmp.noalias() = mat * p;                    // the bottleneck of the algorithm
        timer t1 = wctime();
        t_matvec += elapsed(t0, t1);

        double alpha = absNew / tmp.squaredNorm();         // the amount we travel on dir
        // std::cout << "absNew: " << absNew << "tmp: " << tmp.squaredNorm() << "alpha: " << alpha << std::endl;
        x += alpha * p;                             // update solution
        residual -= alpha * tmp;                    // update residual
        normal_residual = mat.adjoint() * residual;     // update residual of the normal equation

        residualNorm2 = normal_residual.squaredNorm();
        if(verb && ((i<=50 && i % 5 == 0) || (i>50 && i%50 ==0)) ) printf("%d %3.2e\n", i, sqrt(residualNorm2 / rhsNorm2));

        // if(verb  && (i % 50 == 0)) printf("%d: |A'(Ax-b)|/|A'b| = %3.2e <? %3.2e\n", i, sqrt(residualNorm2 / rhsNorm2), tol);
        // if(verb) printf("%d: |A'(Ax-b)|/|A'b| = %3.2e <? %3.2e\n", i, (mat.transpose()*(mat*x-rhs)).norm()/(mat.transpose()*rhs).norm(), tol);
        if(residualNorm2 < threshold) {
            if(verb) printf("Converged!\n");
            break;
        }
     
        timer t2 = wctime();
        z = precond.solve(normal_residual);                           // approximately solve for "A z = residual"
        
        timer t3 = wctime();
        t_preco += elapsed(t2, t3);
        double absOld = absNew;
        absNew = normal_residual.dot(z);                   // update the absolute value of r
        double beta = absNew / absOld;                        // calculate the Gram-Schmidt value used to create the new search direction
        p = z + beta * p;                           // update search direction
        i++;
    }
    iters = i+1;
    if(verb) {
        timer stop = wctime();
        printf("# of iter:  %d\n", iters);
        printf("Total time: %3.2e s.\n", elapsed(start, stop));
        printf("  Matvec:   %3.2e s.\n", t_matvec);
        printf("  Precond:  %3.2e s.\n", t_preco);
    }
    return iters;
};


template int cg(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, int iters, double tol, bool verb);
template int gmres(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, int iters, int restart, double tol_error, bool verb);
template int cgls(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, Eigen::Index& iters, Eigen::VectorXd::RealScalar& tol, bool verb);
template int lscg_eigen(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Eigen::LeastSquareDiagonalPreconditioner<double>& precond, int iters, double tol, bool verb);


