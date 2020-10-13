#ifndef IS_H
#define IS_H

#include <iostream>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include "util.h"
#include "tree.h"

template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
int cg(const MatrixType& mat, const Rhs& rhs, Dest& x, const Preconditioner& precond, int iters, double tol, bool verb);

template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
int gmres(const MatrixType& mat, const Rhs& rhs, Dest& x, const Preconditioner& precond, int iters, int restart, double tol_error, bool verb);


template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
int cgls(const MatrixType& mat, const Rhs& rhs, Dest& x, const Preconditioner& precond, int iters, double tol, bool verb);

template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
int lscg_eigen(const MatrixType& mat, const Rhs& rhs, Dest& x, const Preconditioner& precond, int iters, double tol, bool verb);

#endif
