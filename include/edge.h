#ifndef EDGE_H
#define EDGE_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "util.h"
#include "cluster.h"

struct Edge{
	public: 
		Cluster* n1;
		Cluster* n2;
		Eigen::MatrixXd* A21;
		Eigen::MatrixXd* A12;

		Edge(Cluster* n1_, Cluster* n2_, Eigen::MatrixXd* A, Eigen::MatrixXd* AT): n1(n1_), n2(n2_){
			A21 = A;
			A12 = AT;

			if (A != nullptr){
				assert(A->rows() == n2->rows());
				assert(A->cols() == n1->cols());
			}
			if (AT != nullptr){
				assert(AT->rows() == n1->rows());
				assert(AT->cols() == n2->cols());
			}
		}

};

std::ostream& operator<<(std::ostream& os, const Edge& e);

#endif