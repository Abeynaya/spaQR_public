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

		Edge(Cluster* n1_, Cluster* n2_, Eigen::MatrixXd* A): n1(n1_), n2(n2_){
			A21 = A;
			if (A != nullptr){
				assert(A->rows() == n2->rows());
				assert(A->cols() == n1->cols());
			}
			
		}

		~Edge(){
			if (A21 != nullptr){
				delete A21;
			}
		}
};

std::ostream& operator<<(std::ostream& os, const Edge& e);

#endif