#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/QR>
#include <Eigen/Householder> 
#include <Eigen/SVD>
#include "operations.h"

using namespace std;
using namespace Eigen;

/* QR */
void QR::fwd(){
	MatrixXd Q = MatrixXd::Zero(nrows, ncols);
	VectorXd xc = VectorXd::Zero(nrows);

	int curr_row = 0;
	int i=0;
	for (auto e: c->edgesOut){
		if (e->A21 != nullptr){
			int s = A21_indices[i];
			xc.segment(curr_row, s) = e->n2->get_x()->segment(0, s);
			Q.middleRows(curr_row, s) = *(e->A21); 
			curr_row += s;
			++i;
		}
	}

	larfb(&Q, c->get_T(), &xc);
	curr_row = 0;
	i=0;
	for (auto e: c->edgesOut){
		if (e->A21 != nullptr){
			int s = A21_indices[i];
			e->n2->get_x()->segment(0, s) = xc.segment(curr_row, s);
			curr_row += s;
			++i;
		}
	}
}

void QR::bwd(){
	MatrixXd R;
	int i=0;
	Segment xs = c->get_x()->segment(0, ccols);
	for (auto e: c->edgesIn){
		assert(e->A21 != nullptr);
		int s = A12_indices[i];
		assert(e->A21->cols() == s);
		xs -= (e->A21->topRows(ccols))*(e->n1->get_x()->segment(0, s)); 
		++i;	
	}
	assert((*c->edgesOut.begin())->n2 == c);

	R = ((*c->edgesOut.begin())->A21->topRows(xs.size()));
	trsv(&R, &xs, CblasUpper, CblasNoTrans, CblasNonUnit);

}

/* Reassign rows */
void Reassign::fwd(){
	// cout << c->get_id() << " " << n->get_id() << endl;
	assert(indices.size() == nrows);
	for (int i= nstart; i < nstart+nrows; ++i){
		// n->get_x()->segment(i,1) = c->get_x()->segment(indices[i-nstart],1);
		(*n->get_x())[i] = (*c->get_x())[indices[i-nstart]];

	}

}

/* Scaling */
void Scale::bwd(){
    trsv(R, &xs, CblasUpper, CblasNoTrans, CblasNonUnit);
}

void ScaleD::fwd(){
	assert(t != nullptr);
	ormqr_trans(Q, t, &xsf);
}

void ScaleD::bwd(){
	MatrixXd R = Q->topRows(xs.size());
	trsv(&R, &xs, CblasUpper, CblasNoTrans, CblasNonUnit);
}

/* Sparsification using Orthogonal transformations */
void Orthogonal::bwd(){ormqr_notrans(V, tau, &xs);}

void OrthogonalD::fwd(){ormqr_trans(V, tau, &xs);}
void OrthogonalD::bwd(){ormqr_notrans(V, tau, &xs);}


/* Merging in the cluster heirarchy */
void Merge::fwd(){
	int k=0;
	for (auto c: parent->children){
		for (int i=0; i < c->rows(); ++i){
			(*parent->get_x())[k] = (*c->get_x())[i];
			++k;
		}
	}
	// assert(k== parent->get_x()->size());
}

void Merge::bwd(){
	int k=0;
	// cout << "merge bwd " << parent->get_id() << endl;
	for (auto c: parent->children){
		for (int i=0; i < c->cols(); ++i){
			(*c->get_x())[i] = (*parent->get_x())[k];
			++k;
		}
	}
	// assert(k== parent->get_x()->size());
}

/* Split between coarse and fine nodes */
void Split::fwd(){xsf = xsc.bottomRows(xsf.size());}
void Split::bwd(){xsc_head.bottomRows(xsf.size()) = xsf;}

void SplitD::fwd(){
	xsf = xsc.middleRows(rank, ccols-rank);
	VectorXd xsc_rem = xsc.bottomRows(crows-ccols);
	xsc.middleRows(rank, crows-ccols) = xsc_rem;
	xsc.bottomRows(ccols-rank) = xsf;
}
void SplitD::bwd(){
	VectorXd xsc_rem = xsc.middleRows(rank, crows-ccols);
	xsc.middleRows(rank, ccols-rank) = xsf;
	xsc.bottomRows(crows-ccols) = xsc_rem;
}





