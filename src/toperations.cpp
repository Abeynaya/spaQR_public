#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/QR>
#include <Eigen/Householder> 
#include <Eigen/SVD>
#include "toperations.h"

using namespace std;
using namespace Eigen;

/* QR */
void tQR::fwd(){
	int i=0;
	Segment xs = c->xt->segment(0, ccols);
	assert((*c->edgesOut.begin())->n2 == c);
    Edge* eself = *(c->edgesOut.begin());
	MatrixXd R = (eself->A21)->topRows(xs.size());

	trsv(&R, &xs, CblasUpper, CblasTrans, CblasNonUnit); // Do inv(R')*xs
	for (auto e: c->edgesIn){
		assert(e->A21 != nullptr);
		int s = A12_indices[i];
		assert(e->A21->cols() == s);
		(e->n1->xt->segment(0, s)) -= (e->A21->topRows(ccols)).transpose()*xs; 
		++i;
	}
}

void tQR::bwd(){
	MatrixXd Q = MatrixXd::Zero(nrows, ncols);
	VectorXd xc = VectorXd::Zero(nrows);

	int curr_row = 0;
	int i=0;
	for (auto e: c->edgesOut){
		if (e->A21 != nullptr){
			int s = A21_indices[i];
			xc.segment(curr_row, s) = e->n2->xt->segment(0, s);
			Q.middleRows(curr_row, s) = *(e->A21); 
			curr_row += s;
			++i;
		}
	}

	larfb(&Q, c->get_T(), &xc, 'L', 'N', 'F', 'C');
	curr_row = 0;
	i=0;
	for (auto e: c->edgesOut){
		if (e->A21 != nullptr){
			int s = A21_indices[i];
			e->n2->xt->segment(0, s) = xc.segment(curr_row, s);
			curr_row += s;
			++i;
		}
	}
}

/* Reassign rows */
void tReassign::bwd(){
	// cout << c->get_id() << " " << n->get_id() << endl;
	assert(indices.size() == nrows);
	for (int i= nstart; i < nstart+nrows; ++i){
		(*c->xt)[indices[i-nstart]] = (*n->xt)[i];
	}

}


/* Scaling */
void tScale::fwd(){
    trsv(R, &xs, CblasUpper, CblasTrans, CblasNonUnit);
}

void tScaleD::fwd(){
	MatrixXd R = Q->topRows(xs.size());
	trsv(&R, &xs, CblasUpper, CblasTrans, CblasNonUnit);
}

/* Sparsification using Orthogonal transformations */
void tOrthogonal::fwd(){ormqr_trans(V, tau, &xs);}

void tOrthogonalD::fwd(){
	ormqr_trans(V, tau, &xs);
}
void tOrthogonalD::bwd(){
	ormqr_notrans(V, tau, &xs);
}

/* Merging in the cluster heirarchy */
void tMerge::fwd(){
	int k=0;
	for (auto c: parent->children){
		for (int i=0; i < c->cols(); ++i){
			(*parent->xt)[k] = (*c->xt)[i];
			++k;
		}
	}
}

void tMerge::bwd(){
	int k=0;
	for (auto c: parent->children){
		for (int i=0; i < c->rows(); ++i){
			(*c->xt)[i] = (*parent->xt)[k];
			++k;
		}
	}
	// assert(k== parent->get_x()->size());
}

/* Split between coarse and fine nodes */
void tSplit::fwd(){xsf = xsc_head.bottomRows(xsf.size());}
void tSplit::bwd(){xsc.bottomRows(xsf.size()) = xsf;}

void tSplitD::fwd(){
	xsf = xsc.middleRows(rank, ccols-rank);
	VectorXd xsc_rem = xsc.bottomRows(crows-ccols);
	xsc.middleRows(rank, crows-ccols) = xsc_rem;
	xsc.bottomRows(ccols-rank) = xsf;
}
void tSplitD::bwd(){
	// VectorXd xsf = xsc.bottomRows(ccols-rank);
	VectorXd xsc_rem = xsc.middleRows(rank, crows-ccols);
	xsc.middleRows(rank, ccols-rank) = xsf;
	xsc.bottomRows(crows-ccols) = xsc_rem;
}

