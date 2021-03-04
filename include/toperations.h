#ifndef TOPERATIONS_H
#define TOPERATIONS_H

#include <vector>
#include <list>
#include <set>
#include <string>
#include "util.h"
#include "cluster.h"
#include "edge.h"


/** An operation applied on the matrix **/
struct tOperation {
    public:
        virtual void fwd() = 0;
        virtual void bwd() = 0;
        virtual ~tOperation() {};
        virtual std::string Opname() {return "o";};
};

/* QR factorization */
struct tQR : public tOperation {
private:
	Cluster* c;
	int nrows; // nrows in Q
	int ncols; // ncols in Q = c->csize

	int ccols; // ncols in c
	std::vector<int> A21_indices; 
	std::vector<int> A12_indices;
public:
	tQR(Cluster* c_): c(c_), nrows(0), ncols(c->cols()), ccols(c->cols()){
		for (auto e: c->edgesOut){
			if (e->A21 != nullptr){ // For rectangular matrices, edge->A21 gets deleted (Q part)
				A21_indices.push_back(e->n2->rows());
            	nrows += e->A21->rows();
            }
		}

		for (auto e: c->edgesIn){
			assert(e->A21 != nullptr);
			A12_indices.push_back(e->A21->cols());
		}
	}
	void fwd();
	void bwd();
	std::string Opname(){return "QR";}
	~tQR(){}
};


/* Reassign rows between clusters */
struct tReassign : public tOperation{
private: 
	Cluster* c;
	Cluster* n;
	int nstart;
	int nrows;
	std::vector<int> indices;

public:
	tReassign(Cluster* c_, Eigen::VectorXi ind_,  Cluster* n_, int nstart_, int nrows_): c(c_), n(n_), nstart(nstart_), nrows(nrows_){
		for (int i=0; i < ind_.size(); ++i){
			indices.push_back(c_->cols() + ind_[i]);
		}
	}
	void fwd(){};
	void bwd();
	std::string Opname(){return "Reassign";}

	~tReassign(){}
};


/* Scaling */
struct tScale: public tOperation{
    private:
        Segment xs;
        Eigen::MatrixXd* R;
    public: 
        tScale(Cluster* c, Eigen::MatrixXd* R_) : xs(c->thead()), R(R_){}
        void fwd();
        void bwd(){};
		std::string Opname(){return "Scale";}

        ~tScale(){
            delete R;
        }
};

struct tScaleD: public tOperation{
private:
    Segment xs;
    Segment xsf;
    Eigen::MatrixXd* Q; //has both Q and R

public: 
    tScaleD(Cluster* c, Eigen::MatrixXd* Q_) : xs(c->thead()), xsf(c->tfull()), Q(Q_){};
    
    void fwd();
    void bwd(){};
	std::string Opname(){return "ScaleD";}
    ~tScaleD(){
        delete Q;
    }
};



/* Sparsification using Orthogonal transformations */
struct tOrthogonal : public tOperation {
private: 
	Segment xs;
    Eigen::MatrixXd* V;
    Eigen::VectorXd* tau;
public:
	tOrthogonal(Cluster* c, Eigen::MatrixXd* V_, Eigen::VectorXd* tau_) : 
		xs(c->thead()), V(V_), tau(tau_){}	
	void fwd();
	void bwd(){};
	std::string Opname(){return "Orthogonal";}

	~tOrthogonal(){
		delete V;
		delete tau;
	}
};

struct tOrthogonalD : public tOperation {
private: 
	Segment xs;
    Eigen::MatrixXd* V;
    Eigen::VectorXd* tau;
public:
	tOrthogonalD(Cluster* c, Eigen::MatrixXd* V_, Eigen::VectorXd* tau_) : 
		xs(c->thead()), V(V_), tau(tau_){}	
	void fwd();
	void bwd();
	std::string Opname(){return "OrthogonalD";}

	~tOrthogonalD(){
		delete V;
		delete tau;
	}
};

/* Merge in Cluster heirarchy */
struct tMerge : public tOperation {
private:
	Cluster* parent;
public:
	tMerge(Cluster* p): parent(p){}
	void fwd();
	void bwd();
	std::string Opname(){return "Merge";}

	~tMerge(){}

};

/* Split between coarse and fine nodes */
struct tSplit : public tOperation {
private:
	Segment xsc_head;
	Segment xsc;
	Segment xsf;
public:
	tSplit(Cluster* coarse, Cluster* fine) : xsc_head(coarse->thead()), xsc(coarse->tfull()), xsf(fine->tfull()){}
	void fwd();
	void bwd();
	std::string Opname(){return "Split";}

	~tSplit(){}
};

struct tSplitD : public tOperation {
private:
	Segment xsc;
	Segment xsf;
	int ccols;
	int crows;
	int rank;
public:
	tSplitD(Cluster* coarse, Cluster* fine, int rank_) : xsc(coarse->tfull()), xsf(fine->tfull()){
		crows = coarse->rows();
		ccols = coarse->cols();
		rank = rank_;
	}
	void fwd();
	void bwd();
	std::string Opname(){return "SplitD";}

	~tSplitD(){}
};


#endif