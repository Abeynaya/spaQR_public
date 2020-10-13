#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>
#include <list>
#include <set>
#include <string>
#include "util.h"
#include "cluster.h"
#include "edge.h"

/** An operation applied on the matrix **/
struct Operation {
    public:
        virtual void fwd() = 0;
        virtual void bwd() = 0;
        virtual ~Operation() {};
        virtual std::string Opname() {return "o";};
};

/* QR factorization */
struct QR : public Operation {
private:
	Cluster* c;
	int nrows; // nrows in Q
	int ncols; // ncols in Q = c->csize

	int ccols; // ncols in c
	std::vector<int> A21_indices; 
	std::vector<int> A12_indices;

public:
	QR(Cluster* c_): c(c_), nrows(0), ncols(c->cols()), ccols(c->cols()){
		for (auto e: c->edgesOut){
			if (e->A21 != nullptr) { 
				A21_indices.push_back(e->n2->rows());
                nrows += e->A21->rows();
            }
            if (e->A12 != nullptr){
            	A12_indices.push_back(e->n2->cols());
            }
		}
	}
	void fwd();
	void bwd();
	std::string Opname(){return "QR";}
	~QR(){}
};


/* Reassign rows between clusters */
struct Reassign : public Operation{
private: 
	Cluster* c;
	// Eigen::VectorXd* cseg;
	Cluster* n;
	int nstart;
	int nrows;
	std::vector<int> indices;
public:
	Reassign(Cluster* c_, Eigen::VectorXi ind_,  Cluster* n_, int nstart_, int nrows_): c(c_), n(n_), nstart(nstart_), nrows(nrows_){
		for (int i=0; i < ind_.size(); ++i){
			indices.push_back(c_->cols() + ind_[i]);
		}
	}
	void fwd();
	void bwd(){};
	std::string Opname(){return "Reassign";}
	~Reassign(){}
};

/* Shift extra rows to the end after elimination */
struct Shift : public Operation {
private:
	// Cluster* c;
	Segment xsc;
	Segment xse; 
public:
	Shift(Cluster* c, Cluster* cnew): xsc(c->full()), xse(cnew->full()){}
	void fwd();
	void bwd();
	std::string Opname(){return "Shift";}
	~Shift(){}
};

/* Scaling */
struct Scale: public Operation{
private:
    Segment xs;
    Eigen::MatrixXd* R;
public: 
    Scale(Cluster* c, Eigen::MatrixXd* R_) : xs(c->head()), R(R_){}
    void fwd(){};
    void bwd();
	std::string Opname(){return "Scale";}
    ~Scale(){
        delete R;
    }
};

struct ScaleD: public Operation{
private:
    Segment xs;
    Segment xsf;
    Eigen::MatrixXd* Q; //has both Q and R
    Eigen::VectorXd* t;
public: 
    ScaleD(Cluster* c, Eigen::MatrixXd* Q_, Eigen::VectorXd* t_) : xs(c->head()), 
    															 xsf(c->full()), Q(Q_), t(t_){}
    void fwd();
    void bwd();
	std::string Opname(){return "ScaleD";}
    ~ScaleD(){
        delete Q;
        delete t;
    }
};


/* Sparsification using Orthogonal transformations */
struct Orthogonal : public Operation {
private: 
	Segment xs;
    Eigen::MatrixXd* V;
    Eigen::VectorXd* tau;
public:
	Orthogonal(Cluster* c, Eigen::MatrixXd* V_, Eigen::VectorXd* tau_) : 
		xs(c->head()), V(V_), tau(tau_){}	
	void fwd(){};
	void bwd();
	std::string Opname(){return "Orthogonal";}

	~Orthogonal(){
		delete V;
		delete tau;
	}
};

/* Sparsification using Orthogonal transformations when diag blocks are scaled */
struct OrthogonalD : public Operation {
private: 
	Segment xs;
    Eigen::MatrixXd* V;
    Eigen::VectorXd* tau;
public:
	OrthogonalD(Cluster* c, Eigen::MatrixXd* V_, Eigen::VectorXd* tau_) : 
		xs(c->head()), V(V_), tau(tau_){}	
	void fwd();
	void bwd();
	std::string Opname(){return "OrthogonalD";}

	~OrthogonalD(){
		delete V;
		delete tau;
	}
};

/* Sparsification using Eigenvalue decomposition */
struct EVD : public Operation {
    private: 
        Segment xs; 
        Eigen::MatrixXd* U; 
    public: 
        EVD(Cluster* c, Eigen::MatrixXd* U_) : xs(c->head()), U(U_) {}
        void fwd() {};
        void bwd();
		std::string Opname(){return "EVD";}
        
        ~EVD() {
            delete U;
        }
};

/* Merge in Cluster heirarchy */
struct Merge : public Operation {
private:
	Cluster* parent;
public:
	Merge(Cluster* p): parent(p){}
	void fwd();
	void bwd();
	std::string Opname(){return "Merge";}

	~Merge(){}

};

/* Split between coarse and fine nodes */
struct Split : public Operation {
private:
	Segment xsc_head;
	Segment xsc;
	Segment xsf;
public:
	Split(Cluster* coarse, Cluster* fine) : xsc_head(coarse->head()), xsc(coarse->full()), xsf(fine->full()){}
	void fwd();
	void bwd();
	std::string Opname(){return "Split";}

	~Split(){}
};

struct SplitD : public Operation {
private:
	Segment xsc;
	Segment xsf;
	int crows;
	int ccols;
	int rank; // new ccols
public:
	SplitD(Cluster* coarse, Cluster* fine, int rank_) : xsc(coarse->full()), xsf(fine->full()){
		crows = coarse->rows();
		ccols = coarse->cols();
		rank = rank_;
	}
	void fwd();
	void bwd();
	std::string Opname(){return "SplitD";}

	~SplitD(){}
};


#endif