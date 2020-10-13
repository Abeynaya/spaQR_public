#include "cluster.h"
#include "edge.h"

typedef Eigen::SparseMatrix<double, 0, int> SpMat;

/* Print SepID */
std::ostream& operator<<(std::ostream& os, const SepID& s) {
    os << "(" << s.lvl <<  " " << s.sep << ")";
    return os;
}

/* Print ClusterID */
std::ostream& operator<<(std::ostream& os, const ClusterID& c) {
    os << "(" << c.self << "," << c.part << ":" << c.l << ";" << c.r << ")";
    return os;
}


SepID merge(SepID& s) {
    return SepID(s.lvl + 1, s.sep / 2);
}

ClusterID merge_if(ClusterID& c, int lvl) {
    auto left  = c.l.lvl < lvl ? merge(c.l) : c.l;
    auto right = c.r.lvl < lvl ? merge(c.r) : c.r;
    auto section = c.section;
    if (lvl > 0) 
        section /= 2;
    // auto part = c.part;
    auto part = c.self.lvl == lvl ? 2 : c.part;
    // auto part = c.part;

    return ClusterID(c.self, left, right, part, section);
}


int Cluster::get_order() const {return order;}
int Cluster::get_level() const {return id.self.lvl;} // Get level of the cluster
int Cluster::cols() const {return csize;}
int Cluster::rows() const {return rsize;}
int Cluster::get_cstart() const {return cstart;}
int Cluster::get_rstart() const {return rstart;}
int Cluster::original_rows() const {return rsize_org;};
int Cluster::original_cols() const {return csize_org;};
void Cluster::set_org(int r, int c) {rsize_org = r; csize_org = c;} 
ClusterID Cluster::get_id() const {return id;}
SepID Cluster::get_sepID(){return id.self;};
int Cluster::part(){return id.part;};


/*Heirarchy*/
Cluster* Cluster::get_parent(){return parent;}
ClusterID Cluster::get_parentid(){return parentid;}
void Cluster::set_parentid(ClusterID cid){parentid = cid;}
void Cluster::set_parent(Cluster* p){parent = p;}

void Cluster::add_children(Cluster* c){children.push_back(c);}
void Cluster::add_edgeOut(Edge* e){edgesOut.push_back(e);}
void Cluster::add_edgeIn(Edge* e){edgesIn.push_back(e);}
void Cluster::sort_edgesOut(bool reverse){
    if (reverse){
        edgesOut.sort([](Edge* a, Edge* b){return a->n2->get_order() > b->n2->get_order();});
    }
    else {
        edgesOut.sort([](Edge* a, Edge* b){return a->n2->get_order() < b->n2->get_order();});
    }
}

/*Elimination*/
void Cluster::set_tau(int r, int c){
    int k = std::min(r,c);
    this->tau = new Eigen::VectorXd(k);
    this->T = new Eigen::MatrixXd(k,k);
    this->tau->setZero();
    this->T->setZero();
}

Eigen::VectorXd* Cluster::get_tau(){return this->tau;}
Eigen::MatrixXd* Cluster::get_T(){return this->T;}

void Cluster::set_size(int r, int c){
    rsize = r;
    csize = c;
    delete this->x;
    this->x = new Eigen::VectorXd(r);
    this->x->setZero();

    delete this->xt;
    this->xt = new Eigen::VectorXd(r);
    this->xt->setZero();
}

void Cluster::reset_size(int r, int c){
    rsize = r;
    csize = c;
}

/* Solution to Linear System */
Segment Cluster::head(){
    assert(this->x != nullptr);
    return this->x->segment(0, this->csize);
}

Segment Cluster::full(){
    assert(this->x != nullptr);
    return this->x->segment(0, this->rsize);
}

Eigen::VectorXd* Cluster::get_x(){return this->x;}

void Cluster::set_vector(const Eigen::VectorXd& b){
    assert(x != nullptr);
    assert(this->get_rstart() >=0);
    for (int i=0; i < this->rsize_org; ++i){
        (*this->get_x())[i] = b[this->get_rstart()+i];
    }
}

void Cluster::set_vector_x(const Eigen::VectorXd& b){
    assert(x != nullptr);
    assert(this->get_cstart()>=0);
    for (int i=0; i < this->csize_org; ++i){
        (*this->get_x())[i] = b[this->get_cstart()+i];
    }
}


void Cluster::extract_vector(Eigen::VectorXd& soln){
    assert(x != nullptr);
    // std::cout << this->id << std::endl;
    for (int i=0; i < this->csize_org; ++i){
        soln[this->get_cstart()+i] = (*this->get_x())[i];
        // std::cout << (*this->get_x())[i] << "  " ;
    }
}


/* Destructor */
Cluster::~Cluster(){
    delete x;
    delete tau;
    delete T;
}


