#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h> 
#include <iomanip> 
#include <random>
#include "tree.h"
#include "partition.h"

using namespace std;
using namespace Eigen;

/* Smaller helper functions */
void Tree::set_tol(double tol){this->tol=tol;}
void Tree::set_hsl(double hsl){this->use_matching=hsl;};
void Tree::set_skip(int s){this->skip=s;};
void Tree::set_square(int s){this->square=s;};
void Tree::set_scale(int s){this->scale=s;};
void Tree::set_order(float s){this->order=s;};
void Tree::set_Xcoo(Eigen::MatrixXd* Xcoo) {this->Xcoo = Xcoo;}

/* Access to basic information */
int Tree::rows() const {return nrows;}
int Tree::cols() const {return ncols;}
int Tree::levels() const {return nlevels;}

tuple<int,int> Tree::topsize() {
    int top_csize = 0;
    int top_rsize = 0; // Number of rows
    for (auto self: this->bottom_current()){
        if (self->get_level() == this->nlevels-1){
            top_csize += self->cols();
            top_rsize += self->rows();
        }
    }
    return make_tuple(top_rsize, top_csize);
}


int Tree::get_new_order(){
    int o = this->max_order;
    this->max_order++;
    return o;
}

void initPermutation(int nlevels, const vector<ClusterID>& cmap, VectorXi& cperm){
    // Sort rows&columns according to ND ordering first and the cluster merging process then 
    vector<ClusterID> partmerged = cmap;
    auto compIJ = [&partmerged](int i, int j){return (partmerged[i] < partmerged[j]);};
    stable_sort(cperm.data(), cperm.data()+cperm.size(), compIJ); 
    for(int lvl=1; lvl < nlevels; ++lvl){
        transform(partmerged.begin(), partmerged.end(), partmerged.begin(), [&lvl](ClusterID s){return merge_if(s,lvl);});
        stable_sort(cperm.data(), cperm.data() + cperm.size(), compIJ); 
    }
}

void form_rmap(SpMat& A, vector<ClusterID>& rmap, const vector<ClusterID>& cmap, VectorXi& cperm){
    // cmap is actually cpermed
    auto t0 = wctime();
    int r = A.rows();
    SpMat At = col_perm(A, cperm).transpose();

    ClusterID last_cluster = cmap[A.cols()-1];
    int sum=0;
    
    for (int i=0; i < r; ++i){

        double max_val = -1;
        ClusterID rid = rmap[i];

        if (rid == ClusterID()){
            SpMat::InnerIterator it(At,i);
            for (; it; ){
                ClusterID cid = cmap[it.row()];
                auto it_next = it;
                ++it_next;
                double val = (it.value())*(it.value());
                while (it_next && cid == cmap[it_next.row()]){
                    val += (it_next.value())*(it_next.value());
                    ++it_next;
                }
                if (val > max_val){
                    max_val = val;
                    rid = cid;
                }
                it = it_next;
            }
            if (rid == ClusterID()){
                sum += 1;
                rid = last_cluster;
            } // probably a zero row, so assign to last cluster (top separator)
                
            rmap[i] = rid;
            assert(rid != ClusterID());
        }
        
    }
}

void form_rmap(SpMat& Ap, const vector<ClusterID>& cpermed, VectorXi& rperm, VectorXi& c2r_count, vector<vector<int>> c2r, bool matched){
    int r = Ap.rows();
    int c = Ap.cols();
    SpMat At = Ap.transpose();

    int sum=0;

    VectorXi is_matched(r);
    is_matched.setZero();
    if (matched){ // Matched by bipartite matching
        for (int i=0; i < c; ++i){
            is_matched[rperm[i]] = 1;
            c2r[i].push_back(rperm[i]);
            c2r_count[i]++;
        }
    }
    
    for (int i=0; i < r; ++i){ // all rows

        if(is_matched[i] == 0){ // is not_matched to a column yet
            double max_val = -1;
            int cidx = -1;

            SpMat::InnerIterator it(At,i);
            for (; it; ){
                ClusterID cid = cpermed[it.row()];
                auto it_next = it;
                ++it_next;
                double val = (it.value())*(it.value());
                while (it_next && cid == cpermed[it_next.row()]){
                    val += (it_next.value())*(it_next.value());
                    ++it_next;
                }
                if (val > max_val){
                    max_val = val;
                    cidx = it.row();
                }
                it = it_next;
            }

            if (cidx == -1){
                sum += 1;
                cidx = c-1; // last column
            } // probably a zero row, so assign to last cluster (top separator)
                
            assert(cidx != -1);
            c2r_count[cidx]++; // in the last column belonging to that cluster-- add the row
            c2r[cidx].push_back(i); 
        }
    }

    // Create rperm
    VectorXi rperm_copy = rperm;
    int start = 0;
    for (int i=0; i < c; ++i){
        int nr = c2r_count[i];
        for (int j=0; j < nr; ++j){
            assert(start < r);
            rperm[start] = c2r[i][j];
            ++start;
        }
    }
    assert(start == r);
}


/* Main helper functions */
/* Update fill-in (symbolic factorization) */
bool Tree::check_if_valid(ClusterID cid, ClusterID nid, set<SepID> col_sparsity){
    auto cself = cid.self;
    auto nself = nid.self;
    if (cself.lvl > nself.lvl) return false; // can never happen

    int d = nself.lvl - cself.lvl;
    int csep =  cself.sep/(int)(pow(2,d));

    if (csep != nself.sep) return false;

    // Check the left and right parts
    auto nleft = nid.l;
    auto nright = nid.r;

    int dl = nleft.lvl - cself.lvl;
    int dr = nright.lvl - cself.lvl;

    int sepl=0;
    int sepr=0;

    bool lyes=0;
    bool ryes=0;

    if (nleft.lvl >= cself.lvl) {
        sepl = cself.sep/(int)(pow(2,dl));
        lyes = (sepl == nleft.sep);
    }
    else {
        sepl = nleft.sep/(int)(pow(2,-dl));
        lyes = (sepl == cself.sep);
    }

    if (nright.lvl >= cself.lvl){
        sepr = cself.sep/(int)(pow(2,dr));
        ryes = (sepr == nright.sep);
    }
    else {
        sepr = nright.sep/(int)(pow(2,-dr));
        ryes = (sepr == cself.sep);
    }

    if (!(lyes) && !(ryes)) return false; // atleast one have to be true

    // cannot connect to any other separators
    bool geo = (this->Xcoo != nullptr);
    if (geo) {
        auto found = find(col_sparsity.begin(), col_sparsity.end(), nself);
        if (found == col_sparsity.end()) return false;

        auto foundl = find(col_sparsity.begin(), col_sparsity.end(), nleft);
        auto foundr = find(col_sparsity.begin(), col_sparsity.end(), nright);
        if (foundl == col_sparsity.end() && foundr == col_sparsity.end()) return false;
    }

    return true;

}

void Tree::sparsify_extra_rows(MatrixXd& C, int& rank, double& tol, bool rel){
    VectorXi jpvt = VectorXi::Zero(C.cols());
    VectorXd t= VectorXd::Zero(min(C.rows(), C.cols())); // cols > rows
    VectorXd rii;

    #ifdef USE_MKL
        /* Rank revealing QR on C with rrqr function */ 
        int bksize = min(C.cols(), max((long)3,(long)(0.1 * C.rows())));
        laqps(&C, &jpvt, &t, tol, bksize, rank);
        rii = C.topLeftCorner(rank, rank).diagonal();

    #else
        geqp3(&C, &jpvt, &t);
        rii = C.diagonal();
    #endif

    rank = choose_rank(rii, tol, rel);
    MatrixXd Crank = C.topRows(rank).triangularView<Upper>();
    Crank = Crank * (jpvt.asPermutation().transpose());
    C = Crank;
    return;
}

/* Add an edge between c1 and c2
 such that A(c2, c1) block is non-zero */
Edge* Tree::add_edge(Cluster* c1, Cluster* c2){
    Edge* e;

    MatrixXd* A = new MatrixXd(c2->rows(), c1->cols());
    A->setZero();
    e = new Edge(c1, c2, A);
    c1->cnbrs.insert(c2);
    c1->add_edgeOut(e);
    if(c1 != c2) {c2->add_edgeIn(e);}
    return e; 
}

void Tree::householder(vector<MatrixXd*> H, MatrixXd* Q, VectorXd* t, MatrixXd* T){
    geqrf(Q,t);
    
    if (this->square){ // Copy back to H 
        int curr_row = 0;
        for (int i=0; i< H.size(); ++i){
            *(H[i]) = Q->middleRows(curr_row, H[i]->rows());
            curr_row += H[i]->rows();
        }
    }
    else // Don't store Q. Need only R
        *(H[0]) = Q->topRows(H[0]->cols());

    larft(Q,t,T);
}

/* Update Q^T A after householder on interiors */
int Tree::update_cluster(Cluster* c, Cluster* n, MatrixXd* Q){
    vector<MatrixXd*> N;
    MatrixXd V(Q->rows(), n->cols());
    int curr_row = 0;

    for (auto edge: c->edgesOut){
        Edge* ecn;
        if (edge->A21 != nullptr){// Only the cnbrs + c itself
            Cluster* nbr_c = edge->n2; 
            auto found_out = find_if(n->edgesOut.begin(), n->edgesOut.end(), [&nbr_c](Edge* e){return (e->n2 == nbr_c );});

            // Fill-in => Allocate memory first
            if (found_out == n->edgesOut.end()){
                ecn = add_edge(n, nbr_c);
            }
            else {
                ecn = *(found_out);
            }
            
            assert(ecn->A21 != nullptr);
            N.push_back(ecn->A21);
            V.middleRows(curr_row, ecn->A21->rows()) = *ecn->A21; 
            curr_row += ecn->A21->rows();    
        }
    }

    larfb(Q, c->get_T(), &V);   

    // Copy back to N 
    curr_row = 0;
    for (int i=0; i< N.size(); ++i){
        *(N[i]) = V.middleRows(curr_row, N[i]->rows());
        curr_row += N[i]->rows();
    } 

    return 0;
}

/* Reassign extra rows to other clusters */
void Tree::reassign_rows(Cluster* c){
    if (c->rows() == c->cols()) return;
    assert(c->rows() > c->cols());

    int rsize = c->rows() - c->cols();
    // Assign row to a cluster based on maximum norm heuristic
    vector<tuple<Cluster*, double>> r2c(rsize, make_tuple(c, 0.0));
    for (auto edge: c->edgesIn){
        assert(edge->A21 != nullptr);
        auto n = edge->n1;
        bool is_cnbr = (find_if(c->edgesOut.begin(), c->edgesOut.end(), [&n](Edge* e){return e->n2 == n;}) != c->edgesOut.end());
        if (is_cnbr){
            MatrixXd A = edge->A21->middleRows(c->cols(), rsize);
            auto rnorm = A.rowwise().norm();
            for (int i=0; i<rsize; ++i){
                if (rnorm(i)>get<1>(r2c[i])) {
                    r2c[i] = make_tuple(n, rnorm(i));
                }
            }
        }   
    }

    // Sort rows according to cluster IDs (orders?)
    VectorXi row_perm  = VectorXi::LinSpaced(rsize, 0, rsize-1);
    auto compIJ = [&r2c](int i, int j){return ( (get<0>(r2c[i])->get_order()) < (get<0>(r2c[j]))->get_order());};
    stable_sort(row_perm.data(), row_perm.data() + row_perm.size(), compIJ); 


    PermMat P;
    P.indices() = row_perm;

    for (auto edge: c->edgesIn){
        assert(edge->A21 != nullptr);
        auto nold = edge->n1;
        MatrixXd A12t = P.transpose() * (edge->A21->middleRows(c->cols(),rsize));
        for (int i=0; i<rsize; ){
            auto nnew = get<0>(r2c[row_perm[i]]);
            int inext = i+1;
            while(inext < rsize && nnew == get<0>(r2c[row_perm[inext]])) {inext++;}
            int rinc = inext - i;

            if (nold == nnew) {
                auto found = find_if(nold->edgesOut.begin(), nold->edgesOut.end(), [&nnew](Edge* eold){return eold->n2 == nnew;});
                assert(found != nold->edgesOut.end());
                Edge* e = *found;
                assert(e->A21 != nullptr);

                e->A21->conservativeResize(e->A21->rows()+ rinc, NoChange);
                e->A21->bottomRows(rinc) = A12t.middleRows(i, rinc);

                nnew->resize_x(rinc);

                VectorXi ind = row_perm.segment(i, rinc);
                this->ops.push_back(new Reassign(c, ind, nnew, nnew->rows(), rinc)); // will fail if this not sorted before
                if (!this->square) this->tops.push_back(new tReassign(c, ind, nnew, nnew->rows(), rinc));

            } 
            else {
                auto found = find_if(nold->edgesOut.begin(), nold->edgesOut.end(), [&nnew](Edge* eold){return eold->n2 == nnew;});
                Edge* e = nullptr;
                if(found == nold->edgesOut.end()){ e =  add_edge(nold, nnew);}
                else {e = *found;}

                assert(e != nullptr);
                assert(e->A21 != nullptr);

                e->A21->conservativeResize(e->A21->rows()+rinc, NoChange);
                e->A21->bottomRows(rinc) = A12t.middleRows(i, rinc);
            }
            i = inext;
        }
    }

    // Change rsize, csize and padding with zeros
    for (auto edge: c->edgesOut){
        if (edge->n2 != c){
            auto n = edge->n2;
            auto found = find_if(n->edgesOut.begin(), n->edgesOut.end(), [&n](Edge* e){return e->n2 == n;});
            assert(found != n->edgesOut.end());

            auto mself = (*found)->A21;
            assert(mself != nullptr);
            n->reset_size(mself->rows(), mself->cols());
            for (auto e: n->edgesIn){
                if (!(e->n1->is_eliminated()) && e->A21 != nullptr && e->A21->rows() != n->rows()){
                    int old_r = e->A21->rows();
                    e->A21->conservativeResize(n->rows(), NoChange);
                    e->A21->bottomRows(n->rows()-old_r).setZero();
                }
            }
        }
    }
}

/* Merging interfaces */
void Tree::update_size(Cluster* snew){
    int rsize = 0;
    int csize = 0;
    for (auto sold: snew->children){
        sold->rposparent = rsize;
        sold->cposparent = csize;
        rsize += sold->rows();
        csize += sold->cols();
    }
    snew->set_size(rsize, csize); 
    snew->set_org(rsize, csize);

    if (this->ilvl >= skip) {
        this->profile.rank[this->ilvl-skip][nlevels- snew->get_sepID().lvl-1][snew->get_sepID().sep] += csize;
    }

    this->profile.aspect_ratio[this->ilvl+1].push_back((double)rsize/(double)csize);

    this->ops.push_back(new Merge(snew));
    if (!this->square) this->tops.push_back(new tMerge(snew));
}

void Tree::update_edges(Cluster* snew){
    set<Cluster*> edges_merged;
    for (auto sold: snew->children){
        for (auto eold : sold->edgesOut){
            auto n = eold->n2; 
            if (!(n->is_eliminated())){
                assert(n->get_parent() != nullptr);
                snew->cnbrs.insert(n->get_parent());
            }
        }
    }
   
    // Allocate memory and create new edges
    for (auto n: snew->cnbrs){
        add_edge(snew, n);
    }

    snew->sort_edgesOut();

    // Fill edges, delete previous edges
    for (auto sold: snew->children){
        for (auto eold: sold->edgesOut){
            auto nold = eold->n2;

            if (!(nold->is_eliminated())){
                auto nnew = nold->get_parent();
                auto found = find_if(snew->edgesOut.begin(), snew->edgesOut.end(), [&nnew](Edge* e){return e->n2 == nnew;});
                assert(found != snew->edgesOut.end());  // Must have the edge

                assert(eold->A21 != nullptr);
                assert(eold->A21->rows() == nold->rows());
                assert(eold->A21->cols() == sold->cols());

                (*found)->A21->block(nold->rposparent, sold->cposparent, nold->rows(), sold->cols()) = *(eold->A21);
                delete eold;
            }
            
        }
    }
} 
/* ********************* */


/* Split edges into fine and coarse edges after sparsification */
void Tree::split_edges(Cluster* c, Cluster* f){
    for (auto e: c->edgesIn){
        if (!(e->n1->is_eliminated())){
            assert(e->A21 != nullptr);
            MatrixXd* AT = new MatrixXd();
            *AT = e->A21->bottomRows(f->rows());
            assert(AT->cols() == e->n1->cols());

            Edge* efine = new Edge(e->n1, f, AT);

            // truncate block in Cluster c
            MatrixXd A21temp = e->A21->topRows(c->rows());
            *(e->A21) = A21temp;

            f->add_edgeIn(efine);
            f->rsparsity.insert(e->n1);
            e->n1->add_edgeOut(efine);
        }
    }

    auto found = find_if(c->edgesOut.begin(), c->edgesOut.end(), [&c](Edge* e){return e->n2 == c;});
    assert(found != c->edgesOut.end());
    Edge* e_self = *found;

    // Split e_self into 4 parts [ Acc Acf; Afc Aff]
    MatrixXd* Aff = new MatrixXd();
    MatrixXd* Afc = new MatrixXd();
    MatrixXd* Acf = new MatrixXd();

    *Aff = e_self->A21->bottomRightCorner(f->rows(), f->cols());
    *Afc = e_self->A21->bottomLeftCorner(f->rows(), c->cols());
    *Acf = e_self->A21->topRightCorner(c->rows(), f->cols());
    MatrixXd self_temp = e_self->A21->topLeftCorner(c->rows(), c->cols());
    (*e_self->A21) = self_temp;

    Edge* e_ff = new Edge(f, f, Aff);
    Edge* e_fc = new Edge(f, c, Acf);
    Edge* e_cf = new Edge(c, f, Afc);

    f->add_edgeOut(e_ff);
    f->add_edgeOut(e_fc);
    f->add_edgeIn(e_cf);
    c->add_edgeOut(e_cf);

    f->cnbrs.insert(c);
    f->cnbrs.insert(f);
    f->rsparsity.insert(c);

    /* Sort edges */   
    f->sort_edgesOut(true);
}

/* Block diagonal scaling */
void Tree::diagScale(Cluster* c){
    auto found = find_if(c->edgesOut.begin(), c->edgesOut.end(), [&c](Edge* e){return e->n2 == c;});
    assert(found != c->edgesOut.end());
    Edge* e = *found;

    MatrixXd* Q = new MatrixXd(c->rows(), c->cols());
    *Q = *(e->A21);
    VectorXd* t = new VectorXd(c->cols());
    geqrf(Q, t);

    MatrixXd* T = new MatrixXd(c->cols(), c->cols());
    larft(Q, t, T);

    MatrixXd* R = new MatrixXd(c->cols(), c->cols());
    *R = Q->topRows(c->cols()).triangularView<Upper>();

    (*e->A21).topRows(c->cols()) = MatrixXd::Identity(c->cols(), c->cols());
    (*e->A21).bottomRows(c->rows()-c->cols()) = MatrixXd::Zero(c->rows()-c->cols(),c->cols());

    this->nnzR += 0.5*c->cols()*c->cols();
    this->nnzH += (unsigned long long int)(c->rows()*c->cols());
    for (auto e: c->edgesIn){
        if (!e->n1->is_eliminated()){
            assert(e->A21 != nullptr);
            larfb(Q, T, e->A21);
        }
    }

    for (auto e: c->edgesOut){
        if (!e->n2->is_eliminated() && (e->n2 != c)){
            assert(e->A21 != nullptr);
            trsm_right(R, e->A21, CblasUpper, CblasNoTrans, CblasNonUnit);
        }
    }
 
    if (this->square) {
        this->ops.push_back(new ScaleD(c, Q, t));  
        delete R;
    }
    else {
        this->ops.push_back(new ScaleD(c, R));  
        this->tops.push_back(new tScaleD(c, R));
        delete Q;
    }
     
    delete T;
    return;
}

/* Sparsify only if left and right neighbors are eliminated */
bool Tree::want_sparsify(Cluster* c){
    if (c->get_level() > this->ilvl){
        SepID left = c->get_id().l;
        SepID right = c->get_id().r;
        if (left.lvl <= this->ilvl && right.lvl <= this->ilvl) 
            return true;
    }
    return false;
}

/* ************************************* */
/* Main functions */
void Tree::partition(SpMat& A){
    auto t0 = wctime();
    int r = A.rows();
    int c = A.cols();
    this->nrows = r; 
    this->ncols = c;

    bool geo = (Xcoo != nullptr);
    vector<ClusterID> cmap(c);
    
    if (geo){
        cout << "Using GeometricPartition... " << endl;
        cmap = GeometricPartitionAtA(A, this->nlevels, Xcoo, this->use_matching);
    }
    else{
        #ifdef USE_METIS
            cout << "Using Metis Partition... "<< endl;
            cmap = MetisPartition(A, this->nlevels);
        #else
            cout << "Using Hypergraph Partition... "<< endl;
            cmap = HypergraphPartition(A, this->nlevels);
        #endif
    }
    
    rperm = VectorXi::LinSpaced(r,0,r-1);
    cperm = VectorXi::LinSpaced(c,0,c-1);

    // Apply permutation
    vector<ClusterID> cpermed(c);
    initPermutation(nlevels, cmap, cperm); 
    transform(cperm.data(), cperm.data()+cperm.size(), cpermed.begin(), [&cmap](int i){return cmap[i];});

    SpMat Ap = col_perm(A, this->cperm);
    VectorXi c2r_count(c);
    c2r_count.setZero();
    vector<vector<int>> c2r(c);

    if (!this->square){
        #ifdef HSL_AVAIL
            if (this->use_matching){
                bipartite_row_matching(Ap, cpermed, rperm);
                form_rmap(Ap, cpermed, rperm, c2r_count, c2r, true);
            }
            else {
                form_rmap(Ap, cpermed, rperm, c2r_count, c2r, false);
            }
        #else
            form_rmap(Ap, cpermed, rperm, c2r_count, c2r, false);
        #endif
    }
    else{
        c2r_count.setOnes();

        #ifdef HSL_AVAIL
            if (this->use_matching){
                bipartite_row_matching(Ap, cpermed, rperm);
            }
            else { 
                rperm = cperm;
            }    
        #else 
            rperm = cperm;
        #endif
    }
   
    /* Set the initial clusters: lvl 0 in cluster heirarchy */
    int k=0;
    int l=0;
    for (; k < c;){
        int knext = k+1;
        ClusterID cid = cpermed[k];
        int nr2c = c2r_count[k];

        while(knext < c && cid == cpermed[knext]){
            nr2c += c2r_count[knext];
            knext++;
        }

        assert(nr2c != 0);
        Cluster* self = new Cluster(k, knext-k, l, nr2c, cid, get_new_order());
        this->bottoms[0].push_back(self);

        k = knext; 
        l += nr2c;
    }

    assert(l==r);


    /* Set parent and children and build the cluster heirarchy */
    for (int lvl=1; lvl < nlevels; ++lvl){
        auto begin = find_if(bottoms[lvl-1].begin(), bottoms[lvl-1].end(), [lvl](const Cluster* s){return s->get_level() >= lvl;});
        auto end = bottoms[lvl-1].end();

        for(auto self = begin; self != end; self++){
            assert((*self)->get_level() >= lvl);
            auto cid = (*self)->get_id();
            (*self)->set_parentid(merge_if(cid, lvl));
        }
        for (auto k = begin; k != end; ){
            auto idparent = (*k)->get_parentid();
            vector<Cluster*> children;
            children.push_back(*k);
            int children_csize = (*k)->cols();
            int children_rsize = (*k)->rows();
            int children_cstart = (*k)->get_cstart();
            int children_rstart = (*k)->get_rstart();
            k++;
            while(k != end && idparent == (*k)->get_parentid()){
                children.push_back(*k);
                children_csize += (*k)->cols();
                children_rsize += (*k)->rows();
                k++;
            }
            Cluster* parent = new Cluster(children_cstart, children_csize, children_rstart, children_rsize, idparent, get_new_order());
            // Include parent cluster with all the children
            for (auto c: children){
                assert(parent != nullptr);
                c->set_parent(parent); 
                parent->add_children(c);
            }

            bottoms[lvl].push_back(parent);
        }
    }
    auto t1 = wctime();
    cout << "Time to partition: " << elapsed(t0, t1) << endl;

}

void Tree::assemble(SpMat& A){
    auto astart = wctime();
    int r = A.rows();
    int c = A.cols();
    this->nnzA = A.nonZeros();
    
    // Permute the matrix
    SpMat ApT = col_perm(A, this->cperm).transpose();
    SpMat App = col_perm(ApT, this->rperm).transpose();
    App.makeCompressed(); 

    // Get CSC format  
    int nnz = App.nonZeros();
    VectorXi rowval = Map<VectorXi>(App.innerIndexPtr(), nnz);
    VectorXi colptr = Map<VectorXi>(App.outerIndexPtr(), c + 1);
    VectorXd nnzval = Map<VectorXd>(App.valuePtr(), nnz);

    // Get cmap and rmap
    vector<Cluster*> cmap(c);
    vector<Cluster*> rmap(r);
    for (auto self : bottom_original()){
        for(int k=self->get_cstart(); k< self->get_cstart()+self->cols(); ++k){cmap[k]=self;}
        for(int i=self->get_rstart(); i< self->get_rstart()+self->rows(); ++i){rmap[i]=self;}
    }

    // Assemble edges
    for (auto self: bottom_original()){
        set<Cluster*> cnbrs; // Non-zeros entries in the column belonging to self/another cluster
        for (int j= self->get_cstart(); j < self->get_cstart()+self->cols(); ++j){
            for (SpMat::InnerIterator it(App,j); it; ++it){
                Cluster* n = rmap[it.row()];
                cnbrs.insert(n);
            }
        }
        cnbrs.insert(self);

        for (auto nbr: cnbrs){
            MatrixXd* sA = new MatrixXd(nbr->rows(), self->cols());
            sA->setZero();
            block2dense(rowval, colptr, nnzval, nbr->get_rstart(), self->get_cstart(), nbr->rows(), self->cols(), sA, false); 
            Edge* e = new Edge(self, nbr, sA);
            self->add_edgeOut(e);
            if (self != nbr)
                nbr->add_edgeIn(e);
        }

        self->sort_edgesOut();
        self->cnbrs = cnbrs;
    }
    auto aend = wctime();
    cout << "Time to assemble: " << elapsed(astart, aend)  << endl;
    cout << "Aspect ratio of top separator: " << (double)get<0>(topsize())/(double)get<1>(topsize()) << endl;
    
}

int Tree::eliminate_cluster(Cluster* c){
    if (c->cols()==0){
        c->set_eliminated();
        return 0;
    }

    /* Do Householder reflections */
    vector<MatrixXd*> H;
    int num_rows=0;
    int num_cols=0;
    for (auto edge: c->edgesOut){
        if (edge->A21 != nullptr){
            assert(edge->n2->is_eliminated() == false);
            H.push_back(edge->A21);
            num_rows += edge->A21->rows();
            num_cols = edge->A21->cols(); // Must be the same
        }
    }
    assert(num_rows>0);
    assert(num_cols>0);

    MatrixXd* Q = new MatrixXd(num_rows, num_cols);
    concatenate(H, Q);
    c->set_tau(num_rows, num_cols);
    householder(H, Q, c->get_tau(), c->get_T());
    this->nnzR += (unsigned long long int)(0.5)*(c->cols())*(c->cols());
    this->nnzH += (unsigned long long int)(num_rows*num_cols);

    /* Update each cluster in row_sparsity */
    for (auto n: c->rsparsity){
        if (!(n->is_eliminated()) && c != n) {
            update_cluster(c, n, Q);
            this->nnzR += (unsigned long long int) (c->cols() * (n->cols()));
        }
    }

    if (!this->square){  // Free up memory of Q 
        for (auto edge: c->edgesOut){
            if (edge->n2 != c){
                assert(edge->A21 != nullptr);
                delete edge->A21;
                edge->A21 = nullptr;
            }
        }
    }

    this->ops.push_back(new QR(c));
    if (!this->square) this->tops.push_back(new tQR(c));

    c->set_eliminated();
    delete Q;
    return 0;
}

void Tree::get_sparsity(Cluster* c){
    set<Cluster*> row_sparsity;
    set<SepID> col_sparsity;

    for (auto edge: c->edgesOut){
        assert(edge->A21 != nullptr);
        col_sparsity.insert(edge->n2->get_sepID());
    }
    bool geo = (this->Xcoo != nullptr);

    double eps = 1e-14;

    for (auto edge: c->edgesOut){
        if (edge->A21 != nullptr){
            Cluster* n = edge->n2;
            row_sparsity.insert(n);
            for (auto ein: n->edgesIn){
                if (ein->A21 != nullptr && ein->n1 != c && !ein->n1->is_eliminated()){
                    Cluster* nin = ein->n1;

                    bool valid = true;
                    
                    valid = check_if_valid(c->get_id(), nin->get_id(), col_sparsity);
                    if (!geo && valid && 
                        !(nin->get_id().l == c->get_sepID() || nin->get_id().r == c->get_sepID()) ) {
                        valid = ((*(edge->A21)).transpose()*(*(ein->A21))).norm() > eps;                                    
                    }
                    
                    if (valid) {
                        row_sparsity.insert(nin);
                    }
                }
            }        
        }
    }
    c->rsparsity = row_sparsity; 
}

void Tree::merge_all(){
    current_bottom++;
    for (auto self: this->bottom_current()){
        this->update_size(self);
    }

    for (auto self: this->bottom_current()){
        this->update_edges(self);
    }  
}

void Tree::do_scaling(Cluster* c){ 
    diagScale(c);
}

void Tree::sparsify(Cluster* c){
    if (c->cols() == 0) return;
    vector<MatrixXd*> Apm;
    vector<MatrixXd*> Amp;
    MatrixXd* App;

    for (auto e: c->edgesIn){
        if (!(e->n1->is_eliminated())){
            assert(e->A21 != nullptr);
            Apm.push_back(e->A21);
        }
    }

    for (auto e: c->edgesOut){
        if (!(e->n2->is_eliminated())){
            assert(e->A21 != nullptr);
            if (e->n2 != c) {
                Amp.push_back(e->A21);
            }
            else {App = e->A21;}
        }
    }

    MatrixXd Apmc = Hconcatenate(Apm);
    MatrixXd Ampc = Vconcatenate(Amp);

    MatrixXd C = MatrixXd::Zero(Ampc.cols(), Ampc.rows() + Apmc.cols()); // Block to be compressed

    C.leftCols(Ampc.rows()) = Ampc.transpose();
    if (scale) { // column or diag scale
        C.middleCols(Ampc.rows(), Apmc.cols()) = (*App).transpose()*Apmc;
    }
    else {
        MatrixXd Ap = MatrixXd::Zero(App->rows()+Ampc.rows(),App->cols()); // column Ap
        Ap.middleRows(0, App->rows()) = (*App);
        Ap.middleRows(App->rows(), Ampc.rows()) = Ampc;

        VectorXd S = VectorXd(Ampc.cols());
        gesvd_values(&Ap, &S);
        double s = S(min(Ap.rows(), Ap.cols())-1);
        C.middleCols(Ampc.rows(), Apmc.cols()) = (1/s)*(*App).transpose()*Apmc;
    }

    MatrixXd Ccopy = C;

    int rank = 0;
    VectorXi jpvt = VectorXi::Zero(C.cols());
    VectorXd t = VectorXd(min(C.rows(), C.cols()));
    VectorXd rii;
   
    #ifdef USE_MKL
        /* Rank revealing QR on C with rrqr function */ 
        int bksize = min(C.cols(), max((long)3,(long)(0.1 * C.rows())));
        laqps(&C, &jpvt, &t, this->tol, bksize, rank);
        rii = C.topLeftCorner(rank, rank).diagonal();

    #else
        geqp3(&C, &jpvt, &t);
        rii = C.diagonal();
    #endif

    rank = choose_rank(rii, this->tol);
    if (rank >= C.rows()) return;

    /* Sparsification */
    MatrixXd* v = new MatrixXd(C.rows(), rank);
    VectorXd* h = new VectorXd(rank);
    *v = C.leftCols(rank);
    *h = t.topRows(rank);
    ormqr_notrans_right(v, h, App);
    this->ops.push_back(new Orthogonal(c, v, h)); // push before size of c changes
    if (!this->square) this->tops.push_back(new tOrthogonal(c, v, h)); // push before size of c changes


    /* Create fine nodes */ // keep it square
    int cf_cstart = c->get_cstart()+rank;
    int cf_csize = c->cols()-rank;
    int cf_rstart = c->get_rstart() + c->rows()- cf_csize;
    int cf_rsize = cf_csize;
    Cluster * cf = new Cluster(cf_cstart, cf_csize, cf_rstart, cf_rsize, c->get_id(), get_new_order());

    this->ops.push_back(new Split(c,cf));
    if (!this->square) this->tops.push_back(new tSplit(c,cf));


    c->reset_size(c->rows()-cf_csize, rank);

    /* Split edges between coarse and fine */
    MatrixXd Crank = C.topRows(rank).triangularView<Upper>();
    MatrixXd Ctemp = Crank * (jpvt.asPermutation().transpose());
    MatrixXd Cpart = Ctemp.leftCols(Ampc.rows());

    int curr_col =0;
    for (int i=0; i<Amp.size(); ++i){
        *(Amp[i]) = Cpart.middleCols(curr_col, Amp[i]->rows()).transpose();
        curr_col += Amp[i]->rows();
    }

    split_edges(c, cf);

    this->fine[this->ilvl].push_back(cf);
    eliminate_cluster(cf);
}

// If sparsify after diagScaling
void Tree::sparsifyD(Cluster* c){
    if (c->cols() == 0) return;
    vector<MatrixXd*> Apm;
    vector<MatrixXd*> Amp;
    Edge* e_self = nullptr;

    int spm=0;
    int smp=0;
    for (auto e: c->edgesIn){
        if (!(e->n1->is_eliminated())){
            assert(e->A21 != nullptr);
            Apm.push_back(e->A21);
            spm += e->A21->cols();
        }
    }

    for (auto e: c->edgesOut){
        if (!(e->n2->is_eliminated()) && e->n2 != c){
            assert(e->A21 != nullptr);
            Amp.push_back(e->A21);
            smp += e->A21->rows();
        }
        else if(e->n2 == c){
            e_self = e;
        }
    }

    assert(e_self != nullptr);

    
    MatrixXd C = MatrixXd::Zero(c->cols(), spm+smp); // Block to be compressed

    MatrixXd Apmc = Hconcatenate(Apm);
    MatrixXd Apmc_rest = MatrixXd::Zero(0,0);
    if (!this->square)
        Apmc_rest = Apmc.bottomRows(c->rows()-c->cols()); // Get the remaining rows that won't be affected
    

    C.leftCols(smp) = Vconcatenate(Amp).transpose();
    C.middleCols(smp, spm) = Apmc.topRows(c->cols());

    int rank = 0;
    VectorXi jpvt = VectorXi::Zero(C.cols());
    VectorXd t = VectorXd(min(C.rows(), C.cols()));
    VectorXd rii;
   
    #ifdef USE_MKL
        /* Rank revealing QR on C with rrqr function */ 
        int bksize = min(C.cols(), max((long)3,(long)(0.05 * C.rows())));
        laqps(&C, &jpvt, &t, this->tol, bksize, rank);
        rii = C.topLeftCorner(rank, rank).diagonal();

    #else
        geqp3(&C, &jpvt, &t);
        rii = C.diagonal();
    #endif

    rank = choose_rank(rii, this->tol);
    if (rank >= C.rows()) return;

    this->profile.neighbors[this->ilvl-skip][nlevels - c->get_sepID().lvl-1].push_back(C.cols());


    /* Sparsification */
    MatrixXd* v = new MatrixXd(C.rows(), rank);
    VectorXd* h = new VectorXd(rank);
    *v = C.leftCols(rank);
    *h = t.topRows(rank);

    this->ops.push_back(new OrthogonalD(c, v, h)); // push before size of c changes
    if (!this->square)
        this->tops.push_back(new tOrthogonalD(c, v, h)); // push before size of c changes


    this->nnzQ += (unsigned long long int)(C.rows()*rank);

    /* Create fine nodes */ // keep it square
    int cf_cstart = c->get_cstart()+rank;
    int cf_csize = c->cols()-rank;
    int cf_rstart = c->get_rstart() + c->cols() - cf_csize;
    int cf_rsize = cf_csize;
    Cluster * cf = new Cluster(cf_cstart, cf_csize, cf_rstart, cf_rsize, c->get_id(), get_new_order());

    
    this->ops.push_back(new SplitD(c, cf, rank));
    if (!this->square)
        this->tops.push_back(new tSplitD(c, cf, rank));


    /* Split edges between coarse and fine */
    MatrixXd Crank = C.topRows(rank).triangularView<Upper>();
    MatrixXd Ctemp = Crank * (jpvt.asPermutation().transpose());
    Crank = Ctemp;

    assert(e_self->n2 == c && e_self->n1 == c);
    MatrixXd self = MatrixXd::Zero(c->rows()-cf_rsize, rank);
    self.topRows(rank) = e_self->A21->block(0,0,rank,rank); // Identity

    int rank_rows = rank + Apmc_rest.rows(); 
    c->reset_size(rank_rows, rank);

    int curr_col =0;
    for (int i=0; i<Amp.size(); ++i){
        *(Amp[i]) = Crank.middleCols(curr_col, Amp[i]->rows()).transpose();
        curr_col += Amp[i]->rows();
    }
    int count = curr_col;


    for (int i=0; i<Apm.size(); ++i){
        Apm[i]->conservativeResize(c->rows(), NoChange);
        (*Apm[i]).topRows(c->cols()) = Crank.middleCols(curr_col, Apm[i]->cols());

        if (!this->square)
            (*Apm[i]).bottomRows(c->rows()-c->cols()) = Apmc_rest.middleCols(curr_col - count, Apm[i]->cols());
        curr_col += Apm[i]->cols();
    }

    *(e_self->A21) = self.topRows(c->rows());

    this->fine[this->ilvl].push_back(cf);
    cf->set_eliminated();
}

// Improved scheme - store E2: Use if order=1.5
void Tree::sparsifyD_imp(Cluster* c){
    if (c->cols() == 0) return;
    vector<MatrixXd*> Apm;
    vector<MatrixXd*> Amp;
    Edge* e_self = nullptr;

    int spm=0;
    int smp=0;
    for (auto e: c->edgesIn){
        if (!(e->n1->is_eliminated())){
            assert(e->A21 != nullptr);
            Apm.push_back(e->A21);
            spm += e->A21->cols();
        }
    }

    for (auto e: c->edgesOut){
        if (!(e->n2->is_eliminated()) && e->n2 != c){
            assert(e->A21 != nullptr);
            Amp.push_back(e->A21);
            smp += e->A21->rows();
        }
        else if(e->n2 == c){
            e_self = e;
        }
    }

    assert(e_self != nullptr);

    MatrixXd Apmc = Hconcatenate(Apm);
    MatrixXd C = MatrixXd::Zero(c->cols(), spm+smp); // Block to be compressed
    MatrixXd Apmc_rest = MatrixXd::Zero(0,0);
    if (!this->square)
        Apmc_rest = Apmc.bottomRows(c->rows()-c->cols()); // Get the remaining rows that won't be affected

    MatrixXd Apmc_top = Apmc.topRows(c->cols()); // create a copy
    
    C.leftCols(smp) = Vconcatenate(Amp).transpose();
    C.middleCols(smp, spm) = Apmc.topRows(c->cols());

    int rank = 0;
    VectorXi jpvt = VectorXi::Zero(C.cols());
    VectorXd t = VectorXd(min(C.rows(), C.cols()));
    VectorXd rii;
   
    #ifdef USE_MKL
        /* Rank revealing QR on C with rrqr function */ 
        int bksize = min(C.cols(), max((long)3,(long)(0.05 * C.rows())));
        laqps(&C, &jpvt, &t, this->tol, bksize, rank);
        rii = C.topLeftCorner(rank, rank).diagonal();

    #else
        geqp3(&C, &jpvt, &t);
        rii = C.diagonal();
    #endif
    
    rank = choose_rank(rii, this->tol);
    if (rank >= C.rows()) return;

    this->profile.neighbors[this->ilvl-skip][nlevels - c->get_sepID().lvl-1].push_back(C.cols());

    /* Sparsification */
    MatrixXd* v = new MatrixXd(C.rows(), rank);
    VectorXd* h = new VectorXd(rank);
    *v = C.leftCols(rank);
    *h = t.topRows(rank);

    this->ops.push_back(new OrthogonalD(c, v, h)); // push before size of c changes
    if (!this->square)
        this->tops.push_back(new tOrthogonalD(c, v, h)); // push before size of c changes

    /* Create fine nodes */ // keep it square
    int cf_cstart = c->get_cstart()+rank;
    int cf_csize = c->cols()-rank;
    int cf_rstart = c->get_rstart() + c->cols()- cf_csize;
    int cf_rsize = cf_csize;
    Cluster * cf = new Cluster(cf_cstart, cf_csize, cf_rstart, cf_rsize, c->get_id(), get_new_order());

    
    this->ops.push_back(new SplitD(c, cf, rank));
    if (!this->square)
        this->tops.push_back(new tSplitD(c, cf, rank));

    this->nnzQ += (unsigned long long int)(C.rows()*rank);

    c->reset_size(c->rows()-cf_csize, rank);

    /* Split edges between coarse and fine */
    MatrixXd Crank = C.topRows(rank).triangularView<Upper>();
    MatrixXd Ctemp = Crank * (jpvt.asPermutation().transpose());
    Crank = Ctemp;

    int curr_col =0;
    for (int i=0; i<Amp.size(); ++i){
        *(Amp[i]) = Crank.middleCols(curr_col, Amp[i]->rows()).transpose();
        curr_col += Amp[i]->rows();
    }
    int count = curr_col;

    for (int i=0; i<Apm.size(); ++i){
        Apm[i]->conservativeResize(c->rows(), NoChange);
        (*Apm[i]).topRows(c->cols()) = Crank.middleCols(curr_col, Apm[i]->cols());
        if (!this->square)
            (*Apm[i]).bottomRows(c->rows()-c->cols()) = Apmc_rest.middleCols(curr_col - count, Apm[i]->cols());
        curr_col += Apm[i]->cols();
    }

    MatrixXd* Aff = new MatrixXd(cf->rows(), cf->cols());
    *Aff = (e_self)->A21->topLeftCorner(cf->cols(), cf->cols()); // Identity

    MatrixXd self = (e_self)->A21->bottomRightCorner(c->rows(), c->cols());
    *(e_self->A21) = self;

    Edge* eff = new Edge(cf, cf, Aff);
    cf->add_edgeOut(eff);

    /* Add edges to fine nodes - storing the E2 term */
    ormqr_trans_left(v, h, &Apmc_top);
    curr_col= 0;
    for (auto e: c->edgesIn){
        if (!(e->n1->is_eliminated())){
            assert(e->A21 != nullptr);
            auto m = e->n1;
            MatrixXd* Afm = new MatrixXd(cf->rows(), m->cols());
            *Afm = Apmc_top.block(rank ,curr_col, cf->rows() , m->cols());

            Edge* e = new Edge(m, cf, Afm);
            cf->add_edgeIn(e);
            m->add_edgeOut(e);
            curr_col += m->cols();
        }
    }


    this->fine[this->ilvl].push_back(cf);
    cf->set_tau(cf->rows(), cf->cols());
    this->ops.push_back(new QR(cf));
    if (!this->square) this->tops.push_back(new tQR(cf));

    cf->set_eliminated();
}

// Sparsify only the extra rows after diagScaling
void Tree::sparsify_extra(Cluster* c){
    if (c->cols() == 0) return;
    if (c->rows() <= c->cols()) return;
    // do_scaling before calling this function -- will not work otherwise
    vector<MatrixXd*> Apm;

    int spm=0;
    for (auto e: c->edgesIn){
        if (!(e->n1->is_eliminated())){
            assert(e->A21 != nullptr);
            Apm.push_back(e->A21);
            spm += e->A21->cols();}
    }

    MatrixXd C = MatrixXd::Zero(c->cols(), spm); // Block to be compressed

    MatrixXd Apmc = Hconcatenate(Apm);
    MatrixXd Apmc_rest = Apmc.bottomRows(c->rows()-c->cols()); // Get the remaining rows to sparsify

    int rank = c->cols();
    int rank_rows = 0;
    
    if (Apmc_rest.rows() >0){ 
        sparsify_extra_rows(Apmc_rest, rank_rows, this->tol);
        rank_rows += rank; 
    }

    c->reset_size(rank_rows, rank); // updates c->rows() and c->cols()

    // Resize the diagonal block
    auto found = find_if(c->edgesOut.begin(), c->edgesOut.end(), [&c](Edge* e){return e->n2 == c;});
    assert(found != c->edgesOut.end());
    Edge* eself = *(found);
    MatrixXd* self = eself->A21; 
    MatrixXd self_temp = self->topRows(rank_rows);
    *self = self_temp;

    int curr_col=0;

    for (int i=0; i<Apm.size(); ++i){
        Apm[i]->conservativeResize(c->rows(), NoChange);
        (*Apm[i]).topRows(c->cols()) = Apmc.block(0,curr_col, c->cols(), Apm[i]->cols());
        
        (*Apm[i]).bottomRows(c->rows()-c->cols()) = Apmc_rest.middleCols(curr_col, Apm[i]->cols());
        curr_col += Apm[i]->cols();
    }
    return;
}

int Tree::factorize(){
    auto fstart = wctime();
    for (this->ilvl=0; this->ilvl < nlevels; ++ilvl){
        // Get rsparsity for the clusters to be eliminated next
        auto vstart = wctime();
        {
            for (auto self: this->bottom_current()){
                if (self->get_level() == this->ilvl){
                    this->get_sparsity(self);
                }
            }
        }
        auto vend = wctime();

        auto estart = wctime();
        // Eliminate
        {      
            for (auto self: this->bottom_current()){
                if (self->get_level() == this->ilvl){
                    assert(self->is_eliminated() == false);
                    auto elmn0 = wctime();
                    this->eliminate_cluster(self);
                    auto elmn1 = wctime();
                    this->profile.time_elmn[this->ilvl][self->get_sepID().sep] += (elapsed(elmn0,elmn1));
                }
            }

        }
        auto eend = wctime();
        

        // Shift rows
        auto shstart = wctime();
        {
            // Move extra rows to additional clusters before sparsification/merging process begins
            if (this->ilvl < nlevels-1){
                for (auto self: this->bottom_current()){
                    if (self->is_eliminated() && !(this->square)){
                        assert(self->get_level() == this->ilvl);
                        this->reassign_rows(self);
                    }
                }
            }
        }
        auto shend = wctime();

        // Scale
        auto scstart = wctime();
        {
            if (this->scale && this->ilvl < nlevels -2  && this->ilvl >= skip && this->tol != 0  ){ //&& ((this->ilvl-skip)%2 == 0)

                for (auto self: this->bottom_current()){
                    if (want_sparsify(self)){
                        auto scale0 = wctime();
                        do_scaling(self);
                        auto scale1 = wctime();
                        this->profile.time_scale[this->ilvl-skip][nlevels - self->get_sepID().lvl-1].push_back(elapsed(scale0,scale1));
                    }
                }
            }
        }
        auto scend = wctime();

        // Sparsify
        auto spstart = wctime();
        {
            if (this->ilvl >= skip && this->ilvl < nlevels -2  && this->tol != 0){ 
                for (auto self: this->bottom_current()){
                    if (want_sparsify(self)){
                        if (this->scale  && !(this->square)){
                            sparsify_extra(self);
                        } 
                    }
                }
            
                for(auto self: this->bottom_current()){
                   
                    if (want_sparsify(self) ){
                        this->profile.rank_before[this->ilvl-skip][nlevels - self->get_sepID().lvl-1].push_back(self->cols());

                        auto spars0 = wctime();
                        if (scale){
                            if (this->order == 1.5) sparsifyD_imp(self);
                            else sparsifyD(self); 
                        }
                        else{// No scaling done
                            sparsify(self);
                        }
                        
                        auto spars1 = wctime();
                        this->profile.time_spars[this->ilvl-skip][nlevels - self->get_sepID().lvl-1].push_back(elapsed(spars0,spars1));
                        this->profile.rank_after[this->ilvl-skip][nlevels - self->get_sepID().lvl-1].push_back(self->cols());
                    } 
                }
            }
        }
        auto spend = wctime();

        // Merge 
        auto mstart = wctime();
        {   
            if (this-> ilvl < nlevels-1){
                merge_all();
            }
        }
        auto mend = wctime();
        

        cout << "lvl: " << ilvl << "    " ;  
        cout << fixed << setprecision(3)  
             << "update fill-in: " <<   elapsed(vstart,vend) << "  "
             << "elmn: " <<   elapsed(estart,eend) << "  "
             << "shift: " <<  elapsed(shstart,shend) << "  "
             << "scale: " << elapsed(scstart,scend) << "  "
             << "sprsfy: " << elapsed(spstart,spend) << "  "
             << "merge: " << elapsed(mstart,mend) << "  " 
             << "size top_sep: " <<  get<0>(topsize()) << ", " << get<1>(topsize()) << "   "
             << "a.r top_sep: " << (double)get<0>(topsize())/(double)get<1>(topsize()) << endl;

    }   
    auto fend = wctime();
    cout << "Tolerance set: " << scientific << this->tol << endl;
    cout << "Time to factorize:  " << elapsed(fstart,fend) << endl;
    cout << "Size of top separator: " <<  (*this->bottoms[nlevels-1].begin())->cols() << endl;
    cout << "nnzA: " << this->nnzA << " nnzR: " << this->nnzR << endl;
    cout << "nnzH: " << this->nnzH << " nnzQ: " << this->nnzQ << endl;

    return 0;
}

/* For linear systems */
void Tree::solve(VectorXd b, VectorXd& x) const{
    // Permute the rhs according to this->rperm
    b = this->rperm.asPermutation().transpose()*b;

    // Set solution
    for (auto cluster: bottom_original()){
        cluster->set_vector(b);
    }

    // Fwd
    for(auto io = ops.begin(); io != ops.end(); ++io) {
        (*io)->fwd();
    }
    
    // Bwd
    for(auto io = ops.rbegin(); io != ops.rend(); ++io) {
            (*io)->bwd();
    }

    // Extract solution
    for(auto cluster : bottom_original()) {
        cluster->extract_vector(x);
    }

    // Permute back
    x = this->cperm.asPermutation() * x; 
}

/* For least square problems 
 * Solve the normal equations 
 */
void Tree::solve_nrml(VectorXd b, VectorXd& x) const{ // b = A'*b
    // A^T y = b
    // Permute the RHS according to this->cperm
    b = this->cperm.asPermutation().transpose()*b;

    // Set solution
    for (auto cluster: bottom_original()){
        cluster->tset_vector(b);
    }

    // Fwd
    for(auto io = tops.begin(); io != tops.end(); ++io) {
            (*io)->fwd();
    }

    // Bwd
    for(auto io = tops.rbegin(); io != tops.rend(); ++io) {
        if ((*io)->Opname()!="QR"  && (*io)->Opname()!="ScaleD" && (*io)->Opname()!="OrthogonalD")
            (*io)->bwd();
    }

    // Extract solution
    for(auto cluster : bottom_original()) {
        cluster->textract_vector(); // assign to x from xt
    }

    // A x = y
    // Fwd
    for(auto io = ops.begin(); io != ops.end(); ++io) {
        if ((*io)->Opname()!="QR" && (*io)->Opname()!="ScaleD" && (*io)->Opname()!="OrthogonalD")
            (*io)->fwd();
    }
    
    // Bwd
    for(auto io = ops.rbegin(); io != ops.rend(); ++io) {
        (*io)->bwd();
    }

    // Extract solution
    for(auto cluster : bottom_original()) {
        cluster->extract_vector(x);
    }

    // Permute back
    x = this->cperm.asPermutation() * x;

}

/* Destructor */
Tree:: ~Tree(){
    for (auto o : this->ops){
        delete o;
    }

    for(int lvl = 0; lvl < this->nlevels; lvl++) {
        for(auto s : bottoms[lvl]){
            if (s->get_level() == lvl){
                for(auto e : s->edgesOut) {
                    delete e;
                }
            }
                
            delete s;
           
        }
    }

    for(int lvl = skip; lvl < this->nlevels-1; lvl++) {
        for(auto s : fine[lvl]){
            for(auto e : s->edgesOut) {
                delete e;
            }
            delete s;
        }
    }
}
