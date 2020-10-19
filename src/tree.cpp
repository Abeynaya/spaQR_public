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

void Tree::set_scale(int s){this->scale=s;};
void Tree::set_Xcoo(Eigen::MatrixXd* Xcoo) {this->Xcoo = Xcoo;}
void Tree::set_output(int s, string n){this->output_mat = s; this->name = n;};

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

/* Add an edge between c1 and c2
 * Assumes c1->get_order() <= c2->get_order()
 * if (A21) add block A21
 * if (A12) add block A12 */
Edge* Tree::add_edge(Cluster* c1, Cluster* c2, bool A21, bool A12){
    assert(c1->get_order() <= c2->get_order());
    Edge* e;

    if (A21 && A12){
        MatrixXd* AT = new MatrixXd(c1->rows(), c2->cols());
        AT->setZero();
        MatrixXd* A = new MatrixXd(c2->rows(), c1->cols());
        A->setZero();

        e = new Edge(c1, c2, A, AT);
        c1->cnbrs.insert(c2);
        c1->rnbrs.insert(c2);
    }
    else if(A21){
        MatrixXd* A = new MatrixXd(c2->rows(), c1->cols());
        A->setZero();
        assert(A12 == false); e = new Edge(c1, c2, A, nullptr);
        c1->cnbrs.insert(c2);
    }
    else {
        MatrixXd* AT = new MatrixXd(c1->rows(), c2->cols());
        AT->setZero();
        assert(A12 == true);
        assert(A21 == false); e = new Edge(c1, c2, nullptr, AT);
        c1->rnbrs.insert(c2);
    }

    c1->add_edgeOut(e);
    if(c1 != c2) {c2->add_edgeIn(e);}
    return e; 
}

/* Main helper functions */
void Tree::householder(vector<MatrixXd*> H, MatrixXd* Q, VectorXd* t, MatrixXd* T){
    geqrf(Q,t);
    
    int curr_row = 0;
    for (int i=0; i< H.size(); ++i){
        *(H[i]) = Q->middleRows(curr_row, H[i]->rows());
        curr_row += H[i]->rows();
    }


    larft(Q,t,T);
}

int Tree::update_cluster(Cluster* c, Cluster* n, MatrixXd* Q){
    vector<MatrixXd*> N(c->cnbrs.size());
    MatrixXd V(Q->rows(), n->cols());
    int curr_row = 0;

    int counter = 0;
    bool is_present = false;
    for (auto edge: c->edgesOut){
        if (edge->n2 == n && edge->A12 != nullptr){
            N[counter] = (edge->A12);
            V.middleRows(curr_row, edge->A12->rows()) = *(edge->A12); 
            curr_row += edge->A12->rows();
            counter++;
            is_present = true;
            break;
        }
    }

    // Fill-in block => Allocate memory
    if (!is_present){
        assert(c->get_order() < n->get_order());
        Edge* enew = add_edge(c, n, false, true);
        N[counter]=(enew->A12);
        V.middleRows(curr_row, enew->A12->rows()) = *enew->A12; 
        curr_row += enew->A12->rows();
        counter++;
    }

    for (auto edge: c->edgesOut){
        Edge* ecn;
        if (edge->n2 != c && edge->A21 != nullptr){// Only the cnbrs
            Cluster* nbr_c = edge->n2;
            auto found_out = find_if(n->edgesOut.begin(), n->edgesOut.end(), [&nbr_c](Edge* e){return (e->n2 == nbr_c );});
            auto found_in = n->edgesIn.end();
            if (found_out == n->edgesOut.end()){ 
                found_in = find_if(n->edgesIn.begin(), n->edgesIn.end(), [&nbr_c](Edge* e){return (e->n1 == nbr_c );});
                if (found_in != n->edgesIn.end()) ecn = *found_in;
            }
            else {
                ecn = *(found_out);
            }

            // Fill-in => Allocate memory first
            if (found_out == n->edgesOut.end() &&  found_in == n->edgesIn.end()){
                if (nbr_c->get_order() < n->get_order()) {ecn = add_edge(nbr_c, n, false, true);}
                else {ecn = add_edge(n, nbr_c, true, false);} 
            }
            

            if (nbr_c->get_order() < n->get_order()){ //edgeIn to n
                if (ecn->A12 == nullptr) {
                    MatrixXd* A_T = new MatrixXd(nbr_c->rows(), n->cols());
                    A_T->setZero();
                    ecn->A12 = A_T;
                    nbr_c->rnbrs.insert(n);
                }
                // N.push_back(ecn->A12);
                N[counter] = ecn->A12;
                V.middleRows(curr_row, ecn->A12->rows()) = *ecn->A12; 
                curr_row += ecn->A12->rows();
                counter++;

            }
            else {
                if (ecn->A21 == nullptr){
                    MatrixXd* A = new MatrixXd(nbr_c->rows(), n->cols());
                    A->setZero();
                    ecn->A21 = A;
                    n->cnbrs.insert(nbr_c);
                }
                // N.push_back(ecn->A21);
                N[counter] = ecn->A21;
                V.middleRows(curr_row, ecn->A21->rows()) = *ecn->A21; 
                curr_row += ecn->A21->rows();
                counter++;

            }
        }
    }

    // auto V = Vconcatenate(N);
    larfb(Q, c->get_T(), &V);   

    // Copy back to N 
    curr_row = 0;
    for (int i=0; i< N.size(); ++i){
        *(N[i]) = V.middleRows(curr_row, N[i]->rows());
        curr_row += N[i]->rows();
    } 

    return 0;
}


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

}

void Tree::update_edges(Cluster* snew){
    set<Cluster*> edges_merged;
    for (auto sold: snew->children){
        for (auto n : sold->cnbrs){
            assert(n->get_parent() != nullptr);
            snew->cnbrs.insert(n->get_parent());
        }

        for (auto n : sold->rnbrs){
            assert(n->get_parent() != nullptr);
            if (n->get_parent() != snew && !(n->is_eliminated())) {
                snew->rnbrs.insert(n->get_parent());
            }
        }
    }

    // Allocate memory and create new edges
    for (auto n: snew->cnbrs){
        if (snew->get_order() <= n->get_order() && snew->rnbrs.find(n) != snew->rnbrs.end() ){ // Both cnbr and rnbr
            assert(snew != n);
            add_edge(snew, n, true, true);
        }
        else if (snew->get_order() <= n->get_order()) {add_edge(snew, n, true, false);}
    }

    for (auto n: snew->rnbrs){
        if (snew->cnbrs.find(n) == snew->cnbrs.end() && snew->get_order() < n->get_order()) { // Only a rnbr
            add_edge(snew, n, false, true);
        }
    }
    snew->sort_edgesOut();

    // Fill edges, delete previous edges
    for (auto sold: snew->children){
        for (auto eold: sold->edgesOut){
            auto nold = eold->n2;
            auto nnew = nold->get_parent();
            auto found = find_if(snew->edgesOut.begin(), snew->edgesOut.end(), [&nnew](Edge* e){return e->n2 == nnew;});
            assert(found != snew->edgesOut.end());  // Must have the edge

            if (eold->A21 != nullptr){
                assert(eold->A21->rows() == nold->rows());
                assert(eold->A21->cols() == sold->cols());
                (*found)->A21->block(nold->rposparent, sold->cposparent, nold->rows(), sold->cols()) = *(eold->A21); delete eold->A21;
            }

            if (snew == nnew && eold->A12 != nullptr){
                (*found)->A21->block(sold->rposparent, nold->cposparent, sold->rows(), nold->cols()) = *(eold->A12); delete eold->A12;
            }
            else if (eold->A12 != nullptr){
                (*found)->A12->block(sold->rposparent,nold->cposparent, sold->rows(), nold->cols()) = *(eold->A12); delete eold->A12;
            }
            delete eold;
        }
    }
}  

void Tree::split_edges(Cluster* c, Cluster* f){
    for (auto e: c->edgesIn){
        if (e->A21 != nullptr && !(e->n1->is_eliminated())){
            MatrixXd* AT = new MatrixXd();
            *AT = e->A21->bottomRows(f->rows());
            assert(AT->cols() == e->n1->cols());

            Edge* efine = new Edge(f, e->n1, nullptr, AT);

            // truncate block in Cluster c
            MatrixXd A21temp = e->A21->topRows(c->rows());
            *(e->A21) = A21temp;

            f->add_edgeOut(efine);
            f->rnbrs.insert(e->n1);
        }
    }

    for (auto e: c->edgesOut){
        if (e->A12 != nullptr && !(e->n2->is_eliminated())){
            MatrixXd* AT = new MatrixXd();
            *AT = e->A12->bottomRows(f->rows());
            Edge* efine = new Edge(f, e->n2, nullptr, AT);

            MatrixXd A12temp = e->A12->topRows(c->rows());
            *(e->A12) = A12temp;

            f->add_edgeOut(efine);
            f->rnbrs.insert(e->n2);
        }
    }
    Edge* e_self = *(c->edgesOut.begin());
    assert(e_self->n2 == c);

    // Split e_self into 4 parts [ Acc Acf; Afc Aff]
    MatrixXd* Aff = new MatrixXd();
    MatrixXd* Afc = new MatrixXd();
    MatrixXd* Acf = new MatrixXd();

    *Aff = e_self->A21->bottomRightCorner(f->rows(), f->cols());
    *Afc = e_self->A21->bottomLeftCorner(f->rows(), c->cols());
    *Acf = e_self->A21->topRightCorner(c->rows(), f->cols());
    MatrixXd self_temp = e_self->A21->topLeftCorner(c->rows(), c->cols());
    (*e_self->A21) = self_temp;

    Edge* e_ff = new Edge(f, f, Aff, nullptr);
    Edge* e_fc = new Edge(f, c, Acf, Afc);

    f->add_edgeOut(e_ff);
    f->add_edgeOut(e_fc);

    f->cnbrs.insert(c);
    f->cnbrs.insert(f);
    f->rnbrs.insert(c);

    /* Sort edges */   
    f->sort_edgesOut(true);
}

void Tree::diagScale(Cluster* c){
    Edge* e = *(c->edgesOut.begin()); 
    if (c->rows() < c->cols()) {
        cout << "Less rows than cols in scale_diagBlock" << endl;
        return;
    }

    MatrixXd* Q = new MatrixXd(c->rows(), c->cols());
    *Q = *(e->A21);
    VectorXd* t = new VectorXd(c->cols());
    geqrf(Q, t);

    MatrixXd* T = new MatrixXd(c->cols(), c->cols());
    larft(Q, t, T);

    MatrixXd* R = new MatrixXd(t->size(), t->size());
    *R = Q->topRows(c->cols()).triangularView<Upper>();

    (*e->A21).topRows(c->cols()) = MatrixXd::Identity(c->cols(), c->cols());
    (*e->A21).bottomRows(c->rows()-c->cols()) = MatrixXd::Zero(c->rows()-c->cols(),c->cols());

    this->nnzR += t->size()*t->size();
    this->nnzH += (unsigned long long int)(c->cols(), c->cols());

    for (auto e: c->edgesIn){
        if (e->A12 != nullptr && !e->n1->is_eliminated()){
            trsm_right(R, e->A12, CblasUpper, CblasNoTrans, CblasNonUnit);
        }

        if (e->A21 != nullptr && !e->n1->is_eliminated()){
            larfb(Q, T, e->A21);
        }
    }

    for (auto e: c->edgesOut){
        if (e->A21 != nullptr && !e->n2->is_eliminated() && (e->n2 != c)){
            trsm_right(R, e->A21, CblasUpper, CblasNoTrans, CblasNonUnit);
        }

        if (e->A12 != nullptr && !e->n2->is_eliminated()){
            larfb(Q, T, e->A12);
        }
    }
 

    this->ops.push_back(new ScaleD(c, Q, t));  
    delete R;

     
    delete T;
    return;
}


bool Tree::want_sparsify(Cluster* c){
    if (c->get_level() > this->ilvl){
        SepID left = c->get_id().l;
        SepID right = c->get_id().r;
        if (left.lvl <= this->ilvl && right.lvl <= this->ilvl) 
            return true;
    }
    return false;
}


/* Main functions */
void Tree::partition(SpMat& A){
    int r = A.rows();
    int c = A.cols();
    this->nrows = r; 
    this->ncols = c;

    bool geo = (Xcoo != nullptr);
    tuple<vector<ClusterID>, vector<ClusterID>> maps;
    
    if (geo){
        maps = GeometricPartitionAtA(A, this->nlevels, Xcoo, this->use_matching);
    }
    else{
        #ifdef USE_METIS
            maps = MetisPartition(A, this->nlevels);
        #else
            maps = HypergraphPartition(A, this->nlevels);
        #endif
    }

    auto cmap = get<0>(maps);
    auto rmap = get<1>(maps);

    this->rperm = VectorXi::LinSpaced(r,0,r-1);
    this->cperm = VectorXi::LinSpaced(c,0,c-1);

    // Apply permutation
    vector<ClusterID> cpermed(c);
    initPermutation(nlevels, cmap, cperm); 
    transform(cperm.data(), cperm.data()+cperm.size(), cpermed.begin(), [&cmap](int i){return cmap[i];});

    vector<ClusterID> rpermed(r);
    #ifdef HSL_AVAIL
        if (this->use_matching){
            // vector<int> invcperm(c);
            // for (int i=0; i<c;++i){
            //     invcperm[cperm[i]]=i;
            // }
            // vector<int> cpair_of_r(r);
            // bipartite_row_matching(A, cpair_of_r); // gives row to column mathcing

            // for (int i=0;i<r;++i){
            //     rperm[ invcperm[ cpair_of_r[i] ] ] = i;
            // }
            bipartite_row_matching(A, cmap, rmap);
            initPermutation(nlevels, rmap, rperm);  

        }
        else {
            rmap = cmap;
            initPermutation(nlevels, rmap, rperm);  
        }
        
    #else 
        rmap = cmap;
        initPermutation(nlevels, rmap, rperm);  
    #endif

    
    transform(rperm.data(), rperm.data()+rperm.size(), rpermed.begin(), [&rmap](int i){return rmap[i];});

    /* Set the initial clusters: lvl 0 in cluster heirarchy */
    int k=0;
    int l=0;
    for (; k < c;){
        for (; l < r; ){
            int knext = k+1;
            int lnext = l; // Because the number of rows assigned to a cluster can be zero (? corner case)
            ClusterID cid = cpermed[k];
            ClusterID rid = rpermed[l];
            assert(cid == rid); // Decide what to do when zero rows belong to a cluster - push the cluster to the end? 
            while(knext < c && cid == cpermed[knext]){ knext++;}
            if (rid == cid){lnext = l+1; while(lnext < r && rid == rpermed[lnext]){ lnext++;} }
            Cluster* self = new Cluster(k, knext-k, l, lnext-l, cid, get_new_order());
            this->bottoms[0].push_back(self);
            this->profile.aspect_ratio[0].push_back((double)(self->rows())/(double)(self->cols()));

            k = knext; 
            l = lnext; 
            break;
        }
    }
    
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
}

void Tree::assemble(SpMat& A){
    int r = A.rows();
    int c = A.cols();
    this->nnzA = A.nonZeros();
    

    // Permute the matrix
    SpMat ApT = col_perm(A, this->cperm).transpose();
    SpMat App = col_perm(ApT, this->rperm).transpose();

    SpMat App_T = App.transpose();

    App.makeCompressed(); 
    App_T.makeCompressed();

    // Get CSC format  
    int nnz = App.nonZeros();
    VectorXi rowval = Map<VectorXi>(App.innerIndexPtr(), nnz);
    VectorXi colptr = Map<VectorXi>(App.outerIndexPtr(), c + 1);
    VectorXd nnzval = Map<VectorXd>(App.valuePtr(), nnz);

    VectorXi rowval_T = Map<VectorXi>(App_T.innerIndexPtr(), nnz);
    VectorXi colptr_T = Map<VectorXi>(App_T.outerIndexPtr(), r + 1);
    VectorXd nnzval_T = Map<VectorXd>(App_T.valuePtr(), nnz);

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
                if (self->get_order() <= n->get_order()){ // changed from get_level to get_order
                    cnbrs.insert(n);

                }
            }
        }
        cnbrs.insert(self);

        set<Cluster*> rnbrs; // Non-zeros entries in a row belonging to self/another cluster
        for (int j=self->get_rstart(); j <self->get_rstart()+ self->rows(); ++j){
            for (SpMat::InnerIterator it(App_T, j); it; ++it){
                Cluster* n = cmap[it.row()];

                if (self->get_level() <= n->get_level() &&  self != n){ //
                    rnbrs.insert(n);
                }
            }
        }

        for (auto nbr: cnbrs){
            MatrixXd* sA = new MatrixXd(nbr->rows(), self->cols());
            sA->setZero();
            block2dense(rowval, colptr, nnzval, nbr->get_rstart(), self->get_cstart(), nbr->rows(), self->cols(), sA, false); 
            if (self != nbr && self->get_order() <= nbr->get_order()){ // only go from lower to higher for outgoing edges
                MatrixXd* sAT = nullptr;
                if (rnbrs.find(nbr) != rnbrs.end()){
                    sAT = new MatrixXd(self->rows(), nbr->cols());
                    sAT->setZero();
                    block2dense(rowval_T, colptr_T, nnzval_T, nbr->get_cstart(), self->get_rstart(), nbr->cols(), self->rows(), sAT, true); // check
                }
                Edge* e = new Edge(self, nbr, sA, sAT);
                self->add_edgeOut(e);
                nbr->add_edgeIn(e);
                
            }
            else if (self == nbr){
                Edge* e = new Edge(self, nbr, sA, nullptr);
                self->add_edgeOut(e);
            }
            
            else {
                delete sA;
            }
        }

        // Add edges for non zeros in rows only
        for (auto n: rnbrs){
            if (cnbrs.find(n) == cnbrs.end() && self->get_order() < n->get_order()){
                MatrixXd* sAT = new MatrixXd(self->rows(), n->cols());
                sAT->setZero();
                // this->nnzA += (unsigned long long int)(n->cols()*self->rows());

                block2dense(rowval_T, colptr_T, nnzval_T, n->get_cstart(), self->get_rstart(), n->cols(), self->rows(), sAT, true);
                Edge* e = new Edge(self, n, nullptr, sAT);
                self->add_edgeOut(e);
                n->add_edgeIn(e);
            }
        }
        self->sort_edgesOut();
        self->cnbrs = cnbrs;
        self->rnbrs = rnbrs;
    }
    cout << "nnzA: " << nnzA << endl;
}

int Tree::eliminate_cluster(Cluster* c){
    if (c->cols()==0){
        c->set_eliminated();
        return 0;
    }

    /* Do Householder reflections */
    vector<MatrixXd*> H(c->cnbrs.size());
    int num_rows=0;
    int num_cols=0;
    int counter =0;
    for (auto edge: c->edgesOut){
        if (edge->A21 != nullptr){
            H[counter] = (edge->A21);
            counter ++;
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

    this->ops.push_back(new QR(c));

    c->set_eliminated();
    delete Q;
    return 0;
}

void Tree::get_sparsity(Cluster* c){
    set<Cluster*> row_sparsity;
    set<SepID> col_sparsity;

    for (auto edge: c->edgesOut){
        if (edge->A21 != nullptr){
            col_sparsity.insert(edge->n2->get_sepID());
        }
    }
    bool geo = (this->Xcoo != nullptr);
    
    for (auto edge: c->edgesOut){
        if (edge->A21 != nullptr){
            Cluster* n = edge->n2;
            row_sparsity.insert(n);
                if (n->get_id() != c->get_id()){
                    for (auto ein: n->edgesIn){
                        if (ein->A21 != nullptr && ein->n1 != c && !ein->n1->is_eliminated()){
                            Cluster* nin = ein->n1;
                            // bool valid = true;

                            bool valid = check_if_valid(c->get_id(), nin->get_id(), col_sparsity);
                            if ( !geo && valid) {
                                double tnorm = ((*(edge->A21)).transpose()*(*(ein->A21))).norm();
                                valid = tnorm >(1e-14);
                            }

                            if (valid) row_sparsity.insert(nin);
                        }
                    }

                    for (auto eout: n->edgesOut){
                        if (eout->A12 != nullptr){
                            Cluster* nout = eout->n2;
                            // bool valid = true;

                            bool valid = check_if_valid(c->get_id(), nout->get_id(), col_sparsity);
                            if ( !geo && valid) {
                                valid = ((*(edge->A21)).transpose()*(*(eout->A12))).norm()>(1e-14);
                            }
                            
                            if (valid) row_sparsity.insert(nout);
                        }
                    }
                }
        }
        if (edge->A12 != nullptr){
            row_sparsity.insert(edge->n2);
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
        if (e->A12 != nullptr && !(e->n1->is_eliminated())) {Amp.push_back(e->A12);}
        if (e->A21 != nullptr && !(e->n1->is_eliminated())){Apm.push_back(e->A21);}
    }

    for (auto e: c->edgesOut){
        if (e->A21 != nullptr && !(e->n2->is_eliminated())){
            if (e->n2 != c) {Amp.push_back(e->A21);}
            else {App = e->A21;}
        }
        if (e->A12 != nullptr && !(e->n2->is_eliminated())){Apm.push_back(e->A12);}
    }

    MatrixXd Apmc = Hconcatenate(Apm);
    MatrixXd Ampc = Vconcatenate(Amp);

    MatrixXd C = MatrixXd::Zero(Ampc.cols(), Ampc.rows() + Apmc.cols()); // Block to be compressed

    C.leftCols(Ampc.rows()) = Ampc.transpose();
    if (scale) {
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


    /* Create fine nodes */ // keep it square
    int cf_cstart = c->get_cstart()+rank;
    int cf_csize = c->cols()-rank;
    int cf_rstart = c->get_rstart() + c->rows()- cf_csize;
    int cf_rsize = cf_csize;
    Cluster * cf = new Cluster(cf_cstart, cf_csize, cf_rstart, cf_rsize, c->get_id(), get_new_order());

    this->ops.push_back(new Split(c,cf));

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
    cf->rsparsity = cf->rnbrs;

    eliminate_cluster(cf);
}



// If sparsify after diagScaling
void Tree::sparsifyD(Cluster* c){
    if (c->cols() == 0) return;
    // do_scaling(c);
    vector<MatrixXd*> Apm;
    vector<MatrixXd*> Amp;

    int spm=0;
    int smp=0;
    for (auto e: c->edgesIn){
        if (e->A12 != nullptr && !(e->n1->is_eliminated())) {Amp.push_back(e->A12); smp += e->A12->rows();}
        if (e->A21 != nullptr && !(e->n1->is_eliminated())){Apm.push_back(e->A21); spm += e->A21->cols();}
    }

    for (auto e: c->edgesOut){
        if (e->A21 != nullptr && !(e->n2->is_eliminated()) && e->n2 != c){Amp.push_back(e->A21); smp += e->A21->rows();}
        if (e->A12 != nullptr && !(e->n2->is_eliminated())){Apm.push_back(e->A12); spm += e->A12->cols();}
    }

    
    MatrixXd C = MatrixXd::Zero(c->cols(), spm+smp); // Block to be compressed
    MatrixXd Apmc = Hconcatenate(Apm);
    MatrixXd Apmc_rest = MatrixXd::Zero(0,0);

    C.leftCols(smp) = Vconcatenate(Amp).transpose();
    C.middleCols(smp, spm) = Apmc.topRows(c->cols());


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

    this->profile.neighbors[this->ilvl-skip][nlevels - c->get_sepID().lvl-1].push_back(C.cols());


    /* Sparsification */
    MatrixXd* v = new MatrixXd(C.rows(), rank);
    VectorXd* h = new VectorXd(rank);
    *v = C.leftCols(rank);
    *h = t.topRows(rank);

    this->ops.push_back(new OrthogonalD(c, v, h)); // push before size of c changes
    this->nnzQ += (unsigned long long int)(C.rows()*rank);

    /* Create fine nodes */ // keep it square
    int cf_cstart = c->get_cstart()+rank;
    int cf_csize = c->cols()-rank;
    int cf_rstart = c->get_rstart() + c->cols() - cf_csize;
    int cf_rsize = cf_csize;
    Cluster * cf = new Cluster(cf_cstart, cf_csize, cf_rstart, cf_rsize, c->get_id(), get_new_order());

    
    this->ops.push_back(new SplitD(c, cf, rank));

    /* Split edges between coarse and fine */
    MatrixXd Crank = C.topRows(rank).triangularView<Upper>();
    Crank = Crank * (jpvt.asPermutation().transpose());

    Edge* e_self = *(c->edgesOut.begin());
    MatrixXd self = MatrixXd::Zero(c->rows()-cf_rsize, rank);
    self.topRows(rank) = e_self->A21->topRows(rank); // Identity

   
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
        curr_col += Apm[i]->cols();
    }

    *(e_self->A21) = self.topRows(c->rows());

    this->fine[this->ilvl].push_back(cf);
    cf->set_eliminated();
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
        
        if (output_mat){write_mats_at_lvl(*this, this->name+"_merge.txt", this->ilvl, this->tol);}
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
        

        // Scale
        auto scstart = wctime();
        {
            if (this->scale && this->ilvl < nlevels -2  && this->ilvl >= skip && this->tol != 0  ){

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
            if (this->ilvl >= skip && this->ilvl < nlevels -2  && this->tol != 0  ){ //
                for(auto self: this->bottom_current()){
                    if (want_sparsify(self) ){
                        this->profile.rank_before[this->ilvl-skip][nlevels - self->get_sepID().lvl-1].push_back(self->cols());

                        auto spars0 = wctime();
                        if (scale){ //sparsify with diagonal scaling
                            sparsifyD(self);   
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
        if (output_mat){write_mats_at_lvl(*this, this->name+"_elmn.txt", this->ilvl, this->tol);}
        

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
             << "scale: " << elapsed(scstart,scend) << "  "
             << "sprsfy: " << elapsed(spstart,spend) << "  "
             << "merge: " << elapsed(mstart,mend) << "  " 
             << "s_top: " <<  get<0>(topsize()) << ", " << get<1>(topsize()) << endl;

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


/* Statistics */
list<const Cluster*> Tree::get_clusters() const {
    list<const Cluster*> all;
    for(int l = 0; l < bottoms.size(); l++) {
        for(auto n: bottoms[l]) {
            all.push_back(n);
        }
    }
    return all;
}

list<const Cluster*> Tree::get_clusters_at_lvl(int l) const {
    list<const Cluster*> all;
    for(auto n: bottoms[l]) {
        all.push_back(n);
    }
    return all;
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
                    if (e->A21 != nullptr){
                        delete e->A21;
                    }
                    if (e->A12 != nullptr){
                        delete e->A12;
                    }
                    delete e;
                }
            }
                
            delete s;
           
        }
    }

    for(int lvl = skip; lvl < this->nlevels-1; lvl++) {
        for(auto s : fine[lvl]){
            for(auto e : s->edgesOut) {
                if (e->A21 != nullptr){
                    delete e->A21;
                }
                if (e->A12 != nullptr){
                    delete e->A12;
                }
                delete e;
            }
            delete s;
        }
    }
}
