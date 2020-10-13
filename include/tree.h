#ifndef TREE_H
#define TREE_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <list>
#include <set>
#include <tuple>
#include <utility>
#include <queue>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/QR>
#include <Eigen/Householder> 
#include <Eigen/SVD>
#include <numeric>
#include <assert.h>
#include <limits>

#include "util.h"
#include "edge.h"
#include "cluster.h"
#include "operations.h"
#include "toperations.h"
#include "stats.h"
#include "profile.h"



typedef Eigen::SparseMatrix<double, 0, int> SpMat;
typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PermMat;


// Tree
class Tree{
    private: 
        int ilvl;  // Level [0,...ilvl] have been eliminated; -1 nothing eliminated
        int nlevels; 
        int nrows; // rows in matrix
        int ncols; // cols in matrix 
        double tol;
        int use_matching;
        int skip;   // Skip levels before starting sparsification (0)
        int scale; // Scale (default true)

        int max_order;
        int output_mat; // Output matrices
        std::string name;


        // External data (coordinates)
        Eigen::MatrixXd* Xcoo;
        Eigen::VectorXi cperm; // Column permutation of the matrix
        Eigen::VectorXi rperm; // Row permutation of the matrix (initial)

        // Stores the clusters at each level of the cluster hierarchy
        std::vector<std::list<Cluster*>> bottoms; // bottoms.size() = nlevels
        std::vector<std::list<Cluster*>> fine; // list of fine nodes

        int current_bottom;
        const std::list<Cluster*>& bottom_current() const {return bottoms[current_bottom];};
        const std::list<Cluster*>& bottom_original() const {return bottoms[0];};
        
        /* Store the operations */
        std::vector<Operation*> ops;

        /*Perform QR factorization */
        Edge* add_edge(Cluster* c1, Cluster* c2, bool A21, bool A12);
        void householder(std::vector<Eigen::MatrixXd*> H, Eigen::MatrixXd* Q, Eigen::VectorXd* t,  Eigen::MatrixXd* T);
        int update_cluster(Cluster* c, Cluster* nn, Eigen::MatrixXd* Q);
        int eliminate_cluster(Cluster* c);

      
        /* Merging and updating*/
        void merge_all();
        void update_size(Cluster* snew);
        void update_edges(Cluster* snew);
        bool check_if_valid(ClusterID cid, ClusterID nid, std::set<SepID> col_sparsity);
        void get_sparsity(Cluster* c);

        /* Sparsification */
        bool want_sparsify(Cluster* c);
        void sparsify(Cluster* c);
        void sparsifyD(Cluster* c);

        void split_edges(Cluster* c, Cluster* f);
        void diagScale(Cluster* c);
        void do_scaling(Cluster* c);

        /* Fill in data */
        unsigned long long int nnzA;
        unsigned long long int nnzR;
        unsigned long long int nnzH; // Householder transformations on left
        unsigned long long int nnzQ; // Sparsification Q

        /* Helper functions */
        std::tuple<int,int> topsize(); // size of the top separator at current_bottom
        int get_new_order(); 

        /* Stats*/
        Profile profile = Profile(nlevels, skip);
        

    public: 
        // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Tree(int lvls, int skip_): ilvl(-1), nlevels(lvls), nrows(0), ncols(0), tol(0), 
                        use_matching(0), skip(skip_), scale(0),  max_order(0), 
                        output_mat(0), current_bottom(0), nnzA(0), nnzR(0), nnzH(0), nnzQ(0) {
                            bottoms = std::vector<std::list<Cluster*>>(nlevels);
                            fine = std::vector<std::list<Cluster*>>(nlevels);
                            Xcoo = nullptr;
                            profile = Profile(lvls, skip_);
                        };
            
        std::vector<std::vector<ClusterID>> get_clusters_levels() const;
        
        void set_tol(double tol);
        void set_hsl(double hsl);
        void set_skip(int s);
        void set_scale(int s);
        void set_Xcoo(Eigen::MatrixXd*);
        void set_output(int s, std::string n);


        // Get access to basic info
        int rows() const;
        int cols() const;
        int levels() const;

        // Partition and set-up
        void partition(SpMat& A);
        void assemble(SpMat& A);
 
        // Factorize and solve
        int factorize();
        void solve(Eigen::VectorXd b, Eigen::VectorXd& x) const; // Solve Ax = b

        // Stats
        std::list<const Cluster*> get_clusters() const;
        std::list<const Cluster*> get_clusters_at_lvl(int l) const;

        /** Destructor */
        ~Tree();

};

void initPermutation(int nlevels, const std::vector<ClusterID>& cmap, Eigen::VectorXi& cperm);
void form_rmap(SpMat& A, std::vector<ClusterID>& rmap, const std::vector<ClusterID>& cmap, Eigen::VectorXi& cperm);

#endif