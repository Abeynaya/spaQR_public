#include "partition.h"

#ifdef HSL_AVAIL
extern "C"{
    #include "hsl_mc64d.h"
}
#endif

using namespace Eigen;
using namespace std;

typedef SparseMatrix<double, 0, int> SpMat; 
tuple<vector<int>,vector<int>> SpMat2CSC(SpMat& A) {
    int size = A.cols();
    vector<int> colptr(size+1);
    vector<int> rowval;
    colptr[0] = 0;
    for(int i = 0; i < size; i++) {
        colptr[i+1] = colptr[i];
        for (SpMat::InnerIterator it(A,i); it; ++it) {
            assert(i == it.col());
            int j = it.row();
            // if(i != j) {
                rowval.push_back(j);
                colptr[i + 1] += 1;
            // }
        }
    }
    return make_tuple(colptr, rowval);
}

tuple<vector<int>,vector<int>> SpMat2CSC_noloops(SpMat& A) {
    int size = A.cols();
    vector<int> colptr(size+1);
    vector<int> rowval;
    colptr[0] = 0;
    for(int i = 0; i < size; i++) {
        colptr[i+1] = colptr[i];
        for (SpMat::InnerIterator it(A,i); it; ++it) {
            assert(i == it.col());
            int j = it.row();
            if(i != j) {
                rowval.push_back(j);
                colptr[i + 1] += 1;
            }
        }
    }
    return make_tuple(colptr, rowval);
}

/* Get if a node belonging to parition part_leaf belongs to 
   left or right parition at lvl in the tree
 */
int part_at_lvl(int part_leaf, int lvl, int nlevels) {
    assert(lvl > 0);
    for(int l = 0; l < lvl-1; l++) {
        part_leaf /= 2;
    }
    // assert(part_leaf >= 0 && part_leaf < pow(2, nlevels-lvl));
    return part_leaf;
}

SepID find_highest_common(SepID n1, SepID n2) {
    int lvl1 = n1.lvl;
    int lvl2 = n2.lvl;
    int sep1 = n1.sep;
    int sep2 = n2.sep;

    while (lvl1 < lvl2) {
        lvl1 ++;
        sep1 /= 2;
    }
    while (lvl2 < lvl1) {
        lvl2 ++;
        sep2 /= 2;
    }
    while (sep1 != sep2) {
        lvl1 ++;
        lvl2 ++;
        sep1 /= 2;
        sep2 /= 2;
    }
    assert(lvl1 == lvl2);
    assert(sep1 == sep2);
    return SepID(lvl1, sep1);
}

SepID find_lowest_common(SepID n1, SepID n2) {
    int lvl1 = n1.lvl;
    int lvl2 = n2.lvl;
    int sep1 = n1.sep;
    int sep2 = n2.sep;

    if (lvl1 < lvl2){
        return n1;
    }
    else { // lvl 1 > lvl 2
        assert(lvl1 > lvl2);
        return n2;
    }
}

/* Assign rows to clusters according to the specified parameters */
void rows2clusters(SpMat& A, vector<ClusterID>& cmap, vector<ClusterID>& rmap, int use_matching){
    int nrows = A.rows();
    int ncols = A.cols();

    if (nrows != ncols){
        // Permute cols according the clusters
        vector<ClusterID> cpermed(ncols);
        VectorXi cperm = VectorXi::LinSpaced(ncols,0,ncols-1);
        auto compIJ = [&cmap](int i, int j){return (cmap[i] < cmap[j]);};
        stable_sort(cperm.data(), cperm.data()+cperm.size(), compIJ); 
        transform(cperm.data(), cperm.data()+cperm.size(), cpermed.begin(), [&cmap](int i){return cmap[i];});

        #ifdef HSL_AVAIL
            if (use_matching){
                bipartite_row_matching(A, cmap, rmap);
            }
        #endif 
        
        form_rmap(A, rmap, cpermed, cperm);
    }

    else {
        #ifdef HSL_AVAIL
            if (use_matching)
                bipartite_row_matching(A, cmap, rmap);
            else 
                rmap = cmap;
        #else 
            rmap = cmap;
        #endif 
    }
}

void rmap2cmap(SpMat& A, vector<ClusterID>& cmap, vector<ClusterID>& rmap, int use_matching){
    int ncols = A.cols();

    #ifdef HSL_AVAIL
        if (use_matching)
            bipartite_row_matching(A, cmap, rmap, false);
        else {
            cout << "Proceeding by matching row i to col i ..." << endl;
            cout << "Set --hsl 1 if you want to use bipartite matching routine" << endl;
            for (int i=0; i < ncols; ++i){
                cmap[i] = rmap[i];
            }
        }
    #else 
        cout << "May need HSL bipartite matching routine to work correctly." << endl;
        cout << "Proceeding by matching row i to col i ..." << endl;
        cmap = rmap;
        for (int i=0; i < ncols; ++i){
            cmap[i] = rmap[i];
        }
    #endif 
}


#ifdef HSL_AVAIL
/* Bipartite row matching */
void bipartite_row_matching(SpMat& A, vector<ClusterID>& cmap, vector<ClusterID>& rmap, bool cmap2rmap){
    SpMat Aabs = A.cwiseAbs();
    auto csc = SpMat2CSC(Aabs);
    auto colptr = get<0>(csc);
    auto rowval = get<1>(csc);

    struct mc64_control control;
    struct mc64_info info;
    mc64_default_control(&control);

    int m = A.rows();
    int n = A.cols();
    vector<int> perm(m+n);
    vector<int> invrperm(m);
    vector<int> invcperm(n);

    int job = 3;
    int matrix_type = 1; // unsymmtric or rectangular
    mc64_matching(job, matrix_type, m, n, colptr.data(), rowval.data(), Aabs.valuePtr(),
                        &control, &info, perm.data(), NULL);

    for (int i=0; i < m; ++i ){
        if (perm[i] >= 0) // perm[i] is negative for unmatched rows and columns
            invrperm[perm[i]] = i; 
    }
    for (int j=m; j < m+n; ++j){
        assert(perm[j]>=0); // since we only have rows>= cols. so all cols should be matched.
        invcperm[perm[j]] = j-m; 
    }


    for (int i=0; i< min(m,n); ++i){
        if (cmap2rmap)
            rmap[invrperm[i]] = cmap[invcperm[i]]; 
        else 
            cmap[invcperm[i]] = rmap[invrperm[i]];
    }
}

void bipartite_row_matching(SpMat& A, vector<ClusterID>& cmap, VectorXi& rperm){
    SpMat Aabs = A.cwiseAbs();
    auto csc = SpMat2CSC(Aabs);
    auto colptr = get<0>(csc);
    auto rowval = get<1>(csc);

    struct mc64_control control;
    struct mc64_info info;
    mc64_default_control(&control);

    int m = A.rows();
    int n = A.cols();
    vector<int> perm(m+n);
    vector<int> invrperm(m);
    vector<int> cperm(n);

    int job = 2;
    int matrix_type = 1; // unsymmtric or rectangular
    mc64_matching(job, matrix_type, m, n, colptr.data(), rowval.data(), Aabs.valuePtr(),
                        &control, &info, perm.data(), NULL);

    for (int j=m; j < m+n; ++j){
        assert(perm[j]>=0); // since we only have rows>= cols. so all cols should be matched.
        cperm[perm[j]] = j-m;
    }

    int counter=0;
    for (int i=0; i < m; ++i ){
        if (perm[i] >= 0) // perm[i] is negative for unmatched rows and columns
            rperm[cperm[perm[i]]] = i; // i=0 to n-1 have been filled
        else {
            rperm[n+counter] = i;
            counter++;
        }
    }
}
#endif

/* PARTITIONING */
/* Recursive bissection - Geometric partition */
void bissect_geo(vector<int> &colptr, vector<int> &rowval, vector<int> &dofs, vector<int> &parts, MatrixXd *X, int start) {
    assert(X->cols() == colptr.size()-1);
    assert(X->cols() == parts.size());
    int N = dofs.size();
    if(N == 0)
        return;
    // Get dimension over which to cut
    int bestdim = -1;
    double maxrange = -1.0;
    for(int d = 0; d < X->rows(); d++) {
        double maxi = numeric_limits<double>::lowest();
        double mini = numeric_limits<double>::max();
        for(int i = 0; i < dofs.size(); i++) {
            int idof = dofs[i];
            maxi = max(maxi, (*X)(d,idof));
            mini = min(mini, (*X)(d,idof));
        }
        double range = maxi - mini;
        if(range > maxrange) {
            bestdim = d;
            maxrange = range;
        }
    }
    assert(bestdim >= 0);
    vector<int> dofstmp = dofs;
    // Sort dofs based on X[dim,:]
    std::sort(dofstmp.begin(), dofstmp.end(), [&](int i, int j) { return (*X)(bestdim,i) < (*X)(bestdim,j); });
    int imid = dofstmp[N/2];
    double midv = (*X)(bestdim,imid);
    // X <= midv -> 0, X > midv -> 1
    for(int i = 0; i < N; i++) {
        int j = dofs[i];
        if( (*X)(bestdim,j) <= midv ) {
            parts[j] = start;
        } else {
            parts[j] = start+1;
        }
    }
}

/* Infer Separators */
vector<ClusterID> GeometricPartitionAtA(SpMat& A, int nlevels, MatrixXd* Xcoo, int use_matching){
    SpMat AtA = A.transpose()*A;

    int c = AtA.cols();
    int r = AtA.rows();
    vector<int> parts(c);
    auto csc = SpMat2CSC_noloops(AtA);
    vector<int> colptr = get<0>(csc);
    vector<int> rowval = get<1>(csc);

    fill(parts.begin(), parts.end(), 0);
    for(int depth = 0; depth < nlevels-1; depth++) {
        // Create dofs lists
        vector<vector<int>> dofs(pow(2, depth));
        for(int i = 0; i < parts.size(); i++) {
            assert(0 <= parts[i] && parts[i] < pow(2, depth));
            dofs[parts[i]].push_back(i);
        }
        // Do a geometric partitioning
        for(int k = 0; k < pow(2, depth); k++) {
            bissect_geo(colptr, rowval, dofs[k], parts, Xcoo, 2*k);
        }
    }

    vector<ClusterID> cmap(c, ClusterID());
    for (int depth = 0; depth < nlevels -1; depth++){
        int l = nlevels - depth -1;
        for (int i=0; i < c; ++i){
            if (cmap[i].self.lvl == -1){ // Not yet a separator
                bool sep = false ; 
                int pi = part_at_lvl(parts[i], l, nlevels); 

                if (pi % 2 == 0 ){ // Only look at left partition // Gives 0 or 1 depending on left or right partition
                    for (SpMat::InnerIterator it(AtA,i); it; ++it){
                        assert(i==it.col());
                        int j = it.row(); // Neighbor of i
                        if (cmap[j].self.lvl == -1){
                            int pj = part_at_lvl(parts[j], l, nlevels);
                            if (pj == pi+1){
                                sep = true;
                                break;
                            }
                        }
                    }
                    if(sep){
                        cmap[i].self=SepID(l,pi/2);
                    }
                }
            }
        }
    }

    // Handle leaves
    for (int i=0; i < c; ++i){
        if (cmap[i].self.lvl == -1){ // leaves
            cmap[i].self = SepID(0,parts[i]);
            cmap[i].l = SepID(0,parts[i]);
            cmap[i].r = SepID(0,parts[i]);
        }
    }

    vector<ClusterID> rmap(A.rows(), ClusterID());
    rows2clusters(A, cmap, rmap, use_matching);

    getInterfacesUsingA(A, nlevels, cmap, rmap, parts, use_matching);
    return cmap;
}

#ifdef USE_METIS
/* Metis partition on A^TA */
void partition_metis_RB(SpMat& A, int nlevels, vector<int>& parts){
  int size = A.cols(); 
  int one = 1;
  auto csc = SpMat2CSC_noloops(A);
  vector<int> colptr = get<0>(csc);
  vector<int> rowval = get<1>(csc);

  int objval; 
  int nparts = pow(2,nlevels-1);
  int options[METIS_NOPTIONS];


  options[METIS_OPTION_SEED] = 7103855;
  METIS_SetDefaultOptions(options);
  METIS_PartGraphRecursive(&size, &one, colptr.data(), rowval.data(),
                                              nullptr, nullptr, nullptr, &nparts, nullptr, nullptr, options, &objval, parts.data());
};

vector<ClusterID> MetisPartition(SpMat& A, int nlevels){
    SpMat AtA = A.transpose()*A;
    int c = A.cols();
    int r = A.rows();
    vector<int> parts(c);
    partition_metis_RB(AtA, nlevels, parts);
    
    // getInterfacesUsingATA(AtA, nlevels, cmap, parts);
    vector<ClusterID> cmap(c, ClusterID());

    // Get separators using AtA
    for (int depth = 0; depth < nlevels -1; depth++){
        int l = nlevels - depth -1;
        for (int i=0; i < c; ++i){
            if (cmap[i].self.lvl == -1){ // Not yet a separator
                bool sep = false ; 
                int pi = part_at_lvl(parts[i], l, nlevels); 

                if (pi % 2 == 0 ){ // Only look at left partition // Gives 0 or 1 depending on left or right partition
                    for (SpMat::InnerIterator it(AtA,i); it; ++it){
                        assert(i==it.col());
                        int j = it.row(); // Neighbor of i
                        if (cmap[j].self.lvl == -1){
                            int pj = part_at_lvl(parts[j], l, nlevels);
                            if (pj == pi+1){
                                sep = true;
                                break;
                            }
                        }
                    }
                    if(sep){
                        cmap[i].self=SepID(l,pi/2);
                    }
                }
            }
        }
    }

    // Handle leaves
    for (int i=0; i < c; ++i){
        if (cmap[i].self.lvl == -1){ // leaves
            cmap[i].self = SepID(0,parts[i]);
            cmap[i].l = SepID(0,parts[i]);
            cmap[i].r = SepID(0,parts[i]);
        }
    }

    getInterfacesUsingATA(AtA, nlevels, cmap, parts);    
    return cmap;
}
#else
/* HyperGraph Partition*/
void partition_patoh(SpMat& A, int nlevels, vector<int>& parts){
    int nrows = A.rows();
    int ncols = A.cols(); 
    auto csc = SpMat2CSC(A);
    vector<int> colptr = get<0>(csc);
    vector<int> rowval = get<1>(csc);
    int nconst = 1; 
    int useFixCells = 0;
    vector<int> cwghts(nrows, 1);
    vector<int> nwghts(ncols, 1);

    int nparts = pow(2,nlevels-1);

    PaToH_Parameters args;
    PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT); // Initialize parameters
    // The third parameter can be set to _SPEED, _QUALITY, _DEFAULT
    args._k = nparts;
    args.seed = 42;

    vector<int> partweights(nparts);
    int cut; 

    PaToH_Alloc(&args, nrows, ncols, nconst, cwghts.data(), nullptr, colptr.data(), rowval.data()); // Allocate memory that will be used by partitioning algorithms
    PaToH_Part(&args, nrows, ncols, nconst, useFixCells, cwghts.data(), nullptr, colptr.data(), rowval.data(), nullptr, parts.data(), partweights.data(), &cut);

    PaToH_Free();
}

vector<ClusterID> HypergraphPartition(SpMat& A, int nlevels){
    int r = A.rows();
    int c = A.cols(); 

    vector<int> parts(r,0);
    partition_patoh(A, nlevels, parts);

    // Get separators (columns)
    vector<ClusterID> cmap(c, ClusterID());
    for (int depth=0; depth < nlevels-1; ++depth){
        int l = nlevels - depth - 1;
        // vector<int> sepsizes(pow(2,depth),0);
        for (int i=0; i < c; ++i){
            if (cmap[i].self.lvl == -1){// Not yet a separator
                bool sep = false; 
                int pi = -1;
                for (SpMat::InnerIterator it(A,i); it; ++it){
                    assert(it.col()==i);
                    int j = it.row();
                    int pj = part_at_lvl(parts[j], l, nlevels);
                    if (pi == -1) pi=pj; 
                    else if (pi != pj) {
                        sep = true; 
                        break;
                    }

                }
                if (sep){
                    cmap[i].self = SepID(l, pi/2);
                }
            }
        }
    }

    // Leaves
    for (int i=0; i < c; ++i){
        if(cmap[i].self.lvl == -1){
            for (SpMat::InnerIterator it(A,i); it; ++it){
                int j = it.row();
                cmap[i].self = SepID(0, parts[j]); // parts[j] should be same for all j
                cmap[i].l = SepID(0, parts[j]);
                cmap[i].r = SepID(0, parts[j]);
                break; 
            }
        }
    }

    getInterfacesUsingA_HUND(A, nlevels, cmap, parts);
    return cmap;
}
#endif


/* Infer interfaces 
* Use A to get interfaces of separators -- use only with geometric partition on A^TA 
*/
void getInterfacesUsingA(SpMat& A, int nlevels, vector<ClusterID>& cmap, vector<ClusterID>& rmap, vector<int> parts, int use_matching){
    int nrows = A.rows();

    SpMat At = A.transpose();
    for(int i = 0; i < nrows; i++) {
        SepID self = rmap[i].self;
        // If it's a separator row
        if(self.lvl > 0) {
            // Define left/right part
            // Find left & right SepID
            SepID left = SepID(-1,0);
            SepID right = SepID(-1, 0);
            for (SpMat::InnerIterator ot(At,i); ot; ++ot) { // look through columns of that row
                int j = ot.row();
                SepID nbr = cmap[j].self;

                if(nbr.lvl < self.lvl) {
                    int pj = nbr.sep;
                    pj /= pow(2, self.lvl-nbr.lvl-1);
                    
                    if(pj % 2 == 0) { // Update left. It could be empty.
                        if (left.lvl == -1) left  = nbr;
                        else left  = find_lowest_common(nbr, left);
                      
                    } else { // Update right. It could be empty.
                        if(right.lvl == -1) right = nbr;
                        else right = find_lowest_common(nbr, right);
                    }
                }
            }

            rmap[i].l = left;
            rmap[i].r = right;
        }
    }
    // Go from rmap to cmap
    rmap2cmap(A, cmap, rmap, use_matching);

    for (int i=0; i< nrows; ++i){
        SepID self = rmap[i].self;
        SepID left = rmap[i].l;
        SepID right = rmap[i].r;
        if (right == SepID()){ // I need a right id
            for (SpMat::InnerIterator ot(At,i); ot; ++ot){
                int j = ot.row(); // j column
                if (self == cmap[j].self && cmap[j].r != SepID()){ 
                    if (right == SepID()) right = cmap[j].r;
                    else right = find_highest_common(right, cmap[j].r); 
                    // okay to keep highest common nbr here as this is a distance 2 connection -- keep fewer interfaces
                }
            }
        }
        else if (left == SepID()){
            for (SpMat::InnerIterator ot(At,i); ot; ++ot){
                int j = ot.row();
                if (self == cmap[j].self && cmap[j].l != SepID() ){
                    if (left == SepID()) left = cmap[j].l;
                    else left = find_highest_common(left, cmap[j].l);
                }
            }
        }
        rmap[i].l = left;
        rmap[i].r = right; 
    }

    for (int i=0; i < nrows; ++i){
        if (rmap[i].l == SepID()) rmap[i].l = rmap[i].self;
        if (rmap[i].r == SepID()) rmap[i].r = rmap[i].self;
    }

    // Go from rmap to cmap
    rmap2cmap(A, cmap, rmap, use_matching);
}


/* Infer interfaces */
void getInterfacesUsingA_HUND(SpMat& A, int nlevels, vector<ClusterID>& cmap, vector<int> parts){
    assert(cmap.size() == A.cols());
    int c = A.cols();
    SpMat At = A.transpose();

    // Group together separators using distance 2 neighbors (or dist 1 nbrs in A'*A)
    // vector<tuple<ClusterID, ClusterID>> interfaces(c);
    for (int i=0; i<c; ++i){
        SepID self = cmap[i].self;
        if (self.lvl >0){
            SepID left = SepID(-1, 0);
            SepID right = SepID(-1, 0);
            for (SpMat::InnerIterator it(A,i); it; ++it){
                int j = it.row();
                int pj = part_at_lvl(parts[j], self.lvl, nlevels);

                for (SpMat::InnerIterator ot(At,j); ot; ++ot){
                    int k = ot.row();
                    auto nbr = cmap[k].self;
                    if (nbr.lvl < self.lvl){
                        if (pj % 2 == 0){ // Update left
                            if (left.lvl == -1){left = nbr;}
                            else {left = find_highest_common(nbr, left);}
                        }  
                        else {// Update right
                            if (right.lvl == -1){right = nbr;}
                            else {right = find_highest_common(nbr, right);}
                        }
                    }
                }
            }
            cmap[i].l = left;
            cmap[i].r = right;    
        }
    }

    for (int i=0; i < c; ++i){
        if (cmap[i].l == SepID()) cmap[i].l = cmap[i].self;
        if (cmap[i].r == SepID()) cmap[i].r = cmap[i].self;
    }

}

/*  Using A^TA to get interfaces 
 * A = A^TA
 */
void getInterfacesUsingATA(SpMat& A, int nlevels, vector<ClusterID>& cmap, vector<int> parts){
    int size = A.cols();
    assert(A.cols() == A.rows());

    // Leaf-clusters
    for(int i = 0; i < size; i++) {
        SepID self = cmap[i].self;
        // If it's a separator
        if(self.lvl > 0) {
            // Define left/right part
            // Find left & right SepID
            SepID left  = SepID(0, parts[i]);
            // SepID left = SepID(-1,0);
            SepID right = SepID(-1, 0);
            int pi = part_at_lvl(parts[i], self.lvl, nlevels);
            assert(self.sep == pi/2);
            for (SpMat::InnerIterator it(A,i); it; ++it) {
                int j = it.row();
                SepID nbr = cmap[j].self;
                if(nbr.lvl < self.lvl) {
                    int pj = part_at_lvl(parts[j], self.lvl, nlevels);
                    assert(pi == pj || pi+1 == pj);
                    if(pj <= pi) { // Update left. It could be empty.
                        left  = find_highest_common(nbr, left);
                    } else { // Update right. It cannot be empty.
                        if(right.lvl == -1) {
                            right = nbr;
                        } else {
                            right = find_highest_common(nbr, right);
                        }
                    }
                }
            }
            assert(right.lvl != -1); // *cannot* be empty
            cmap[i].l = left;
            cmap[i].r = right;
            cmap[i].section = parts[i];
        }
    }
    for(int i = 0; i < size; i++) {
        SepID self = cmap[i].self;
        if(self.lvl == 0) {
            cmap[i].l = self;
            cmap[i].r = self;
            cmap[i].section = parts[i];
        }
    }
}



