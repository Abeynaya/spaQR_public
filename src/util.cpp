#include <random>
#include "util.h"

using namespace Eigen;
using namespace std;

bool are_connected(VectorXi &a, VectorXi &b, SpMat &A) {
    int  bsize = b.size();
    auto b_begin = b.data();
    auto b_end   = b.data() + b.size();
    for(int ia = 0; ia < a.size(); ia++) {
        // Look at all the neighbors of ia
        int node = a[ia];
        for(SpMat::InnerIterator it(A,node); it; ++it) {
            auto neigh = it.row();
            auto id = lower_bound(b_begin, b_end, neigh);
            int pos = id - b_begin;
            if(pos < bsize && b[pos] == neigh) // Found one in b! They are connected.
                return true;
        }
    }
    return false;
}

// lvl=0=leaf
// assumes a ND binary tree
bool should_be_disconnected(int lvl1, int lvl2, int sep1, int sep2) {
    while (lvl2 > lvl1) {
        lvl1 += 1;
        sep1 /= 2;
    }
    while (lvl1 > lvl2) {
        lvl2 += 1;
        sep2 /= 2;
    }
    if (sep1 != sep2) {
        return true;
    } else {
        return false;
    }
}

/** 
 * Given A, returns |A|+|A^T|+I
 */
SpMat symmetric_graph(SpMat& A) {
    assert(A.rows() == A.cols());
    int n = A.rows();
    vector<Triplet<double>> vals(2 * A.nonZeros() + n);
    int l = 0;
    for (int k=0; k < A.outerSize(); ++k) {
        vals[l++] = Triplet<double>(k, k, 1.0);
        for (SpMat::InnerIterator it(A,k); it; ++it) {
            vals[l++] = Triplet<double>(it.col(), it.row(), abs(it.value()));
            vals[l++] = Triplet<double>(it.row(), it.col(), abs(it.value()));
        }
    }
    assert(l == vals.size());
    SpMat AAT(n,n);
    AAT.setFromTriplets(vals.begin(), vals.end());
    return AAT;
}

double elapsed(timeval& start, timeval& end) {
    return (end.tv_sec + end.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6);
}

timer wctime() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time;
}

// All are base-0
void swap2perm(Eigen::VectorXi* swap, Eigen::VectorXi* perm) {
    int n = perm->size();
    assert(swap->size() == n);
    for(int i = 0; i < n; i++) {
        (*perm)[i] = i;
    }
    for(int i = 0; i < n; i++) {
        int ipiv = (*swap)[i];
        int tmp = (*perm)[ipiv];
        (*perm)[ipiv] = (*perm)[i];
        (*perm)[i] = tmp;
    }
}

bool isperm(Eigen::VectorXi* perm) {
    int n = perm->size();
    VectorXi count = VectorXi::Zero(n);
    for(int i = 0;i < n; i++) {
        int pi = (*perm)[i];
        if(pi < 0 || pi >= n) { return false; }
        count[pi] += 1;
    }
    return (count.cwiseEqual(1)).all();
}

size_t hashv(vector<size_t> vals) {
    size_t seed = 0;
    for (size_t i = 0; i < vals.size(); ++i) {
      seed ^= vals[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

void concatenate(vector<MatrixXd*> H, MatrixXd* V){
    assert(H.size()>0);
    int size = H.size();
    int num_cols = H[0]->cols();
    int num_rows = H[0]->rows();
    for (int i=1; i< size; ++i){
        assert(H[i]->cols()==num_cols);
        num_rows += H[i]->rows();
    }
    // MatrixXd V(num_rows, num_cols);
    assert(V->rows()==num_rows);
    assert(V->cols()==num_cols);
    
    int curr_row = 0;
    for (int i=0; i<size; ++i){
        V->middleRows(curr_row, H[i]->rows()) = *(H[i]);
        curr_row += H[i]->rows();
    }
}

MatrixXd Vconcatenate(vector<MatrixXd*> H){
    assert(H.size()>0);
    int size = H.size();
    int num_cols = H[0]->cols();
    int num_rows = H[0]->rows();
    for (int i=1; i< size; ++i){
        assert(H[i]->cols()==num_cols);
        num_rows += H[i]->rows();
    }
    MatrixXd V(num_rows, num_cols);
    
    int curr_row = 0;
    for (int i=0; i<size; ++i){
        V.middleRows(curr_row, H[i]->rows()) = *(H[i]);
        curr_row += H[i]->rows();
    }
    return V;
}

/* Reverse vertical concatenation */
void rVconcatenate(vector<MatrixXd*> H, MatrixXd& Hc){
    assert(H.size()>0);
    int size = H.size();
    int curr_row = 0;
    for (int i=0; i<size; ++i){
        *(H[i]) = Hc.middleRows(curr_row, H[i]->rows());
        curr_row += H[i]->rows();
    }
}

VectorXd Vconcatenate(vector<VectorXd*> H){
    assert(H.size()>0);
    int size = H.size();
    int num_cols = H[0]->cols();
    int num_rows = H[0]->rows();
    for (int i=1; i< size; ++i){
        assert(H[i]->cols()==num_cols);
        num_rows += H[i]->rows();
    }
    assert(num_cols==1);
    VectorXd V(num_rows, num_cols);
    int curr_row = 0;
    for (int i=0; i<size; ++i){
        V.middleRows(curr_row, H[i]->rows()) = *(H[i]);
        curr_row += H[i]->rows();
    }
    return V;
}

MatrixXd Hconcatenate(vector<MatrixXd*> H){
    // assert(H.size()>0);
    int size = H.size();
    int num_cols = H[0]->cols();
    int num_rows = H[0]->rows();
    for (int i=1; i< size; ++i){
        assert(H[i]->rows()==num_rows);
        num_cols += H[i]->cols();
    }
    MatrixXd V(num_rows, num_cols);
    int curr_col = 0;
    for (int i=0; i<size; ++i){
        V.middleCols(curr_col, H[i]->cols()) = *(H[i]);
        curr_col += H[i]->cols();
    }
    return V;
}

/* Reverse horizontal concatenation */
void rHconcatenate(vector<MatrixXd*> H, MatrixXd& Hc){
    assert(H.size()>0);
    int size = H.size();
     int curr_col = 0;
    for (int i=0; i<size; ++i){
        *(H[i]) = Hc.middleCols(curr_col, H[i]->cols());
        curr_col += H[i]->cols();
    }
}

/**
 * C = alpha A^(/T) * B^(/T) + beta C
 */
double rcond_1_getf(Eigen::MatrixXd* A_LU, double A_1_norm) {
    int m = A_LU->rows();
    int n = A_LU->cols();
    // assert(A_LU->cols() == n);
    double rcond = 10.0;
    int info = -1;
    if (m >= n)
        info = LAPACKE_dgecon(LAPACK_COL_MAJOR, '1', n, A_LU->data(), m, A_1_norm, &rcond);
    else {
         info = LAPACKE_dgecon(LAPACK_COL_MAJOR, 'I', m, A_LU->transpose().data(), n, A_1_norm, &rcond);
    }
    assert(info == 0);
    return rcond;
}

double rcond_1_potf(Eigen::MatrixXd* A_LLT, double A_1_norm) {
    int n = A_LLT->rows();
    assert(A_LLT->cols() == n);
    double rcond = 10.0;
    int info = LAPACKE_dpocon(LAPACK_COL_MAJOR, 'L', n, A_LLT->data(), n, A_1_norm, &rcond);
    assert(info == 0);
    return rcond;
}

void trsm_right(MatrixXd* L, MatrixXd* B, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag) {
    int m = B->rows();
    int n = B->cols();
    assert(L->rows() == n);
    assert(L->cols() == n);
    if (m == 0 || n == 0)
        return;
    cblas_dtrsm(CblasColMajor, CblasRight, uplo, trans, diag, m, n, 1.0, L->data(), n, B->data(), m);
}

void trsm_left(MatrixXd* L, MatrixXd* B, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag) {
    int m = B->rows();
    int n = B->cols();
    assert(L->rows() == m);
    assert(L->cols() == m);
    if (m == 0 || n == 0)
        return;
    cblas_dtrsm(CblasColMajor, CblasLeft, uplo, trans, diag, m, n, 1.0, L->data(), m, B->data(), m);
}

void trsv(MatrixXd* LU, Segment* x, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag) {
    int n = LU->rows();
    assert(LU->cols() == n);
    assert(x->size() == n);
    if (n == 0)
        return;
    cblas_dtrsv(CblasColMajor, uplo, trans, diag, n, LU->data(), n, x->data(), 1);
}

void trmv_trans(MatrixXd* L, Segment* x) {
    int n = L->rows();
    int m = L->cols();
    assert(x->size() == n);
    assert(n == m);
    if (n == 0)
        return;
    cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, L->rows(), L->data(), L->rows(), x->data(), 1);
}

// A <- L^T * A
void trmm_trans(MatrixXd* L, MatrixXd* A) {
    int m = A->rows();
    int n = A->cols();
    assert(L->rows() == m);
    assert(L->cols() == m);
    if (m == 0 || n == 0)
        return;
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, m, n, 1.0, L->data(), m, A->data(), m);
}

// x2 -= A21 * x1
void gemv_notrans(MatrixXd* A21, Segment* x1, Segment* x2) {
    int m = A21->rows();
    int n = A21->cols();
    assert(x1->size() == n);
    assert(x2->size() == m);
    if (n == 0 || m == 0)
        return;
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, -1.0, A21->data(), m, x1->data(), 1, 1.0, x2->data(), 1);
}

// x2 -= A12^T x1
void gemv_trans(MatrixXd* A12, Segment* x1, Segment* x2) {
    int m = A12->rows();
    int n = A12->cols();
    assert(x1->size() == m);
    assert(x2->size() == n);
    if (n == 0 || m == 0)
        return;
    cblas_dgemv(CblasColMajor, CblasTrans, m, n, -1.0, A12->data(), m, x1->data(), 1, 1.0, x2->data(), 1);
}

// x <- Q * x
void ormqr_notrans(MatrixXd* v, VectorXd* h, Segment* x) {
    int m = v->rows();
    // int n = min(v->rows(), v->cols());
    int n = v->cols();
    assert(h->size() == n);
    assert(x->size() == m);
    if (m == 0) 
        return;
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'N', m, 1, n, v->data(), m, h->data(), x->data(), m); 
    assert(info == 0);
}

// x <- Q^T * x
void ormqr_trans(MatrixXd* v, VectorXd* h, Segment* x) {
    int m = x->size();
    // n = 1
    int k = v->cols();
    assert(h->size() == k);
    assert(v->rows() == m);
    if (m == 0) 
        return;
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', m, 1, k, v->data(), m, h->data(), x->data(), m); 
    assert(info == 0);
}

// A <- Q^T * A
void ormqr_trans_left(MatrixXd* v, VectorXd* h, MatrixXd* A) {
    int m = A->rows();
    int n = A->cols();
    int k = v->cols();
    assert(h->size() == k);
    assert(v->rows() == m);
    if (m == 0 || n == 0)
        return;
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', m, n, k, v->data(), m, h->data(), A->data(), m);
    assert(info == 0);
}

// A <- Q * A
void ormqr_notrans_left(MatrixXd* v, VectorXd* h, MatrixXd* A) {
    int m = A->rows();
    int n = A->cols();
    int k = v->cols();
    assert(h->size() == k);
    assert(v->rows() == m);
    if (m == 0 || n == 0)
        return;
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'N', m, n, k, v->data(), m, h->data(), A->data(), m);
    assert(info == 0);
}

// A <- A * Q
void ormqr_notrans_right(MatrixXd* v, VectorXd* h, MatrixXd* A) {
    int m = A->rows();
    int n = A->cols();
    // int k = min(v->rows(), v->cols());
    int k = v->cols();
    assert(h->size() == k);
    assert(v->rows() == n);
    if (m == 0 || n == 0)
        return;
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'R', 'N', m, n, k, v->data(), n, h->data(), A->data(), m);
    assert(info == 0);
}

// A <- (Q^/T) * A * (Q^/T)
void ormqr(MatrixXd* v, VectorXd* h, MatrixXd* A, char side, char trans) {
    int m = A->rows();
    int n = A->cols();
    int k = v->cols(); // number of reflectors
    assert(h->size() == k);
    if (m == 0 || n == 0)
        return;
    if(side == 'L') // Q * A or Q^T * A
        assert(k <= m);
    if(side == 'R') // A * Q or A * Q^T
        assert(k <= n);
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, side, trans, m, n, k, v->data(), v->rows(), h->data(), A->data(), m);
    assert(info == 0);
}

// Create the thin Q in v
void orgqr(Eigen::MatrixXd* v, Eigen::VectorXd* h) {
    int m = v->rows();
    int k = v->cols();
    assert(h->size() == k);
    if(m == 0)
        return;
    int info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, k, k, v->data(), m, h->data());
    assert(info == 0);
}

/* Finds upper triangular matrix T such that,
    H  =  I - V * T * V^T
    using Householder vectors V and their norms tau
    H = Householder reflector matrix 
*/
void larft(MatrixXd* V, VectorXd* tau, MatrixXd* T){
    int m = V->rows();
    int k = V->cols();
    assert(tau->size() == k);
    assert(T->rows()==k);
    assert(T->cols()==k);
    int info = LAPACKE_dlarft(LAPACK_COL_MAJOR, 'F','C',  m, k, V->data(), m, tau->data(), T->data(), k);
    assert(info==0);
}

/* Apply householder vectors on a rectangular matrix 
V = [1    *     *   * *
     v(1) 1     *   * *
     v(1) v(2)  *   * *
     v(1) v(2) v(3) * *]; 
H = H(1) H(2) H(3) ... H(k) = I - V * T * V^T
C <- H^T C
*/
void larfb(MatrixXd* V, MatrixXd* T, MatrixXd* C){
    int m = C->rows();
    int n = C->cols();
    int k = V->cols();
    assert(m == V->rows());
    assert(T != nullptr);
    assert(T->rows() == k);
    if (m==0 || n==0){
        return;
    }
    #ifdef USE_MKL
    char side = 'L';
    char trans = 'T';
    char direct = 'F';
    char storev = 'C';
    MatrixXd work(n,k);
    work.setZero();
    dlarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V->data(), &m, T->data(), &k, C->data(), &m, work.data(), &n);
    #else 
    int info = LAPACKE_dlarfb(LAPACK_COL_MAJOR, 'L', 'T', 'F', 'C', m, n, k, V->data(), m, T->data(), k, C->data(), m);
    assert(info==0);
    #endif
}

void larfb(MatrixXd* V, MatrixXd* T, VectorXd* C, char side, char trans, char direct, char storev){
    int m = C->rows();
    int n = C->cols();
    int k = V->cols();
    assert(m == V->rows());
    assert(T != nullptr);
    assert(T->rows() == k);


    #ifdef USE_MKL
    MatrixXd work(n,k);
    work.setZero();
    dlarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V->data(), &m, T->data(), &k, C->data(), &m, work.data(), &n);
    #else 
    int info = LAPACKE_dlarfb(LAPACK_COL_MAJOR, side, trans, direct, storev, m, n, k, V->data(), m, T->data(), k, C->data(), m);
    assert(info==0);
    #endif
}


// RRQR
void geqp3(MatrixXd* A, VectorXi* jpvt, VectorXd* tau) {
    int m = A->rows();
    int n = A->cols();
    if (m == 0 || n == 0)
        return;
    assert(jpvt->size() == n);
    assert(tau->size() == min(m,n));
    int info = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, m, n, A->data(), m, jpvt->data(), tau->data());
    assert(info == 0);
    for (int i = 0; i < jpvt->size(); i++)
        (*jpvt)[i] --;
}

// Householder vector
double house(VectorXd& x){
    int m = x.size();
    double sigma = x.bottomRows(m-1).squaredNorm();
    double beta = 0;
    VectorXd v(m);
    v.bottomRows(m-1) = x.bottomRows(m-1);
    v.setZero();
    // if (x(0)>=0.0){
    //     v(0) = x(0)+x.norm();
    // }
    // else {v(0) = x(0)-x.norm();}
    // v /= v(0);
    // beta = 2/v.squaredNorm();
    v(0)=1;
    v.bottomRows(m-1) = x.bottomRows(m-1);
    if (sigma == 0 && x(0) >= 0.0) beta = 0;
    else if (sigma == 0 && x(0) < 0.0) beta = -2;
    else {
        double mu = x.norm();
        if (x(0) <= 0.0) v(0) = x(0)-mu;
        else v(0) = -sigma/(x(0)+mu);
        beta = 2*v(0)*v(0)/(sigma + v(0)*v(0));
        v /= v(0);
    }
    x = v;
    return beta;
}

// RRQR with truncation when the max R_ii < tol
void rrqr(MatrixXd* A, VectorXi* jpvt, VectorXd* tau, double& tol, int& rank) {
    int m = A->rows();
    int n = A->cols();
    if (m == 0 || n == 0)
        return;
    assert(jpvt->size() == n);
    assert(tau->size() == min(m,n));

    // Calucate norms of each column
    VectorXd cnrm = A->colwise().squaredNorm();
    int k;
    double t = cnrm.maxCoeff(&k);
    int r = 0;
    double tolsq = tol*tol;
    double beta = 0.0;

    jpvt->setLinSpaced(n, 0, n-1);
    cout << "cnrm " << cnrm.transpose() << endl; 

    // cout << *A << endl;
    while (t > tolsq ){
        // permute the column
        // cout << cnrm.transpose() << endl;
        // A->col(r).swap(A->col(k));

        cblas_dswap(m, A->col(r).data(), 1, A->col(k).data(), 1);
        jpvt->row(r).swap(jpvt->row(k));
        cnrm.row(r).swap(cnrm.row(k));

        VectorXd v = A->col(r).tail(m-r);
        // auto beta = house(v); // Find householder vector
        // (*tau)(r) = beta;

        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m-r, 1, v.data(), m-r, &beta);
        (*tau)(r) = beta;

        MatrixXd Atemp = A->block(r,r,m-r,n-r); 
        int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', m-r, n-r, 1, v.data(), m-r, &beta, Atemp.data(), m-r); // Apply householder transform on the rest of the matrix
        assert(info == 0);

        A->block(r,r,m-r,n-r) = Atemp;
        // A->block(r,r,m-r,n-r) -= beta*v*v.transpose()*A->block(r,r,m-r,n-r);
        A->col(r).tail(m-r-1) = v.tail(m-r-1); // store the vector in A
     


        r += 1;
        if (r < min(m,n)) {
            // Update the norms of the other columns
            cnrm.tail(n-r) -= (A->row(r-1).tail(n-r)).cwiseAbs2();
            t = cnrm.tail(n-r).maxCoeff(&k);
            k = k+r;
            cout << "cnrm " << cnrm.transpose() << endl; 
            // k = jpvt(k);
        }
        else break;
    }
    rank = r;
    // cout << endl <<  "rank " << r << endl;
    // cout << cnrm.transpose() << endl;
    // cout << "rrqr " << A->diagonal().transpose() << endl;
}

/*  Randomized RRQR - Code originally from Bazyli Klockiewicz, Leopold Cambier */
// Generate random matrices
// template<typename T>
// Eigen::MatrixXd get_gaussian(const int rows, const int cols, T* gen) {
//     std::normal_distribution<double> norm_dist(0.0, 1.0);
//     Eigen::MatrixXd W(rows, cols);
//     for(int j = 0; j < cols; j++){
//         for(int i = 0; i < rows; i++){
//             W(i,j) = norm_dist(*gen);
//         }
//     }
//     Eigen::VectorXd col_norms = W.colwise().norm();
//     assert(col_norms.size() == cols);
//     assert(col_norms.minCoeff() >= 0);
//     if (col_norms.minCoeff() > 0) {
//         return W.normalized();
//     }
//     return W;
// }

template<> MatrixXd get_gaussian(const int rows, const int cols, std::default_random_engine* gen);
template<> MatrixXd get_gaussian(const int rows, const int cols, std::minstd_rand* gen);



MatrixXd get_uniform(const int rows, const int cols) {
    MatrixXd W = MatrixXd::Random(rows,cols);
    VectorXd col_norms = W.colwise().norm();
    assert(col_norms.size() == cols);
    assert(col_norms.minCoeff() >= 0);
    if (col_norms.minCoeff() > 0) {
        return W.normalized();
    }
    return W;
}

// Randomized RRQR - Algo 4.2: Adaptive randomized range finder
void random_rrqr(const MatrixXd* A, MatrixXd* v, VectorXd* h,  double tol, std::function<MatrixXd(int,int)> gen_gaussian){
    const int rows = A->rows();
    const int cols = A->cols();
    if(rows == 0 || cols == 0) {
        *v = MatrixXd::Zero(rows, 0);
        *h = VectorXd::Zero(0);
        return;
    }

    MatrixXd Q_all; // Holds [Q_1, Q_2, ...], ie, where we concatenate all the Q's
    // this is lower than 2-norm
    const double target_norm = tol * A->colwise().norm().maxCoeff() * 0.1 * sqrt(3.14 / 2);
    // using 3 gives probability of > 0.999
    int r= 3;
    const int step = max(r,min(cols,static_cast<int>(0.05 * rows)));

    // Intial Q
    int initial_step = max(r,static_cast<int>(0.4 * rows));
    // timer t_rand = wctime();
    initial_step = min(rows,initial_step);
    MatrixXd W = gen_gaussian(cols,initial_step); 
    timer t_gemm = wctime();
    Q_all.noalias() = (*A) * W; // With EIGEN using Blas & Lapack this should be fine
    // timer t_qr = wctime();
    // Q_all = MatrixXd(A->rows(), W.cols());
    // gemm(A, &W, &Q_all, CblasNoTrans, CblasNoTrans, 1.0, 0.0);
    VectorXd h_tmp = VectorXd::Zero(initial_step);
    geqrf(&Q_all, &h_tmp);
    orgqr(&Q_all, &h_tmp);
    // timer t_end = wctime();  
    int rank = Q_all.cols();
    assert(rank <= rows);
    while (rank < rows) {
        // timer t_rand = wctime();
        int actual_step = min(step, rows - rank);
        assert(actual_step >= 1);
        MatrixXd W = gen_gaussian(cols,actual_step);
        // timer t_gemm = wctime();
        MatrixXd AW = (*A) * W; 
        AW -= Q_all * (Q_all.transpose() * AW); // FIXME: is this stable ? Cf Gramm-Schmidt
        // With EIGEN using Blas & Lapack this should be fine
        // MatrixXd AW = MatrixXd(A->rows(),W.cols());
        // MatrixXd QTAW = MatrixXd(Q_all.cols(),W.cols());
        // gemm(A,      &W,      &AW,   CblasNoTrans, CblasNoTrans,  1.0, 0.0);
        // gemm(&Q_all, &AW,     &QTAW, CblasTrans,   CblasNoTrans,  1.0, 0.0);
        // gemm(&Q_all, &QTAW,   &AW,   CblasNoTrans, CblasNoTrans, -1.0, 1.0);
        // timer t_qr = wctime();
        assert(AW.size() > 0); // Minimum 1 col & 1 row
        double norm = AW.colwise().norm().maxCoeff();
        // cout << "norm " << norm << endl;
        if (norm < target_norm)  {
            break;
        }
        int new_rank = Q_all.cols() + actual_step;
        assert(new_rank <= rows);
        VectorXd h_tmp = VectorXd::Zero(new_rank);
        MatrixXd Q_tmp(Q_all.rows(), new_rank);
        Q_tmp << Q_all, AW;
        geqrf(&Q_tmp, &h_tmp);
        orgqr(&Q_tmp, &h_tmp);
        Q_all = Q_tmp;
        rank = new_rank;
        assert(rank == Q_all.cols());
        // timer t_end = wctime();
    } 
    assert(Q_all.cols() == rank);
    assert(Q_all.cols() <= rows);
    assert(Q_all.rows() == rows);
    *v = Q_all;
    *h = VectorXd::Zero(rank);
    geqrf(v, h); 
}

// Rank revealing QR using dlaqps
void laqps(MatrixXd* A, VectorXi* jpvt, VectorXd* tau, double& tol, int block_size, int& rank) {
    #ifdef USE_MKL
    int m = A->rows();
    int n = A->cols();
    if (m == 0 || n == 0)
        return;
    assert(jpvt->size() == n);
    assert(tau->size() == min(m,n));

    
    int offset = 0;
    int kb =0;
    VectorXd vn1 = A->colwise().norm();
    VectorXd vn2 = vn1;

    double t=1.0 ;
    double r0 = vn1.maxCoeff();
    VectorXd aux(block_size); // Auxilary vector
    aux.setZero();
    jpvt->setLinSpaced(n, 1, n);

    int nt = n;
    if (block_size > min(m,n)){
        block_size = min(m,n);
    }

    while (t > tol){
        MatrixXd F(nt, block_size);
        F.setZero();

        dlaqps(&m, &nt, &offset, &block_size, &kb, A->block(0,offset,m, nt).data(), &m, jpvt->tail(nt).data(), 
               tau->segment(offset, block_size).data(), vn1.tail(nt).data(), vn2.tail(nt).data(), aux.data(), F.data(), &nt);

        offset += kb;
        nt -= kb;

        if (offset < min(m,n)) {
            t = vn1.tail(n-offset).maxCoeff()/r0; // relative tolerance
            if (offset + block_size > min(m,n)){
                block_size = min(m,n) - offset;
            }
        }
        else break;
    }

    rank = offset;

    for (int i = 0; i < jpvt->size(); i++)
        (*jpvt)[i] --;

    #endif
}

// Full EVD
void geevd(Eigen::MatrixXd* A, Eigen::MatrixXd* U, Eigen::VectorXd* S) {
    int m = A->rows();
    int n = A->cols();
    assert(m=n);
    int k = m;
    assert(U->rows() == m && U->cols() == m);
    assert(S->size() == k);
    if(k == 0)
        return;
    VectorXd superb(k-1);
    int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'N', m, n, A->data(), m, S->data(), U->data(), m, nullptr, n, superb.data());
    assert(info == 0);
}

// Full SVD
void gesvd(Eigen::MatrixXd* A, Eigen::MatrixXd* U, Eigen::VectorXd* S, Eigen::MatrixXd* VT) {
    int m = A->rows();
    int n = A->cols();
    int k = min(m,n);
    assert(U->rows() == m && U->cols() == m);
    assert(VT->rows() == n && VT->cols() == n);
    assert(S->size() == k);
    if(k == 0)
        return;
    VectorXd superb(k-1);
    int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, A->data(), m, S->data(), U->data(), m, VT->data(), n, superb.data());
    assert(info == 0);
}

// Only singular values
void gesvd_values(Eigen::MatrixXd* A,  Eigen::VectorXd* S) {
    int m = A->rows();
    int n = A->cols();
    int k = min(m,n);
    assert(S->size() == k);
    if(k == 0)
        return;
    VectorXd superb(k-1);
    MatrixXd* U = new MatrixXd(A->rows(), A->rows());
    MatrixXd* VT = new MatrixXd(A->cols(), A->cols());

    int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'N', m, n, A->data(), m, S->data(), U->data(), m, VT->data(), n, superb.data());
    assert(info == 0);
    delete U;
    delete VT;
}

// QR
void geqrf(MatrixXd* A, VectorXd* tau) {
    int m = A->rows();
    int n = A->cols();
    if (m == 0 || n == 0)
        return;
    assert(tau->size() == min(m,n));
    int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A->data(), m, tau->data());
    assert(info == 0);
}

int choose_rank(VectorXd& s, double tol, bool rel) {
    if (tol == 0) {
        return s.size();
    } else if (tol >= 1.0) {
        return 0;
    } else {
        if (s.size() <= 1) {
            return s.size();
        } else {
            double sref;
            if (rel) sref=abs(s[0]);
            else sref = 1.0;
            
            int rank = 1;
            while(rank < s.size() && abs(s[rank]) / sref >= tol) {
                rank++;
            }
            assert(rank <= s.size());
            return rank;
        }
    }
}

void block2dense(VectorXi &rowval, VectorXi &colptr, VectorXd &nnzval, int i, int j, int li, int lj, MatrixXd *dst, bool transpose) {
    if(transpose) {
        assert(dst->rows() == lj && dst->cols() == li);
    } else {
        assert(dst->rows() == li && dst->cols() == lj);
    }
    for(int col = 0; col < lj; col++) {
        // All elements in column c
        int start_c = colptr[j + col];
        int end_c = colptr[j + col + 1];
        int size = end_c - start_c;
        auto start = rowval.data() + start_c;
        auto end = rowval.data() + end_c;
        // Find i
        auto found = lower_bound(start, end, i);
        int id = distance(start, found);
        // While we are between i and i+i...
        while(id < size) {
            int row = rowval[start_c + id];
            if(row >= i + li) {
                break;
            }
            row = row - i;
            double val = nnzval[start_c + id];
            if(transpose) {
                (*dst)(col,row) = val;
            } else {
                (*dst)(row,col) = val;
            }
            id ++;
        }
    }
}

MatrixXd linspace_nd(int n, int dim) {
    MatrixXd X = MatrixXd::Zero(dim, pow(n, dim));
    if (dim == 1) {
        for(int i = 0; i < n; i++) {
            X(0,i) = double(i);
        }
    } else if (dim == 2) {
        int id = 0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                X(0,id) = i;
                X(1,id) = j;
                id ++;
            }
        }
    } else if (dim == 3) {
        int id = 0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                for(int k = 0; k < n; k++) {
                    X(0,id) = i;
                    X(1,id) = j;
                    X(2,id) = k;
                    id ++;
                }
            }
        }
    }
    return X;
}

// Compute A[p,p]
SpMat symm_perm(SpMat &A, VectorXi &p) {
    // Create inverse permutation
    VectorXi pinv(p.size());
    for(int i = 0; i < p.size(); i++)
        pinv[p[i]] = i;
    // Initialize A[p,p]
    int n = A.rows();
    int nnz = A.nonZeros();
    assert(n == A.cols()); 
    SpMat App(n, n);
    App.reserve(nnz);
    // Create permuted (I, J, V) values
    vector<Triplet<double>> vals(nnz);
    int l = 0;
    for (int k = 0; k < A.outerSize(); k++){
        for (SpMat::InnerIterator it(A, k); it; ++it){
            int i = it.row();
            int j = it.col();
            double v = it.value();
            vals[l] = Triplet<double>(pinv[i],pinv[j],v);
            l ++;
        }
    }
    // Create App
    App.setFromTriplets(vals.begin(), vals.end());
    return App;
}

SpMat col_perm(SpMat &A, VectorXi &p) {
    // Create inverse permutation
    VectorXi pinv(p.size());
    for(int i = 0; i < p.size(); i++)
        pinv[p[i]] = i;
    
    // Initialize A[p,p]
    int c = A.cols();
    int r = A.rows();
    int nnz = A.nonZeros();

    SpMat App(r,c);
    App.reserve(nnz);
    // Create permuted (I, J, V) values
    vector<Triplet<double>> vals(nnz);
    int l = 0;
    for (int k = 0; k < A.outerSize(); k++){
        for (SpMat::InnerIterator it(A, k); it; ++it){
            int i = it.row();
            int j = it.col();
            double v = it.value();
            vals[l] = Triplet<double>(i,pinv[j],v);
            l ++;
        }
    }
    // Create App
    App.setFromTriplets(vals.begin(), vals.end());
    return App;
}

// Random [-1,1]
VectorXd random(int size, int seed) {
    mt19937 rng;
    rng.seed(seed);
    uniform_real_distribution<double> dist(-1.0,1.0);
    VectorXd x(size);
    for(int i = 0;i < size; i++) {
        x[i] = dist(rng);
    }
    return x;
}

MatrixXd random(int rows, int cols, int seed) {
    mt19937 rng;
    rng.seed(seed);
    uniform_real_distribution<double> dist(-1.0,1.0);
    MatrixXd A(rows, cols);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            A(i,j) = dist(rng);
        }
    }
    return A;
}
