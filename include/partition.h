#ifndef PARTITION_H
#define PARTITION_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <assert.h>
// #include "hsl_mc64d.h"


#ifdef USE_METIS
	#include <metis.h>
#else
	#include "patoh.h"
#endif

#include "tree.h"


using namespace Eigen;
using namespace std;

typedef SparseMatrix<double, 0, int> SpMat; 
tuple<vector<int>,vector<int>> SpMat2CSC(SpMat& A);
tuple<vector<int>,vector<int>> SpMat2CSC_noloops(SpMat& A);

/* Bipartite matching */
void bipartite_row_matching(SpMat& A, vector<ClusterID>& cmap, vector<ClusterID>& rmap, bool cmap2rmap = true);
void bipartite_row_matching(SpMat& A, vector<ClusterID>& cmap, VectorXi& rperm);

/* Rows to clusters */
void rows2clusters(SpMat& A, vector<ClusterID>& cmap, vector<ClusterID>& rmap, int use_matching);
void rmap2cmap(SpMat& A, vector<ClusterID>& cmap, vector<ClusterID>& rmap, int use_matching);

/* Geometric Partition */
void bissect_geo(vector<int> &colptr, vector<int> &rowval, vector<int> &dofs, vector<int> &parts, MatrixXd *X, int start);
vector<ClusterID> GeometricPartitionAtA(SpMat& A, int nlevels, MatrixXd* Xcoo, int use_matching);
void getInterfacesUsingA(SpMat& A, int nlevels, vector<ClusterID>& cmap, vector<ClusterID>& rmap, vector<int> parts, int use_matching);

/* Metis partition */
void partition_metis_RB(SpMat& A, int nlevels, vector<int>& parts);
vector<ClusterID> MetisPartition(SpMat& A, int nlevels);
void getInterfacesUsingATA(SpMat& A, int nlevels, vector<ClusterID>& cmap, vector<int> parts);


/* Hypergraph partitioning */
void partition_patoh(SpMat& A, int nlevels, vector<int>& parts);
vector<ClusterID> HypergraphPartition(SpMat& A, int nlevels);
void getInterfacesUsingA_HUND(SpMat& A, int nlevels, vector<ClusterID>& cmap, vector<int> parts);





#endif