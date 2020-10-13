#ifndef PROFILE_H
#define PROFILE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <vector>
#include <list>
#include "util.h"

struct Cluster;
struct Profile{
public:
	int nlevels; // Total levels
	int skip; // skip levels

	// Time to eliminate per separator: [i][j]: sep j at level i
	std::vector<std::vector<double>> time_elmn; 

	 // Time to scale an interface: [i][j][k]: sparsfication at level i, interface k at level j
	std::vector<std::vector<std::list<double>>> time_scale;
	std::vector<std::vector<std::list<double>>> time_spars;
	std::vector<std::vector<std::list<double>>> time_rrqr;

	// Ranks of separators at every level: [i][j][k]: after sparsification at level i, separator k at level j
	std::vector<std::vector<std::vector<unsigned long int>>> rank;

	// Ranks of each interface that gets sparsified at every level: [i][j]: before sparsification at level i, interface j
	std::vector<std::vector<std::list<unsigned long int>>> rank_before;
	std::vector<std::vector<std::list<unsigned long int>>> neighbors;

	std::vector<std::vector<std::list<unsigned long int>>> rank_after;

	// Aspect ratio of the diagonal blocks. [i][j] after merging at level [i], interface [j] 
	std::vector<std::list<double>> aspect_ratio; 



	void write_telmn();
	/* SEPARATOR DETAILS */
	/* Output mean, median, quartile data */
	void write_ranks();
	void write_mean_rank_per_interface();
	void write_neighbors();
	
	void write_mean_tss(); // both spars and scale
	void write_median_tss(); // both spars and scale
	void write_iqr_tss(); // inter quartile range for both spars and scale
	void write_quartile_tss();

	/* INTERFACE DETAILS */
	void write_mean_tspars();


	void write_quartile_tspars();
	void write_quartile_tscale();
	void write_tcomparison(); // Comparing time of sparsification with rrqr time

	// Aspect ratio 
	void write_aspect_ratio();

	Profile(int lvls, int s): nlevels(lvls), skip(s) {
		time_scale = std::vector<std::vector<std::list<double>>>(lvls-skip-1);
		time_spars = std::vector<std::vector<std::list<double>>>(lvls-skip-1);
		time_rrqr = std::vector<std::vector<std::list<double>>>(lvls-skip-1);

	 	rank_before = std::vector<std::vector<std::list<unsigned long int>>>(lvls-skip-1);
	 	rank_after = std::vector<std::vector<std::list<unsigned long int>>>(lvls-skip-1);


	 	neighbors = std::vector<std::vector<std::list<unsigned long int>>>(lvls-skip-1);
		rank = std::vector<std::vector<std::vector<unsigned long int>>>(lvls-skip-1);

		aspect_ratio = std::vector<std::list<double>>(nlevels);



		int L = lvls-skip-1;
		for (int i= 0; i < L; ++i){
			std::vector<std::list<double>> t(L-i);
			std::vector<std::list<unsigned long int>> rb(L-i);

			time_scale[i] = t;
			time_spars[i] = t;
			time_rrqr[i] = t;
			// if (i < L-1){
				rank_before[i] = rb;
				rank_after[i] = rb;
				neighbors[i] = rb;
			// }

			for (int j=0; j < L-i; ++j){
				// std::vector<double> t(pow(2, j), 0.0);
				std::vector<unsigned long int> r(pow(2, j), 0);
				rank[i].push_back(r);
			}
		}

		for (int i =0; i < nlevels; ++i){ // leaves at level 0
			std::vector<double> t(pow(2, nlevels-i-1), 0.0);
			time_elmn.push_back(t);
		}
	};

};

#endif