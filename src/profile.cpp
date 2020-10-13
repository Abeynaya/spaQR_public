#include "profile.h"
using namespace std;

template <typename T1, typename T2> typename T1::value_type quant(const T1 &x, T2 q)
{
    assert(q >= 0.0 && q <= 1.0);

    const auto n  = x.size();
    const auto id = (n - 1) * q;
    const auto lo = floor(id);
    const auto hi = ceil(id);
    const auto qs = x[lo];
    const auto h  = (id - lo);

    return (1.0 - h) * qs + h * x[hi];
}

/* Elimination time */
void Profile::write_telmn(){
	cout << "Mean and quartiles for time of elmn of separators at a level" << endl ;

	cout << "Mean 	  " << "Q1 	  	" << "Q2/ Median 	 " << "Q3 " << endl;
	cout << "\n----------------------------------" << endl ;

	
	for (int i=0; i < nlevels; ++i){
		auto v = time_elmn[i];
		stable_sort(v.begin(), v.end());
		cout << nlevels-i-1 << " 		" << (double)(accumulate(v.begin(), v.end(), 0.0)/v.size());
		cout << "  		" << quant(v,0.25); 
		cout << "  		" << quant(v,0.5); 
		cout << "  		" << quant(v,0.75) <<endl; 

	}

	cout << "\n----------------------------------" << endl ;

}

/* INTERFACE DETAILS */
// Mean, Median, quartiles per interface per level of sparsification
void Profile::write_ranks(){
	cout << "Mean, Median and quartiles of ranks per interface per level of sparsification" << endl;
	cout << "Level      Mean 	  " << "Q1 	  	" << "Q2/ Median 	 " << "Q3 " << endl;
	cout << "\n----------------------------------" << endl ;

	int L = nlevels-skip-1;

	for (int i= 0; i < L; ++i){
		cout << L-i;
		vector<unsigned long int> ranks_per_level; 

		for (int j=0; j < L-i; ++j){
			auto v = rank_before[i][j]; 
			ranks_per_level.insert(ranks_per_level.end(), v.begin(), v.end());	
		}
		stable_sort(ranks_per_level.begin(), ranks_per_level.end());
		if (ranks_per_level.size() >0){
			cout << "       " << (unsigned long int)(accumulate(ranks_per_level.begin(), ranks_per_level.end(), 0.0)/ranks_per_level.size());
			cout << "  		" << (unsigned long int)quant(ranks_per_level,0.25); 
			cout << "  		" << (unsigned long int)quant(ranks_per_level,0.5); 
			cout << "  		" << (unsigned long int)quant(ranks_per_level,0.75) << endl; 
		}
	}
	cout << "\n----------------------------------" << endl ;
}

// Mean rank per interface per level of sparsification (before, after, total before, total after)
void Profile::write_mean_rank_per_interface(){
	cout << "Mean rank per interface per level of sparsification" << endl;

	cout << " Lvl| Rk bfr | Rk aft | Tot bfr | Tot aft" << endl;
	cout << "\n----------------------------------" << endl ;

	int L = nlevels-skip-1;


	for (int i= 0; i < L; ++i){
		cout << L-i;
		int sum_bfr = 0;
		int sum_aftr = 0;
		int n=0; 
		for (int j=0; j < L-i; ++j){
			auto v = rank_before[i][j];
			auto a = rank_after[i][j];
			// begins with top separator 
			if (v.size() > 0){
				sum_bfr += accumulate(v.begin(), v.end(), 0.0);
				sum_aftr += accumulate(a.begin(), a.end(), 0.0);
				n += v.size();
			}

			// stable_sort(v.begin(), v.end());
			// cout << "	 | " << (int)(accumulate(v.begin(), v.end(), 0.0)/v.size());
			
		}
		if (n > 0){
			cout << "   | " << (int)(sum_bfr/n) << "     | " << (int) (sum_aftr/n) 
		    << "     | " << (int)(sum_bfr) << "     | " << (int) (sum_aftr) << endl;
		}
 	}
	cout << "\n----------------------------------" << endl ;	
}

// Mean, Median, quartiles of neighbors per interface per level of sparsification
void Profile::write_neighbors(){
	cout << "Mean, Median and quartiles of neighbors of an interface interface per level of sparsification" << endl;
	cout << "Level      Mean 	  " << "Q1 	  	" << "Q2/ Median 	 " << "Q3 " << endl;
	cout << "\n----------------------------------" << endl ;

	int L = nlevels-skip-1;

	for (int i= 0; i < L; ++i){
		cout << L-i;
		vector<unsigned long int> n_per_level; 

		for (int j=0; j < L-i; ++j){
			auto v = neighbors[i][j]; 
			n_per_level.insert(n_per_level.end(), v.begin(), v.end());	
		}
		stable_sort(n_per_level.begin(), n_per_level.end());
		if (n_per_level.size() >0){
			cout << "       " << (unsigned long int)(accumulate(n_per_level.begin(), n_per_level.end(), 0.0)/n_per_level.size());
			cout << "  		" << (unsigned long int)quant(n_per_level,0.25); 
			cout << "  		" << (unsigned long int)quant(n_per_level,0.5); 
			cout << "  		" << (unsigned long int)quant(n_per_level,0.75) << endl; 
		}
	}
	cout << "\n----------------------------------" << endl ;
}

/* ASPECT RATIO OF DIAGONAL BLOCKS */
// Mean, Median, quartiles per level 
void Profile::write_aspect_ratio(){
	cout << "Mean and quartiles of the aspect ratio of the diagonal blocks for the separators remaining at a level" << endl ;

	cout << "Level          Mean    	  " << "Min        "   <<    " Q1 	  	" << "Q2/ Median 	 " << "Q3        "  << "Max" << endl;
	cout << "\n----------------------------------" << endl ;

	
	for (int i=0; i < nlevels; ++i){
		auto vl = aspect_ratio[i]; // list
		vector<double> v(vl.begin(), vl.end());

		stable_sort(v.begin(), v.end());
		cout << nlevels-i-1 << " 		" << (double)(accumulate(v.begin(), v.end(), 0.0)/v.size());
		cout << "  		" << quant(v,0);
		cout << "  		" << quant(v,0.25); 
		cout << "  		" << quant(v,0.5); 
		cout << "  		" << quant(v,0.75); 
		cout << "  		" << quant(v,1) <<endl; 

	}

	cout << "\n----------------------------------" << endl ;

}


