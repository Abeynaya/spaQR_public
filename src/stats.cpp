#include "stats.h"

using namespace std;
using namespace Eigen;

/* Write cluster sizes */
void write_stats(const Tree& t, string fn) {
    ofstream stats_s(fn);
    stats_s << t.rows() << ";" << t.cols() << ";" << t.levels() << "\n";
    stats_s << "id;rows;cols;newrows;rank" << "\n";

    for (auto n: t.get_clusters_at_lvl(0)){
    	stats_s << n->get_order() << ";" << n->original_rows() << ";" << n->original_cols() << ";" << n->rows() << ";" << n->cols() << endl;
    }
    stats_s.close();
}

void write_basic_info(const Tree& t, string fn) {
    ofstream stats_s(fn);
    stats_s << t.rows() << ";" << t.cols() << ";" << t.levels() << "\n";
    stats_s.close();
}


void write_mats(const Tree& t, string fn){
	ofstream mats_s(fn);
	mats_s << "id; {edgesOut->A12}, {edgesOut->A21}" << "\n";

	for (auto n: t.get_clusters()){
		mats_s << n->get_order() << ";" << n->get_id() << "\n";
		for (auto e: n->edgesOut){
			if (e->A12 != nullptr){
				mats_s << e->n2->get_order() << ";" << e->n2->get_id() << ",";
			}
		}
		mats_s << "\n";
		for (auto e: n->edgesOut){
			if (e->A21 != nullptr){
				mats_s << e->n2->get_order() << ";" << e->n2->get_id() << ",";
			}
		}
		mats_s << "\n";
	}
	mats_s.close();
}

void write_mats_at_lvl(const Tree& t, string fn, int l, double tol){
	ofstream mats_s;
	mats_s.open(fn, ios::out | ios::app);

	for (auto n: t.get_clusters_at_lvl(l)){
		mats_s << l << ":" << n->get_order() << ";" << n->rows() << ";" << n->cols() << ";";

		for (auto e: n->edgesOut){
			if (e->A12 != nullptr){
				MatrixXd A12t = *e->A12;
				VectorXi jpvt = VectorXi::Zero(e->A12->cols());
    			VectorXd t = VectorXd(min(e->A12->rows(), e->A12->cols()));
    			geqp3(&A12t, &jpvt, &t);
    			VectorXd rii = A12t.diagonal();
    			int r = choose_rank(rii, tol);
    			double color = (double)r/(double)t.size();
				mats_s << e->n2->get_order() << "|" << color  << ",";
			}
		}
		mats_s << ";";
		for (auto e: n->edgesOut){
			if (e->A21 != nullptr){
				MatrixXd A21t = *e->A21;
				VectorXi jpvt = VectorXi::Zero(e->A21->cols());
    			VectorXd t = VectorXd(min(e->A21->rows(), e->A21->cols()));
    			geqp3(&A21t, &jpvt, &t);
    			VectorXd rii = A21t.diagonal();

    			int r = choose_rank(rii, tol);
    			double color = (double)r/(double)t.size();
				mats_s << e->n2->get_order() << "|" << color << ",";
			}
		}
		mats_s << "\n";
	}
	mats_s.close();
}