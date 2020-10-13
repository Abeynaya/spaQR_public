#ifndef STATS_H
#define STATS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include "cluster.h"
#include "tree.h"
#include "util.h"

/**
* Statistics 
**/

class Tree;

void write_stats(const Tree& t, std::string fn);
void write_basic_info(const Tree& t, std::string fn); 
void write_mats(const Tree& t, std::string fn);
void write_mats_at_lvl(const Tree& t, std::string fn, int l, double tol);




#endif