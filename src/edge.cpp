#include "edge.h"

/* Print Edge*/
std::ostream& operator<<(std::ostream& os, const Edge& e) {
    os << e.n1->get_id() << " " << e.n2->get_id();
    return os;
}