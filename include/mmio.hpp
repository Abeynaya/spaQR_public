/*
# linalgCpp
Collection of basic tools or examples for linear algebra in c++
Original code from https://github.com/leopoldcambier/linalgCpp
*/

#ifndef MMIO_HPP
#define MMIO_HPP

#include <assert.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <complex>
#include <string>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace mmio {

    /** Read a line real / cplx **/
    template<typename V>
    V read_entry(std::istringstream& vals) {
        V v;
        vals >> v;
        return v;
    };
    template<>
    std::complex<double> read_entry(std::istringstream& vals) {
        double v1, v2;
        vals >> v1 >> v2;
        return std::complex<double>(v1, v2);
    };
    template<>
    std::complex<float> read_entry(std::istringstream& vals) {
        float v1, v2;
        vals >> v1 >> v2;
        return std::complex<float>(v1, v2);
    };

    /** Read a line real / cplx of coordinate values **/
    template<typename V, typename I>
    Eigen::Triplet<V,I> read_line(std::istringstream& vals) {
        I i, j;
        V v;
        vals >> i >> j >> v;
        return Eigen::Triplet<V,I>(i-1,j-1,v);
    };
    template<>
    Eigen::Triplet<std::complex<double>,int> read_line(std::istringstream& vals) {
        int i, j; double v1, v2;
        vals >> i >> j >> v1 >> v2;
        return Eigen::Triplet<std::complex<double>,int>(i-1,j-1,std::complex<double>(v1, v2));
    };
    template<>
    Eigen::Triplet<std::complex<float>,int> read_line(std::istringstream& vals) {
        int i, j; float v1, v2;
        vals >> i >> j >> v1 >> v2;
        return Eigen::Triplet<std::complex<float>,int>(i-1,j-1,std::complex<float>(v1, v2));
    };

    /** Write a line real / cplx of coordinate values **/
    template<typename V, typename I>
    std::string get_line(I i, I j, V v) {
        return std::to_string(i+1) + " " + std::to_string(j+1) + " " + std::to_string(v);
    };
    template<>
    std::string get_line(int i, int j, std::complex<double> v) {
        return std::to_string(i+1) + " " + std::to_string(j+1) + " " + std::to_string(v.real()) + " " + std::to_string(v.imag());
    };
    template<>
    std::string get_line(int i, int j, std::complex<float> v) {
        return std::to_string(i+1) + " " + std::to_string(j+1) + " " + std::to_string(v.real()) + " " + std::to_string(v.imag());
    };

    /** Symmetric (real) / hermitian (cplx) **/
    template<typename V, typename I>
    Eigen::Triplet<V,I> symmetric(Eigen::Triplet<V,I>& a) {
        return Eigen::Triplet<V,I>(a.col(), a.row(), a.value());
    }
    template<>
    Eigen::Triplet<std::complex<double>,int> symmetric(Eigen::Triplet<std::complex<double>,int>& a) {
        return Eigen::Triplet<std::complex<double>,int>(a.col(), a.row(), std::conj(a.value()));
    }
    template<>
    Eigen::Triplet<std::complex<float>,int> symmetric(Eigen::Triplet<std::complex<float>,int>& a) {
        return Eigen::Triplet<std::complex<float>,int>(a.col(), a.row(), std::conj(a.value()));
    }

    /** Skew-symmetric (real only, really) **/
    template<typename V, typename I>
    Eigen::Triplet<V,I> skew_symmetric(Eigen::Triplet<V,I>& a) {
        return Eigen::Triplet<V,I>(a.col(), a.row(), - a.value());
    }

    enum class format {coordinate, array};
    enum class type {real, integer, complex, pattern};
    enum class property {general, symmetric, hermitian, skew_symmetric};

    std::string prop2str(property p) {
        if(p == property::general) return "general";
        else if(p == property::symmetric) return "symmetric";
        else if(p == property::hermitian) return "hermitian";
        else return "skew_symmetric";
    }

    template<typename V>
    struct V2str {
        static std::string value() {
            if (std::is_same<V,std::complex<double>>::value || std::is_same<V,std::complex<float>>::value) {
                return "complex";
            } else if (std::is_integral<V>::value) {
                return "integer";
            } else {
                return "real";
            }
        }
    };

    struct Header {
        bool bannerOK;
        bool objectOK;
        format f;
        type   t;
        property p;
        Header(std::istringstream& header) {
            std::string banner, object, format, type, properties;
            header >> banner >> object >> format >> type >> properties;
            std::transform(object.begin(),      object.end(),       object.begin(),       ::tolower);
            std::transform(format.begin(),      format.end(),       format.begin(),       ::tolower);
            std::transform(type.begin(),        type.end(),         type.begin(),         ::tolower);
            std::transform(properties.begin(),  properties.end(),   properties.begin(),   ::tolower);
            this->bannerOK = ! banner.compare("%%MatrixMarket");
            this->objectOK = ! object.compare("matrix");
            assert(this->bannerOK);
            assert(this->objectOK);
            if(! format.compare("coordinate")) {
                this->f = format::coordinate;
            } else if(! format.compare("array")) {
                this->f = format::array;
            } else {
                assert(false);
            }
            if (! type.compare("real")) {
                this->t = type::real;
            } else if (! type.compare("integer")) {
                this->t = type::integer;
            } else if (! type.compare("complex")) {
                this->t = type::complex;
            } else if (! type.compare("pattern")) {
                this->t = type::pattern;
            } else { 
                assert(false);
            }
            if (! properties.compare("general")) {
                this->p = property::general;
            } else if (! properties.compare("symmetric")) {
                this->p = property::symmetric;
            } else if (! properties.compare("skew-symmetric")) {
                this->p = property::skew_symmetric;
            } else if (! properties.compare("hermitian")) {
                this->p = property::hermitian;
            } else { 
                assert(false);
            }
        }
    };

    /**
     * Read a sparse matrix in MM format
     */
    template<typename V, typename I>
    Eigen::SparseMatrix<V, Eigen::ColMajor, I> sp_mmread(std::string filename) {
        std::ifstream mfile(filename);
        if (mfile.is_open()) {
            std::string line;
            /** Header **/
            std::getline(mfile, line);
            std::istringstream header(line);
            Header h(header);
            assert(h.f == format::coordinate);
            // assert(h.t != type::pattern);
            /** Find M N K row **/
            while(std::getline(mfile, line)) {
                if(line.size() == 0 || line[0] == '%') continue;
                else break;
            }
            I M, N, K;
            std::istringstream MNK(line);
            MNK >> M >> N >> K;
            std::vector<Eigen::Triplet<V,I>> data;
            if(h.p != property::general) {
                data.reserve(2*K);
            } else {
                data.reserve(K);
            }
            /** Read data **/
            int lineread = 0;
            while(std::getline(mfile, line)) {
                if(line.size() == 0 || line[0] == '%') continue;
                if (h.t == type::pattern) line = line + " 1";
                std::istringstream vals(line);
                Eigen::Triplet<V,I> dataline = read_line<V,I>(vals);
                data.push_back(dataline);
                if(dataline.row() != dataline.col() && (h.p == property::symmetric || h.p == property::hermitian)) {
                    Eigen::Triplet<V,I> dataline2 = symmetric(dataline);
                    data.push_back(dataline2);
                }
                if(dataline.row() != dataline.col() && (h.p == property::skew_symmetric)) {
                    Eigen::Triplet<V,I> dataline2 = skew_symmetric(dataline);
                    data.push_back(dataline2);
                }
                if(h.p == property::skew_symmetric && dataline.row() == dataline.col()) {
                    assert(false);
                }
                if(h.p != property::general && dataline.row() < dataline.col()) {
                    assert(false);
                }
                lineread ++;
            }
            assert(lineread == K);
            Eigen::SparseMatrix<V, Eigen::ColMajor, I> A(M, N);
            A.setFromTriplets(data.begin(), data.end());
            return std::move(A);
        } else {
            throw("Couldn't open file");
        }
    }

    /**
     * Reads a dense matrix in MM format
     */
    template<typename V>
    Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> dense_mmread(std::string filename) {
        std::ifstream mfile(filename);
        if (mfile.is_open()) {
            std::string line;
            /** Header **/
            std::getline(mfile, line);
            std::istringstream header(line);
            Header h(header);
            assert(h.p == property::general);
            assert(h.f == format::array);
            assert(h.t != type::pattern);
            /** Find M N row **/
            while(std::getline(mfile, line)) {
                if(line.size() == 0 || line[0] == '%') continue;
                else break;
            }
            int M, N;
            std::istringstream MNK(line);
            MNK >> M >> N;
            Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> A(M, N); 
            /** Read data **/
            int lineread = 0;
            while(std::getline(mfile, line)) {
                if(line.size() == 0 || line[0] == '%') continue;
                std::istringstream vals(line);
                V v = read_entry<V>(vals);
                int i = (lineread % M);
                int j = (lineread / M);
                A(i,j) = v;
                lineread ++;
            }
            assert(lineread == M*N);
            return std::move(A);
        } else {
            throw("Couldn't open file");
        }
    }

    /**
     * Reads a dense vector in MM format
     */
    template<typename V>
    Eigen::Matrix<V, Eigen::Dynamic, 1> vector_mmread(std::string filename) {
        std::ifstream mfile(filename);
        if (mfile.is_open()) {
            std::string line;
            /** Header **/
            std::getline(mfile, line);
            std::istringstream header(line);
            Header h(header);
            assert(h.p == property::general);
            assert(h.f == format::array);
            assert(h.t != type::pattern);
            /** Find M N row **/
            while(std::getline(mfile, line)) {
                if(line.size() == 0 || line[0] == '%') continue;
                else break;
            }
            int M, N;
            std::istringstream MNK(line);
            MNK >> M >> N;
            assert(N ==1);
            Eigen::Matrix<V, Eigen::Dynamic, 1> b(M); 
            /** Read data **/
            int lineread = 0;
            while(std::getline(mfile, line)) {
                if(line.size() == 0 || line[0] == '%') continue;
                std::istringstream vals(line);
                V v = read_entry<V>(vals);
                int i = (lineread % M);
                // int j = (lineread / M);
                // A(i,j) = v;
                b(i) = v;
                lineread ++;
            }
            assert(lineread == M*N);
            return std::move(b);
        } else {
            throw("Couldn't open file");
        }
    }

    /**
     * Writes a sparse matrix in MM format, using the optional property p.
     * Wether the matrix satisfies or not p is not verified
     */
    template<typename V, int S, typename I>
    void sp_mmwrite(std::string filename, Eigen::SparseMatrix<V,S,I> mat, property p = property::general) {
        std::ofstream mfile;
        mfile.open (filename);
        if (mfile.is_open()) {
            std::string type = V2str<V>::value();
            std::string prop = prop2str(p);
            mfile << "%%MatrixMarket matrix coordinate " << type << " " << prop << "\n";
            int NNZ = 0;
            for (int k = 0; k < mat.outerSize(); ++k) {
                for (typename Eigen::SparseMatrix<V,S,I>::InnerIterator it(mat,k); it; ++it) {
                    if( (p == property::symmetric || p == property::hermitian) && (it.row() < it.col()) ) continue;
                    if( (p == property::skew_symmetric) && (it.row() <= it.col()) ) continue;
                    NNZ ++;
                }
            }
            mfile << mat.rows() << " " << mat.cols() << " " << NNZ << "\n";
            for (int k = 0; k < mat.outerSize(); ++k) {
                for (typename Eigen::SparseMatrix<V,S,I>::InnerIterator it(mat,k); it; ++it) {
                    if( (p == property::symmetric || p == property::hermitian) && (it.row() < it.col()) ) continue;
                    if( (p == property::skew_symmetric) && (it.row() <= it.col()) ) continue;
                    mfile << get_line(it.row(), it.col(), it.value()) << "\n";
                }
            }
        } else {
            throw("Couldn't open file");
        }
    }

    /**
     * Writes a dense matrix in MM format, using the optional property p.
     * Wether the matrix satisfies or not p is not verified
     */
    template<typename V>
    void dense_mmwrite(std::string filename, Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> mat, property p = property::general) {
        std::ofstream mfile;
        mfile.open (filename);
        if (mfile.is_open()) {
            std::string type = V2str<V>::value();
            std::string prop = prop2str(p);
            mfile << "%%MatrixMarket matrix array " << type << " " << prop << "\n";
            mfile << mat.rows() << " " << mat.cols() << "\n";
            for(int j = 0; j < mat.cols(); j++) {
                for(int i = 0; i < mat.rows(); i++) {
                    if( (p == property::symmetric || p == property::hermitian) && (i < j) ) continue;
                    if( (p == property::skew_symmetric) && (i <= j) ) continue;
                    mfile << mat(i,j) << "\n";
                }
            }
        } else {
            throw("Couldn't open file");
        }
    }

}

#endif