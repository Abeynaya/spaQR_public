include Makefile.conf
INCDIR = include
SRCDIR = src
OBJDIR = obj

#Necessary
INCLUDE = -I$(INCDIR) -I$(EIGENDIR) -I$(BLASDIR) -I$(MMIODIR) 
LDFLAGS +=  -L$(BLASLIB) 

CFLAGS = -std=c++11 -Wall -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE
ifdef DEBUG
    CFLAGS += -g -O0 -D_GLIBCXX_DEBUG -fsanitize=address  
else
    CFLAGS += -g -O2 -DEIGEN_NO_DEBUG -DNDEBUG -Wno-unused-variable  
endif

#Metis or Patoh
ifeq ($(USE_METIS),1)
	CFLAGS += -DUSE_METIS
	INCLUDE += -I$(METISDIR)
	LDFLAGS +=  -L$(METISLIB) -lmetis
else
	INCLUDE += -I$(PATOHDIR) 
	LDFLAGS += -L$(PATOHLIB) -lpatoh
endif

#MKL or openblas
ifeq ($(USE_MKL),1)
    CFLAGS += -DUSE_MKL -DEIGEN_USE_MKL_ALL
    LDFLAGS += -Wl,-rpath,$(BLASLIB) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
else
    LDFLAGS += -lopenblas -llapack
endif

#if HSL_MC64 routine is available
ifeq ($(HSL_AVAIL),1)
	CFLAGS += -DHSL_AVAIL
	INCLUDE += -I$(HSLDIR) 
	LDFLAGS +=  -L$(FORTRANLIB) -lgfortran -lquadmath -lm -L$(HSLLIB) -lhsl_mc64
endif


DEPS = $(INCDIR)/util.h  $(INCDIR)/tree.h $(INCDIR)/partition.h  $(INCDIR)/cluster.h $(INCDIR)/edge.h $(INCDIR)/operations.h  $(INCDIR)/stats.h $(INCDIR)/profile.h $(INCDIR)/is.h 
OBJ  = $(OBJDIR)/util.o  $(OBJDIR)/tree.o $(OBJDIR)/partition.o $(OBJDIR)/cluster.o $(OBJDIR)/edge.o $(OBJDIR)/operations.o  $(OBJDIR)/stats.o  $(OBJDIR)/profile.o $(OBJDIR)/is.o

all: spaQR

# Tests
$(OBJDIR)/%.o: %.cpp $(DEPS) 
	$(CC) -c -o $@ $< $(CFLAGS) $(INCLUDE)

# Source
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(INCLUDE) 

# Executables
spaQR: spaQR.cpp $(OBJ) 
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)  $(INCLUDE)  
.PHONY: clean

clean:
	rm -f $(OBJDIR)/*.o spaQR

