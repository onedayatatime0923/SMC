/**CFile****************************************************************
  FileName    [global.hpp]
  SystemName  [Neural Networks Verifier.]
  PackageName [Global declarations.]
  Synopsis    []
  Author      [Kevin Chiaming Chang.]
  Affiliation [USC]
  Date        [Ver. 1.0. Started - May 19, 2022.]
***********************************************************************/

#ifndef NNV__misc__global_h
#define NNV__misc__global_h

////////////////////////////////////////////////////////////////////////
///                          INCLUDES                                ///
////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <cmath>
#include <functional>
#include <gurobi_c++.h>
#include <iostream>
#include <map>
#include <math.h> 
#include <numeric>
#include <opencv2/opencv.hpp>
#include <set>
#include <stack>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <z3++.h>
#include <zlib.h>
#include "argparse.hpp"
#include "command.hpp"
#include "usage.hpp"


#pragma GCC diagnostic ignored "-Wunused-result"
////////////////////////////////////////////////////////////////////////
///                         NAMESPACES                               ///
////////////////////////////////////////////////////////////////////////

#define NNV_NAMESPACING_START namespace NNV{ using namespace std;
#define NNV_NAMESPACING_END };
#define USING_NNV_NAMESPACING using namespace NNV;
#define RED             "\033[0;31m"
#define BOLD_RED        "\033[1;31m"
#define Green           "\033[0;32m"
#define BOLD_GREEN      "\033[1;32m"
#define YELLOW          "\033[0;33m"
#define BOLD_YELLOW     "\033[1;33m"
#define BLUE            "\033[0;34m"
#define BOLD_BLUE       "\033[1;34m"
#define MAGENTA         "\033[0;35m"
#define BOLD_MAGENTA    "\033[1;35m"
#define CYAN            "\033[0;36m"
#define BOLD_CYAN       "\033[1;36m"
#define RESET           "\033[0m"
#define PROMPT BOLD_BLUE" NNV" BOLD_MAGENTA " > " RESET

////////////////////////////////////////////////////////////////////////
///                         PARAMETERS                               ///
////////////////////////////////////////////////////////////////////////

NNV_NAMESPACING_START

////////////////////////////////////////////////////////////////////////
///                         BASIC TYPES                              ///
////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
///                         helper function                          ///
////////////////////////////////////////////////////////////////////////

inline void check                   (bool expr) { assert(expr); };

template<typename ItType>
struct IterCmp {
    bool operator() (ItType l, ItType q) const {
        return &*l < &*q;
    }
};

////////////////////////////////////////////////////////////////////////
///                         RELATION OPERATOR                        ///
////////////////////////////////////////////////////////////////////////

template <class T> static inline bool operator != (const T& x, const T& y) { return !(x == y); };
template <class T> static inline bool operator >  (const T& x, const T& y) { return y < x;     };
template <class T> static inline bool operator <= (const T& x, const T& y) { return !(y < x);  };
template <class T> static inline bool operator >= (const T& x, const T& y) { return !(x < y);  };


NNV_NAMESPACING_END

#endif
////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////
