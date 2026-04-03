/**CFile****************************************************************
  FileName    [global.hpp]
  SystemName  [NNV.]
  PackageName [Global declarations.]
  Synopsis    []
  Author      [Kevin Chiaming Chang.]
  Affiliation [USC]
  Date        [Ver. 1.0. Started - May 19, 2022.]
***********************************************************************/

#ifndef NNV__db__image_hpp
#define NNV__db__image_hpp

#include "dbGlobal.hpp"

// #define DEBUG
NNV_NAMESPACING_START

class CImage {
public:
    CImage() {};

    int                         Run             (int argc, char **argv);
    void                        Load            (const string & image_path);

    const torch::Tensor         Data            () const { return data_; }
private:

    torch::Tensor                   data_;
};

NNV_NAMESPACING_END
#endif
