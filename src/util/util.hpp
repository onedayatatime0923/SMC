#ifndef NNV__util__util_hpp
#define NNV__util__util_hpp

#include "utilGlobal.hpp"

NNV_NAMESPACING_START

string  InsertSpaceAtBeginOfLine    (const string& data, int num_space);
bool    CheckStartwith              (const string& data, const string& target);
bool    CheckEndwith                (const string& data, const string& target);
bool    CheckDigit                  (const string& data);
void    WriteFile                   (vector<torch::Tensor>& data_v, const string& file);

NNV_NAMESPACING_END
#endif
