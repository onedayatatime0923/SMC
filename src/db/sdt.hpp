/**CFile****************************************************************
  FileName    [global.hpp]
  SystemName  [NNV.]
  PackageName [Global declarations.]
  Synopsis    []
  Author      [Kevin Chiaming Chang.]
  Affiliation [USC]
  Date        [Ver. 1.0. Started - May 19, 2022.]
***********************************************************************/

#ifndef NNV__db__sdt_hpp
#define NNV__db__sdt_hpp

#include "dbGlobal.hpp"
#include "db/image.hpp"
#include "solver/solver.hpp"

// #define DEBUG
// #define VERBOSE
NNV_NAMESPACING_START

class CInnerNode;
class CLeafNode;

class CNode {
public:
    CNode() : left_(nullptr), right_(nullptr) {};
    friend class CInnerNode;
    friend class CLeafNode;
    friend class CSDT;

    bool                IsInner         () { return !IsLeaf(); }
    bool                IsLeaf          () { return (left_ == nullptr and right_ == nullptr); }
    virtual string      Str             () const = 0;

private:
    CNode*  left_;
    CNode*  right_;
};

class CInnerNode: public CNode {
public:
    friend class CSDT;

    virtual string      Str             () const;
private:
    torch::Tensor   weight_;
    torch::Tensor   bias_;
};

class CLeafNode: public CNode {
public:
    friend class CSDT;

    virtual string      Str             () const;
private:
    torch::Tensor   distribution_;
};

class CSDT {
public:

    CSDT(CSolver& solver) : solver_(solver), root_(nullptr) {};

    void            LoadModel               (const string & model_path);

    vector<list<CSolver::SatVar*>>& SatOutputVariable       () { return sat_output_var_vl_; }
    vector<GRBVar>&                 ConvexInputVariable     () { return convex_input_var_v_; }


    torch::Tensor   Forward                 (const torch::Tensor& input);
    int             Target                  (const torch::Tensor& input) { return torch::argmax(Forward(input)).item<int>(); }

    string          Str                     (bool matrix = false) const;
    string          SatStr                  () const;
    string          ConvexStr               () const;
private:

    void            LoadTorchModel  (const string & model_path);
    void            InitVariable    ();

    CSolver&                        solver_;

    int                             input_dim_;
    int                             output_dim_;

    list<CInnerNode>                inner_node_l_;
    list<CLeafNode>                 leaf_node_l_;
    CNode*                          root_;

    vector<list<CSolver::SatVar*>>  sat_output_var_vl_;
    vector<GRBVar>                  convex_input_var_v_;
};

NNV_NAMESPACING_END
#endif
