
#include "util.hpp"

NNV_NAMESPACING_START

string InsertSpaceAtBeginOfLine(const string& data, int num_space) {
    string res = string(num_space, ' ');
    for (int i = 0; i < (int)data.size(); ++i) {
        if (data[i] == '\n') {
            res += data[i];
            res += string(num_space, ' ');
        }
        else {
            res += data[i];
        }
    }
    return res;
};

bool CheckStartwith(const string& data, const string& target) {
    return (data.size() >= target.size()) and (data.compare(0, target.size(), target) == 0);
};

bool CheckEndwith(const string& data, const string& target) {
    return (data.size() >= target.size()) and (data.compare(data.size() - target.size(), target.size(), target) == 0);
};

bool CheckDigit(const string& data) {
    return all_of(data.begin(), data.end(), ::isdigit);
};

void WriteFile(vector<torch::Tensor>& data_v, const string& file) {
    FILE *fptr;
    if ((fptr = fopen(file.c_str(),"a")) == NULL){
        printf("Error! opening file");
        exit(1);
    }
    else {
        fprintf(fptr, "data\n");
        for(int i = 0; i < (int)data_v.size(); ++i) {
            torch::Tensor d = torch::flatten(data_v[i]);
            // cout << d << endl;
            fprintf(fptr, "%lf", d.index({0}).item<double>());
            for (int j = 1; j < d.sizes()[0]; ++j) {
                fprintf(fptr, " %lf", d.index({j}).item<double>());
            }
            fprintf(fptr, "\n");
       }
       fclose(fptr); 
   }
};
NNV_NAMESPACING_END
