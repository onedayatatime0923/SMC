
#ifndef COMMAND_HPP
#define COMMAND_HPP

#include <functional>
#include <list>
#include <map>
#include <readline/readline.h>
#include <readline/history.h>
#include <sstream>
#include <string>
#include "usage.hpp"

using namespace std;

extern Usage usage;

class Command {
public:
    struct Parameter {
        int     n_last_looked =   10;  // do not add history if the same entry appears among the last entries
    } parameter_;

    Command();

    void    Interactive     ();
    void    AddCommand      (const string& name, function<int(int, char**)> func) { m_commands_.emplace(name, func); }

    string  GetInput        ();
    int     ExecuteCommand  (const string& command);
// private:

    void    AddHistory      (const string& command);
    string  SplitLine       (const string& command, vector<char*>& argv);
    int     ApplyAlias      (vector<char*>& argv, int& loop);
    void    ApplySet        (vector<char*>& argv);
    int     DispatchCommand (int argc, char** argv);
    bool    CheckCommandInShell(const string& command);
    void    FreeArgv        (vector<char*>& argv);

    int     Alias           (int argc, char **argv);
    void    PrintAliasTable ();
    void    PrintAlias      (const string& command);
    int     Help            (int argc, char **argv);
    int     History         (int argc, char **argv);
    int     Quit            (int argc, char **argv);
    int     Source          (int argc, char **argv);
    int     Set             (int argc, char **argv);
    void    PrintSetTable   ();
    void    PrintSet        (const string& command);
    int     Unalias         (int argc, char **argv);
    int     Unset           (int argc, char **argv);

    map<string, function<int(int, char**)>>     m_commands_;

    map<string, vector<string>>                 m_alias_;
    map<string, string>                         m_set_;


};

#endif
