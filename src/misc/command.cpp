#include "command.hpp"
#include "global.hpp"

Command::Command() {
    AddCommand("alias", bind(&Command::Alias, this, placeholders::_1, placeholders::_2));
    AddCommand("help", bind(&Command::Help, this, placeholders::_1, placeholders::_2));
    AddCommand("history", bind(&Command::History, this, placeholders::_1, placeholders::_2));
    AddCommand("memory", bind(&Memory::Run, &usage.memory_, placeholders::_1, placeholders::_2));
    AddCommand("quit", bind(&Command::Quit, this, placeholders::_1, placeholders::_2));
    AddCommand("set", bind(&Command::Set, this, placeholders::_1, placeholders::_2));
    AddCommand("source", bind(&Command::Source, this, placeholders::_1, placeholders::_2));
    AddCommand("time", bind(&Time::Run, &usage.time_, placeholders::_1, placeholders::_2));
    AddCommand("unalias", bind(&Command::Unalias, this, placeholders::_1, placeholders::_2));
    AddCommand("unset", bind(&Command::Unset, this, placeholders::_1, placeholders::_2));
    AddCommand("usage", bind(&Usage::Run, &usage, placeholders::_1, placeholders::_2));
}
void Command::Interactive() {
    // start interactive mode
    usage.time_.Pause();
    while (!feof(stdin)) {
        // print command line prompt and
        // get the command from the user
        string command = GetInput();

        // execute the user's command
        int status = ExecuteCommand(command);

        // stop if the user quitted or an error occurred
        if (status == -1 || status == -2)
            break;
    }
}
string Command::GetInput() {
    char * line = NULL;
    line = readline(PROMPT);  
    if (line == NULL){
        printf("***EOF***\n"); exit(0);
    }
    else {
        string s_line(line);
        s_line.erase(s_line.find_last_not_of("\n") + 1);

        free(line);
        return s_line;
    }
}
int Command::ExecuteCommand(const string& command) {
    AddHistory(command);

    vector<char*> argv;
    int f_status = 0;
    string next_command = command;
    while (f_status == 0 && next_command.size() > 0) {
        next_command = SplitLine(next_command, argv);

        int loop = 0;
        f_status = ApplyAlias(argv, loop);
        ApplySet(argv);
        // for (auto c : argv) {
        //     printf(" %s", c);
        // }
        // printf("\n");
        if ( f_status == 0 ) 
            f_status = DispatchCommand(argv.size(), &argv[0]);
        FreeArgv(argv);
    } 
    return f_status;
}
void Command::AddHistory(const string& command) {
    string s = command;
    s.erase(s.find_last_not_of("\n") + 1);

    HISTORY_STATE* history_info = history_get_history_state();
    HIST_ENTRY** a_history = history_list();
    for (int i = 0; i < history_info->length; ++i) { /* output history list */
        if ((string)a_history[i]->line == s) {
            HIST_ENTRY* entry = remove_history(i);
            assert(entry);
            free(entry->line);
            free(entry);
            break;
        }
    }
    add_history(s.c_str());
    return;
}
string Command::SplitLine(const string& command, vector<char*>& argv) {
    string result = command;
    while (true) {
        // skip leading white space 
        result.erase(0, result.find_first_not_of(" \t\n\v\f\r"));

        // skip until end of this token 
        int i;
        bool single_quote = false, double_quote = false;
        for (i = 0; i < (int)result.size(); ++i) {
            if ((result[i] == ';' || result[i] == '#' || result[i] == ' ' || result[i] == '\t' || result[i] == '\n' ||
                    result[i] == '\v' || result[i] == '\f' || result[i] == '\r') && !single_quote && !double_quote) {
                break;
            }
            else if ( result[i] == '\'' ) {
                single_quote = !single_quote;
            }
            else if ( result[i] == '"' ) {
                double_quote = !double_quote;
            }
        }
        if ( single_quote || double_quote ) {
            throw std::runtime_error("** cmd warning: ignoring unbalanced quote ...\n");
        }
        if (i == 0) {
            break;
        }

        char* new_arg = (char*) malloc(sizeof(char) * (i + 1));
        int k = 0;
        for (int j = 0; j < i; ++j) {
            if (( result[j] != '\'') && (result[j] != '\"')) {
                new_arg[k++] = isspace((int)result[j]) ? ' ' : result[j];
            }
        }
        new_arg[k] = '\0';
        argv.emplace_back(new_arg);
        result = result.substr(i);
    }
    result.erase(0, result.find_first_not_of(";"));
    if (result[0] == '#') {
        result = "";
    }
    // cout<< result.size()<< endl;
    // cout<< result<< endl;
    // getchar();
    return result;
}
int Command::ApplyAlias(vector<char*>& argv, int& loop) {
    if (argv.size() == 0)
        return 0;

    bool stop = false;
    for (; loop < 100; ++loop) {
        map<string, vector<string>>::iterator it = m_alias_.find(argv[0]);
        if (stop || it == m_alias_.end()) {
            return 0;
        }
        vector<string>& alias_argv = it->second;

        stop |= ((string)argv[0] == alias_argv[0]);

        vector<char*> added_argv;
        for (int i = 0; i < (int)alias_argv.size(); ++i) {
            string argument = alias_argv[i];

            vector<char*> tmp_argv;
            int f_status = 0;
            while ( f_status == 0 ) {
                argument = SplitLine(argument, tmp_argv);
                /*
                 * If there's a complete `;' terminated command in `arg',
                 * when split_line() returns arg[0] != '\0'.
                 */
                if (argument.size() == 0) { /* just a bunch of words */
                    break;
                }
                else {
                    f_status = ApplyAlias(tmp_argv, loop);
                    ApplySet(argv);
                    if (f_status == 0) {
                        f_status = DispatchCommand(tmp_argv.size(), &tmp_argv[0]);
                    }
                    FreeArgv(tmp_argv);
                }
            }
            if (f_status != 0) {
                FreeArgv(added_argv);
                assert(tmp_argv.size() == 0);
                return 1;
            }

            added_argv.insert(added_argv.end(), tmp_argv.begin(), tmp_argv.end());
            tmp_argv.clear();
        }
        assert(added_argv.size() >= 1);
        free(argv[0]);
        argv[0] = added_argv[0];
        argv.insert(argv.begin() + 1, added_argv.begin() + 1, added_argv.end());
        added_argv.clear();
    }

    printf("** cmd warning: alias loop.\n");
    return 1;
}
void Command::ApplySet(vector<char*>& argv) {
    if (argv.size() > 0 && (string)argv[0] != "unset") {
        for (int i = 1; i < (int)argv.size(); ++i) {
            map<string, string>::iterator it = m_set_.find((string)argv[i]);
            if (it != m_set_.end()) {
                // printf("command: %s\n", argv[i]);
                // printf("name: %s, value: %s\n", it->first.c_str(), it->second.c_str());
                free(argv[i]);
                argv[i] = (char*) malloc(sizeof(char) * (it->second.size() + 1));
                strcpy(argv[i], it->second.c_str());
                // for (int j = 0; j < (int)argv.size(); ++j) {
                //     cout << argv[j] << endl;
                // }
            }
        }
    }
}
int Command::DispatchCommand(int argc, char** argv) {
    if (argc == 0) {
        return 0;
    }

    // get the command
    map<string, function<int(int, char**)>>::iterator it = m_commands_.find((string)argv[0]);
    if (it != m_commands_.end()) {
        // execute the command
        usage.time_.Play();
        int status = (it->second)(argc, argv);
        usage.time_.Pause();
        return status;
    }
    else if (CheckCommandInShell(argv[0])) {
        string cmd = argv[0];
        for (int i = 1; i < argc; ++i) {
            cmd = cmd + " " + argv[i];
        }
        system(cmd.c_str());
        return 0;
    }
    else {   // the command is not in the table
        fprintf(stderr, "** cmd error: unknown command '%s'\n", argv[0] );
        return 1;
    }
}
bool Command::CheckCommandInShell(const string& command) {
    string check_command = "which " + command + " > /dev/null 2>&1";
    return !(system(check_command.c_str()));
}

void Command::FreeArgv(vector<char*>& argv) {
    for (int i = 0; i < (int)argv.size(); ++i) {
        free(argv[i]);
    }
    argv.clear();
}
int Command::Alias(int argc, char **argv) {
    ArgumentParser program("alias");
    program.AddArgument("command")
        .Remaining();

    bool res;
    try {
    res = program.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    if (!program.Present("command")) {
        PrintAliasTable();
    }
    else {
        vector<string> command = program.Get<vector<string>>("command");
        if (command.size() == 1) {
            PrintAlias(command[0]);
        }
        else {
            vector<string>& alias_command = m_alias_[command[0]];
            alias_command.clear();
            alias_command.insert(alias_command.begin(), command.begin() + 1, command.end());
        }
    }
    return 0;
}
void Command::PrintAliasTable() {
    vector<string> v_alias;
    for (auto it = m_alias_.begin(); it != m_alias_.end(); ++it) {
        v_alias.emplace_back(it->first);
    }
    sort(v_alias.begin(), v_alias.end());

    for (int i = 0; i < (int)v_alias.size(); ++i) {
        PrintAlias(v_alias[i]);
    }
}
void Command::PrintAlias(const string& command) {
    map<string, vector<string>>::iterator it = m_alias_.find(command);
    if (it != m_alias_.end()) {
        printf("%-15s", it->first.c_str());
        vector<string>& alias = it->second;
        for(int i = 0; i < (int)alias.size(); ++i) {
            printf(" %s", alias[i].c_str());
        }
        printf("\n");
    }
}
int Command::Help(int argc, char **argv) {
    ArgumentParser program("help");
    program.AddDescription("prints the list of available commands by group.");
    program.AddArgument("-d")
        .Help("print usage details to all commands [default = no].")
        .DefaultValue(false)
        .ImplicitValue(true);

    bool res;
    try {
    res = program.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    vector<string> v_command;
    for (auto it = m_commands_.begin(); it != m_commands_.end(); ++it) {
        v_command.emplace_back(it->first);
    }
    sort(v_command.begin(), v_command.end());

    int max_length = 0;
    for (int i = 0; i < (int)v_command.size(); ++i) {
        max_length = max(max_length, (int)v_command[i].size());
    }
    int n_columns = 79 / (max_length + 2);
    for (int i = 0; i < (int)v_command.size(); ++i) {
        if (i % n_columns == 0) {
            printf("\n" ); 
        }
        // print this command
        printf(" %-*s", max_length, v_command[i].c_str());
    }
    if (program.Get<bool>("-d")) {
    // print help messages for all commands in the previous groups
        printf("\n");
        for (int i = 0; i < (int)v_command.size(); ++i) {
            printf("\n");
            std::stringstream stream;
            stream << v_command[i] << " -h";
            // cout << stream.str() << endl;
            ExecuteCommand(stream.str());
        }
    }
    printf("\n");
    return 0;
}
int Command::History(int argc, char **argv) {
    ArgumentParser program("history");
    program.AddDescription("lists the last commands entered on the command line.");
    program.AddArgument("-n")
        .Help("the maximum number of entries to show [default = 20].")
        .DefaultValue(20)
        .Action([](const string& value) { return std::stoi(value); });

    bool res;
    try {
    res = program.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    HISTORY_STATE* history_info = history_get_history_state();
    HIST_ENTRY** a_history = history_list();
    int i = max(history_info->length - program.Get<int>("-n"), 0);
    for (; i < history_info->length; ++i) {
        printf("%2d : %s\n", history_info->length - i, a_history[i]->line);
    }
    return 0;
};
int Command::Quit(int argc, char **argv) {
    ArgumentParser program("quit");
    program.AddArgument("-s")
        .Help("frees all the memory before quitting.")
        .ImplicitValue(true)
        .DefaultValue(false);

    bool res;
    try {
    res = program.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    if (program.Get<bool>("-s")) {
        return -2;
    }
    else {
        return -1;
    }
};
int Command::Set(int argc, char **argv) {
    ArgumentParser program("set");
    program.AddDescription("sets the value of parameter <name>");

    program.AddArgument("name")
        .Nargs(1)
        .DefaultValue((string)"");

    program.AddArgument("value")
        .Nargs(1)
        .DefaultValue((string)"");

    bool res;
    try {
    res = program.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    string name;
    if (program.Get<string>("name") == "") {
        PrintSetTable();
        return 0;
    }
    else {
        name = program.Get<string>("name");
    }
    // printf("name: %s, value: %s\n", name.c_str(), program.Get<string>("value").c_str());

    m_set_.emplace(name, program.Get<string>("value"));
    return 0;
}
void Command::PrintSetTable() {
    vector<string> v_set;
    for (auto it = m_set_.begin(); it != m_set_.end(); ++it) {
        v_set.emplace_back(it->first);
    }
    sort(v_set.begin(), v_set.end());

    for (int i = 0; i < (int)v_set.size(); ++i) {
        PrintSet(v_set[i]);
    }
}
void Command::PrintSet(const string& command) {
    map<string, string>::iterator it = m_set_.find(command);
    if (it != m_set_.end()) {
        printf("%-15s %s\n", it->first.c_str(), it->second.c_str());
    }
}
int Command::Source(int argc, char **argv ) {
    ArgumentParser program("source");
    program.AddArgument("file_name").Required();
    program.AddArgument("-s").Help("silently ignore nonexistant file [default = no].")
        .DefaultValue(false).ImplicitValue(true);
    program.AddArgument("-x").Help("echo each line as it is executed [default = no].")
        .DefaultValue(false).ImplicitValue(true);

    bool res;
    try {
    res = program.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    bool silent = program.Get<bool>("-s");
    bool echo = program.Get<bool>("-x");

    FILE* p_file = fopen(program.Get<string>("file_name").c_str(), "r");
    if (p_file == NULL) {
        if (!silent) {
            printf("Cannot open file \"%s\".\n", program.Get<string>("file_name").c_str());
        }
        return !silent;     /* error return if not silent */
    }
    else {
        int status = 0;
        while ( status == 0 ) {
            int max_str = 1 << 15;
            char line[max_str];
            if (fgets(line, max_str, p_file) == NULL) {
                status = 0; /* successful end of 'source' ; loop? */
                break;
            }

            // cout << line << endl;
            if (echo) {
                printf("%s%s", PROMPT, line );
            }
            status = ExecuteCommand(line);
        }

        if (status > 0) {
            printf("** cmd error: aborting 'source %s'\n", program.Get<string>("file_name").c_str());
        }
        fclose(p_file);
        return status;
    }
}
int Command::Unalias(int argc, char **argv) {
    ArgumentParser program("unalias");
    program.AddArgument("alias_names")
        .Remaining();

    bool res;
    try {
    res = program.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    if (!program.Present("alias_names")) {
        cout << program;
    }
    else {
        vector<string> command = program.Get<vector<string>>("alias_names");
        for (int i = 0; i < (int)command.size(); ++i) {
            m_alias_.erase(command[i]);
        }
    }
    return 0;
}
int Command::Unset(int argc, char **argv) {
    ArgumentParser program("unset");
    program.AddDescription("removes the value of parameter <set_names>");

    program.AddArgument("set_names")
        .Remaining();

    bool res;
    try {
    res = program.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    if (!program.Present("set_names")) {
        cout << program;
    }
    else {
        vector<string> command = program.Get<vector<string>>("set_names");
        for (int i = 0; i < (int)command.size(); ++i) {
            // cout << command[i] << endl;
            m_set_.erase(command[i]);
        }
    }
    return 0;
}
