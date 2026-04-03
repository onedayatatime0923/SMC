#include "argparse.hpp"

ostream& operator<<(ostream &stream, const Argument& argument) {
    stringstream strStream;
    for (int i = 0; i < (int)argument.name_v_.size(); ++i) {
        strStream << argument.name_v_[i] << " ";
    }
    stream.width(argument.program_p_->longest_argument_name_length_);
    stream << strStream.str();
    strStream.str("");

    if (!argument.value_v_.empty()) {
        for (int i = 0; i < (int)argument.value_v_.size(); ++i) {
            strStream << Argument::ConvertStr(argument.value_v_[i]) << " ";
        }
    }
    else if (!argument.default_value_.empty()) {
        strStream << Argument::ConvertStr(argument.default_value_) << " ";
    }
    stream.width(argument.program_p_->longest_argument_value_length_);
    stream << strStream.str();

    stream << argument.help_;
    if (argument.is_required_)
        stream << "[Required]";
    stream << "\n";
    return stream;
}

Argument& Argument::Nargs(int num_args) {
    if (num_args < 0)
        throw logic_error("Number of arguments must be non-negative.\n");
    num_args_ = num_args;
    return *this;
}
vector<string>::const_iterator Argument::Consume(vector<string>::const_iterator start, vector<string>::const_iterator end, const string& used_name) {
    if (is_used_) {
      throw runtime_error("Duplicate argument.\n");
    }
    is_used_ = true;
    used_name_ = used_name;

    if (num_args_ == 0) {
        value_v_.emplace_back(implicit_value_);
        return start;
    }
    else if (num_args_ <= distance(start, end)) {
        if (num_args_ != -1) {
            end = next(start, num_args_);
        }

        if (any_of(start, end, Argument::IsOptional)) {
            throw runtime_error("Optional argument in parameter sequence.\n");
        }

        transform(start, end, back_inserter(value_v_), action_);
        return end;
    }
    else if (!default_value_.empty()) {
        return start;
    }
    else {
        throw runtime_error("Too few arguments.\n");
    }
}

void Argument::Validate() const {
    if (is_optional_) {
        stringstream stream;
        if (is_used_ && num_args_ != -1 && (int)value_v_.size() != num_args_ && default_value_.empty()) {
            stream << used_name_ << ": expected " << num_args_ << " argument(s). " << value_v_.size() << " provided.\n";
            throw runtime_error(stream.str());
        }
        else if (!is_used_ && default_value_.empty() && is_required_) {
            stream << name_v_[0] << ": required.\n";
            throw runtime_error(stream.str());
        }
        else if (is_used_ && is_required_ && value_v_.size() == 0) {
            stream << used_name_ << ": no value provided.\n";
            throw runtime_error(stream.str());
        }
    } else {
        if (num_args_ != -1 && (int)value_v_.size() != num_args_ && default_value_.empty()) {
            stringstream stream;
            if (!used_name_.empty()) {
                stream << used_name_ << ": ";
            }
            stream << num_args_ << " argument(s) expected. " << value_v_.size() << " provided.\n";
            throw runtime_error(stream.str());
        }
    }
}

int Argument::GetArgumentNameLength() const {
    return accumulate(begin(name_v_), end(name_v_), int(0),
           [](int sum, const string &s) { return sum + s.size() + 1; // +1 for space between names
           });
}

int Argument::GetArgumentValueLength() const {
    return accumulate(begin(value_v_), end(value_v_), int(0),
           [](int sum, const any &s) { 
               return sum + ConvertStr(s).size() + 1; // +1 for space between names
           });
}

int Argument::Lookahead(const string& s) {
    if (s.empty())
        return EOF;
    else
        return (int)(unsigned)(s[0]);
}

bool Argument::IsDecimalLitral(string s) {
    auto IsDigit = [](char c) -> bool {
        switch (c) {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                return true;
            default:
                return false;
        }
    };

    // precondition: we have consumed or will consume at least one digit
    auto ConsumeDigits = [=](const string& s) {
        auto it = find_if_not(begin(s), end(s), IsDigit);
        return s.substr(it - begin(s));
    };

    switch (Lookahead(s)) {
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9': {
            s = ConsumeDigits(s);
            if (s.empty())
                return true;
            else
                goto integer_part_consumed;
        }
        case '.': {
            s.erase(0, 1);
            goto post_decimal_point;
        }
        default:
            return false;
    }

    integer_part_consumed:
        switch (Lookahead(s)) {
            case '.': {
                s.erase(0, 1);
                if (IsDigit(Lookahead(s)))
                    goto post_decimal_point;
                else
                    goto exponent_part_opt;
            }
            case 'e':
            case 'E': {
                s.erase(0, 1);
                goto post_e;
            }
            default:
                return false;
        }

    post_decimal_point:
        if (IsDigit(Lookahead(s))) {
            s = ConsumeDigits(s);
            goto exponent_part_opt;
        }
        else {
            return false;
        }

    exponent_part_opt:
        switch (Lookahead(s)) {
            case EOF:
                return true;
            case 'e':
            case 'E': {
                s.erase(0, 1);
                goto post_e;
            }
            default:
                return false;
            }

    post_e:
        switch (Lookahead(s)) {
            case '-':
            case '+':
                s.erase(0, 1);
        }
        if (IsDigit(Lookahead(s))) {
            s = ConsumeDigits(s);
            return s.empty();
        }
        else {
            return false;
        }
}

bool Argument::IsPositional(string name) {
    switch (Lookahead(name)) {
        case EOF:
            return true;
        case '-': {
            name.erase(0, 1);
            if (name.empty())
                return true;
            else
                return IsDecimalLitral(name);
        }
        default:
            return true;
        }
}

bool Argument::Present() const {
    if (!default_value_.empty())
      throw logic_error("Argument with default value always presents.\n");
    else 
        return !value_v_.empty();
}

string Argument::ConvertStr(const any& value) {
    if (string(value.type().name()) == "b") {
        return ((any_cast<bool>(value))? "true": "false");
    }
    else if (string(value.type().name()) == "d") {
        return to_string(any_cast<double>(value));
    }
    else if (string(value.type().name()) == "i") {
        return to_string(any_cast<int>(value));
    }
    else if (string(value.type().name()) == "NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE") {
        return any_cast<string>(value);
    }
    else assert(false);
};

ostream& operator<<(ostream &stream, const ArgumentParser& parser) {
    stream.setf(ios_base::left);
    stream << "Usage: " << parser.program_name_ << " [options] ";

    for (auto it = parser.positional_argument_l_.begin(); it != parser.positional_argument_l_.end(); ++it) {
        stream << it->name_v_.front() << " ";
    }
    stream << "\n\n";

    if(!parser.description_.empty())
        stream << parser.description_ << "\n\n";

    if (!parser.positional_argument_l_.empty()) {
        stream << "Positional arguments:\n";

        for (auto it = parser.positional_argument_l_.begin(); it != parser.positional_argument_l_.end(); ++it) {
            stream << (*it);
        }
        stream << "\n";
    }

    if (!parser.optional_argument_l_.empty()) {
        stream << "Optional arguments:\n";

        for (auto it = parser.optional_argument_l_.begin(); it != parser.optional_argument_l_.end(); ++it) {
            stream << (*it);
        }
    }

    if(!parser.epilog_.empty())
        stream << parser.epilog_ << "\n\n";

    return stream;
}
ArgumentParser::ArgumentParser(const string& program_name, const string& version) : program_name_(program_name), version_(version), longest_argument_name_length_(0), longest_argument_value_length_(0) {
    AddArgument("-h", "--help").Help("shows help message and exits").Nargs(0);
    AddArgument("-v", "--version").Help("prints version information and exits").Nargs(0);
}

ArgumentParser& ArgumentParser::AddDescription(const string& description) {
    description_ = description;
    return *this;
}
ArgumentParser& ArgumentParser::AddEpilog(const string& epilog) {
    epilog_ = move(epilog);
    return *this;
}
int ArgumentParser::ParseArgs(int argc, const char *const argv[]) {
    vector<string> arguments;
    copy(argv, argv + argc, back_inserter(arguments));
    return ParseArgs(arguments);
}
int ArgumentParser::ParseArgs(const vector<string>& arguments) {
    if (ParseArgsInternal(arguments)) {
        return 1;
    }
    else {
        ParseArgsValidate();
        ParseLengthOfLongestArgument();
        return 0;
    }
  }
bool ArgumentParser::Present(const string& name) const {
    return (*this)[name].Present();
}
Argument& ArgumentParser::operator[](const string& name) const {
    auto it = argument_m_.find(name);
    if (it != argument_m_.end())
      return *(it->second);
    else 
        throw logic_error("No such argument.\n");
}
int ArgumentParser::ParseArgsInternal(const vector<string>& v_arguments) {
    if (program_name_.empty() && !v_arguments.empty()) {
      program_name_ = v_arguments.front();
    }

    // clear argument
    for (auto it = positional_argument_l_.begin(); it != positional_argument_l_.end(); ++it) {
        it->value_v_.clear();
        it->is_used_ = false;
    }
    for (auto it = optional_argument_l_.begin(); it != optional_argument_l_.end(); ++it) {
        it->value_v_.clear();
        it->is_used_ = false;
    }

    list_iterator positional_argument_it = positional_argument_l_.begin();
    for (auto it = next(v_arguments.begin()); it != v_arguments.end();) {
        const string& argument = *it;
        if (Argument::IsPositional(argument)) {
            if (positional_argument_it == positional_argument_l_.end()) {
                throw runtime_error("Maximum number of positional arguments exceeded.\n");
            }
            it = positional_argument_it->Consume(it, v_arguments.end());
            ++positional_argument_it;
        }
        else {
            auto argIt = argument_m_.find(argument);
            if (argIt != argument_m_.end()) {
                list_iterator arg_list_it = argIt->second;

                // the first optional argument is --help
                if (arg_list_it == optional_argument_l_.begin()) {
                    cout << *this;
                    return 1;
                }
                // the second optional argument is --version 
                else if (arg_list_it == next(optional_argument_l_.begin())) {
                    cout << version_ << "\n";
                    return 1;
                }

                it = arg_list_it->Consume(next(it), v_arguments.end(), argIt->first);
            } else if (argument.size() > 1 && argument[0] == '-' && argument[1] != '-') {
                ++it;
                for (size_t j = 1; j < argument.size(); j++) {
                    auto hypothetical_argument = string{'-', argument[j]};
                    auto compound_arg_it = argument_m_.find(hypothetical_argument);
                    if (compound_arg_it != argument_m_.end()) {
                        list_iterator arg_list_it = compound_arg_it->second;
                        it = arg_list_it->Consume(it, v_arguments.end(), compound_arg_it->first);
                    } else {
                        cout << argument << endl;
                        throw runtime_error("Unknown argument.\n");
                    }
                }
            }
            else {
                cout << argument << endl;
                throw runtime_error("Unknown argument.\n");
            }
        }
    }
    return 0;
}
void ArgumentParser::ParseArgsValidate() {
// Check if all arguments are parsed
    for (auto it = argument_m_.begin(); it != argument_m_.end(); ++it) {
        it->second->Validate();
    }
}
void ArgumentParser::ParseLengthOfLongestArgument() {
    longest_argument_name_length_ = 0;
    longest_argument_value_length_ = 0;
    for (auto it = argument_m_.begin(); it != argument_m_.end(); ++it) {
        longest_argument_name_length_ = max(longest_argument_name_length_, it->second->GetArgumentNameLength());
        longest_argument_value_length_ = max(longest_argument_value_length_, it->second->GetArgumentValueLength());
    }
}
void ArgumentParser::IndexArgument(list_iterator it) {
    vector<string>& v_names = it->name_v_;
    for (int i = 0; i < (int)v_names.size(); ++i) {
        if (argument_m_.find(v_names[i]) != argument_m_.end())
            throw runtime_error("Duplicate argument.\n");
        argument_m_.emplace(v_names[i], it);
    }
}
