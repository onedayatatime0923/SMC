#ifndef ARGPARSER_HPP
#define ARGPARSER_HPP

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <list>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>
#include "any.hpp"

using namespace std;

namespace ArgparseDetails {
    template<typename T, typename V = void, typename = void>
    struct CastFromContainer {
        static T Cast(const vector<any>& v_values) {
            return any_cast<T>(v_values.front());
        }
    };
    
    template <>
    struct CastFromContainer<string> {
        static string Cast(const vector<any>& v_values) {
            return any_cast<string>(v_values.front());
        }
    };

    template<typename ...>
    using void_t = void;
    template<typename T>
    struct CastFromContainer<T, typename T::value_type,
        void_t<decltype(declval<T>().begin()), decltype(declval<T>().end()),
        typename T::value_type >> {

        static T Cast(const vector<any>& v_values) {
            using ValueType = typename T::value_type;

            T tResult;
            transform(begin(v_values), end(v_values), back_inserter(tResult),
                [](const any &value) { return any_cast<ValueType>(value); });
            return tResult;
        }
    };
};

class ArgumentParser;

class Argument {
public:
    friend class ArgumentParser;
    friend ostream& operator<<(ostream &stream, const Argument& argument);
    friend ostream& operator<<(ostream &stream, const ArgumentParser& parser);

    using ValuedAction = function<any(const string &)>;

    template <size_t N>
    Argument(ArgumentParser* program_p, const string(&&a)[N]);

    Argument&   Help            (const string& help)        { help_ = help; return *this; }
    Argument&   DefaultValue    (const any& default_value)  { default_value_ = default_value; return *this; }
    Argument&   ImplicitValue   (const any& implicit_value) { implicit_value_ = implicit_value; num_args_ = 0; return *this; }
    Argument&   Action          (ValuedAction action)       { action_ = action; return *this; }

    Argument&   Nargs           (int num_args);
    Argument&   Remaining       ()                          { num_args_ = -1; return *this; }
    Argument&   Required        ()                          { is_required_ = true; return *this; }

    vector<string>::const_iterator Consume(vector<string>::const_iterator start, vector<string>::const_iterator end, const string& used_name = "");
    void                        Validate() const;

    int         Nargs                   ()                          { return num_args_; }
    int         GetArgumentNameLength   () const;
    int         GetArgumentValueLength  () const;

    string      str             () { stringstream out; out << *this; return out.str(); }
private:
    static int      Lookahead           (const string& s);
    static bool     IsDecimalLitral     (string s);
    static bool     IsOptional          (string name)    { return !IsPositional(name); }
    static bool     IsPositional        (string name);

    static string   ConvertStr          (const any& value);

    template <typename T> 
    T               Get                 () const;
    bool            Present             () const;


    ArgumentParser*     program_p_;

    vector<string>      name_v_;
    string              used_name_;
    string              help_;
    any                 default_value_;
    any                 implicit_value_;

    ValuedAction        action_;
    vector<any>         value_v_;
    int                 num_args_;
    bool                is_optional_;
    bool                is_required_;
    bool                is_used_;
};

class ArgumentParser {
public:
    friend ostream& operator<<(ostream &stream, const Argument& argument);
    friend ostream& operator<<(ostream &stream, const ArgumentParser& parser);

    using list_iterator = list<Argument>::iterator;

    explicit            ArgumentParser      (const string& program_name = {}, const string& version = "1.0");

    template <typename... Targs>
    Argument&           AddArgument         (Targs... Fargs);

    ArgumentParser&     AddDescription      (const string& description);
    ArgumentParser&     AddEpilog           (const string& epilog);

    int                 ParseArgs           (int argc, const char *const argv[]);
    int                 ParseArgs           (const vector<string>& arguments);

    template <typename T = string>
    T           Get                         (const string& name) const;
    bool        Present                     (const string& name) const;
    Argument&   operator[]                  (const string& name) const;

    string      str             () { stringstream out; out << *this; return out.str(); }
private:
    int         ParseArgsInternal               (const vector<string>& v_arguments);
    void        ParseArgsValidate               ();
    void        ParseLengthOfLongestArgument    ();

    void        IndexArgument       (list_iterator it);


    string                      program_name_;
    string                      version_;
    string                      description_;
    string                      epilog_;
    list<Argument>              positional_argument_l_;
    list<Argument>              optional_argument_l_;
    map<string, list_iterator>  argument_m_;


    int                         longest_argument_name_length_;
    int                         longest_argument_value_length_;
};
template <size_t N>
Argument::Argument(ArgumentParser* program_p, const string(&&a)[N]) : program_p_(program_p), 
    action_([](const string& value) { return value; }),
    num_args_(1), is_required_(false), is_used_(false) {
    bool is_optional = false;
    for (int i = 0; i < (int)N; ++i) {
        name_v_.emplace_back(a[i]);
        is_optional |= IsOptional(a[i]);
    }
    sort(name_v_.begin(), name_v_.end(), [](const string& lhs, const string& rhs) {
          return lhs.size() == rhs.size() ? lhs < rhs : lhs.size() < rhs.size(); });
    is_optional_ = is_optional;
};

template <typename T> 
T Argument::Get() const {
    if (!value_v_.empty()) {
        return ArgparseDetails::CastFromContainer<T>::Cast(value_v_);
    }
    else if (!default_value_.empty()) {
        return any_cast<T>(default_value_);
    }
    throw logic_error("No value provided.\n");
};


template <typename... Targs>
Argument& ArgumentParser::AddArgument(Targs... Fargs) {
    Argument argument(this, (string[sizeof...(Targs)]){Fargs...});
    list_iterator it;
    if (argument.is_optional_) {
        optional_argument_l_.emplace_back(argument);
        it = prev(optional_argument_l_.end());
    }
    else {
        positional_argument_l_.emplace_back(argument);
        it = prev(positional_argument_l_.end());
    }
    IndexArgument(it);
    return *it;
}
template <typename T>
T ArgumentParser::Get(const string& name) const {
    return (*this)[name].Get<T>();
};
#endif
