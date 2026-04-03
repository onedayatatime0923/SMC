
#ifndef USAGE_HPP
#define USAGE_HPP

#include <string>
#include <cassert>
#include <cstring>

#include "argparse.hpp"


class Time {
public:
    enum Type { FULL, PARTIAL };
    struct State {
        State(double r = 0, double u = 0, double s = 0) : realTime(r), userTime(u), sysTime(s) {}
        bool    operator == (const State& s);
        bool    operator != (const State& s) { return !((*this) == s); }
        State   operator +  (const State& s);
        State   operator -  (const State& s);

        double realTime, userTime, sysTime;
    };

    Time    ();

    int                 Run             (int argc, char **argv);
    void                Clear           ()      { Check(partial_stamp_); };
    void                Pause           ()      { if (pause_stamp_ == State()) { Check(pause_stamp_); } };
    void                Play            ();
    void                Show            ();
    State               Get             (Type type = PARTIAL);
    void                CheckTimeout    (double timeout, Type type = PARTIAL);
private:
    static void         Check           (State& st);


    static constexpr double    TIME_SCALE   = 1000000.0;

    State           full_stamp_;
    State           partial_stamp_;
    State           pause_stamp_;
    ArgumentParser  program_;
};

class Memory {
public:
    struct State {
        State(double p = 0, double c = 0) : peakMem(p), currMem(c) {}
        double peakMem, currMem;
    };

    Memory  ();

    int                 Run             (int argc, char **argv);
    void                Show            ();
    State               Get             ();
private:
    static void         Check               (State& st);

    static constexpr double    MEMORY_SCALE = 1024.0;
    ArgumentParser  program_;

};

class Usage {
public:
    Usage();

    int                 Run                 (int argc, char **argv);
    void                ClearTime           () { time_.Clear(); };
    void                PauseTime           () { time_.Pause(); };
    void                PlayTime            () { time_.Play(); };
    void                ShowTime            () { time_.Show(); };
    void                ShowMemory          () { memory_.Show(); };
    void                ShowUsage           (const std::string& comment);

    Time::State         GetTime             (Time::Type type = Time::PARTIAL) { return time_.Get(type); };
    Memory::State       GetMemory           () { return memory_.Get(); };

    void                CheckTimeout        (double timeout, Time::Type type = Time::FULL) { time_.CheckTimeout(timeout, type); }

    Time    time_;
    Memory  memory_;

private:
    ArgumentParser  program_;
};


#endif // USAGE_HPP
