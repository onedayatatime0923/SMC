
#include <sys/time.h>
#include <sys/resource.h>

#include "usage.hpp"

bool Time::State::operator == (const State& s) { 
    return realTime == s.realTime && userTime == s.userTime && sysTime == s.sysTime;
}
Time::State Time::State::operator +(const State& s) {
    return State(realTime + s.realTime, userTime + s.userTime, sysTime + s.sysTime);
};
Time::State Time::State::operator -(const State& s) {
    return State(realTime - s.realTime, userTime - s.userTime, sysTime - s.sysTime);
};
Time::Time() : program_("time") {
    Check(full_stamp_);
    Check(partial_stamp_);
    
    program_.AddDescription("print the runtime since the last call.");
    program_.AddArgument("-c")
        .Help("clears the elapsed time without printing it.")
        .ImplicitValue(true)
        .DefaultValue(false);
    program_.AddArgument("-pause")
        .Help("pauses the elapsed time without printing it.")
        .ImplicitValue(true)
        .DefaultValue(false);
    program_.AddArgument("-play")
        .Help("plays the elapsed time without printing it.")
        .ImplicitValue(true)
        .DefaultValue(false);
}
int Time::Run(int argc, char **argv) {
    bool res;
    try {
    res = program_.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    if (program_.Get<bool>("-c")) {
        Clear();
    }
    if (program_.Get<bool>("-pause")) {
        Pause();
    }
    if (program_.Get<bool>("-play")) {
        Play();
    }
    if (argc ==  1) {
        Show();
    }
    return 0;
};
void Time::Play() {
    if(pause_stamp_ != State()) { 
        if (partial_stamp_.realTime > pause_stamp_.realTime) {
            Check(partial_stamp_);
        }
        else {
            State curSt;
            Check(curSt);
            partial_stamp_ = partial_stamp_ + (curSt - pause_stamp_);
        }
        pause_stamp_ = State();
    }
}
void Time::Show() {
    State fullTime = Get(FULL);
    State partTime = Get(PARTIAL);
    // printf("--------------------------------------------------------------------\n");
    printf("Time:\n");
    printf("    Full    Real: %fs; User: %fs; System: %fs\n", fullTime.realTime, fullTime.userTime, fullTime.sysTime);
    printf("    Partial Real: %fs; User: %fs; System: %fs\n", partTime.realTime, partTime.userTime, partTime.sysTime);
    // printf("--------------------------------------------------------------------\n");
    fflush(stdout);
}
Time::State Time::Get(Type type) {
    State curSt;
    Check(curSt);
    State dur = (type == FULL) ? curSt - full_stamp_ : curSt - partial_stamp_;
    return dur;
}

void Time::CheckTimeout(double timeout, Type type) {
    double realTime = Get(type).realTime;
    // printf("real time: %f\n", realTime);
    // printf("timeout: %f\n", timeout);
    if (realTime > timeout) {
        throw std::runtime_error("Timeout");
    }
}

void Time::Check(State& st) {
    rusage tUsg;
    getrusage(RUSAGE_SELF, &tUsg);
    timeval tReal;
    gettimeofday(&tReal, NULL);
    st.userTime = tUsg.ru_utime.tv_sec + tUsg.ru_utime.tv_usec / TIME_SCALE;
    st.sysTime  = tUsg.ru_stime.tv_sec + tUsg.ru_stime.tv_usec / TIME_SCALE;
    st.realTime = tReal.tv_sec + tReal.tv_usec / TIME_SCALE;
}

Memory::Memory() : program_("memory") {
    program_.AddDescription("print the memory.");
}
int Memory::Run(int argc, char **argv) {
    bool res;
    try {
    res = program_.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;
    Show();
    return 0;
};
void Memory::Show() {
    State curSt = Get();
    // printf("--------------------------------------------------------------------\n");
    printf("Memory:\n");
    printf("    Peak: %lf MB\n", curSt.peakMem);
    printf("    Curr: %lf MB\n", curSt.currMem);
    // printf("--------------------------------------------------------------------\n");
    fflush(stdout);
}
Memory::State Memory::Get() {
    State curSt;
    Check(curSt);
    return curSt;
}
void Memory::Check(State& st) {
    #ifdef __linux__
        FILE* fmem = fopen("/proc/self/status", "r");
        char membuf[128];
        char* ch;

        while (fgets(membuf, 128, fmem)) {
            if ((ch = strstr(membuf, "VmPeak:"))) {
                st.peakMem = atol(ch + 7) / MEMORY_SCALE;
                continue;
            }
            else if ((ch = strstr(membuf, "VmSize:"))) {
                st.currMem = atol(ch + 7) / MEMORY_SCALE;
                continue;
            }
        }
        fclose(fmem);
    #else
    st.peakMem = -1;
    st.currMem = -1;
    #endif
}

Usage::Usage() : program_("usage") {
    program_.AddDescription("print the usage.");
    program_.AddArgument("-t")
        .Help("print the runtime since the last call.")
        .ImplicitValue(true)
        .DefaultValue(false);
    program_.AddArgument("-m")
        .Help("print the memory.")
        .ImplicitValue(true)
        .DefaultValue(false);
    program_.AddArgument("-c")
        .Help("specify the usage.")
        .DefaultValue((string)"");
    
}
int Usage::Run(int argc, char **argv) {
    bool res;
    try {
    res = program_.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    if (program_.Get<bool>("-t")) {
        ShowTime();
    }
    else if (program_.Get<bool>("-m")) {
        ShowMemory();
    }
    else {
        ShowUsage(program_.Get<string>("-c"));
    }
    return 0;
};
void Usage::ShowUsage(const std::string& comment) {
    // printf("#################### %-20s Usage ####################\n", comment.c_str());
    printf("-------------------- %-20s Usage --------------------\n", comment.c_str());
    ShowTime();
    ShowMemory();
    printf("--------------------------------------------------------------------\n");
    // printf("####################################################################\n");
    fflush(stdout);
}




