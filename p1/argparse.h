#include <string>
#include <unordered_map>
#include <unordered_set>

#ifndef _ARGPARSE_H_
#define _ARGPARSE_H_

std::unordered_map<std::string, std::string> parse_args(
    int argc, char **argv, std::unordered_set<std::string> flags,
    std::unordered_set<std::string> opts);

#endif