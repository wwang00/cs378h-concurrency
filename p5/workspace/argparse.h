#ifndef _ARGPARSE_H_
#define _ARGPARSE_H_

#include <string>
#include <unordered_map>
#include <unordered_set>

std::unordered_map<std::string, std::string>
parse_args(int argc, char **argv, const std::unordered_set<std::string> &flags,
           const std::unordered_set<std::string> &opts);

#endif