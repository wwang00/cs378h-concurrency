#include "argparse.h"

#include <stdio.h>

using namespace std;

unordered_map<string, string> parse_args(int argc, char **argv,
                                         unordered_set<string> flags,
                                         unordered_set<string> opts) {
  unordered_map<string, string> result;
  for (int i = 1; i < argc; i++) {
    string token(argv[i]);
    if (flags.count(token)) {
      result[token] = "";
    } else if (opts.count(token)) {
      i++;
      if (i >= argc) {
        printf("option %s missing value\n", token.c_str());
        result.clear();
        return result;
      }
      string val(argv[i]);
      result[token] = val;
    } else {
      printf("invalid argument %s\n", token.c_str());
      result.clear();
      return result;
    }
  }
  return result;
}