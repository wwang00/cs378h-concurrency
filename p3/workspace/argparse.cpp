#include "argparse.h"

#include <iostream>

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
        cout << "option " << token << " missing value" << endl;
        result.clear();
        return result;
      }
      string val(argv[i]);
      result[token] = val;
    } else {
      cout << "invalid argument " << token << endl;
      result.clear();
      return result;
    }
  }
  return result;
}