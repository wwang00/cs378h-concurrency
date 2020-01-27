#include <pthread.h>
#include <stdio.h>

#include "argparse.h"

using namespace std;

int N;
int type;

unordered_set<string> flags{"-v", "-s"};
unordered_set<string> opts{"-n", "-i", "-o"};

int main(int argc, char **argv) {
  unordered_map<string, string> args = parse_args(argc, argv, flags, opts);
}
