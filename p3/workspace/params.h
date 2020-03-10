#ifndef _PARAMS_H_
#define _PARAMS_H_

struct Parameters {
  int         points;
  int         clusters;
  int         dims;
  int         iterations;
  float       threshold;
  bool        output_centroids;
  int         seed;
};

extern Parameters P;

#endif
