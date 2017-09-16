#include "functions.h"
#include <random>
std::default_random_engine& myGenerator()
{
  static std::default_random_engine gene;
  return gene;
};
