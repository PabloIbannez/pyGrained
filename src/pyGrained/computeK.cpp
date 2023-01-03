#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

std::vector<double> computeK(const std::vector<double>& distances,const int& Nbeads){

  std::vector<double> K(Nbeads,0);

  for(int bead_i=0; bead_i<Nbeads; bead_i++){
  for(int bead_j=0; bead_j<Nbeads; bead_j++){
      if(bead_i != bead_j){ // for numerical reasons
        if(distances[bead_j]<distances[bead_i]){
          K[bead_i]++;
        }
      }
    }}

  return K;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(computeK,m)
{
  m.doc() = "computeK";

  m.def("computeK", &computeK, "computeK");
}
