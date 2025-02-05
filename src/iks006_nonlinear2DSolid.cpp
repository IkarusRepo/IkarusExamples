// SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <config.h>

#include <dune/alugrid/grid.hh>
#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/dunepython.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/iga/nurbsbasis.hh>
#include <dune/iga/nurbsgrid.hh>

#include <Eigen/Core>

#include <ikarus/assembler/simpleassemblers.hh>
#include <ikarus/controlroutines/loadcontrol.hh>
#include <ikarus/finiteelements/fefactory.hh>
#include <ikarus/finiteelements/ferequirements.hh>
#include <ikarus/finiteelements/mechanics/materials/svk.hh>
#include <ikarus/finiteelements/mechanics/materials/vanishingstress.hh>
#include <ikarus/finiteelements/mechanics/nonlinearelastic.hh>
#include <ikarus/io/resultevaluators.hh>
#include <ikarus/io/resultfunction.hh>
#include <ikarus/io/vtkwriter.hh>
#include <ikarus/solver/nonlinearsolver/newtonraphson.hh>
#include <ikarus/solver/nonlinearsolver/nonlinearsolverfactory.hh>
#include <ikarus/solver/nonlinearsolver/trustregion.hh>
#include <ikarus/utils/algorithms.hh>
#include <ikarus/utils/basis.hh>
#include <ikarus/utils/dirichletvalues.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/init.hh>
#include <ikarus/utils/listener/controlvtkwriter.hh>
#include <ikarus/utils/listener/nonlinearsolverlogger.hh>
#include <ikarus/utils/nonlinearoperator.hh>
#include <ikarus/utils/pythonautodiffdefinitions.hh>

// The following grid types (gridType) are included in this example
enum class GridType
{
  ALUGrid,
  YaspGrid,
  NURBSGrid
};

// The following solver types (solverType) are included in this example
enum class SolverType
{
  NewtonRaphson,
  TrustRegion
};

template <GridType gt, SolverType st>
auto run() {
  using namespace Ikarus;
  constexpr int gridDim = 2;

  auto grid = []() {
    if constexpr (gt == GridType::ALUGrid) {
      using Grid = Dune::ALUGrid<gridDim, 2, Dune::simplex, Dune::conforming>;
      auto alg   = Dune::GmshReader<Grid>::read("auxiliaryFiles/unstructuredTrianglesfine.msh", false);
      alg->globalRefine(1);
      return alg;
    } else if constexpr (gt == GridType::YaspGrid) {
      using Grid        = Dune::YaspGrid<gridDim>;
      const double L    = 1;
      const double h    = 1;
      const size_t elex = 10;
      const size_t eley = 10;

      Dune::FieldVector<double, 2> bbox       = {L, h};
      std::array<int, 2> elementsPerDirection = {elex, eley};
      auto yg                                 = std::make_shared<Grid>(bbox, elementsPerDirection);
      return yg;
    } else if constexpr (gt == GridType::NURBSGrid) {
      constexpr auto dimworld              = 2;
      const std::array<int, gridDim> order = {2, 2};

      const std::array<std::vector<double>, gridDim> knotSpans = {
          {{0, 0, 0, 1, 1, 1}, {0, 0, 0, 1, 1, 1}}
      };

      using ControlPoint = Dune::IGA::NURBSPatchData<gridDim, dimworld>::ControlPointType;

      const std::vector<std::vector<ControlPoint>> controlPoints = {
          {  {.p = {0, 0}, .w = 5},    {.p = {0.5, 0}, .w = 1},   {.p = {1, 0}, .w = 1}},
          {{.p = {0, 0.5}, .w = 1}, {.p = {0.5, 0.5}, .w = 10}, {.p = {1, 0.5}, .w = 1}},
          {  {.p = {0, 1}, .w = 1},    {.p = {0.5, 1}, .w = 1},   {.p = {1, 1}, .w = 1}}
      };

      std::array<int, gridDim> dimsize = {(int)(controlPoints.size()), (int)(controlPoints[0].size())};

      auto controlNet = Dune::IGA::NURBSPatchData<gridDim, dimworld>::ControlPointNetType(dimsize, controlPoints);
      using Grid      = Dune::IGA::NURBSGrid<gridDim, dimworld>;

      Dune::IGA::NURBSPatchData<gridDim, dimworld> patchData;
      patchData.knotSpans     = knotSpans;
      patchData.degree        = order;
      patchData.controlPoints = controlNet;
      auto ng                 = std::make_shared<Grid>(patchData);
      ng->globalRefine(2);
      return ng;
    }
  }();

  auto gridView        = grid->leafGridView();
  const auto& indexSet = gridView.indexSet();

  Dune::BitSetVector<1> neumannVertices(gridView.size(2), false);

  std::string lambdaNeumannVertices = std::string("lambda x: ( x[0]>0.999 )");
  Python::start();
  Python::Reference main = Python::import("__main__");
  Python::run("import math");

  Python::runStream() << std::endl << "import sys" << std::endl << "import os" << std::endl;

  auto pythonNeumannVertices = Python::make_function<bool>(Python::evaluate(lambdaNeumannVertices));

  for (auto&& vertex : vertices(gridView)) {
    bool isNeumann                          = pythonNeumannVertices(vertex.geometry().corner(0));
    neumannVertices[indexSet.index(vertex)] = isNeumann;
  }

  BoundaryPatch<decltype(gridView)> neumannBoundary(gridView, neumannVertices);

  using namespace Dune::Functions::BasisFactory;
  auto basis = [&]() {
    if constexpr (gt == GridType::ALUGrid or gt == GridType::YaspGrid)
      return Ikarus::makeBasis(gridView, power<gridDim>(lagrange<1>()));
    else if constexpr (gt == GridType::NURBSGrid)
      return Ikarus::makeBasis(gridView, power<gridDim>(nurbs()));
  }();

  std::cout << "This gridview contains: " << std::endl;
  std::cout << gridView.size(2) << " vertices" << std::endl;
  std::cout << gridView.size(1) << " edges" << std::endl;
  std::cout << gridView.size(0) << " elements" << std::endl;
  std::cout << basis.flat().size() << " Dofs" << std::endl;

  // draw(gridView);

  auto localView = basis.flat().localView();

  auto matParameter = Ikarus::toLamesFirstParameterAndShearModulus({.emodul = 1000, .nu = 0.3});

  Materials::StVenantKirchhoff matSVK(matParameter);
  auto reducedMat = Materials::planeStress(matSVK, 1e-8);

  auto vL = [](auto& globalCoord, auto& lamb) {
    Eigen::Vector2d fext;
    fext.setZero();
    return fext;
  };

  auto neumannBl = [](auto& globalCoord, auto& lamb) {
    Eigen::Vector2d fext;
    fext.setZero();
    fext[1] = lamb / 40;
    return fext;
  };
  auto sk = skills(nonLinearElastic(reducedMat), volumeLoad<2>(vL), neumannBoundaryLoad(&neumannBoundary, neumannBl));

  using FEType = decltype(makeFE(basis, sk));
  std::vector<FEType> fes;
  for (auto&& ge : elements(gridView)) {
    fes.emplace_back(makeFE(basis, sk));
    fes.back().bind(ge);
  }

  auto basisP = std::make_shared<const decltype(basis)>(basis);
  Ikarus::DirichletValues dirichletValues(basisP->flat());

  dirichletValues.fixBoundaryDOFs([&](auto& dirichletFlags, auto&& localIndex, auto&& localView, auto&& intersection) {
    if (std::abs(intersection.geometry().center()[1]) < 1e-8)
      dirichletFlags[localView.index(localIndex)] = true;
  });

  auto sparseAssembler = makeSparseFlatAssembler(fes, dirichletValues);

  Eigen::VectorXd d;
  d.setZero(basis.flat().size());
  double lambda = 0.0;

  auto req = typename FEType::Requirement();
  req.insertGlobalSolution(d).insertParameter(lambda);

  sparseAssembler->bind(req, Ikarus::AffordanceCollections::elastoStatics, Ikarus::DBCOption::Full);

  auto nonlinSolver = [&]() {
    if constexpr (st == SolverType::NewtonRaphson) {
      auto linSolver = Ikarus::LinearSolver(Ikarus::SolverTypeTag::sd_UmfPackLU);
      NewtonRaphsonConfig<decltype(linSolver)> nrConfig{
          .parameters = {.tol = 1e-8, .maxIter = 100},
            .linearSolver = linSolver
      };
      Ikarus::NonlinearSolverFactory nrFactory(nrConfig);
      return nrFactory.create(sparseAssembler);
    } else if constexpr (st == SolverType::TrustRegion) {
      TrustRegionConfig<> trConfig{
          .parameters = {.verbosity = 1,
                         .maxIter   = 30,
                         .grad_tol  = 1e-8,
                         .corr_tol  = 1e-8,
                         .useRand   = false,
                         .rho_reg   = 1e6,
                         .Delta0    = 1}
      };
      Ikarus::NonlinearSolverFactory trFactory(trConfig);
      return trFactory.create(sparseAssembler);
    }
  }();

  auto nonLinearSolverObserver = NonLinearSolverLogger();

  auto vtkWriter = ControlSubsamplingVertexVTKWriter<std::remove_cvref_t<decltype(basis.flat())>>(basis.flat(), d, 2);
  vtkWriter.setFileNamePrefix("iks006_nonlinear2DSolid");
  vtkWriter.setFieldInfo("Displacement", Dune::VTK::FieldInfo::Type::vector, 2);

  auto lc = Ikarus::LoadControl(nonlinSolver, 20, {0, 2000});

  nonLinearSolverObserver.subscribeTo(lc.nonlinearSolver());
  vtkWriter.subscribeTo(lc);

  // Postprocessing
  auto vonMisesFunction =
      Ikarus::makeResultFunction<ResultTypes::PK2Stress>(sparseAssembler, ResultEvaluators::VonMises{});

  using Ikarus::Vtk::DataTag::asPointData;
  Ikarus::Vtk::Writer writer(sparseAssembler);

  writer.template addResult<Ikarus::ResultTypes::PK2Stress>(asPointData);
  writer.addResultFunction(std::move(vonMisesFunction), asPointData);
  writer.addInterpolation(d, basis.flat(), "displacement", asPointData);

  writer.write("iks006_nonlinear2DSolid_Result_" + Dune::className(st));
}

int main(int argc, char** argv) {
  Ikarus::init(argc, argv);
  run<GridType::ALUGrid, SolverType::NewtonRaphson>();
  run<GridType::ALUGrid, SolverType::TrustRegion>();
}