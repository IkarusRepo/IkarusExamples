// SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <config.h>

#include <matplot/matplot.h>

#include <dune/foamgrid/foamgrid.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/localfefunctions/eigenDuneTransformations.hh>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <autodiff/forward/dual/dual.hpp>

#include <ikarus/assembler/simpleassemblers.hh>
#include <ikarus/controlroutines/loadcontrol.hh>
#include <ikarus/finiteelements/autodifffe.hh>
#include <ikarus/finiteelements/fefactory.hh>
#include <ikarus/finiteelements/physicshelper.hh>
#include <ikarus/solver/linearsolver/linearsolver.hh>
#include <ikarus/solver/nonlinearsolver/newtonraphson.hh>
#include <ikarus/solver/nonlinearsolver/nonlinearsolverfactory.hh>
#include <ikarus/utils/basis.hh>
#include <ikarus/utils/dirichletvalues.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/eigendunetransformations.hh>
#include <ikarus/utils/init.hh>
#include <ikarus/utils/nonlinearoperator.hh>
#include <ikarus/utils/observer/controlvtkwriter.hh>
#include <ikarus/utils/observer/genericobserver.hh>
#include <ikarus/utils/observer/nonlinearsolverlogger.hh>
#include <ikarus/utils/pythonautodiffdefinitions.hh>

using namespace Ikarus;

template <typename PreFE, typename FE>
class Truss;

struct TrussPre
{
  double EA;

  template <typename PreFE, typename FE>
  using Skill = Truss<PreFE, FE>;
};

template <typename PreFE, typename FE>
class Truss
{
public:
  using Traits       = typename PreFE::Traits;
  using BasisHandler = typename Traits::BasisHandler;
  using FlatBasis    = typename Traits::FlatBasis;
  using Requirement =
      FERequirementsFactory<FESolutions::displacement, FEParameter::loadfactor, Traits::useEigenRef>::type;
  using LocalView = typename Traits::LocalView;
  using Geometry  = typename Traits::Geometry;
  using Element   = typename Traits::Element;
  using Pre       = TrussPre;

  Truss(Pre pre)
      : EA{pre.EA} {}

protected:
  template <template <typename, int, int> class RT>
  requires Dune::AlwaysFalse<RT<double, 1, 1>>::value
  auto calculateAtImpl(const Requirement& req, const Dune::FieldVector<double, Traits::mydim>& local,
                       Dune::PriorityTag<0>) const {}

  template <typename ScalarType>
  auto calculateScalarImpl(const Requirement& par, ScalarAffordance affo,
                           const std::optional<std::reference_wrapper<const Eigen::VectorX<ScalarType>>>& dx =
                               std::nullopt) const -> ScalarType {
    const auto& d         = par.globalSolution();
    const auto& lambda    = par.parameter();
    const auto& localView = underlying().localView();
    const auto& tree      = localView.tree();
    auto& ele             = localView.element();
    const auto X1         = Dune::toEigen(ele.geometry().corner(0));
    const auto X2         = Dune::toEigen(ele.geometry().corner(1));

    Eigen::Matrix<ScalarType, Traits::worlddim, 2> u;
    u.setZero();
    if (dx) {
      for (int i = 0; i < 2; ++i)
        for (int k2 = 0; k2 < Traits::worlddim; ++k2)
          u.col(i)(k2) =
              dx.value().get()[Traits::worlddim * i + k2] + d[localView.index(tree.child(k2).localIndex(i))[0]];
    } else {
      for (int i = 0; i < 2; ++i)
        for (int k2 = 0; k2 < Traits::worlddim; ++k2)
          u.col(i)(k2) = d[localView.index(tree.child(k2).localIndex(i))[0]];
    }

    const Eigen::Vector2<ScalarType> x1 = X1 + u.col(0);
    const Eigen::Vector2<ScalarType> x2 = X2 + u.col(1);

    const double LRefsquared  = (X1 - X2).squaredNorm();
    const ScalarType lsquared = (x1 - x2).squaredNorm();

    const ScalarType Egl = 0.5 * (lsquared - LRefsquared) / LRefsquared;

    return 0.5 * EA * sqrt(LRefsquared) * Egl * Egl;
  }

private:
  //> CRTP
  const auto& underlying() const { return static_cast<const FE&>(*this); }
  auto& underlying() { return static_cast<FE&>(*this); }
  double EA;
};

auto truss(double EA) { return TrussPre(EA); }

int main(int argc, char** argv) {
  Ikarus::init(argc, argv);
  /// Construct grid
  Dune::GridFactory<Dune::FoamGrid<1, 2, double>> gridFactory;
  const double h = 1.0;
  const double L = 2.0;
  gridFactory.insertVertex({0, 0});
  gridFactory.insertVertex({L, h});
  gridFactory.insertVertex({2 * L, 0});
  gridFactory.insertElement(Dune::GeometryTypes::line, {0, 1});
  gridFactory.insertElement(Dune::GeometryTypes::line, {1, 2});
  auto grid     = gridFactory.createGrid();
  auto gridView = grid->leafGridView();
  // draw(gridView);

  /// Construct basis
  using namespace Dune::Functions::BasisFactory;
  auto basis = Ikarus::makeBasis(gridView, power<2>(lagrange<1>()));

  /// Create finite elements
  const double EA  = 100;
  auto sk          = skills(truss(EA));
  using AutoDiffFE = Ikarus::AutoDiffFE<decltype(makeFE(basis, sk))>;
  std::vector<AutoDiffFE> fes;
  for (auto&& ge : elements(gridView)) {
    fes.emplace_back(AutoDiffFE(makeFE(basis, sk)));
    fes.back().bind(ge);
  }

  /// Collect dirichlet nodes
  auto basisP = std::make_shared<const decltype(basis)>(basis);
  Ikarus::DirichletValues dirichletValues(basisP->flat());
  dirichletValues.fixBoundaryDOFs(
      [&](auto& dirichletFlags, auto&& globalIndex) { dirichletFlags[globalIndex] = true; });

  /// Create assembler
  auto denseFlatAssembler = makeDenseFlatAssembler(fes, dirichletValues);

  /// Create non-linear operator
  double lambda = 0;
  Eigen::VectorXd d;
  d.setZero(basis.flat().size());

  auto req = AutoDiffFE::Requirement();
  req.insertGlobalSolution(d).insertParameter(lambda);
  denseFlatAssembler->bind(req, Ikarus::AffordanceCollections::elastoStatics, Ikarus::DBCOption::Full);

  /// Choose linear solver
  auto linSolver = Ikarus::LinearSolver(Ikarus::SolverTypeTag::d_LDLT);

  /// Create Nonlinear solver for controlroutine, i.e. a Newton-Rahpson object
  NewtonRaphsonConfig<decltype(linSolver)> nrConfig{
      .parameters = {.tol = 1e-8, .maxIter = 100},
        .linearSolver = linSolver
  };

  Ikarus::NonlinearSolverFactory nrFactory(nrConfig);
  auto nr = nrFactory.create(denseFlatAssembler);

  /// Create Observer to write information of the non-linear solver on the
  /// console
  auto nonLinearSolverObserver = std::make_shared<NonLinearSolverLogger>();

  const int loadSteps = 10;
  Eigen::Matrix3Xd lambdaAndDisp;
  lambdaAndDisp.setZero(Eigen::NoChange, loadSteps + 1);
  /// Create Observer which executes when control routines messages
  /// SOLUTION_CHANGED
  auto lvkObserver = std::make_shared<Ikarus::GenericObserver<Ikarus::ControlMessages>>(
      Ikarus::ControlMessages::SOLUTION_CHANGED, [&](int step) {
        lambdaAndDisp(0, step) = lambda;
        lambdaAndDisp(1, step) = d[2];
        lambdaAndDisp(2, step) = d[3];
      });

  /// Create Observer which writes vtk files when control routines messages
  /// SOLUTION_CHANGED
  auto vtkWriter = std::make_shared<ControlSubsamplingVertexVTKWriter<std::remove_cvref_t<decltype(basis.flat())>>>(
      basis.flat(), d, 2);
  vtkWriter->setFieldInfo("displacement", Dune::VTK::FieldInfo::Type::vector, 2);
  vtkWriter->setFileNamePrefix("iks007_vonMisesTruss");

  /// Create loadcontrol
  auto lc = Ikarus::LoadControl(nr, loadSteps, {0, 30});
  lc.nonlinearSolver().subscribeAll(nonLinearSolverObserver);
  lc.subscribeAll({vtkWriter, lvkObserver});

  /// Execute!
  lc.run();

  /// Postprocess
  using namespace matplot;
  Eigen::VectorXd lambdaVec = lambdaAndDisp.row(0);
  Eigen::VectorXd dVec      = -lambdaAndDisp.row(2);
  auto f                    = figure(true);

  title("Load-Displacement Curve");
  xlabel("y-Displacement");
  ylabel("LoadFactor");

  auto analyticalLoadDisplacementCurve = [&](auto& w) {
    const double Ltruss = std::sqrt(h * h + L * L);
    return 2.0 * EA * Dune::power(h, 3) / Dune::power(Ltruss, 3) *
           (w / h - 1.5 * Dune::power(w / h, 2) + 0.5 * Dune::power(w / h, 3));
  };

  std::vector<double> x  = linspace(0.0, dVec.maxCoeff());
  std::vector<double> y1 = transform(x, [&](auto x) { return analyticalLoadDisplacementCurve(x); });
  auto p                 = plot(x, y1, dVec, lambdaVec);
  p[0]->line_width(3);
  p[1]->line_width(2);
  p[1]->marker(line_spec::marker_style::asterisk);
  // save("vonMisesTruss.png");
  // f->draw();
  // using namespace std::chrono_literals;
  // std::this_thread::sleep_for(5s);
}
