// SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <config.h>

#include <matplot/matplot.h>

#include <dune/alugrid/grid.hh>
#include <dune/foamgrid/foamgrid.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/localfefunctions/cachedlocalBasis/cachedlocalBasis.hh>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <autodiff/forward/dual/dual.hpp>

#include <ikarus/assembler/simpleassemblers.hh>
#include <ikarus/finiteelements/autodifffe.hh>
#include <ikarus/finiteelements/febase.hh>
#include <ikarus/finiteelements/fefactory.hh>
#include <ikarus/finiteelements/physicshelper.hh>
#include <ikarus/utils/algorithms.hh>
#include <ikarus/utils/basis.hh>
#include <ikarus/utils/dirichletvalues.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/init.hh>

using namespace Ikarus;
using namespace Dune::Indices;

template <typename PreFE, typename FE>
class IncompressibleSolid;

struct IncompressibleSolidPre
{
  double emod;
  double nu;

  template <typename PreFE, typename FE>
  using Skill = IncompressibleSolid<PreFE, FE>;
};

template <typename PreFE, typename FE>
class IncompressibleSolid
{
public:
  using Traits       = typename PreFE::Traits;
  using BasisHandler = typename Traits::BasisHandler;
  using FlatBasis    = typename Traits::FlatBasis;
  using Requirement =
      FERequirementsFactory<FESolutions::displacement, FEParameter::loadfactor, Traits::useEigenRef>::type;

  using LocalView   = typename Traits::LocalView;
  using Geometry    = typename Traits::Geometry;
  using Element     = typename Traits::Element;
  using GlobalIndex = typename Traits::GlobalIndex;
  using Pre         = IncompressibleSolidPre;
  IncompressibleSolid(Pre pre)
      : emod_{pre.emod},
        nu_{pre.nu} {
    mu_        = emod_ / (2 * (1 + nu_));
    lambdaMat_ = convertLameConstants({.emodul = emod_, .nu = nu_}).toLamesFirstParameter();
  }

protected:
  template <template <typename, int, int> class RT>
  requires Dune::AlwaysFalse<RT<double, 1, 1>>::value
  auto calculateAtImpl(const Requirement& req, const Dune::FieldVector<double, Traits::mydim>& local,
                       Dune::PriorityTag<0>) const {}

  template <class ScalarType>
  auto calculateScalarImpl(const Requirement& par, ScalarAffordance affo,
                           const std::optional<std::reference_wrapper<const Eigen::VectorX<ScalarType>>>& dx =
                               std::nullopt) const -> ScalarType {
    const auto& d         = par.globalSolution();
    const auto& lambda    = par.parameter();
    const auto& localView = underlying().localView();
    const auto& tree      = localView.tree();
    Eigen::VectorX<ScalarType> localDisp(localView.size());
    localDisp.setZero();
    auto& displacementNode = tree.child(_0, 0);
    auto& pressureNode     = tree.child(_1);
    const auto& feDisp     = displacementNode.finiteElement();
    const auto& fePressure = pressureNode.finiteElement();
    Eigen::Matrix<ScalarType, Traits::dimension, Eigen::Dynamic> disp;
    disp.setZero(Eigen::NoChange, feDisp.size());
    Eigen::Vector<ScalarType, Eigen::Dynamic> pN;
    pN.setZero(fePressure.size());

    if (dx) {
      for (auto i = 0U; i < feDisp.size(); ++i)
        for (auto k2 = 0U; k2 < Traits::mydim; ++k2)
          disp.col(i)(k2) =
              dx.value().get()[i * Traits::mydim + k2] + d[localView.index(tree.child(_0, k2).localIndex(i))[0]];
      for (auto i = 0U; i < fePressure.size(); ++i)
        pN[i] =
            dx.value().get()[Traits::mydim * feDisp.size() + i] + d[localView.index(tree.child(_1).localIndex(i))[0]];
    } else {
      for (auto i = 0U; i < feDisp.size(); ++i)
        for (auto k2 = 0U; k2 < Traits::mydim; ++k2)
          disp.col(i)(k2) = d[localView.index(tree.child(_0, k2).localIndex(i))[0]];
      for (auto i = 0U; i < fePressure.size(); ++i)
        pN[i] = d[localView.index(tree.child(_1).localIndex(i))[0]];
    }

    ScalarType energy = 0.0;

    const int order  = 2 * (feDisp.localBasis().order());
    const auto& rule = Dune::QuadratureRules<double, Traits::mydim>::rule(localView.element().type(), order);
    const auto geo   = localView.element().geometry();
    Dune::CachedLocalBasis localBasisDisp(feDisp.localBasis());
    Dune::CachedLocalBasis localBasisPressure(fePressure.localBasis());
    Eigen::Matrix<double, Eigen::Dynamic, Traits::mydim> dNdisp;
    Eigen::VectorXd Ndisp;
    Eigen::VectorXd Npressure;
    for (auto&& gp : rule) {
      const auto J = Dune::toEigen(geo.jacobianTransposed(gp.position())).transpose().eval();
      localBasisDisp.evaluateFunctionAndJacobian(gp.position(), Ndisp, dNdisp);
      localBasisPressure.evaluateFunction(gp.position(), Npressure);
      const Eigen::Vector<double, Traits::worlddim> X = Dune::toEigen(geo.global(gp.position()));
      Eigen::Vector<ScalarType, Traits::worlddim> x   = X;
      for (auto i = 0U; i < feDisp.size(); ++i)
        x += disp.col(i) * Ndisp[i];

      ScalarType pressure = pN.dot(Npressure);

      const auto gradu      = (disp * dNdisp.template cast<ScalarType>() * J.inverse()).eval();
      const auto symgradu   = Dune::sym(gradu);
      const ScalarType divU = gradu.trace();

      Eigen::Vector<double, Traits::worlddim> fext;
      fext.setZero();
      fext[1] = lambda;
      fext[0] = 0 * lambda;

      energy += (0.5 * (2 * mu_ * symgradu.squaredNorm() - 1 / lambdaMat_ * Dune::power(pressure, 2)) +
                 pressure * divU - x.dot(fext)) *
                geo.integrationElement(gp.position()) * gp.weight(); // plane strain for 2D
    }
    return energy;
  }

private:
  //> CRTP
  const auto& underlying() const { return static_cast<const FE&>(*this); }
  auto& underlying() { return static_cast<FE&>(*this); }
  double emod_;
  double nu_;
  double mu_;
  double lambdaMat_;
};

int main(int argc, char** argv) {
  using namespace Dune::Functions;
  /// Construct grid
  Ikarus::init(argc, argv);
  using namespace Ikarus;
  using namespace Dune::Indices;
  constexpr int gridDim = 2;

  using Grid        = Dune::YaspGrid<gridDim>;
  const double L    = 1;
  const double h    = 1;
  const size_t elex = 20;
  const size_t eley = 20;

  Dune::FieldVector<double, 2> bbox       = {L, h};
  std::array<int, 2> elementsPerDirection = {elex, eley};
  auto grid                               = std::make_shared<Grid>(bbox, elementsPerDirection);
  auto gridView                           = grid->leafGridView();
  // draw(gridView);

  using namespace Dune::Functions::BasisFactory;
  /// Construct basis
  auto basis = Ikarus::makeBasis(gridView, composite(power<2>(lagrange<1>()), lagrange<0>()));

  /// Create finite elements
  const double Emod = 2.1e1;
  const double nu   = 0.5;
  auto sk           = skills(IncompressibleSolidPre(Emod, nu));
  using AutoDiffFE  = Ikarus::AutoDiffFE<decltype(makeFE(basis, sk))>;

  std::vector<AutoDiffFE> fes;
  for (auto&& ge : elements(gridView)) {
    fes.emplace_back(AutoDiffFE(makeFE(basis, sk)));
    fes.back().bind(ge);
  }

  /// Collect dirichlet nodes
  auto basisP = std::make_shared<const decltype(basis)>(basis);
  Ikarus::DirichletValues dirichletValues(basisP->flat());

  dirichletValues.fixDOFs([](auto& basis_, auto& dirichletFlags) {
    Dune::Functions::forEachBoundaryDOF(subspaceBasis(basis_, _0),
                                        [&](auto&& localIndex, auto&& localView, auto&& intersection) {
                                          if (std::abs(intersection.geometry().center()[1]) < 1e-8)
                                            dirichletFlags[localView.index(localIndex)] = true;
                                        });
  });

  /// Create assembler
  auto sparseFlatAssembler = SparseFlatAssembler(fes, dirichletValues);

  /// Create function for external forces and stiffness matrix
  double lambda = 1;
  Eigen::VectorXd d;
  d.setZero(basis.flat().size());

  auto req = AutoDiffFE::Requirement();
  req.insertGlobalSolution(d).insertParameter(lambda);

  sparseFlatAssembler.bind(req);
  sparseFlatAssembler.bind(Ikarus::AffordanceCollections::elastoStatics);
  sparseFlatAssembler.bind(Ikarus::DBCOption::Reduced);

  auto& K = sparseFlatAssembler.matrix();
  auto& R = sparseFlatAssembler.vector();
  Eigen::SparseLU solver(K);

  if (solver.info() != Eigen::Success)
    DUNE_THROW(Dune::MathError, "Failed Compute");

  d -= sparseFlatAssembler.createFullVector(solver.solve(R));
  if (solver.info() != Eigen::Success)
    DUNE_THROW(Dune::MathError, "Failed Solve");

  /// Postprocess
  auto disp = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, 2>>(
      subspaceBasis(basis.flat(), _0), d);
  auto pressure = Dune::Functions::makeDiscreteGlobalBasisFunction<double>(subspaceBasis(basis.flat(), _1), d);
  Dune::VTKWriter vtkWriter(gridView, Dune::VTK::nonconforming);
  vtkWriter.addVertexData(disp, Dune::VTK::FieldInfo("displacement", Dune::VTK::FieldInfo::Type::vector, 2));
  vtkWriter.addVertexData(pressure, Dune::VTK::FieldInfo("pressure", Dune::VTK::FieldInfo::Type::scalar, 1));
  vtkWriter.write("iks003_incompressibleLinearElasticity");
}
