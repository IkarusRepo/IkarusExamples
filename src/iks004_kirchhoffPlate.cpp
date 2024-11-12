// SPDX-FileCopyrightText: 2021-2024 The Ikarus Developers mueller@ibb.uni-stuttgart.de
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <config.h>

#include <matplot/matplot.h>
#include <numbers>

#include <dune/common/indices.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/gridfunctions/analyticgridviewfunction.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/iga/nurbsbasis.hh>
#include <dune/iga/nurbsgrid.hh>
#include <dune/localfefunctions/cachedlocalBasis/cachedlocalBasis.hh>
#include <dune/localfefunctions/eigenDuneTransformations.hh>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ikarus/assembler/simpleassemblers.hh>
#include <ikarus/finiteelements/autodifffe.hh>
#include <ikarus/finiteelements/fefactory.hh>
#include <ikarus/finiteelements/physicshelper.hh>
#include <ikarus/utils/algorithms.hh>
#include <ikarus/utils/basis.hh>
#include <ikarus/utils/concepts.hh>
#include <ikarus/utils/dirichletvalues.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/eigendunetransformations.hh>
#include <ikarus/utils/init.hh>
#include <ikarus/utils/nonlinearoperator.hh>
#include <ikarus/utils/observer/controlvtkwriter.hh>
#include <ikarus/utils/pythonautodiffdefinitions.hh>

using namespace Ikarus;

template <typename PreFE, typename FE>
class KirchhoffPlate;

struct KirchhoffPlatePre
{
  double Emodul;
  double nu;
  double thickness;

  template <typename PreFE, typename FE>
  using Skill = KirchhoffPlate<PreFE, FE>;
};

template <typename PreFE, typename FE>
class KirchhoffPlate
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
  using Pre       = KirchhoffPlatePre;

  KirchhoffPlate(Pre pre)
      : Emodul{pre.Emodul},
        nu{pre.nu},
        thickness{pre.thickness} {}

  void bind() {}

  static Eigen::Matrix<double, 3, 3> constitutiveMatrix(double Emod, double p_nu, double p_thickness) {
    const double factor = Emod * Dune::power(p_thickness, 3) / (12.0 * (1.0 - p_nu * p_nu));
    Eigen::Matrix<double, 3, 3> D;
    D.setZero();
    D(0, 0) = 1;
    D(0, 1) = D(1, 0) = p_nu;
    D(1, 1)           = 1;
    D(2, 2)           = (1 - p_nu) / 2.0;
    D *= factor;
    return D;
  }

protected:
  void updateStateImpl(const Requirement& /* par */,
                       typename Traits::template VectorTypeConst<> /* correction */) const {}

  template <template <typename, int, int> class RT>
  requires Dune::AlwaysFalse<RT<double, 1, 1>>::value
  auto calculateAtImpl(const Requirement& req, const Dune::FieldVector<double, Traits::mydim>& local,
                       Dune::PriorityTag<0>) const {}

  template <typename ScalarType>
  auto calculateScalarImpl(const Requirement& par, ScalarAffordance affo,
                           const std::optional<std::reference_wrapper<const Eigen::VectorX<ScalarType>>>& dx =
                               std::nullopt) const -> ScalarType {
    const auto geometry   = underlying().localView().element().geometry();
    const auto& wGlobal   = par.globalSolution();
    const auto& lambda    = par.parameter();
    const auto D          = constitutiveMatrix(Emodul, nu, thickness);
    ScalarType energy     = 0.0;
    const auto& localView = underlying().localView();
    const auto& tree      = localView.tree();
    auto& ele             = localView.element();
    auto& fe              = tree.finiteElement();
    Eigen::VectorX<ScalarType> wNodal;
    wNodal.setZero(fe.size());
    Dune::CachedLocalBasis localBasis(fe.localBasis());
    const auto& rule = Dune::QuadratureRules<double, 2>::rule(ele.type(), 2 * localBasis.order());

    localBasis.bind(rule, Dune::bindDerivatives(0, 2));
    if (dx) {
      for (auto i = 0U; i < fe.size(); ++i)
        wNodal(i) = dx.value().get()[i] + wGlobal[localView.index(tree.localIndex(i))[0]];
    } else {
      for (auto i = 0U; i < fe.size(); ++i)
        wNodal(i) = wGlobal[localView.index(tree.localIndex(i))[0]];
    }

    /// Calculate Kirchhoff plate energy
    for (auto&& [gpIndex, gp] : localBasis.viewOverIntegrationPoints()) {
      auto& N          = localBasis.evaluateFunction(gpIndex);
      auto& ddN        = localBasis.evaluateSecondDerivatives(gpIndex);
      auto& ddN_xixi   = ddN.col(0);
      auto& ddN_etaeta = ddN.col(1);
      auto& ddN_xieta  = ddN.col(2);

      const auto Jinv = Dune::toEigen(geometry.jacobianInverseTransposed(gp.position())).transpose().eval();

      Eigen::VectorXd ddN_xx(fe.size());
      Eigen::VectorXd ddN_yy(fe.size());
      Eigen::VectorXd ddN_xy(fe.size());
      using Dune::power;
      // The following derivative transformation assumes a non-distorted grid, otherwise there would be non-linear terms
      for (auto i = 0U; i < fe.size(); ++i) {
        ddN_xx[i] = ddN_xixi[i] * power(Jinv(0, 0), 2);
        ddN_yy[i] = ddN_etaeta[i] * power(Jinv(1, 1), 2);
        ddN_xy[i] = ddN_xieta[i] * Jinv(0, 0) * Jinv(1, 1);
      }
      Eigen::Vector<ScalarType, 3> kappa;
      kappa << ddN_xx.dot(wNodal), ddN_yy.dot(wNodal), 2 * ddN_xy.dot(wNodal);
      ScalarType w = N.dot(wNodal);

      energy += (0.5 * kappa.dot(D * kappa) - w * lambda) * geometry.integrationElement(gp.position()) * gp.weight();
    }

    /// Clamp boundary using penalty method
    const double penaltyFactor = 1e8;
    if (ele.hasBoundaryIntersections())
      for (auto& intersection : intersections(localView.globalBasis().gridView(), ele))
        if (intersection.boundary()) {
          const auto& rule1 = Dune::QuadratureRules<double, 1>::rule(intersection.type(), 2 * localBasis.order());
          Eigen::MatrixX2d dN_xi_eta;
          for (auto& gp : rule1) {
            const auto& gpInElement = intersection.geometryInInside().global(gp.position());
            localBasis.evaluateJacobian(gpInElement, dN_xi_eta);
            Eigen::VectorXd dN_x(fe.size());
            Eigen::VectorXd dN_y(fe.size());
            const auto Jinv = Dune::toEigen(geometry.jacobianInverseTransposed(gpInElement)).transpose().eval();
            for (auto i = 0U; i < fe.size(); ++i) {
              dN_x[i] = dN_xi_eta(i, 0) * Jinv(0, 0);
              dN_y[i] = dN_xi_eta(i, 1) * Jinv(1, 1);
            }
            const ScalarType w_x = dN_x.dot(wNodal);
            const ScalarType w_y = dN_y.dot(wNodal);

            energy += 0.0 * 0.5 * penaltyFactor * (w_x * w_x + w_y * w_y);
          }
        }

    return energy;
  }

private:
  //> CRTP
  const auto& underlying() const { return static_cast<const FE&>(*this); }
  auto& underlying() { return static_cast<FE&>(*this); }
  double Emodul;
  double nu;
  double thickness;
};

auto klPlate(double E, double nu, double thickness) { return KirchhoffPlatePre(E, nu, thickness); }

int main(int argc, char** argv) {
  Ikarus::init(argc, argv);

  /// Create 2D nurbs grid
  using namespace Ikarus;
  constexpr int griddim                                    = 2;
  constexpr int dimworld                                   = 2;
  const std::array<std::vector<double>, griddim> knotSpans = {
      {{0, 0, 1, 1}, {0, 0, 1, 1}}
  };

  using ControlPoint = Dune::IGA::NURBSPatchData<griddim, dimworld>::ControlPointType;

  const double Lx                                            = 10;
  const double Ly                                            = 10;
  const std::vector<std::vector<ControlPoint>> controlPoints = {
      { {.p = {0, 0}, .w = 1},  {.p = {0, Ly}, .w = 1}},
      {{.p = {Lx, 0}, .w = 1}, {.p = {Lx, Ly}, .w = 1}}
  };

  std::array<int, griddim> dimsize = {2, 2};

  std::vector<size_t> dofsVec;
  std::vector<double> l2Evcector;
  auto controlNet = Dune::IGA::NURBSPatchData<griddim, dimworld>::ControlPointNetType(dimsize, controlPoints);
  using Grid      = Dune::IGA::NURBSGrid<griddim, dimworld>;

  Dune::IGA::NURBSPatchData<griddim, dimworld> patchData;
  patchData.knotSpans     = knotSpans;
  patchData.degree        = {1, 1};
  patchData.controlPoints = controlNet;
  /// Increase polynomial degree in each direction
  patchData = Dune::IGA::degreeElevate(patchData, 0, 1);
  patchData = Dune::IGA::degreeElevate(patchData, 1, 1);
  Grid grid(patchData);

  for (int ref = 0; ref < 5; ++ref) {
    auto gridView = grid.leafGridView();
    // draw(gridView);
    using namespace Dune::Functions::BasisFactory;
    /// Create nurbs basis with extracted preBase from grid
    auto basis = Ikarus::makeBasis(gridView, nurbs());
    /// Fix complete boundary (simply supported plate)
    auto basisP = std::make_shared<const decltype(basis)>(basis);
    Ikarus::DirichletValues dirichletValues(basisP->flat());
    dirichletValues.fixBoundaryDOFs(
        [&](auto& dirichletFlags, auto&& globalIndex) { dirichletFlags[globalIndex] = true; });

    /// Create finite elements
    auto localView         = basis.flat().localView();
    const double Emod      = 2.1e8;
    const double nu        = 0.3;
    const double thickness = 0.1;

    auto sk          = skills(klPlate(Emod, nu, thickness));
    using AutoDiffFE = Ikarus::AutoDiffFE<decltype(makeFE(basis, sk))>;
    std::vector<AutoDiffFE> fes;
    for (auto&& ge : elements(gridView)) {
      fes.emplace_back(AutoDiffFE(makeFE(basis, sk)));
      fes.back().bind(ge);
    }

    /// Create assembler
    auto denseAssembler = DenseFlatAssembler(fes, dirichletValues);

    /// Create non-linear operator with potential energy
    Eigen::VectorXd w;
    w.setZero(basis.flat().size());

    double totalLoad = 2000 * thickness * thickness * thickness;

    auto req = AutoDiffFE::Requirement();
    req.insertGlobalSolution(w).insertParameter(totalLoad);
    denseAssembler.bind(req, Ikarus::AffordanceCollections::elastoStatics);

    const auto& K = denseAssembler.matrix();
    const auto& R = denseAssembler.vector();
    Eigen::LDLT<Eigen::MatrixXd> solver;
    solver.compute(K);
    w -= solver.solve(R);

    // Output solution to vtk
    auto wGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<double>(basis.flat(), w);
    Dune::SubsamplingVTKWriter vtkWriter(gridView, Dune::refinementLevels(2));
    vtkWriter.addVertexData(wGlobalFunc, Dune::VTK::FieldInfo("w", Dune::VTK::FieldInfo::Type::scalar, 1));
    vtkWriter.write("iks004_kirchhoffPlate");

    /// Create analytical solution function for the simply supported case
    const double D = Emod * Dune::power(thickness, 3) / (12 * (1 - Dune::power(nu, 2)));
    // https://en.wikipedia.org/wiki/Bending_of_plates#Simply-supported_plate_with_uniformly-distributed_load
    auto wAna = [&](auto x) {
      double w                = 0.0;
      const int seriesFactors = 40;
      const double pi         = std::numbers::pi;
      auto oddFactors =
          std::ranges::iota_view(1, seriesFactors) | std::views::filter([](auto i) { return i % 2 != 0; });
      for (auto m : oddFactors)
        for (auto n : oddFactors)
          w += sin(m * pi * x[0] / Lx) * sin(n * pi * x[1] / Ly) /
               (m * n * Dune::power(m * m / (Lx * Lx) + n * n / (Ly * Ly), 2));

      return 16 * totalLoad / (Dune::power(pi, 6) * D) * w;
    };

    /// Displacement at center of clamped square plate
    // clamped sol http://faculty.ce.berkeley.edu/rlt/reports/clamp.pdf
    const double wCenterClamped = 1.265319087 / (D / (totalLoad * Dune::power(Lx, 4)) * 1000.0);
    auto wGlobalFunction =
        Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, 1>>(basis.flat(), w);
    auto wGlobalAnalyticFunction = Dune::Functions::makeAnalyticGridViewFunction(wAna, gridView);
    auto localw                  = localFunction(wGlobalFunction);
    auto localwAna               = localFunction(wGlobalAnalyticFunction);

    /// Calculate L_2 error for simply supported case
    double l2_error  = 0.0;
    double l2_normEx = 0.0;
    for (auto& ele : elements(gridView)) {
      localView.bind(ele);
      localw.bind(ele);
      localwAna.bind(ele);
      const auto geo   = localView.element().geometry();
      const auto& rule = Dune::QuadratureRules<double, 2>::rule(
          ele.type(), 2U * localView.tree().finiteElement().localBasis().order());
      for (auto gp : rule) {
        const auto intElement = ele.geometry().integrationElement(gp.position()) * gp.weight();
        const auto w_ex       = localwAna(gp.position());
        const auto w_fe       = localw(gp.position());
        l2_error += Dune::power(w_ex - w_fe, 2) * intElement;
        l2_normEx += w_ex * intElement;
      }
    }

    l2_error = std::sqrt(l2_error) / std::sqrt(l2_normEx);
    std::cout << "l2_error: " << l2_error << " Dofs:: " << basis.flat().size() << std::endl;
    dofsVec.push_back(basis.flat().size());
    l2Evcector.push_back(l2_error);
    grid.globalRefine(1);
  }
  /// Draw L_2 error over dofs count
  using namespace matplot;
  auto f  = figure(true);
  auto ax = gca();
  ax->y_axis().label("L2_error");

  ax->x_axis().label("#Dofs");
  auto p = ax->loglog(dofsVec, l2Evcector);
  p->line_width(2);
  p->marker(line_spec::marker_style::asterisk);
  // save("kirchhoffPlate.png");
  // f->draw();
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(5s);
}
