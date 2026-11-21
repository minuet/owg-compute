/* ASPECT v3.0.0 boundary velocity plugin "force controlled" for tools/aspect_brune.py (Brune et al. 2016 M1).
   The boundary moves as a rigid plate end: the horizontal velocity v is the same at every depth (the plate is pulled
   as a whole and the stress in each layer follows from its strength at the common strain rate, as in Kusznir & Park
   1987, ref. 11 of Brune et al. 2016). Brune et al. keep the boundary force constant and let v follow; here, at the
   start of every time step, v is rescaled from the force F_m measured on the previous solution:
       v_new = v_old * clamp((F / F_m)^m, 1 / f_max, f_max)
   F_m is the force in excess of lithostatic, F_m = integral over the boundary of (sigma_xx - sigma_yy) dy
   = integral of 2 eta (eps_xx - eps_yy) dy, the definition of Brune et al.'s analytical model (tau = F / 2D).
   The pressure cancels, so F_m does not depend on the free-surface height at the wall. Only v_old is state, and it is
   read back from the solution (mean boundary velocity), so a restart from a checkpoint needs nothing extra.
   Parameters and their derivation: PARAM_AUDIT 11.15. */
#include <aspect/boundary_velocity/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/material_model/interface.h>
#include <aspect/global.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <algorithm>
#include <cmath>

namespace aspect
{
  namespace BoundaryVelocity
  {
    using namespace dealii;

    template <int dim>
    class ForceControlled : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void update () override
        {
          const types::boundary_id bid =
            this->get_geometry_model().translate_symbolic_boundary_name_to_id(boundary_name);
          if (this->get_timestep_number() == 0 || this->get_timestep_number() == numbers::invalid_unsigned_int)
            {
              velocity = initial_velocity;
              return;
            }
          const QGauss<dim-1> quadrature(this->get_fe().base_element(this->introspection().base_elements.velocities)
                                         .degree + 1);
          FEFaceValues<dim> fe_face_values(this->get_mapping(), this->get_fe(), quadrature,
                                           update_values | update_gradients | update_quadrature_points |
                                           update_JxW_values);
          double force = 0, sxx = 0, vx = 0, length = 0;
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned() && cell->at_boundary())
              for (const unsigned int f : cell->face_indices())
                if (cell->at_boundary(f) && cell->face(f)->boundary_id() == bid)
                  {
                    fe_face_values.reinit(cell, f);
                    MaterialModel::MaterialModelInputs<dim> in(fe_face_values, cell, this->introspection(),
                                                               this->get_solution());
                    MaterialModel::MaterialModelOutputs<dim> out(fe_face_values.n_quadrature_points,
                                                                 this->n_compositional_fields());
                    in.requested_properties = MaterialModel::MaterialProperties::viscosity;
                    this->get_material_model().evaluate(in, out);
                    for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                      {
                        const double w = fe_face_values.JxW(q), eta = out.viscosities[q];
                        force += 2 * eta * (in.strain_rate[q][0][0] - in.strain_rate[q][1][1]) * w;
                        sxx += (2 * eta * in.strain_rate[q][0][0] - in.pressure[q]) * w;
                        vx += in.velocity[q][0] * w;
                        length += w;
                      }
                  }
          const MPI_Comm comm = this->get_mpi_communicator();
          force = dealii::Utilities::MPI::sum(force, comm);
          sxx = dealii::Utilities::MPI::sum(sxx, comm);
          vx = dealii::Utilities::MPI::sum(vx, comm) / dealii::Utilities::MPI::sum(length, comm);
          const double old_velocity = vx > 0 ? vx : initial_velocity;
          const double factor = force > 0 ? std::pow(target_force / force, exponent) : max_factor;
          velocity = old_velocity * std::min(max_factor, std::max(1. / max_factor, factor));
          this->get_pcout() << "     force controlled: t_yr " << this->get_time() / year_in_seconds
                            << " v_old_mm_yr " << old_velocity * year_in_seconds * 1e3
                            << " F_dev_TN_m " << force / 1e12 << " int_sxx_TN_m " << sxx / 1e12
                            << " v_new_mm_yr " << velocity * year_in_seconds * 1e3 << std::endl;
        }

        Tensor<1,dim> boundary_velocity (const types::boundary_id, const Point<dim> &) const override
        {
          Tensor<1,dim> u;
          u[0] = velocity;
          return u;
        }

        static void declare_parameters (ParameterHandler &prm)
        {
          prm.enter_subsection("Boundary velocity model");
          prm.enter_subsection("Force controlled");
          prm.declare_entry("Boundary", "right", Patterns::Anything(), "Boundary the force acts on.");
          prm.declare_entry("Force", "9e12", Patterns::Double(0), "Target force in excess of lithostatic, N/m.");
          prm.declare_entry("Initial velocity", "1e-3", Patterns::Double(0), "Velocity of the first step, m/yr.");
          prm.declare_entry("Exponent", "1", Patterns::Double(0), "m in (F / F_m)^m.");
          prm.declare_entry("Maximum factor", "2", Patterns::Double(1), "Largest change of v in one step.");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void parse_parameters (ParameterHandler &prm) override
        {
          prm.enter_subsection("Boundary velocity model");
          prm.enter_subsection("Force controlled");
          boundary_name = prm.get("Boundary");
          target_force = prm.get_double("Force");
          initial_velocity = prm.get_double("Initial velocity") / year_in_seconds;
          exponent = prm.get_double("Exponent");
          max_factor = prm.get_double("Maximum factor");
          prm.leave_subsection();
          prm.leave_subsection();
        }

      private:
        std::string boundary_name;
        double target_force, initial_velocity, exponent, max_factor, velocity = 0;
    };
  }
}

namespace aspect
{
  namespace BoundaryVelocity
  {
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(ForceControlled, "force controlled",
                                            "Rigid boundary velocity rescaled every time step so that the force "
                                            "in excess of lithostatic stays at a target value.")
  }
}
