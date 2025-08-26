    // =============================================================================
    // PROJECT CHRONO - http://projectchrono.org
    //
    // Copyright (c) 2019 projectchrono.org
    // All rights reserved.
    //
    // Use of this source code is governed by a BSD-style license that can be found
    // in the LICENSE file at the top level of the distribution and at
    // http://projectchrono.org/license-chrono.txt.
    //
    // =============================================================================
    // Authors: Nic Olsen, Ruochun Zhang
    // =============================================================================
    // Chrono::Gpu demo using SMC method. A body whose geometry is described by an
    // OBJ file is time-integrated in Chrono and interacts with a granular wave tank
    // in Chrono::Gpu via the co-simulation framework. The entire simulation consists
    // of 2 runs: the settling phase (which outputs a checkpoint file), and a restarted
    // phase (which load the checkpoint file and then drop the ball, literally).
    // =============================================================================

    #include <iostream>
    #include <vector>
    #include <string>
    #include <iomanip>
    #include <algorithm>
    #include <fstream>
    #include "chrono/core/ChGlobal.h"
    #include "chrono/physics/ChSystemSMC.h"
    #include "chrono/physics/ChBody.h"
    #include "chrono/physics/ChForce.h"
    #include "chrono/timestepper/ChTimestepper.h"
    #include "chrono/utils/ChUtilsSamplers.h"
    #include "chrono/utils/ChUtilsCreators.h"
    #include "chrono/assets/ChVisualShapeSphere.h"

    #include "chrono/geometry/ChTriangleMeshConnected.h"
    #include "chrono/assets/ChVisualShapeTriangleMesh.h"


    #include "chrono_gpu/physics/ChSystemGpu.h"
    #include "chrono_gpu/utils/ChGpuJsonParser.h"

    #ifdef CHRONO_VSG
        #include "chrono_gpu/visualization/ChGpuVisualizationVSG.h"
    #endif

    #include "chrono_thirdparty/filesystem/path.h"

    using namespace chrono;
    using namespace chrono::gpu;


    #include <sstream>
    #include <fstream>
    #include <algorithm>
    #include <map>



    // --- Units & conversions (cgs) ---
    inline double dyn_to_N(double F_dyn) { return F_dyn * 1e-5; }   // dyn → N
    inline double cm_to_m(double z_cm)   { return z_cm * 1e-2; }    // cm → m
    inline double erg_to_J(double E_erg) { return E_erg * 1e-7; }   // erg → J

    // --- Schedule types & parameters ---
    enum class SchedType { BangBang, DwellBurst };

    struct SchedParams {
        SchedType type = SchedType::BangBang;
        // dwell-and-burst knobs:
        double tau_d_ms = 0.0;     // dwell time [ms], 0..50
        double T_b_ms   = 60.0;    // burst duration [ms], 30..120
        double frac_preload = 0.25;// fraction of stroke devoted to preload (0..1)
        // common
        double settle_s = 0.05;    // wait on bed [s]
        double A_cm     = 8.0;     // nominal stroke [cm]
    };

    // ease [0,1] → [0,1], C1; derivative w.r.t s is 6s(1-s)
    inline double ease01(double s){ s = std::clamp(s, 0.0, 1.0); return s*s*(3.0 - 2.0*s); }
    inline double dease01_ds(double s){ s = std::clamp(s, 0.0, 1.0); return 6.0*s*(1.0 - s); }

    // rest-length command and its time derivative
    struct LrCmd { double Lr; double Lr_dot; };  // [cm], [cm/s]

    // returns command at time t with amplitude scale (A_scale)
    inline LrCmd rest_length_cmd(double t, const SchedParams& p, double Lnat_equil, double A_scale){
        const double A = p.A_cm * A_scale;
        const double t0 = p.settle_s;

        if (t < t0) return { Lnat_equil, 0.0 };

        if (p.type == SchedType::DwellBurst && p.tau_d_ms <= 1e-12) {
        const double Tb = p.T_b_ms * 1e-3;
        const double t1 = t0 + Tb;
        const double t2 = t1 + 0.04;
        const double t3 = t2 + Tb;
        auto ramp = [&](double tstart, double T, double Lstart, double Lend){
            if (T <= 0){ return LrCmd{Lend, 0.0}; }
            double s = ease01((t - tstart)/T);
            double dsdt = (1.0/T) * dease01_ds((t - tstart)/T);
            double Lr = Lstart + (Lend - Lstart) * s;
            double Lr_dot = (Lend - Lstart) * dsdt; // [cm/s]
            return LrCmd{Lr, Lr_dot};
        };
        if (t < t1) return ramp(t0, Tb, Lnat_equil, Lnat_equil + A);
        if (t < t2) return { Lnat_equil + A, 0.0 };
        if (t < t3) return ramp(t2, Tb, Lnat_equil + A, Lnat_equil);
        return { Lnat_equil, 0.0 };
    }

        auto ramp = [&](double tstart, double T, double Lstart, double Lend){
            if (T <= 0){ return LrCmd{Lend, 0.0}; }
            double s = ease01((t - tstart)/T);
            double dsdt = (1.0/T) * dease01_ds((t - tstart)/T);
            double Lr = Lstart + (Lend - Lstart) * s;
            double Lr_dot = (Lend - Lstart) * dsdt; // [cm/s]
            return LrCmd{Lr, Lr_dot};
        };

        if (p.type == SchedType::BangBang){
            // single fast burst (up), hold, return
            const double Tb = p.T_b_ms * 1e-3;
            const double t1 = t0 + Tb;
            const double t2 = t1 + 0.04;
            const double t3 = t2 + Tb;
            if (t < t1) return ramp(t0, Tb, Lnat_equil, Lnat_equil + A);
            if (t < t2) return { Lnat_equil + A, 0.0 };
            if (t < t3) return ramp(t2, Tb, Lnat_equil + A, Lnat_equil);
            return { Lnat_equil, 0.0 };
        } else {
            // dwell (slow preload) then fast burst, hold, return
            const double Td = p.tau_d_ms * 1e-3;
            const double Tb = p.T_b_ms   * 1e-3;
            const double A_pre = p.frac_preload * A;
            const double A_bst = (1.0 - p.frac_preload) * A;

            const double t1 = t0 + Td;
            const double t2 = t1 + Tb;
            const double t3 = t2 + 0.04;
            const double t4 = t3 + Tb;
            if (t < t1) return ramp(t0, Td, Lnat_equil,          Lnat_equil + A_pre);
            if (t < t2) return ramp(t1, Tb, Lnat_equil + A_pre,  Lnat_equil + A_pre + A_bst);
            if (t < t3) return { Lnat_equil + A_pre + A_bst, 0.0 };
            if (t < t4) return ramp(t3, Tb, Lnat_equil + A_pre + A_bst, Lnat_equil);
            return { Lnat_equil, 0.0 };
        }
    }

    // simple LS slope of Fn_now vs z_pen over a window [z0,z1] (cm, dyn)
    inline double slope_F_vs_z(const std::vector<double>& zpen_cm,
                            const std::vector<double>& Fn_dyn,
                            double z0_cm, double z1_cm){
        double sxx=0, sxy=0;
        int n=0;
        for (size_t i=0;i<zpen_cm.size();++i){
            double z = zpen_cm[i];
            if (z>=z0_cm && z<=z1_cm){
                sxx += z*z; sxy += z*Fn_dyn[i]; ++n;
            }
        }
        return (n>=4 && sxx>0)? (sxy/sxx) : 0.0;
    }

    static constexpr bool LOG_PROBE_STEPS = false; 
    // per-run metrics
    struct RunMetrics {
        double E_act_erg     = 0.0;  // ∫(-Fs)Lr_dot dt   (erg)
        double E_damp_erg    = 0.0;  // ∫ c Ldot^2 dt     (erg)
        double Fn_peak_dyn   = 0.0;
        double contact_s     = 0.0;
        double dz_apex_cm    = 0.0;
        double dFdz_dyn_per_cm = 0.0;  // dyn/cm over 5–10 mm
        double A_scale       = 1.0;
        double dFdz_dwell_dyn_per_cm = 0.0;     
        double impulse_dyn_s = 0.0;   // ∫ Fn dt  [dyn*s]
        double sink_cm       = 0.0; double t_liftoff_s            = -1.0;
        double vz_body_takeoff_cm_s   = 0.0;
        double vz_com_takeoff_cm_s    = 0.0;
        double apex_com_pred_m        = 0.0;
          // final penetration depth [cm]  // NEW

    };

    // Minimal, conclusive grid: 6 burst durations × 4 dwells = 30 runs (incl. baselines)
    static const std::vector<double> TB_MS = {30.0, 45.0, 60.0, 90.0};
    static const std::vector<double> TD_MS = {0.0, 20.0, 40.0};     // Td=0 is a useful control

    
    // --- parse a Chrono::Gpu particle CSV and estimate surface height near impact axis
    //     We take the top-quantile of z within an annulus to avoid outliers.
    //
    // csv_path: path to particle CSV written by gpu_sys.WriteParticleFile(...)
    // cx,cy   : impact axis (usually 0,0)
    // r_in/out: radial window (cm) over which to probe the surface (e.g., 1–2 projectile diameters)
    // q       : quantile in [0,1], e.g., 0.90 -> 90th percentile height
    static double EstimateSurfaceHeightCSV(const std::string& csv_path,
                                        float cx, float cy,
                                        float r_in, float r_out,
                                        double q = 0.90) {
        std::ifstream in(csv_path);
        if (!in) {
            std::cerr << "[zsurf] cannot open " << csv_path << std::endl;
            return 0.0;
        }
        std::string line; 
        std::getline(in, line);  // header "X,Y,Z"
        std::vector<double> zs;
        zs.reserve(10000);
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            double x,y,z; char c1,c2;
            if (!(ss >> x >> c1 >> y >> c2 >> z)) continue;
            double dx = x - cx, dy = y - cy;
            double r2 = dx*dx + dy*dy;
            if (r2 >= r_in*r_in && r2 <= r_out*r_out) zs.push_back(z);
        }
        if (zs.empty()) {
            std::cerr << "[zsurf] no particles in annulus" << std::endl;
            return 0.0;
        }
        size_t k = std::clamp<size_t>(size_t(q * zs.size()), 0, zs.size()-1);
        std::nth_element(zs.begin(), zs.begin() + k, zs.end());
        return zs[k];
    }



    // Output frequency
    float out_fps = 50;

    // Enable/disable run-time visualization
    constexpr bool ENABLE_VIS = false; 
    bool render = ENABLE_VIS;
    float render_fps = 2000;

    // ---- Units: cgs (cm, g, s). Everything must follow this. ----
    constexpr double g_z          = -980.0;      // gravity [cm/s^2]
    constexpr float  sphere_rad   = 3.0f;        // foot radius [cm] (3 cm ~ 0.03 m)
    constexpr double mass_foot_g  = 100.0;       // [g]  (0.1 kg)
    constexpr double mass_body_g  = 300.0;       // [g]  (0.3 kg)
    const    ChVector3d box_hsize = {3.0, 3.0, 3.0}; // [cm] half-sizes (matches 0.04 m)
    constexpr double natural_len  = 13.2;        // [cm] (0.132 m)

    constexpr double k_spring     = 8.94e5;      // [g/s^2] (894 N/m in SI)
    constexpr double c_spring     = 5.00e3;      // [g/s]   (0.5  N*s/m in SI)

    // --- Disc geometry & mass (cgs: cm, g, s) ---
    const float R_disc = sphere_rad;   // same planform radius as sphere foot (e.g., 3.0 cm)    
    const float H_disc = 1.0f;         // thickness (e.g., 1 cm) — thin, but not paper-thin
    const double m_disc_g = mass_foot_g;  // keep the same foot mass for fair shape comparison


    // ---- Runtime toggles ----
                // turn off runtime visualization
    constexpr bool SAVE_PARTICLE_SNAPSHOTS = false; // turn off heavy particle CSVs
    constexpr bool SAVE_MESH_SNAPSHOTS     = false; // turn off mesh OBJ dumps
    constexpr bool LOG_CSV                 = true;  // enable lightweight CSV logging

    // Logging rate (<= out_fps). Keep small for speed, large enough for analysis.
    constexpr float log_fps = 500.0f;


    // ---- Hop command (in centimeters and seconds; you're in cgs) ----
    constexpr double precompress = 2.0;  // cm shorter than natural_len during hold
    constexpr double t_hold      = 0.25; // s - hold time at precompression
    constexpr double t_ramp      = 0.05; // s - ramp back to natural_len (actuation)


        // --- Dwell preload controller knobs ---
    constexpr double PRELOAD_FORCE_GAIN = 1.30;     // target preload ≈ 1.3 × (foot+body) weight
    constexpr double LRB_RATE_LIMIT_CM_S = 30.0;    // max |rest-length| change rate during dwell (cm/s)


    inline double smoothstep01(double s) {
        s = std::clamp(s, 0.0, 1.0);
        return s*s*(3.0 - 2.0*s); // C1 smooth
    }

    void runBallDrop(ChSystemGpuMesh& gpu_sys,
                    ChGpuSimulationParameters& params,
                    const SchedParams& sched = SchedParams{},
                    double A_scale = 1.0,
                    const std::string& tag = "run",
                    RunMetrics* out = nullptr) {
        // Add a ball mesh to the GPU system

        // Build a diagonal scale matrix: diag(R, R, H)
        ChMatrix33<float> S_disc;
        // Option A: Eigen-style fill
        S_disc << R_disc, 0.f,    0.f,
                0.f,    R_disc, 0.f,
                0.f,    0.f,    H_disc;


        const std::string foot_obj = GetChronoDataFile("models/CONCAVE_DSIC_2 v0.obj");
        gpu_sys.AddMesh(GetChronoDataFile("models/cylinderZ.obj"), ChVector3f(0), S_disc, (float)m_disc_g); //GetChronoDataFile("models/cylinderZ.obj")
        // gpu_sys.AddMesh(GetChronoDataFile("models/sphere.obj"), ChVector3f(0), ChMatrix33<float>(sphere_rad), mass_foot_g);


        // One more thing: we need to manually enable mesh in tout_dhis run, because we disabled it in the settling phase,
        // let's overload that option.
        gpu_sys.EnableMeshCollision(true);

        // Re-apply granular params in ONE_STEP before Initialize()
        gpu_sys.SetKn_SPH2SPH(params.normalStiffS2S);
        gpu_sys.SetKn_SPH2WALL(params.normalStiffS2W);
        gpu_sys.SetKn_SPH2MESH(params.normalStiffS2M);

        gpu_sys.SetGn_SPH2SPH(params.normalDampS2S);
        gpu_sys.SetGn_SPH2WALL(params.normalDampS2W);
        gpu_sys.SetGn_SPH2MESH(params.normalDampS2M);

        gpu_sys.SetKt_SPH2SPH(params.tangentStiffS2S);
        gpu_sys.SetKt_SPH2WALL(params.tangentStiffS2W);
        gpu_sys.SetKt_SPH2MESH(params.tangentStiffS2M);

        gpu_sys.SetGt_SPH2SPH(params.tangentDampS2S);
        gpu_sys.SetGt_SPH2WALL(params.tangentDampS2W);
        gpu_sys.SetGt_SPH2MESH(params.tangentDampS2M);

        gpu_sys.SetStaticFrictionCoeff_SPH2SPH(params.static_friction_coeffS2S);
        gpu_sys.SetStaticFrictionCoeff_SPH2WALL(params.static_friction_coeffS2W);
        gpu_sys.SetStaticFrictionCoeff_SPH2MESH(params.static_friction_coeffS2M);

        // Use sane rolling (your file had 0.57 by accident — that’s huge)
        // Try ~0.06 for S2S and ~0.02 for S2W/S2M
        gpu_sys.SetRollingCoeff_SPH2SPH(0.06f);
        gpu_sys.SetRollingCoeff_SPH2WALL(0.02f);
        gpu_sys.SetRollingCoeff_SPH2MESH(0.02f);

        // Cohesion / adhesion
        gpu_sys.SetCohesionRatio(params.cohesion_ratio);
        gpu_sys.SetAdhesionRatio_SPH2WALL(params.adhesion_ratio_s2w);
        gpu_sys.SetAdhesionRatio_SPH2MESH(params.adhesion_ratio_s2m);

        // Enable mesh contacts and then Initialize
        gpu_sys.EnableMeshCollision(true);



        gpu_sys.Initialize();
        std::cout << gpu_sys.GetNumMeshes() << " meshes" << std::endl;

            // ---- CONFIG (edit these if you change sizes) ----
        const double Rp       = 3.0;     // cm
        const double v_drop   = 100.0;   // cm/s (1.0 m/s)
        const double clearance= 0.3;     // cm (~one grain radius)

        // Estimate bed top from the settling fill_top heuristic
        const double fill_top     = params.box_Z / 4.0;          // cm
        const double settle_drop  = 2.0 * params.sphere_radius;  // cm
        const double bed_top_est  = fill_top - settle_drop;

        ChVector3d p0(0, 0, bed_top_est + Rp + clearance);



        // Create rigid foot simulation
        ChSystemSMC sys;
        sys.SetContactForceModel(ChSystemSMC::ContactForceModel::Hooke);
        sys.SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT);
        sys.SetGravitationalAcceleration(ChVector3d(0, 0, g_z));

        double inertia = 2.0 / 5.0 * mass_foot_g * sphere_rad * sphere_rad;
        // double fill_top = params.box_Z / 4.0;
        // const double clearance = 2.0 * params.sphere_radius;   // ~1 grain diameter * 2
    
        auto foot = chrono_types::make_shared<ChBody>();
        const double R = R_disc;
        const double H = H_disc;
        const double Ixx = (1.0/12.0) * m_disc_g * (3.0*R*R + H*H);
        const double Izz = 0.5 * m_disc_g * R*R;
        foot->SetMass(mass_foot_g);
        foot->SetInertiaXX(ChVector3d(Ixx, Ixx, Izz));
        // double settle_drop = 2.0 * params.sphere_radius;   // heuristic: bed settled down ~ 1 grain diameter
        double z_surface_est = fill_top - settle_drop;
        double clr = 0.5 * params.sphere_radius;          // 0.5 grain radius
        double z0 = z_surface_est + sphere_rad + clr;
        foot->SetPos(ChVector3d(0,0,-13.5));
        // foot->SetPosDt(ChVector3d(0, 0, -v_drop));
        auto vis_cyl = chrono_types::make_shared<ChVisualShapeCylinder>(R, H);
        foot->AddVisualShape(vis_cyl);
        // auto sph = chrono_types::make_shared<ChVisualShapeSphere >(sphere_rad);
        // foot->AddVisualShape(sph);   
        sys.AddBody(foot);

        // Create the visual triangle mesh for the foot (NO geometry:: prefix)
        // auto tri = chrono_types::make_shared<ChTriangleMeshConnected>();
        // tri->LoadWavefrontMesh(foot_obj, /*load_normals*/ true, /*load_uv*/ true);
        // tri->Transform(ChVector3d(0,0,0), ChMatrix33<>(ChVector3d(R_disc, R_disc, H_disc)));

        // auto foot_vis = chrono_types::make_shared<ChVisualShapeTriangleMesh>();
        // foot_vis->SetMesh(tri);
        // foot_vis->SetBackfaceCull(true);
        // foot_vis->SetName("FootVis");
        // foot->AddVisualShape(foot_vis);

        auto ground = chrono_types::make_shared<ChBody>();
        ground->SetFixed(true);
        sys.AddBody(ground);

        // --- Body (box)
        auto body = chrono_types::make_shared<ChBody>();
        body->SetName("body");
        body->SetMass(mass_body_g);
        ChVector3d full = box_hsize * 2.0;
        ChVector3d I_box( (1.0/12.0) * mass_body_g * (full.y()*full.y() + full.z()*full.z()),
                        (1.0/12.0) * mass_body_g * (full.x()*full.x() + full.z()*full.z()),
                        (1.0/12.0) * mass_body_g * (full.x()*full.x() + full.y()*full.y()) );
        body->SetInertiaXX(I_box);
        // place to mirror your FSI (natural_length + sphere_radius above foot center)
        body->SetPos(foot->GetPos() + ChVector3d(0, 0, natural_len + H_disc/2));
        // body->SetPos(foot->GetPos() + ChVector3d(0, 0, natural_len + sphere_rad));
        body->SetRot(QUNIT);
        body->EnableCollision(false);  // GPU handles mesh contacts
        auto box = chrono_types::make_shared<ChVisualShapeBox >(box_hsize);
        body->AddVisualShape(box);
        sys.AddBody(body);

        // Vertical-only motion (add ground prismatic)
        auto prism_to_ground = chrono_types::make_shared<ChLinkLockPrismatic>();
        ChFrame<> pr_frame(ChVector3d(0,0,0), QUNIT); // Z axis
        prism_to_ground->Initialize(foot, ground, false, pr_frame, pr_frame);
        sys.AddLink(prism_to_ground);

        // --- Prismatic joints (Z axis), same as FSI intent
        auto prism = chrono_types::make_shared<ChLinkLockPrismatic>();
        ChFrame<> prism_frame(foot->GetPos(), QUNIT);   // Z-up
        prism->Initialize(foot, body, false, prism_frame, prism_frame);
        sys.AddLink(prism);


        auto spring = chrono_types::make_shared<ChLinkTSDA>();
        spring->Initialize(body, foot, true, ChVector3d(0,0,-box_hsize.z()/2), ChVector3d(0,0,H_disc/2));
        // spring->Initialize(body, foot, true, ChVector3d(0,0,-box_hsize.z()/2), ChVector3d(0,0,sphere_rad));
        const double rest_len_nat = natural_len - ((mass_foot_g + mass_body_g) * (-g_z)) / k_spring;
        double Lr_prev_cmd_cm = rest_len_nat;
        const double rest_len_pre = rest_len_nat - precompress;
        // initialize; the schedule will update every step inside the loop
        spring->SetRestLength(rest_len_nat);

        double rest_len = natural_len - ((mass_foot_g + mass_body_g) * (-g_z)) / k_spring;
        // spring->SetRestLength(rest_len);
        spring->SetSpringCoefficient(k_spring);
        spring->SetDampingCoefficient(c_spring);
        // visual coil
        auto coil = chrono_types::make_shared<ChVisualShapeSpring>(1, 100, 20);
        coil->SetColor(ChColor(1.f, 0.f, 0.f));
        spring->AddVisualShape(coil);
        sys.AddLink(spring);

        // Create a run-time visualizer
        std::shared_ptr<ChVisualSystem> visSys;

    #ifdef CHRONO_VSG
        // GPU plugin
    if (ENABLE_VIS) {
        auto visGPU = chrono_types::make_shared<ChGpuVisualizationVSG>(&gpu_sys);
        // VSG visual system (attach visGPU as plugin)
        auto visVSG = chrono_types::make_shared<vsg3d::ChVisualSystemVSG>();
        visVSG->AttachSystem(&sys);
        visVSG->AttachPlugin(visGPU);
        visVSG->SetWindowTitle("Chrono::Gpu ball cosim demo");
        visVSG->SetWindowSize(1280, 800);
        visVSG->SetWindowPosition(100, 100);
        visVSG->AddCamera(ChVector3d(0, -200, 100), ChVector3d(0, 0, 0));
        visVSG->SetLightIntensity(0.9f);
        visVSG->SetLightDirection(CH_PI_2, CH_PI / 6);

        visVSG->Initialize();
        visSys = visVSG;
    }
    #else
        render = false;
    #endif

        std::string out_dir = GetChronoOutputPath() + "GPU/";
        filesystem::create_directory(filesystem::path(out_dir));
        out_dir = out_dir + params.output_dir;
        filesystem::create_directory(filesystem::path(out_dir));

        float iteration_step = params.step_size;
        std::string log_path = out_dir + "/" + tag + ".csv";
        bool enable_step_log = LOG_CSV;
            if (!LOG_PROBE_STEPS && tag == std::string("probe")) enable_step_log = false;
        std::ofstream log;
        if (enable_step_log) {
        log.open(log_path, std::ios::out | std::ios::trunc);
        // make writes immediate (disable internal buffering)
        log.rdbuf()->pubsetbuf(nullptr, 0);          // unbuffered
        log.setf(std::ios::unitbuf);                 // flush after each insertion
        log << std::fixed << std::setprecision(6);   // consistent formatting
        if (!log.is_open()) {
            std::cerr << "[error] cannot open " << log_path << std::endl;
        } else {
            std::cout << "[log] writing to " << log_path << std::endl;
            log << "t,t_cmd,A_scale,foot_z,foot_vz,body_z,body_vz,Fn_gpu,Fs_spring,rest_len,Lr_dot,contact,"
    << "Wact_cum_J,Impulse_cum_Ns\n";


        }
    }
        int   log_frame = 0;

        // Force accumulator on the foot (to apply GPU contact forces)
        auto accumulator_index = foot->AddAccumulator();

        std::cout << "Output at    " << out_fps << " FPS" << std::endl;
        std::cout << "Rendering at " << render_fps << " FPS" << std::endl;

        int sim_frame = 0;
        int render_frame = 0;
        int out_frame = 0;

            // ---- per-run metrics/state (function scope so they survive the loop) ----
        // ---- per-run metrics/state (function scope so they reset every run) ----
        RunMetrics M_local;
        M_local.A_scale = A_scale;
        // Dwell force-tracking offset on top of the schedule (cm)
        double Lr_bias_cm = 0.0;
        double Lr_bias_prev_cm = 0.0;

        bool   c_active        = false;
        bool   have_zc         = false;
        double t_c_start       = 0.0, t_c_end = 0.0;
        double z_contact       = 0.0;

        double z_body0         = body->GetPos().z();
        double z_body_apex     = z_body0;

        // Penetration samples for stiffness estimates (declare once, before loop)
        std::vector<double> zpen_cm, Fn_dyn_series;
        std::vector<double> zpen_cm_dwell, Fn_dyn_series_dwell;


        // Contact detection & penetration bookkeeping (reset per run)
        bool   contact_started  = false;
        bool   rest_recorded    = false;
        double t_contact        = 0.0;
        double v_at_contact     = 0.0;
        double z_at_contact     = 0.0;
        double z_final          = 0.0;

        // --- IMPORTANT: start the foot at the estimated surface, not a hardcoded -13 ---
        {
            double z_surface_est = (params.box_Z / 4.0) - 2.0 * params.sphere_radius; // fill_top - settle_drop
            double clr = 0.5 * params.sphere_radius;                                   // small clearance
            double z0  = z_surface_est + sphere_rad + clr;                             // foot center height
            foot->SetPos(ChVector3d(0, 0, -13.5));
            body->SetPos(foot->GetPos() + ChVector3d(0, 0, natural_len + H_disc / 2));
        }

        // --- main time loop ---
        clock_t start = std::clock();
        int calm_count = 0;                   // rest detection counter (per run)

        // window for stiffness proxy in the first 5–10 mm
        const double ZWIN_LO_CM = 0.5;        // 5 mm
        const double ZWIN_HI_CM = 1.0;        // 10 mm
        double z_contact0_cm = 0.0;
        bool   z_contact0_set = false;
        double sink_max_cm = 0.0;  // track maximum (z_contact0 - z_now) while in contact




        for (double t = 0; t < (double)params.time_end; t += iteration_step) {
            gpu_sys.ApplyMeshMotion(0, foot->GetPos(), foot->GetRot(),
                                    foot->GetPosDt(), foot->GetAngVelParent());

            ChVector3d ball_force, ball_torque;
            gpu_sys.CollectMeshContactForces(0, ball_force, ball_torque);

            // Contact detection (cgs)
            const double Fn_now     = ball_force.z();                   // upward (+)
            const double Fn_eps_now = 1e-3 * foot->GetMass() * 980.0;   // ~0.1% weight
            const bool   in_contact = (std::abs(Fn_now) > Fn_eps_now);

            if (in_contact) {
                if (!z_contact0_set) {
                    z_contact0_cm = foot->GetPos().z();   // reference at first touch
                    z_contact0_set = true;
                }
                const double depth_now_cm = z_contact0_cm - foot->GetPos().z();
                if (depth_now_cm > sink_max_cm)
                    sink_max_cm = depth_now_cm;
            }


            // Track impulse only when positive normal load
            if (in_contact && Fn_now > 0.0)
                M_local.impulse_dyn_s += Fn_now * iteration_step;
 

            // First contact timestamp (optional: ignore t==0 if you don't want spawn hits)
            if (!contact_started && in_contact /* && t > 1e-5 */) {
                contact_started = true;
                t_contact    = t;
                v_at_contact = foot->GetPosDt().z();
                z_at_contact = foot->GetPos().z();
                std::cout << "[impact] t=" << t_contact << "s, vz=" << v_at_contact
                        << " cm/s, z_contact=" << z_at_contact << " cm\n";
            }


            const double t_cmd = t;  // pass wall time; rest_length_cmd respects settle_s


            // Final command
            // Base schedule
            LrCmd cmd = rest_length_cmd(t_cmd, sched, rest_len_nat, A_scale);

            // Start from schedule values
            double Lr_final_cm      = cmd.Lr;       // [cm]
            double Lr_final_dot_cm_s= cmd.Lr_dot;   // [cm/s]

            // Are we in the dwell window of DB?
            
            const double t_since_sched = t - sched.settle_s;

            // Are we in the dwell window?
            const bool in_dwell = (sched.type == SchedType::DwellBurst) &&
                                (t_since_sched >= 0.0) &&
                                (t_since_sched <  sched.tau_d_ms * 1e-3);


            // Only adjust during dwell *and* while actually in contact
            if (in_dwell && in_contact) {
                // Target preload ~ 1.2–1.5× total weight
                const double W_N     = (mass_foot_g + mass_body_g) * 1e-3 * 9.81;  // [N]
                const double F_tgt_N = PRELOAD_FORCE_GAIN * W_N;                    // [N]
                const double F_now_N = dyn_to_N(Fn_now);                            // [N]
                const double err_N   = F_tgt_N - F_now_N;                           // +ve → need *more* force

                // Convert spring k to SI [N/m]; k_spring is dyn/cm in cgs
                const double k_Npm   = dyn_to_N(k_spring) / cm_to_m(1.0);           // [N/m] (≈894)

                // For Hooke: Fn ≈ k * (L - Lr). To increase Fn (err>0), we must DECREASE Lr.
                // So dLr = -err/k. Work in cm with a rate limit.
                double dLr_cmd_m     = - err_N / std::max(1e-9, k_Npm);             // [m]
                double dLr_cmd_cm    = dLr_cmd_m * 100.0;                           // [cm]

                // Rate-limit the bias update
                const double max_step_cm = LRB_RATE_LIMIT_CM_S * iteration_step;    // [cm] per step
                dLr_cmd_cm = std::clamp(dLr_cmd_cm, -max_step_cm, max_step_cm);

                // Integrate bias
                Lr_bias_cm += dLr_cmd_cm;
                // Effective derivative contributed by bias this step
                const double Lr_bias_dot_cm_s = (Lr_bias_cm - Lr_bias_prev_cm) / iteration_step;

                // Add bias to schedule_local
                Lr_final_cm       += Lr_bias_cm;
                Lr_final_dot_cm_s += Lr_bias_dot_cm_s;

                // Remember for next step
                Lr_bias_prev_cm    = Lr_bias_cm;
            }

            // Apply effective rest length (schedule + bias)
            spring->SetRestLength(Lr_final_cm);


            // Rest detection (|vz| small over ~50 ms)
            const double vz = foot->GetPosDt().z();
            calm_count = (std::abs(vz) < 1.0) ? (calm_count + 1) : 0;


            if (!rest_recorded && contact_started && calm_count > int(0.05 / iteration_step)) {
                rest_recorded = true;
                z_final = foot->GetPos().z();
                const double depth = z_at_contact - z_final;
                M_local.sink_cm = depth;
                std::cout << std::fixed << std::setprecision(3)
                        << "[penetration] z_contact=" << z_at_contact
                        << "  z_final=" << z_final
                        << "  depth=" << depth << " cm"
                        << "  (depth/Dp=" << depth / (2.0 * sphere_rad ) << ")\n";
                std::ofstream sum(out_dir + "/drop_summary.csv", std::ios::app);
                if (sum.tellp() == 0)
                    sum << "t_contact,vz_contact_cm_s,z_contact_cm,z_final_cm,depth_cm,depth_over_Dp\n";
                sum << t_contact << "," << v_at_contact << ","
                    << z_at_contact << "," << z_final << ","
                    << depth << "," << depth / (2.0 * sphere_rad) << "\n";
            }

            // Apply GPU contact forces to the foot
            foot->EmptyAccumulator(accumulator_index);
            foot->AccumulateForce(accumulator_index, ball_force, foot->GetPos(), false);
            foot->AccumulateTorque(accumulator_index, ball_torque, false);

            // Spring force (Hooke + viscous) + energies (cgs)
            const double L    = spring->GetLength();
            const double Ldot = spring->GetVelocity();
            const double Fs   = k_spring * (L - spring->GetRestLength()) + c_spring * Ldot;
            M_local.E_act_erg  += (-Fs * Lr_final_dot_cm_s) * iteration_step;   // uses schedule + bias   // actuator work
            M_local.E_damp_erg +=  (c_spring * Ldot * Ldot) * iteration_step;

            // --- Step-energy correction (captures instantaneous ΔLr not seen by Lr_dot * dt) ---
            // Use current commanded rest length and its continuous prediction.
            const double Lr_now_cm     = Lr_final_cm;                               // command used this step
            const double dLr_cont_cm   = Lr_final_dot_cm_s * iteration_step;        // what we already integrated
            const double dLr_actual_cm = (Lr_now_cm - Lr_prev_cmd_cm);              // what actually changed
            const double dLr_err_cm    = dLr_actual_cm - dLr_cont_cm;               // "missing" jump (if any)
            if (std::abs(dLr_err_cm) > 1e-12) {
                // Energy to enforce exact W_act: E += ∫(−Fs dLr) → add (−Fs)*ΔLr for the unaccounted jump
                M_local.E_act_erg += (-Fs) * dLr_err_cm;                            // dyn*cm = erg
            }
            // Update tracker
            Lr_prev_cmd_cm = Lr_now_cm;


            if (!c_active && in_contact) { c_active = true;  t_c_start = t; z_contact = foot->GetPos().z(); have_zc = true; }
            if ( c_active && !in_contact){ c_active = false; t_c_end   = t; }
            if (in_contact)               M_local.Fn_peak_dyn = std::max(M_local.Fn_peak_dyn, Fn_now);

            // Penetration stiffness samples (overall + dwell-only)
            if (have_zc && in_contact) {
                double zpen_now = (z_contact - foot->GetPos().z());   // cm
                zpen_cm.push_back(zpen_now);
                Fn_dyn_series.push_back(Fn_now);
                if (in_dwell) {
                    zpen_cm_dwell.push_back(zpen_now);
                    Fn_dyn_series_dwell.push_back(Fn_now);
                }
            }

            // Apex (body)
            z_body_apex = std::max(z_body_apex, body->GetPos().z());

            if (enable_step_log && t >= (double)log_frame / (double)log_fps) {
                const int contact_bit = in_contact ? 1 : 0;
                log << t << "," << t_cmd << "," << A_scale << ","
                    << foot->GetPos().z()   << "," << foot->GetPosDt().z()   << ","
                    << body->GetPos().z()   << "," << body->GetPosDt().z()   << ","
                    << Fn_now               << "," << Fs                    << ","
                    // in the logging block:
                    << spring->GetRestLength() << "," << Lr_final_dot_cm_s << ","
                    << contact_bit 
                    << "," << erg_to_J(M_local.E_act_erg)                 // Wact_cum_J
                    << "," << (dyn_to_N(1.0) * M_local.impulse_dyn_s)     // Impulse_cum_Ns
                    << "\n";
                log_frame++;
            }

            if (render && t >= render_frame / render_fps) {
                if (!visSys->Run()) break;
                visSys->Render();
                render_frame++;
            }

            gpu_sys.AdvanceSimulation(iteration_step);
            sys.DoStepDynamics(iteration_step);
            sim_frame++;
        }

        // --- finalize per-run metrics ---
        M_local.contact_s   = (t_c_end > t_c_start) ? (t_c_end - t_c_start) : 0.0;
        M_local.dz_apex_cm  = (z_body_apex - z_body0);

        // Slope over first 5–10 mm (fallbacks if sparse)
        auto slope_F_vs_z = [&](const std::vector<double>& zcm,
                                const std::vector<double>& Fdyn,
                                double z0_cm, double z1_cm) {
            double sxx=0, sxy=0; int n=0;
            for (size_t i=0;i<zcm.size();++i){
                double z = zcm[i];
                if (z>=z0_cm && z<=z1_cm){ sxx += z*z; sxy += z*Fdyn[i]; ++n; }
            }
            return (n>=4 && sxx>0)? (sxy/sxx) : 0.0;
        };

        double slope_all = slope_F_vs_z(zpen_cm,       Fn_dyn_series,       ZWIN_LO_CM, ZWIN_HI_CM);
        if (slope_all == 0.0) slope_all = slope_F_vs_z(zpen_cm,       Fn_dyn_series,       0.0, 1.0);
        M_local.dFdz_dyn_per_cm = slope_all;

        double slope_dw  = slope_F_vs_z(zpen_cm_dwell, Fn_dyn_series_dwell, ZWIN_LO_CM, ZWIN_HI_CM);
        if (slope_dw == 0.0) slope_dw  = slope_F_vs_z(zpen_cm_dwell, Fn_dyn_series_dwell, 0.0, 1.0);
        M_local.dFdz_dwell_dyn_per_cm = slope_dw;

        if (sink_max_cm > M_local.sink_cm){
            M_local.sink_cm = sink_max_cm;
        }

        const double Wact_J  = erg_to_J(M_local.E_act_erg);
        const double EPE_J   = ((mass_foot_g + mass_body_g) * 1e-3 * 9.81) * cm_to_m(M_local.dz_apex_cm);
        const double eta     = (Wact_J > 0.0) ? (EPE_J / Wact_J) : 0.0;
        const double Fn_pk_N = dyn_to_N(M_local.Fn_peak_dyn);
        const double mg_N    = (mass_foot_g + mass_body_g) * 1e-3 * 9.81;
        const double Fn_norm = (mg_N > 0.0) ? (Fn_pk_N / mg_N) : 0.0;               // normalized peak
        const double Imp_Ns  = dyn_to_N(1.0) * M_local.impulse_dyn_s;
        const double G_idx   = (Imp_Ns > 0.0) ? (Fn_pk_N / Imp_Ns) : 0.0;           // gentleness index
        const double HJ      = (Wact_J > 0.0) ? (cm_to_m(M_local.dz_apex_cm) / Wact_J) : 0.0; // height per Joule

        // Summary (units converted)
        std::ofstream s(out_dir + "/" + tag + "_summary.txt", std::ios::trunc);
        s << std::fixed << std::setprecision(6)
        << "A_scale="          << M_local.A_scale
        << "  Wact[J]="        << Wact_J
        << "  EPE[J]="         << EPE_J
        << "  eta="            << eta
        << "  Fn_peak[N]="     << Fn_pk_N
        << "  Fn_norm="        << Fn_norm
        << "  contact[s]="     << M_local.contact_s
        << "  dz_apex[m]="     << cm_to_m(M_local.dz_apex_cm)
        << "  dFdz_all[N/m]="  << (dyn_to_N(M_local.dFdz_dyn_per_cm)       / cm_to_m(1.0))
        << "  dFdz_dwell[N/m]="<< (dyn_to_N(M_local.dFdz_dwell_dyn_per_cm) / cm_to_m(1.0))
        << "  impulse[N*s]="   << Imp_Ns
        << "  G="              << G_idx
        << "  HJ="             << HJ
        << "  sink[cm]="       << M_local.sink_cm
        << "\n";



        if (out) *out = M_local;
        if (enable_step_log) log.close();
        clock_t end = std::clock();
        std::cout << "Time: " << ((double)(end - start)) / CLOCKS_PER_SEC << " seconds\n";
    }

    int main(int argc, char* argv[]) {
        SetChronoDataPath("/home/mahir/chrono/chrono_build/data/");
        std::string inputJson = GetChronoDataFile("gpu/ballCosim_test.json");
        if (argc == 2) {
            inputJson = std::string(argv[1]);
        } else if (argc > 2) {
            std::cout << "Usage:\n./demo_GPU_ballCosim <json_file>" << std::endl;
            return 1;
        }

        ChGpuSimulationParameters params;
        if (!ParseJSON(inputJson, params)) {
            std ::cout << "ERROR: reading input file " << inputJson << std::endl;
            return 1;
        }

        if (params.run_mode != CHGPU_RUN_MODE::FRICTIONLESS && params.run_mode != CHGPU_RUN_MODE::ONE_STEP) {
            std::cout << "ERROR: unknown run_mode specified" << std::endl;
            return 1;
        }

        // Output directory
        std::string out_dir = GetChronoOutputPath() + "GPU/";
        filesystem::create_directory(filesystem::path(out_dir));
        out_dir = out_dir + params.output_dir;
        filesystem::create_directory(filesystem::path(out_dir));

        std::string checkpoint_file = out_dir + "/checkpoint.dat";

        constexpr double EQUAL_WORK_TOL = 0.01;   // 1%
        constexpr int    MAX_BISECT_IT  = 8;


        if (params.run_mode == CHGPU_RUN_MODE::ONE_STEP) {
            // Load the settled bed once per run; we reload per run to avoid any extra settling stages.
            auto NperM  = [&](double dyn_per_cm){ return dyn_to_N(dyn_per_cm) / cm_to_m(1.0); };
            auto eta_of = [&](const RunMetrics& M){
                const double W = erg_to_J(M.E_act_erg);
                const double Epe = ( (mass_foot_g + mass_body_g) * 1e-3 * 9.81 ) * cm_to_m(M.dz_apex_cm);
                return (W > 0.0) ? (Epe / W) : 0.0;
            };

            // Report CSV
            std::string sweep_csv = out_dir + "/sweep_min_db_vs_bb.csv";
            std::ofstream sweep(sweep_csv, std::ios::app);
            if (sweep.tellp() == 0) {
                sweep << "tag,sched,tau_d_ms,Tb_ms,A_scale,"
                      << "Wact_J,eta,apex_m,impulse_Ns,Fn_peak_N,dFdz_Npm,sink_cm,"
                      << "base_tag,base_W_J,base_eta,base_apex_m,base_Fn_N,"
                      << "dz_gain_pct,eta_gain_pct,Fn_change_pct,dFdz_change_pct,E_rel_err_pct,"
                      << "Fn_norm,base_Fn_norm,G,base_G,HJ,base_HJ\n";
            }

            
            

            const std::string shape = "disc"; // fixed geometr

            std::map<int,double> last_A_for_Tb;   // key = int(Tb_ms), value = last matched A*



            for (double Tb_ms : TB_MS) {
                // --- 1) Baseline BB at A_scale=1.0 ---
                SchedParams sp_bb; 
                sp_bb.type = SchedType::BangBang; 
                sp_bb.T_b_ms = Tb_ms; 
                sp_bb.tau_d_ms = 0.0;
                sp_bb.settle_s = 0.08;
                RunMetrics M_bb;
                {
                    ChSystemGpuMesh g(checkpoint_file);
                    runBallDrop(g, params, sp_bb, 1.0, (shape + "_bb_Tb" + std::to_string(int(Tb_ms))).c_str(), &M_bb);
                }
                const std::string base_tag = shape + "_bb_Tb" + std::to_string(int(Tb_ms));
                const double Wtarget = erg_to_J(M_bb.E_act_erg);

                const double mg_N   = (mass_foot_g + mass_body_g) * 1e-3 * 9.81;
                const double Fn_bb  = dyn_to_N(M_bb.Fn_peak_dyn);
                const double Fn_bb_norm = (mg_N > 0.0) ? (Fn_bb / mg_N) : 0.0;
                const double Imp_bb = dyn_to_N(1.0) * M_bb.impulse_dyn_s;
                const double G_bb   = (Imp_bb > 0.0) ? (Fn_bb / Imp_bb) : 0.0;
                const double HJ_bb  = (Wtarget > 0.0) ? (cm_to_m(M_bb.dz_apex_cm) / Wtarget) : 0.0;

                sweep << base_tag << ",BB,0," << Tb_ms << ",1.0,"
                    << Wtarget << "," << eta_of(M_bb) << "," << cm_to_m(M_bb.dz_apex_cm) << ","
                    << Imp_bb << "," << Fn_bb << ","
                    << NperM(M_bb.dFdz_dyn_per_cm) << "," << M_bb.sink_cm << ","
                    << base_tag << "," << Wtarget << "," << eta_of(M_bb) << "," << cm_to_m(M_bb.dz_apex_cm) << "," << Fn_bb << ","
                    << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ","
                    << Fn_bb_norm << "," << Fn_bb_norm << ","   // Fn_norm, base_Fn_norm
                    << G_bb << "," << G_bb << ","               // G, base_G
                    << HJ_bb << "," << HJ_bb                    // HJ, base_HJ
                    << "\n";





                // --- 2) DB at td ∈ {30,40} ms (equal-energy to BB) ---
                for (double td_ms : TD_MS) {
                    SchedParams sp_db; 
                    sp_db.type = SchedType::DwellBurst; 
                    sp_db.T_b_ms = Tb_ms; 
                    sp_db.tau_d_ms = td_ms;
                    sp_db.settle_s = 0.08;
                    sp_db.frac_preload = 0.25;  // no rewind; dwell uses 25% of total stroke, burst uses the rest

                    // ---- Equal-work bisection with warm-start & ≤1% tolerance ----
                    auto eval_W = [&](double A)->double{
                        RunMetrics Mp;
                        ChSystemGpuMesh g(checkpoint_file);
                        runBallDrop(g, params, sp_db, A, "probe", &Mp);    // step log suppressed for "probe"
                        return erg_to_J(Mp.E_act_erg);
                    };

                    // warm-start guess from previous DB at the same Tb (if any)
                    double A_guess = 1.0;
                    auto itTb = last_A_for_Tb.find((int)Tb_ms);
                    if (itTb != last_A_for_Tb.end())
                        A_guess = itTb->second;

                    // initial bracket around the guess
                    double Alo = std::max(0.30, 0.70 * A_guess);
                    double Ahi = std::min(2.00, 1.50 * A_guess);

                    // ensure bracket contains target by expansion (at most a couple tries)
                    double Wlo = eval_W(Alo);
                    double Whi = eval_W(Ahi);
                    int expand = 0;
                    while ((Wlo > Wtarget || Whi < Wtarget) && expand < 3) {
                        if (Wlo > Wtarget) { Alo = std::max(0.30, 0.70 * Alo); Wlo = eval_W(Alo); }
                        if (Whi < Wtarget) { Ahi = std::min(2.50, 1.50 * Ahi); Whi = eval_W(Ahi); }
                        ++expand;
                    }

                    // bisection
                    double Astar = 0.5 * (Alo + Ahi);
                    double Wmid  = 0.0;
                    for (int it = 0; it < MAX_BISECT_IT; ++it) {
                        Astar = 0.5 * (Alo + Ahi);
                        Wmid  = eval_W(Astar);
                        const double err = (Wmid - Wtarget) / std::max(1e-12, Wtarget);
                        if (std::abs(err) <= EQUAL_WORK_TOL)
                            break;
                        (Wmid < Wtarget ? Alo : Ahi) = Astar;
                    }
                    last_A_for_Tb[(int)Tb_ms] = Astar;   // cache for next td at same Tb


                    RunMetrics M_db;
                    std::ostringstream tag; tag << shape << "_db_td" << int(td_ms) << "_Tb" << int(Tb_ms)
                                                << "_A" << std::fixed << std::setprecision(2) << Astar;
                    {
                        ChSystemGpuMesh g(checkpoint_file);
                        runBallDrop(g, params, sp_db, Astar, tag.str(), &M_db);
                    }

                    // Deltas vs baseline
                    const double dz_db = cm_to_m(M_db.dz_apex_cm);
                    const double dz_bb = cm_to_m(M_bb.dz_apex_cm);
                    const double Fn_db = dyn_to_N(M_db.Fn_peak_dyn);
                    const double dFdz_db = NperM(M_db.dFdz_dyn_per_cm);
                    const double dFdz_bb = NperM(M_bb.dFdz_dyn_per_cm);
                    const double eta_db = eta_of(M_db), eta_bb = eta_of(M_bb);
                    const double W_db   = erg_to_J(M_db.E_act_erg);
                    const double E_rel_err = (Wtarget>0) ? 100.0*std::abs(W_db - Wtarget)/Wtarget : 0.0;

                    const double mg_N   = (mass_foot_g + mass_body_g) * 1e-3 * 9.81;
                    const double Fn_db_norm = (mg_N > 0.0) ? (Fn_db / mg_N) : 0.0;
                    const double Imp_db = dyn_to_N(1.0) * M_db.impulse_dyn_s;
                    const double G_db   = (Imp_db > 0.0) ? (Fn_db / Imp_db) : 0.0;
                    const double HJ_db  = (W_db > 0.0) ? (dz_db / W_db) : 0.0;
                    const double Imp_bb = dyn_to_N(1.0) * M_bb.impulse_dyn_s;
                    const double Fn_bb  = dyn_to_N(M_bb.Fn_peak_dyn);
                    const double Fn_bb_norm = (mg_N > 0.0) ? (Fn_bb / mg_N) : 0.0;
                    const double G_bb   = (Imp_bb > 0.0) ? (Fn_bb / Imp_bb) : 0.0;
                    const double HJ_bb  = (Wtarget > 0.0) ? (cm_to_m(M_bb.dz_apex_cm) / Wtarget) : 0.0;


                    const auto pct = [](double a,double b){ return (b!=0)? 100.0*(a-b)/b : 0.0; };

                    sweep << tag.str() << ",DB," << td_ms << "," << Tb_ms << "," << Astar << ","
                            << W_db << "," << eta_db << "," << dz_db << ","
                            << Imp_db << "," << Fn_db << ","
                            << dFdz_db << "," << M_db.sink_cm << ","
                            << base_tag << "," << Wtarget << "," << eta_bb << "," << dz_bb << "," << Fn_bb << ","
                            << pct(dz_db, dz_bb) << "," << pct(eta_db, eta_bb) << "," << pct(Fn_db, Fn_bb) << "," << pct(dFdz_db, dFdz_bb) << ","
                            << E_rel_err << ","
                            << Fn_db_norm << "," << Fn_bb_norm << ","
                            << G_db << "," << G_bb << ","
                            << HJ_db << "," << HJ_bb
                            << "\n";


               
                }
            }

            sweep.close();

            return 0;
}



        // run_mode = CHGPU_RUN_MODE::FRICTIONLESS, this is a newly started run. We have to set all simulation params.
        ChSystemGpuMesh gpu_sys(params.sphere_radius, params.sphere_density,
                                ChVector3f(params.box_X, params.box_Y, params.box_Z));

        printf(
            "Now run_mode == FRICTIONLESS, this run is particle settling phase.\n"
            "After it is done, you will have a settled bed of granular material.\n"
            "A checkpoint file will be generated in the output directory to store this state.\n"
            "Next, edit the JSON file, change 'run_mode' from 0 (FRICTIONLESS) to 1 (ONE_STEP),\n"
            "then run this demo again to proceed with the ball drop part of this demo.\n\n");

        float iteration_step = params.step_size;
        double fill_bottom = -params.box_Z / 2.0;
        double fill_top = params.box_Z / 4.0;

        chrono::utils::ChPDSampler<float> sampler(2.4f * params.sphere_radius);
        // chrono::utils::ChHCPSampler<float> sampler(2.05 * params.sphere_radius);

        // leave a 4cm margin at edges of sampling
        ChVector3d hdims(params.box_X / 2 - 0.3, params.box_Y / 2 - 0.3, 0);
        ChVector3d center(0, 0, fill_bottom + 2.0 * params.sphere_radius);
        std::vector<ChVector3f> body_points;

        // Shift up for bottom of box
        center.z() += 3 * params.sphere_radius;
        while (center.z() < fill_top) {
            // You can uncomment this line to see a report on particle creation process.
            std::cout << "Create layer at " << center.z() << std::endl;
            auto points = sampler.SampleBox(center, hdims);
            body_points.insert(body_points.end(), points.begin(), points.end());
            center.z() += 2.3 * params.sphere_radius;
        }
        std::cout << body_points.size() << " particles sampled!" << std::endl;

        gpu_sys.SetParticles(body_points);

        gpu_sys.SetKn_SPH2SPH(params.normalStiffS2S);
        gpu_sys.SetKn_SPH2WALL(params.normalStiffS2W);
        gpu_sys.SetKn_SPH2MESH(params.normalStiffS2M);

        gpu_sys.SetGn_SPH2SPH(params.normalDampS2S);
        gpu_sys.SetGn_SPH2WALL(params.normalDampS2W);
        gpu_sys.SetGn_SPH2MESH(params.normalDampS2M);

        gpu_sys.SetKt_SPH2SPH(params.tangentStiffS2S);
        gpu_sys.SetKt_SPH2WALL(params.tangentStiffS2W);
        gpu_sys.SetKt_SPH2MESH(params.tangentStiffS2M);

        gpu_sys.SetGt_SPH2SPH(params.tangentDampS2S);
        gpu_sys.SetGt_SPH2WALL(params.tangentDampS2W);
        gpu_sys.SetGt_SPH2MESH(params.tangentDampS2M);

        gpu_sys.SetCohesionRatio(params.cohesion_ratio);
        gpu_sys.SetAdhesionRatio_SPH2MESH(params.adhesion_ratio_s2m);
        gpu_sys.SetAdhesionRatio_SPH2WALL(params.adhesion_ratio_s2w);

        gpu_sys.SetGravitationalAcceleration(ChVector3f(params.grav_X, params.grav_Y, params.grav_Z));

        gpu_sys.SetFixedStepSize(params.step_size);
        gpu_sys.SetFrictionMode(CHGPU_FRICTION_MODE::MULTI_STEP);
        gpu_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::CENTERED_DIFFERENCE);


        gpu_sys.SetStaticFrictionCoeff_SPH2SPH(params.static_friction_coeffS2S);
        gpu_sys.SetStaticFrictionCoeff_SPH2WALL(params.static_friction_coeffS2W);
        gpu_sys.SetStaticFrictionCoeff_SPH2MESH(params.static_friction_coeffS2M);

        // (optional, if available in your build) enable rolling resistance
        gpu_sys.SetRollingCoeff_SPH2SPH(0.06f);
        gpu_sys.SetRollingCoeff_SPH2WALL(0.07f);
        gpu_sys.SetRollingCoeff_SPH2MESH(0.07f);

        // gpu_sys.SetRollingCoeff_SPH2SPH(params.rolling_friction_coeffS2S);
        // gpu_sys.SetRollingCoeff_SPH2WALL(params.rolling_friction_coeffS2W);
        // gpu_sys.SetRollingCoeff_SPH2MESH(params.rolling_friction_coeffS2M);

        gpu_sys.SetParticleOutputMode(CHGPU_OUTPUT_MODE::NONE);
        gpu_sys.SetVerbosity(params.verbose);
        gpu_sys.SetBDFixed(true);

        // In the settling run we disable the mesh.
        gpu_sys.EnableMeshCollision(false);


        
        // We could prescribe the motion of the big box domain. But here in this demo we will not do that.
        // std::function<double3(float)> pos_func_wave = [&params](float t) {
        //     double3 pos = {0, 0, 0};

        //     double t0 = 0.5;
        //     double freq = CH_PI / 4;

        //     if (t > t0) {
        //         pos.x = 0.1 * params.box_X * std::sin((t - t0) * freq);
        //     }
        //     return pos;
        // };

        // gpu_sys.setBDWallsMotionFunction(pos_func_wave);
        

        gpu_sys.Initialize();

        int sim_frame = 0;
        int out_frame = 0;

        for (double t = 0; t < (double)params.time_end; t += iteration_step) {
        
            if (t >= out_frame / out_fps) {
                std::cout << "Output frame " << sim_frame + 1 << std::endl;
                char filename[100];
                // sprintf(filename, "%s/step%06d.csv", out_dir.c_str(), sim_frame);
                // gpu_sys.WriteParticleFile(std::string(filename));

                out_frame++;
            }

            gpu_sys.AdvanceSimulation(iteration_step);

            sim_frame++;
        }

        // This is settling phase, so output a checkpoint file
        gpu_sys.WriteCheckpointFile(checkpoint_file);

        return 0;
    }
