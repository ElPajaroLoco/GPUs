#include "GSimulation.hpp"
#include "cpu_time.hpp"
#include <sycl/sycl.hpp>


GSimulation :: GSimulation()
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(16000); 
  set_nsteps(10);
  set_tstep(0.1); 
  set_sfreq(1);
}

void GSimulation :: set_number_of_particles(int N)  
{
  set_npart(N);
}

void GSimulation :: set_number_of_steps(int N)  
{
  set_nsteps(N);
}

void GSimulation :: init_pos()  
{
  std::random_device rd;	//random number generator
  std::mt19937 gen(42);      
  std::uniform_real_distribution<real_type> unif_d(0.0,1.0);
  
  particles->pos_x = new real_type[get_npart()];
  particles->pos_y = new real_type[get_npart()];
  particles->pos_z = new real_type[get_npart()];

  for(int i=0; i<get_npart(); ++i)
  {
    particles->pos_x[i] = unif_d(gen);
    particles->pos_y[i] = unif_d(gen);
    particles->pos_z[i] = unif_d(gen);
  }
}

void GSimulation :: init_vel()  
{
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(-1.0,1.0);

  particles->vel_x = new real_type[get_npart()];
  particles->vel_y = new real_type[get_npart()];
  particles->vel_z = new real_type[get_npart()];

  for(int i=0; i<get_npart(); ++i)
  {
    particles->vel_x[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_y[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_z[i] = unif_d(gen) * 1.0e-3f;
  }
}

void GSimulation :: init_acc() 
{
  particles->acc_x = new real_type[get_npart()];
  particles->acc_y = new real_type[get_npart()];
  particles->acc_z = new real_type[get_npart()];

  for(int i=0; i<get_npart(); ++i)
  {
    particles->acc_x[i] = 0.f;
    particles->acc_y[i] = 0.f;
    particles->acc_z[i] = 0.f;
  }
}

void GSimulation :: init_mass() 
{
  real_type n   = static_cast<real_type> (get_npart());
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0.0,1.0);

  particles->mass = new real_type[get_npart()];

  for(int i=0; i<get_npart(); ++i)
  {
    particles->mass[i] = (n * unif_d(gen)); 
  }
  std::cout << real_type(particles->mass[0]) << std::endl;
}

void GSimulation :: get_acceleration(sycl::queue Q, int n, ParticleSoA *ptcs)
{
  const float softeningSquared = 1e-3f;
  const float G = 6.67259e-11f;

  const int TILE_SIZE = 64;
  size_t size = n;

  auto global_range = sycl::nd_range<1>(
    {(size + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE},
    {TILE_SIZE});

    Q.submit([&](sycl::handler &h) { 

      // Memoria shared del SM (en caso de CUDA)
      sycl::local_accessor<real_type> local_posx(TILE_SIZE, h);
      sycl::local_accessor<real_type> local_posy(TILE_SIZE, h);
      sycl::local_accessor<real_type> local_posz(TILE_SIZE, h);
      sycl::local_accessor<real_type> local_mass(TILE_SIZE, h);


      h.parallel_for(global_range, [=](sycl::nd_item<1> item) {
        const int lid_x = item.get_local_id(0);
        const int gid_x = item.get_global_id(0);

        if (gid_x >= n) return; // Evitamos que se ejecuten hilos extra
        
        // Inicializamos los acumuladores locales
        real_type ax = 0.0f, ay = 0.0f, az = 0.0f; 

        // Cargamos la posicion de una particula para compararla luego con las demas.
        const real_type xi = ptcs->pos_x[gid_x];
        const real_type yi = ptcs->pos_y[gid_x];
        const real_type zi = ptcs->pos_z[gid_x];

        // Pasamos por todas las demas particulas para hacer los calculos
        for (int tile_base = 0; tile_base < n; tile_base += TILE_SIZE) {
            int tile_idx = tile_base + lid_x;

            // Cargamos las posiciones y la masa en memoria local
            if (tile_idx < n) {
              local_posx[lid_x] = ptcs->pos_x[tile_idx];
              local_posy[lid_x] = ptcs->pos_y[tile_idx];
              local_posz[lid_x] = ptcs->pos_z[tile_idx];
              local_mass[lid_x] = ptcs->mass[tile_idx];
            }else {
              local_posx[lid_x] = 0.0f;
              local_posy[lid_x] = 0.0f;
              local_posz[lid_x] = 0.0f;
              local_mass[lid_x] = 0.0f; 
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Calculo para evitar sobrepasar limites
            int tile_limit = sycl::min(TILE_SIZE, n - tile_base);
            // Bucle que realiza los calculos de una particula con todas las demas.
            for (int k = 0; k < tile_limit; ++k) {

              real_type dx = local_posx[k] - xi;
              real_type dy = local_posy[k] - yi;
              real_type dz = local_posz[k] - zi;

              real_type dist_sqr = dx*dx + dy*dy + dz*dz + softeningSquared;
              real_type inv_dist = sycl::rsqrt(dist_sqr);
              real_type inv_dist_cube = inv_dist * inv_dist * inv_dist;
              real_type force = G * local_mass[k] * inv_dist_cube;

              ax += dx * force;
              ay += dy * force;
              az += dz * force;
            }
            item.barrier(sycl::access::fence_space::local_space);
        }

        // Pasamos el resultado de la acceleracion en memoria global.
        ptcs->acc_x[gid_x] = ax;
        ptcs->acc_y[gid_x] = ay;
        ptcs->acc_z[gid_x] = az;
    });
    }).wait();
}

real_type GSimulation :: updateParticles(sycl::queue Q, const int n, real_type dt, ParticleSoA *ptcs, real_type *energy)
{
  int i;
  *energy = 0;

  // Operacion de reduccion de sycl para sumar la energia (cumple la funcionn de atomic add)
  auto reduction_energy = sycl::reduction(energy, sycl::plus<real_type>());
  const int TILE_SIZE = 64;
  size_t size = n;

  
  auto global_range = sycl::nd_range<1>(
    {(size + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE},
    {TILE_SIZE});
    
    Q.submit([&](sycl::handler &h) { 

    h.parallel_for(global_range, reduction_energy, [=](sycl::nd_item<1> item, auto &energy_sum) {
      const int lid_x = item.get_local_id(0);
      const int gid_x = item.get_global_id(0);

      if (gid_x >= size) return;


      // Actualiza velocidades
      ptcs->vel_x[gid_x] += ptcs->acc_x[gid_x] * dt;
      ptcs->vel_y[gid_x] += ptcs->acc_y[gid_x] * dt;
      ptcs->vel_z[gid_x] += ptcs->acc_z[gid_x] * dt;

      // Actualiza posiciones
      ptcs->pos_x[gid_x] += ptcs->vel_x[gid_x] * dt;
      ptcs->pos_y[gid_x] += ptcs->vel_y[gid_x] * dt;
      ptcs->pos_z[gid_x] += ptcs->vel_z[gid_x] * dt;

      // Reestablece las aceleraciones
      ptcs->acc_x[gid_x] = 0.0f;
      ptcs->acc_y[gid_x] = 0.0f;
      ptcs->acc_z[gid_x] = 0.0f;

      // Suma para la energia kinetica total
      energy_sum += ptcs->mass[gid_x] * (ptcs->vel_x[gid_x] * ptcs->vel_x[gid_x] + ptcs->vel_y[gid_x] * ptcs->vel_y[gid_x] + ptcs->vel_z[gid_x] * ptcs->vel_z[gid_x]);
    });
  }).wait();

  return *energy;
}

void GSimulation :: start(sycl::queue Q) 
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();

  particles = new ParticleSoA;

  init_pos();
  init_vel();
  init_acc();
  init_mass();
  
  print_header();
  
  _totTime = 0.; 
  
  
  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ( (11. + 18. ) * nd*nd  +  nd * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;

  ParticleSoA *ptcs = sycl::malloc_device<ParticleSoA>(1, Q);

  // Usamos malloc_device porque asi la memoria se queda siempre en GPU y no se sincroniza. Siendo mas eficiente
  ParticleSoA h_ptcs;
  h_ptcs.acc_x = sycl::malloc_device<real_type>(get_npart(),Q);
  h_ptcs.acc_y = sycl::malloc_device<real_type>(get_npart(),Q);
  h_ptcs.acc_z = sycl::malloc_device<real_type>(get_npart(),Q);
  h_ptcs.vel_x = sycl::malloc_device<real_type>(get_npart(),Q);
  h_ptcs.vel_y = sycl::malloc_device<real_type>(get_npart(),Q);
  h_ptcs.vel_z = sycl::malloc_device<real_type>(get_npart(),Q);
  h_ptcs.pos_x = sycl::malloc_device<real_type>(get_npart(),Q);
  h_ptcs.pos_y = sycl::malloc_device<real_type>(get_npart(),Q);
  h_ptcs.pos_z = sycl::malloc_device<real_type>(get_npart(),Q);
  h_ptcs.mass = sycl::malloc_device<real_type>(get_npart(),Q);

  real_type *energy_usm = sycl::malloc_shared<real_type>(1, Q);

  // Copy initialized host data to device
  Q.memcpy(h_ptcs.vel_x, particles->vel_x, n * sizeof(real_type)).wait();
  Q.memcpy(h_ptcs.vel_y, particles->vel_y, n * sizeof(real_type)).wait();
  Q.memcpy(h_ptcs.vel_z, particles->vel_z, n * sizeof(real_type)).wait();
  Q.memcpy(h_ptcs.pos_x, particles->pos_x, n * sizeof(real_type)).wait();
  Q.memcpy(h_ptcs.pos_y, particles->pos_y, n * sizeof(real_type)).wait();
  Q.memcpy(h_ptcs.pos_z, particles->pos_z, n * sizeof(real_type)).wait();
  Q.memcpy(h_ptcs.mass, particles->mass, n * sizeof(real_type)).wait();

  // Initialize accelerations to zero on device
  Q.memset(h_ptcs.acc_x, 0, n * sizeof(real_type)).wait();
  Q.memset(h_ptcs.acc_y, 0, n * sizeof(real_type)).wait();
  Q.memset(h_ptcs.acc_z, 0, n * sizeof(real_type)).wait();

  Q.memcpy(ptcs, &h_ptcs, sizeof(ParticleSoA)).wait();

  /* 
  Cabe recalcar que estas transferencias podrian optimizarse para solaparlas 
  pero debido a que en este programa esas copias solo se hacen una vez y el rendimiento
  que medimos es solo de la computacion no del programa entero he decidido no optimizarlo.
  */

const double t0 = time.start();
for (int s=1; s<=get_nsteps(); ++s)
{   
  
  *energy_usm = 0;
  
  ts0 += time.start(); 
  
  get_acceleration(Q, n, ptcs);

  energy = updateParticles(Q, n, dt, ptcs, energy_usm);
  
  _kenergy = 0.5 * energy; 
  
  ts1 += time.stop();


  if(!(s%get_sfreq()) ) 
    {
      nf += 1;      
      std::cout << " " 
      <<  std::left << std::setw(8)  << s
      <<  std::left << std::setprecision(5) << std::setw(8)  << s*get_tstep()
      <<  std::left << std::setprecision(5) << std::setw(12) << _kenergy
      <<  std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
      <<  std::left << std::setprecision(5) << std::setw(12) << gflops*get_sfreq()/(ts1 - ts0)
      <<  std::endl;
      if(nf > 2) 
      {
        av  += gflops*get_sfreq()/(ts1 - ts0);
        dev += gflops*get_sfreq()*gflops*get_sfreq()/((ts1-ts0)*(ts1-ts0));
      }
      
      ts0 = 0;
      ts1 = 0;
    }
  
  } 
  
  Q.memcpy(&h_ptcs, ptcs, sizeof(ParticleSoA)).wait();  
  sycl::free(h_ptcs.acc_x, Q);
  sycl::free(h_ptcs.acc_y, Q);
  sycl::free(h_ptcs.acc_z, Q);
  sycl::free(h_ptcs.vel_x, Q);
  sycl::free(h_ptcs.vel_y, Q);
  sycl::free(h_ptcs.vel_z, Q);
  sycl::free(h_ptcs.pos_x, Q);
  sycl::free(h_ptcs.pos_y, Q);
  sycl::free(h_ptcs.pos_z, Q);
  sycl::free(h_ptcs.mass, Q);
  sycl::free(ptcs, Q);
  
  const double t1 = time.stop();
  _totTime  = (t1-t0);
  _totFlops = gflops*get_nsteps();
  
  av/=(double)(nf-2);
  dev=sqrt(dev/(double)(nf-2)-av*av);
  

  std::cout << std::endl;
  std::cout << "# Total Time (s)      : " << _totTime << std::endl;
  std::cout << "# Average Performance : " << av << " +- " <<  dev << std::endl;
  std::cout << "===============================" << std::endl;

}


void GSimulation :: print_header()
{
	    
  std::cout << " nPart = " << get_npart()  << "; " 
	    << "nSteps = " << get_nsteps() << "; " 
	    << "dt = "     << get_tstep()  << std::endl;
	    
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " " 
	    <<  std::left << std::setw(8)  << "s"
	    <<  std::left << std::setw(8)  << "dt"
	    <<  std::left << std::setw(12) << "kenergy"
	    <<  std::left << std::setw(12) << "time (s)"
	    <<  std::left << std::setw(12) << "GFlops"
	    <<  std::endl;
  std::cout << "------------------------------------------------" << std::endl;


}

GSimulation :: ~GSimulation()
{
  delete [] particles->acc_x;
  delete [] particles->acc_y;
  delete [] particles->acc_z;
  delete [] particles->pos_x;
  delete [] particles->pos_y;
  delete [] particles->pos_z;
  delete [] particles->vel_x;
  delete [] particles->vel_y;
  delete [] particles->vel_z;
  delete [] particles->mass;
  delete particles;
}