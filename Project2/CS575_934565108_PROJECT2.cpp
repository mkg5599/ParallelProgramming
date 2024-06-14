#include <iostream>
#include <cmath>
#include <omp.h>

// Constants for the simulation
const float GRAIN_GROWS_PER_MONTH = 12.0;
const float ONE_DEER_EATS_PER_MONTH = 1.0;
const float ONE_PEST_EATS_PER_MONTH = 0.05; 
const float AVG_PRECIP_PER_MONTH = 7.0;
const float AMP_PRECIP_PER_MONTH = 6.0;
const float RANDOM_PRECIP = 2.0;
const float AVG_TEMP = 60.0;
const float AMP_TEMP = 20.0;
const float RANDOM_TEMP = 10.0;
const float MIDTEMP = 40.0;
const float MIDPRECIP = 10.0;

// Global state variables
int NowYear = 2024;
int NowMonth = 0; // 0 to 11
float NowPrecip; // inches of rain per month
float NowTemp; // temperature this month in degrees Fahrenheit
float NowHeight = 5.0; // initial grain height in inches
int NowNumDeer = 2; // initial number of deer in the population
int NowNumPests = 5; // initial number of pests in the population

// Synchronization variables
omp_lock_t	Lock;
volatile int	NumInThreadTeam;
volatile int	NumAtBarrier;
volatile int	NumGone;

// Function to initialize barriers
void InitBarrier(int n) {
    NumInThreadTeam = n;
    NumAtBarrier = 0;
    omp_init_lock(&Lock);
}

void setInitialTempPrecip(){
 float ang = (30.0 * (float)NowMonth + 15.0) * (M_PI / 180.0);
    NowTemp = AVG_TEMP - AMP_TEMP * cos(ang) + ((2.0 * rand() / (float)RAND_MAX) - 1.0) * RANDOM_TEMP;
    NowPrecip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang) + ((2.0 * rand() / (float)RAND_MAX) - 1.0) * RANDOM_PRECIP;
    if (NowPrecip < 0.0) NowPrecip = 0.0;
}

// Function to wait at the barrier
void WaitBarrier() {
    omp_set_lock(&Lock);
    NumAtBarrier++;
    if (NumAtBarrier == NumInThreadTeam) {
        NumGone = 0;
        NumAtBarrier = 0;
        while (NumGone != NumInThreadTeam - 1);
        omp_unset_lock(&Lock);
        return;
    }
    omp_unset_lock(&Lock);
    while (NumAtBarrier != 0); // Wait for all threads
    #pragma omp atomic
    NumGone++;
}

float SQR(float x) {
    return x * x;
}

void Deer() {
    while (NowYear < 2030) {
        int nextNumDeer = NowNumDeer;
        int carryingCapacity = (int)(NowHeight);

        if (nextNumDeer < carryingCapacity)
            nextNumDeer++;
        else if (nextNumDeer > carryingCapacity)
            nextNumDeer--;

        if (nextNumDeer < 0)
            nextNumDeer = 0;

        WaitBarrier();

        NowNumDeer = nextNumDeer;

        WaitBarrier();

        WaitBarrier();
    }
}

void Grain() {
    while (NowYear < 2030) {
        float tempFactor = exp(-SQR((NowTemp - MIDTEMP) / 10));
        float precipFactor = exp(-SQR((NowPrecip - MIDPRECIP) / 10));

        float nextHeight = NowHeight;
        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
        nextHeight -= (float)NowNumPests * ONE_PEST_EATS_PER_MONTH;

        if (nextHeight < 0.0)
            nextHeight = 0.0;

        WaitBarrier();

        NowHeight = nextHeight;

        WaitBarrier();

        WaitBarrier();
    }
}

void Watcher() {
    while (NowYear < 2030) {
        WaitBarrier();

        WaitBarrier();

        std::cout << NowMonth + 1 + (NowYear - 2024) * 12 << "," 
          << NowYear << ","
          << ((5.0 / 9.0) * (NowTemp - 32)) << "," // Convert °F to °C
          << (NowPrecip * 2.54) << "," // Convert inches to cm
          << (NowHeight * 2.54) << "," // Convert inches to cm
          << NowNumDeer << "," //Deers
          << NowNumPests << "\n"; //Pests
          

        NowMonth++;
        if (NowMonth > 11) {
            NowMonth = 0;
            NowYear++;
        }

        setInitialTempPrecip();

        WaitBarrier();
    }
}

void MyAgent() { // Custom agent logic for pests

    while (NowYear < 2030) {
        int nextNumPests = NowNumPests;

        int carryingCapacity = (int)(NowHeight);

        if (nextNumPests < carryingCapacity)
            nextNumPests  = nextNumPests + (rand() % 4 + 1);    
        else if (nextNumPests > carryingCapacity)
            nextNumPests = nextNumPests - 2; //

        if (nextNumPests < 0)
            nextNumPests = 0;
       
        WaitBarrier();

        NowNumPests = nextNumPests;

        WaitBarrier();

        WaitBarrier();
    }
}




int main() {
    setInitialTempPrecip();
    omp_set_num_threads(4);
    InitBarrier(4);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            Deer();
        }
        #pragma omp section
        {
            Grain();
        }
        #pragma omp section
        {
            Watcher();
        }
        #pragma omp section
        {
            MyAgent();
        }
    }

    return 0;
}


