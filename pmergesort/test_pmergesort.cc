/* qsort example */
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* qsort */
#include "pmergesort.h"
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <thread>

using namespace std;


//int values[] = { 40, 10, 100, 90, 20, 25 };

int compare (const void * a, const void * b)
{
  return ( *(size_t*)a - *(size_t*)b );
}

struct timer {
	timeval i_start_, i_stop_;
	void start() {
		gettimeofday(&i_start_, NULL);
	}
	
	void stop() {
		gettimeofday(&i_stop_, NULL);
	}
	
	double interval() {
		double t1 = i_start_.tv_sec + i_start_.tv_usec/1e6;
		double t2 = i_stop_.tv_sec + i_stop_.tv_usec/1e6;
		return (t2-t1);
	}
};

int main ()
{
  //int n;
  //qsort (values, 6, sizeof(int), compare);
  unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
  printf("Number of cores: %lu\n", concurentThreadsSupported);

  srand (1541241);
  timer tm;
  tm.start();
  vector<size_t> vv(1024*1024*100*4);
  generate(vv.begin(), vv.end(), rand);
  tm.stop();
  printf("Time to generate: %.2f\n", tm.interval());
  
  vector<size_t> qsort_vec = vv;
  vector<size_t> symsort_vec = vv;

  tm.start();
  qsort (qsort_vec.data(), qsort_vec.size(), sizeof(size_t), compare);
  tm.stop();
  printf("qsort time: %.2f\n", tm.interval());

  tm.start();
  pmergesort (symsort_vec.data(), symsort_vec.size(), sizeof(size_t), compare);
  tm.stop();
  printf("symmergesort time: %.2f\n", tm.interval());

  for(size_t i = 0; i < vv.size(); ++i) {
    if(qsort_vec[i] != symsort_vec[i]) {
        printf("Worng value.\n");
    }
  }
  
  //symmergesort(values, 6, sizeof(int), compare);
  printf("\n");
  return 0;
}
