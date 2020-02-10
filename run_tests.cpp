//#define INCLUDE_TESTS
#define DEBUG_OUTPUT "output/"

#include <iostream>
#include "gtest/gtest.h"
#include <opt_cnn.hpp>
#include <sstream>


namespace Tests {

      	
	class OptimizationTests :  public ::testing::Test {
		
	};

	TEST_F(OptimizationTests, level_0_fc) {
		fc_test_activate<opt_fc_layer_t>(1,1,1,1,1,1);
		//fc_test_calc_grads<opt_fc_layer_t>(1,1,1,1,1,1);
		fc_test_fix_weights<opt_fc_layer_t>(1,1,1,1,1,1);
		fc_test<opt_fc_layer_t>(1,1,1,1,1,1);
	}	
			  
	TEST_F(OptimizationTests, level_1_fc) {
		fc_test_activate<opt_fc_layer_t>(4,  4,  4,  4, 4, 1);
		fc_test_activate<opt_fc_layer_t>(4,  4,  2,  2, 8, 1);
		fc_test_activate<opt_fc_layer_t>(8,  8,  2,  2, 16,1);
		fc_test_activate<opt_fc_layer_t>(32, 32, 8,  8, 4, 1);
		fc_test_activate<opt_fc_layer_t>(64, 64, 16, 8, 8, 1);

#if (0)
		fc_test_calc_grads<opt_fc_layer_t>(4,  4,  4,  4, 4, 1);
		fc_test_calc_grads<opt_fc_layer_t>(4,  4,  2,  2, 8, 1);
		fc_test_calc_grads<opt_fc_layer_t>(8,  8,  2,  2, 16,1);
		fc_test_calc_grads<opt_fc_layer_t>(32, 32, 8,  8, 4, 1);
		fc_test_calc_grads<opt_fc_layer_t>(64, 64, 16, 8, 8, 1);

		fc_test_fix_weights<opt_fc_layer_t>(4,  4,  4,  4, 4, 1);
		fc_test_fix_weights<opt_fc_layer_t>(4,  4,  2,  2, 8, 1);
		fc_test_fix_weights<opt_fc_layer_t>(8,  8,  2,  2, 16,1);
		fc_test_fix_weights<opt_fc_layer_t>(32, 32, 8,  8, 4, 1);
		fc_test_fix_weights<opt_fc_layer_t>(64, 64, 16, 8, 8, 1);

		fc_test<opt_fc_layer_t>(4,  4,  4,  4, 4, 1);
		fc_test<opt_fc_layer_t>(4,  4,  2,  2, 8, 1);
		fc_test<opt_fc_layer_t>(8,  8,  2,  2, 16,1);
		fc_test<opt_fc_layer_t>(32, 32, 8,  8, 4, 1);
		fc_test<opt_fc_layer_t>(64, 64, 16, 8, 8, 1);
#endif
		
	}
	TEST_F(OptimizationTests, level_2_fc) {
		fc_test_activate<opt_fc_layer_t>(4,  6,  6,  6,  6,  1);
		fc_test_activate<opt_fc_layer_t>(4,  8,  2,  2,  2,  1);
		fc_test_activate<opt_fc_layer_t>(12, 12, 3,  2,  3,  1);
		fc_test_activate<opt_fc_layer_t>(24, 48, 24, 12, 12, 1);
		fc_test_activate<opt_fc_layer_t>(16, 96, 2,  2,  12, 1);

#if (0)
		fc_test_calc_grads<opt_fc_layer_t>(4,  6,  6,  6,  6,  1);
		fc_test_calc_grads<opt_fc_layer_t>(4,  8,  2,  2,  2,  1);
		fc_test_calc_grads<opt_fc_layer_t>(12, 12, 3,  2,  3,  1);
		fc_test_calc_grads<opt_fc_layer_t>(24, 48, 24, 12, 12, 1);
		fc_test_calc_grads<opt_fc_layer_t>(16, 96, 2,  2,  12, 1);

		fc_test_fix_weights<opt_fc_layer_t>(4,  6,  6,  6,  6,  1);
		fc_test_fix_weights<opt_fc_layer_t>(4,  8,  2,  2,  2,  1);
		fc_test_fix_weights<opt_fc_layer_t>(12, 12, 3,  2,  3,  1);
		fc_test_fix_weights<opt_fc_layer_t>(24, 48, 24, 12, 12, 1);
		fc_test_fix_weights<opt_fc_layer_t>(16, 96, 2,  2,  12, 1);

		fc_test<opt_fc_layer_t>(4,  6,  6,  6,  6,  1);
		fc_test<opt_fc_layer_t>(4,  8,  2,  2,  2,  1);
		fc_test<opt_fc_layer_t>(12, 12, 3,  2,  3,  1);
		fc_test<opt_fc_layer_t>(24, 48, 24, 12, 12, 1);
		fc_test<opt_fc_layer_t>(16, 96, 2,  2,  12, 1);
#endif		
	}

	TEST_F(OptimizationTests, level_3_fc) {
		fc_test_activate<opt_fc_layer_t>(3,  7,  13, 3, 7,  1);
		fc_test_activate<opt_fc_layer_t>(5,  9,  17, 5, 11, 1);
		fc_test_activate<opt_fc_layer_t>(31, 29, 5,  5, 13, 1);
		fc_test_activate<opt_fc_layer_t>(89, 31, 7,  7, 19, 1);
		fc_test_activate<opt_fc_layer_t>(3,  17, 31, 3, 23, 1);
		
#if (0)
		fc_test_calc_grads<opt_fc_layer_t>(3,  7,  13, 3, 7,  1);
		fc_test_calc_grads<opt_fc_layer_t>(5,  9,  17, 5, 11, 1);
		fc_test_calc_grads<opt_fc_layer_t>(31, 29, 5,  5, 13, 1);
		fc_test_calc_grads<opt_fc_layer_t>(89, 31, 7,  7, 19, 1);
		fc_test_calc_grads<opt_fc_layer_t>(3,  17, 31, 3, 23, 1);

		fc_test_fix_weights<opt_fc_layer_t>(3,  7,  13, 3, 7,  1);
		fc_test_fix_weights<opt_fc_layer_t>(5,  9,  17, 5, 11, 1);
		fc_test_fix_weights<opt_fc_layer_t>(31, 29, 5,  5, 13, 1);
		fc_test_fix_weights<opt_fc_layer_t>(89, 31, 7,  7, 19, 1);
		fc_test_fix_weights<opt_fc_layer_t>(3,  17, 31, 3, 23, 1);

		fc_test<opt_fc_layer_t>(3,  7,  13, 3, 7,  1);
		fc_test<opt_fc_layer_t>(5,  9,  17, 5, 11, 1);
		fc_test<opt_fc_layer_t>(31, 29, 5,  5, 13, 1);
		fc_test<opt_fc_layer_t>(89, 31, 7,  7, 19, 1);
		fc_test<opt_fc_layer_t>(3,  17, 31, 3, 23, 1);
#endif
		
	}

	TEST_F(OptimizationTests, level_4_fc) {
		for (int i = 0; i < 20; i++) {
			srand(i);
			int x = RAND_LARGE(16);
			int y = RAND_LARGE(24);
			int z = RAND_LARGE(24);
			int b = RAND_LARGE(16);
			int out = RAND_LARGE(8);
			
			fc_test_activate<opt_fc_layer_t>(x,y,z,b,out,1);
#if (0)
			fc_test_calc_grads<opt_fc_layer_t>(x,y,z,b,out,1);
			fc_test_fix_weights<opt_fc_layer_t>(x,y,z,b,out,1);
			fc_test<opt_fc_layer_t>(x,y,z,b,out,1);
#endif
		}
		
	}

}

int main(int argc, char **argv) {
	if (argc >= 2) {
		if (!strcmp(argv[1], "--print-deltas")) {
			tensor_t<double>::diff_prints_deltas = true;
			argc--;
			argv++;
		}
	}
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
