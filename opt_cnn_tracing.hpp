#pragma once
#include"CNN/canela.hpp"

#if(1)
// These three macros are useful for tracing accesses.  See
// `example/stabilize.cpp` for an example of how to use them.
#define DUMP_ACCESS(t,x,y,z,b) do {					\
		trace << t.linearize(x,y,z,b) << " "			\
		      << " "						\
		      << &t.get(x,y,z,b) << " ";			\
	} while(0)

#define END_TRACE_LINE() do {trace << "\n";}while(0)
#define OPEN_TRACE(filename)  std::fstream trace; trace.open (filename, std::fstream::out);


// This one is customized for the stabilization code.  It prints out
// the linear index and address of each tensor element that's
// accessed.
#define DUMP_ACCESSES() do {\
		DUMP_ACCESS(in, i,b,0,0); \
		DUMP_ACCESS(weights, i,n,0,0);\
		DUMP_ACCESS(activator_input, n, 0,0, b); \
		END_TRACE_LINE();			 \
	} while(0)
#else
// By default, you get these versions, which do nothing.
#define DUMP_ACCESS(t,x,y,z,b)
#define END_TRACE_LINE
#define OPEN_TRACE(filename)
#define DUMP_ACCESSES()
#endif


// This class replaces its parent classes in the implementation of the learning
// model for this lab.  If you override functions in the baseclass by
// implementing them here, then the code here will run instead of the baseclass
// code.
//
// You should copy the functions you want to optimize into these classes, and
// confirm that the correctness tests pass.  Then, you can start modifying them
// to make them faster.
//
// The source code Canela is in /course/CSE141pp-SimpleCNN/CNN
class opt_fc_layer_t : public fc_layer_t
{
public:
	opt_fc_layer_t( tdsize in_size, int out_size ) : fc_layer_t(in_size, out_size) {

	}

	void activate( tensor_t<double>& in ) {
        	OPEN_TRACE("activate.trace");
	        copy_input(in);

		tdsize old_size = in.size;
		tdsize old_out_size = out.size;

		// cast to correct shape
		in.size.x = old_size.x * old_size.y * old_size.z;
		in.size.y = old_size.b;
		in.size.z = 1;
		in.size.b = 1;

		out.size.x = old_out_size.x * old_out_size.y * old_out_size.z;
		out.size.y = old_out_size.b;
		out.size.z = 1;
		out.size.b = 1;

		for ( int b = 0; b < activator_input.size.b; b++) {
			for ( int n = 0; n < activator_input.size.x; n++ ) {
				activator_input(n, 0, 0, b) = 0;
			}
		}

		for ( int b = 0; b < in.size.y; b++ ) {
			for ( int i = 0; i < in.size.x; i++ ) {
				for ( int n = 0; n < out.size.x; n++ ) {
					double in_val = in(i, b, 0);
					double weight_val = weights( i, n, 0 );
					double mul_val = in_val * weight_val;
					double acc_val = activator_input(n, 0, 0, b) + mul_val;
					activator_input(n, 0, 0, b) = acc_val;
					DUMP_ACCESSES();
				}
			}
		}

		// finally, apply the activator function.
		for ( unsigned int n = 0; n < activator_input.element_count(); n++ ) {
			out.data[n] = activator_function( activator_input.data[n] );
		}

		// don't forget to reset the shapes
		in.size = old_size;
		out.size = old_out_size;
	}
			
};

