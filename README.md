# Caching Optimizations

In this lab, you will applying caching optimizations to the
perceptron-based ML model you studied in the previous lab.

This lab will be completed on your own.

Check gradescope for due date(s).

## Grading

Your grade for this lab will be based on your completion of the data
collection steps described in this document and the completed `worksheet.pdf`.

| Part                       | value |
|----------------------------|-------|
| Optimizations              | 70%   |
| Worksheet                  | 25%   |
| Reflection                 | 5%    |

The optimizations portions of the grade is based on you successfully
implemented a series of optimizations in Canela, our CNN library.

The optimizations are broken into three tiers.  A total of 100 points are possible.

* Tier 1: Applying loop re-ordering and tiling to
   `fc_layer_t::activate()`. (75 points)

* Tier 2: Applying loop re-ordering and tiling to
   `fc_layer_t::calc_grads()`.  (10 points)

* Tier 3: Applying additional optimizations to those two functions and
   any others you wish. (15 points)

For Tiers 1 and 2, your score is determined by whether you correctly
implement the optimization specified.  They are all-or-nothing: You
will receive full credit or zero credit depending on whether your
implementation is correct.

For Tier 3, your score will vary depending on how much speedup your
optimizations provide for training the neural network.

Your code must pass the regression tests or you'll receive no points
on the lab.

Depending on how things go, we may lower (but will not raise) the
target speedup for Tier 3.  This will only help you.

## The Leader Board

There is a leader board set up for this lab.  It records the speedup of your
code vs the starter code for neural net training.  You can use it to guage your
progress.

For this lab, the leader board does not impact your grade.

## Example Code and Lecture Slides

The `example` directory contains the image stabilization example from the lab
lecture slides.  You shouldn't (and won't need to) use any of the code from
that example for this lab.

The lecture slides are in `lecture-slides.pdf`.

## Skills to Learn and Practice

1. Applying loop reordering.

2. Applying loop tiling.

3. Testing code.

4. Quantifying the benefits of optimizations

5. Interpreting performance counters

## Software You Will Need

1. A computer with Docker installed (either the cloud docker container
via ssh, or your own laptop).  See the intro lab for details.

2. The lab for the github classroom assignment for this lab.  Find the
link on the course home page: https://github.com/CSE141pp/Home/.

3. A PDF annotator/editor to fill out `worksheet.pdf`.  You'll
submit this via a *a separate assignment* in Gradescope.  We *will
not* look at the version in your repo.

## Notes, Hints, and Suggestions

* You should not assume that the optimizations you performed
  for Tier 1 and Tier 2 are necesarily part of the best (or even a
  good) solution to Tier 3.

* The prefetcher has been disabled for this lab.  However, you
  prefetch in software if you are clever (and careful).

## Tasks to Perform

### Inspect The Code

There are three source files in the lab, but you'll only be editting one:

1.  `main.cpp` -- The driver code is in `main()` (mostly data
collection and command line parsing).  Also, code for creating,
training, and running ML models. 

2.  `opt_cnn.hpp` -- A soon-to-be-optimized (by you) version of two CNN primitives.  

There is a `code.cpp` but you won't use it in this lab.

You will only be editing `opt_cnn.hpp`.  Changes to the `main.cpp`
will have no effect on the autograder's output.

The basic flow is like this:

* Execution starts in `main.cpp` which loads the input dataset.

* `main.cpp` executes the "canary" to verify the machine is performing properly.

* It measures the performance of your implementation of
  `fc_layer_t::activate()` for Tier 1 grading.

* It measures the performance of your implementation of
  `fc_layer_t::calc_grads()` for Tier 2 grading.

* It measures the performance of neural net training using your
  optimized functions for Tier 3 grading.

You'll find two skeleton classes in `opt_cnn.hpp`.  They inherit from
the corresponding classes in Canela.  Right now, they have no code, so
using them is the same as using original Canela classes.

To optimize them, you should copy the functions from Canela you want
to optimize into these classes.  Any changes you make will effect the
performance correctness of the code in `main.cpp`.

### Test Locally

Like last time, get started by checking out the code and checking it locally with 

```
runlab --devel
```

The code will run for a while.  On our machine, the starter lab runs
for about 140s.  Your local machine may be slower or faster.

You'll get a few files:

1. `regression.out` has the report from the regression suite.

2. `benchmark.csv` is the csv file used to measure performance.
`CMD_LINE_ARGS` has no effect.

3. `code.csv` is similar to `benchmark.csv` but `CMD_LINE_ARGS` has its
normal effect.

4. `code.gprof` and `benchmark.gprof` are not here now, but if you set
`GPROF=yes` they will appear.

You can submit it to the autograder for good measure, if you want.

### Command Line Options

Your executable takes a few useful command line options we haven't discussed:

* `--scale` this sets the input size for the input data set.  The bigger the
  scale, the more inputs we run through the model.  This only affects the
  execution of `train_model`.
  
* `--reps` how many times to run the individual functions for Tier 1 and Tier
  2.

### Read the Source

You need to get acquainted with the code you'll be optimizing.  The
slides from the lab lecture are an important resource here, especially
for memory layout of `tensor_t` and how the functions in `fc_layer_t`
work.

The baseline version of Canela is in your docker repo in
`/course/CSE141pp-SimpleCNN/CNN`.  You should read through the
commented code in these files:

* `tensor_t.hpp`

* `types.hpp`

* `fc_layer_t.hpp`

* `layer_t.hpp`

In each of these files there's some code near the top that has lots
comments.  This is the code you should focus on.  There's a lot more
code down below, but it's all utility functions and debugging
support. It's not important to the operation of the library or your
optimizations.

The point is not deeply understand the code at this point.  Rather,
it's to become acquainted with where the code is.  This will make it
easier to answer questions about the code that you have later.

`tensor_t` is especially important as far as memory optimazitons go
because it's the data structure that holds almost all the memory in
the models.  Understanding how it maps `x`,`y`,`z`, `b` to linear
indexes into the internal array that holds the data is going to be
very important.  This was covered in the lecture in detail.

### Finding Cache sizes of the Skylake Processor

You can find cache information of the skylake processor here : [Skylake Processor Cache Information](https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(client)#Memory_Hierarchy)

### Tier 1: Optimizing One Function

To get you started, we will walk you through the process for optimizing one
function: `fc_layer_t::activate()`.  Please see the slides in the lab repo for
more details.  They contain detailed description of how the code works.

The code for the baseline implementation lives in
`/course/CSE141pp-SimpleCNN/CNN/fc_layer_t.hpp`.  Copy that version of
`activate()` into your `opt_cnn.hpp`.  Make it a method of
`opt_fc_layer`.

Make sure this didn't break anything by doing `runlab --no-validate`.
It should finish with

```
[  PASSED  ] 5 tests.
```

Which means that your implementation matches the result of the
baseline (which is no surprise because you copied the baseline).

These tests are your best friend, since they provide a quick and easy
way of telling whether your code is correct.  `runlab` runs the tests
every time, and if you the last line shows any failures, you should
look at `regressions.out` for a full report.

**Note** Regressions are always built without optimizations (`-O0`) to make
them debuggable.

#### Loop Reordering

The nesting order for the main loop is initially `b`, `i`, `n`.  Modify the
code so make the nesting order `b`, `n`, `i`.  This is very similar to changes
we discussed in class regarding the stabilization example.  Run the regressions
and run the code through the autograder.

#### Loop Tiling

Next, we will tile the `n` loop.  Proceed in two stages:

**Stage 1** Create new loop that wraps the `n` loop indexed with `nn`.  The
variable `nn` should start at 0 and increment by a constant `TILE_SIZE`.  `nn`
will track the beginning index of the current tile.

Rewrite the `n` loop's initialization and termination condition so that:

1.  It starts at the beginning of the tile.
2.  It stops at the end of the tile *or* the end of the tensor (i.e., the original loop bound).

The resuling nesting order should be `b`, `nn`, `n`, `i`.  Run the regressions
to ensure that you didn't break anything.

Add a `#define TILE_SIZE 4` above the loop body.

**Stage 2** Make the `nn` loop the outermost loop in the main loop.  The
resuling nesting order should be `nn`, `b`, `n`, `i`.  Run the regressions to
ensure that you didn't break anything.

Submit to the autograder.  If you've done everything correctly, your code
should pass Tier 1.  The precise target for the speedup is listed in gradescope
output.

### Tier 2: Optimizing calc_grads

For Tier 2, you need apply the same two optimizations to `calc_grads`.

First, reorder the loops so that the nesting order in the triply-nested loop is
`b`, `n`, `i`.

And apply tiling on `n` and so you have `nn`, `b`, `n`, `i`.

If you do that successfully, your code should pass Tier 2.  The precise target
for the speedup is listed in gradescope output.

### Tier 3:  Other optimizations

Go forth and optimize!

There are more opportunities to apply loop reordering tiling across
`activate()` and `calc_grads()`.  There may also be other functions
worth looking at (How could you find them?).  You can apply whatever
optimizations you want with the following restrictions:

1. You can't modify `main.cpp`

2. No threads (that's a later lab).

3. To explicit vectorization (that's a later lab).

The target speedup for Tier 3 is 3x on the full training function with
includes, `activate()` and `calc_grads`.

## Testing

The lab includes a regression test in `run_tests.cpp`.  The tests are
written using the `googletest` testing framework (which is pretty
cool, and if you ever need to test C/C++ code, you should conisder
using it.)

A test case looks like this:

```
TEST_F(OptimizationTests, level_0_fc) {
        fc_test<opt_fc_layer_t>(1,1,1,1,1,1);
}
```

The name of this test is `level_0_fc`.  The rest of the strange
function signature is `googletest` boilerplate.

There are several sets of tests (`level_0...`, `level_1...` etc. for
each kind of layer that Canela supports.  The idea is that the
`level_1` tests are harder than the `level_0` tests and so on.

Each set of tests calls a `<layer>_test` with several parameters.  The
parameters are input size, output size, and parameters for that type
of layer.  For details, look at the `.hpp` file for the layer type in
`/course/CSE141pp-SimpleCNN/CNN/`

The final level of tests runs some randomized tests.  `RAND_LARGE(x)`
returns a random number between x and 2*x.

You can and should run the test suite locally.  You don't need cloud machines to test
correctness.  `runlab` runs it automatically.

You can also run it by hand with `run_tests.exe`.  You can also debug it using `gdb`.

You can also add tests, but be aware that `run_test.cpp` is not copied to the
cloud, so your new tests won't run there.

## Debugging 

Canela has some debugging support built in.  If you a regression
fails, you'll get a very lengthy report about it.  For instance:

```
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from OptimizationTests
[ RUN      ] OptimizationTests.level_1_fc
/home/root/CSE141pp-SimpleCNN/CNN/fc_layer.hpp:174: Failure
Here's what's different. '#' denotes a position where your result is incorrect.
Diff of ->in: <identical>
Diff of ->out: <identical>
Diff of ->grads_in: <identical>
Diff of ->input: <identical>
Diff of ->weights:
z = 0:
################################################################
################################################################
################################################################
################################################################

Diff of ->gradients: <identical>


```

The line with `[ RUN    ]` gives the test that failed.

What follows is a map showing what parts of the layer's data
structures are incorrect.  In this case `in`, `out`, 'grads_in`,
`input`, and `gradients` are correct.  However, `weights` (which is
65x4x1 tensor) is wrong in every position, if a position was correct,
there would be a `.` instead of a `#'.

THe `z = 0` means that this is 'slice' through the tensor at `z=0`.
If the tensor more was more 3-dimensional, there would be several
blocks of `.` and `#`.

You to see how wrong the values are, you can pass `--print-deltas` as
the _first_ argument to `run_tests.exe`.  You'll get something like this:

```
/home/root/CSE141pp-SimpleCNN/CNN/fc_layer.hpp:174: Failure
Here's what's different. '#' denotes a position where your result is incorrect.
Diff of ->in: <identical>
Diff of ->out: <identical>
Diff of ->grads_in: <identical>
Diff of ->input: <identical>
Diff of ->weights: 
z = 0: 
-0.0014 -0.0011 -0.0015 -0.00095 -0.0018 -0.0015 -0.00073 -0.0016 -0.001 -0.0019 -0.0016 -0.00094 -0.0013 -0.00036 -0.00057 -0.0014 -0.00062 -0.0011 -0.0013 -0.0011 -0.0013 -0.00055 -0.0015 -0.00094 -0.00017 -6.9e-06 -0.00039 -0.00055 -0.00018 -0.00083 -0.0017 -0.00026 -0.0014 -0.0013 -0.00023 -0.00031 -0.0012 -0.00063 -0.0018 -0.00088 -0.0005 -0.0016 -0.0015 -0.00034 -0.00021 -0.0018 -0.0017 -0.0011 -0.00038 -0.00055 -0.0013 -0.00052 -0.00049 -0.00036 -0.0017 -0.0016 -0.0017 -0.00063 -0.001 -0.0013 -0.00069 -0.0017 -0.0017 -0.00094 
-0.0012 -0.00094 -0.0012 -0.00081 -0.0015 -0.0013 -0.00063 -0.0014 -0.00087 -0.0016 -0.0014 -0.00081 -0.0011 -0.00031 -0.00048 -0.0012 -0.00054 -0.00097 -0.0011 -0.00094 -0.0011 -0.00047 -0.0013 -0.0008 -0.00014 -6.1e-06 -0.00033 -0.00047 -0.00015 -0.00071 -0.0015 -0.00023 -0.0012 -0.0011 -0.0002 -0.00027 -0.0011 -0.00054 -0.0015 -0.00076 -0.00042 -0.0013 -0.0013 -0.00029 -0.00018 -0.0016 -0.0015 -0.00098 -0.00033 -0.00047 -0.0011 -0.00045 -0.00042 -0.00031 -0.0015 -0.0014 -0.0014 -0.00054 -0.00089 -0.0011 -0.00059 -0.0015 -0.0014 -0.0008 
-0.00029 -0.00023 -0.00031 -0.0002 -0.00037 -0.00031 -0.00015 -0.00035 -0.00021 -0.0004 -0.00035 -0.0002 -0.00028 -7.6e-05 -0.00012 -0.00029 -0.00013 -0.00024 -0.00028 -0.00023 -0.00028 -0.00012 -0.00031 -0.0002 -3.5e-05 -1.5e-06 -8.2e-05 -0.00012 -3.8e-05 -0.00018 -0.00036 -5.6e-05 -0.0003 -0.00027 -4.9e-05 -6.6e-05 -0.00026 -0.00013 -0.00038 -0.00019 -0.00011 -0.00033 -0.00032 -7.3e-05 -4.5e-05 -0.00039 -0.00037 -0.00024 -8.1e-05 -0.00012 -0.00028 -0.00011 -0.0001 -7.7e-05 -0.00037 -0.00034 -0.00035 -0.00013 -0.00022 -0.00028 -0.00015 -0.00037 -0.00035 -0.0002 
-0.00036 -0.00029 -0.00038 -0.00025 -0.00046 -0.00038 -0.00019 -0.00043 -0.00026 -0.00049 -0.00043 -0.00025 -0.00034 -9.3e-05 -0.00015 -0.00036 -0.00016 -0.00029 -0.00035 -0.00029 -0.00034 -0.00014 -0.00038 -0.00024 -4.4e-05 -1.9e-06 -0.0001 -0.00014 -4.7e-05 -0.00022 -0.00045 -6.9e-05 -0.00037 -0.00033 -6e-05 -8.1e-05 -0.00032 -0.00016 -0.00047 -0.00023 -0.00013 -0.00041 -0.00039 -8.9e-05 -5.5e-05 -0.00048 -0.00045 -0.0003 -0.0001 -0.00014 -0.00034 -0.00014 -0.00013 -9.4e-05 -0.00045 -0.00042 -0.00043 -0.00016 -0.00027 -0.00034 -0.00018 -0.00046 -0.00043 -0.00024 

Diff of ->gradients: <identical>

Failure: fc_test(4, 4, 4, 4, 1);

```

You should focus on one failing test at a time.  You can see the tests that are availble with:

```
$ ./run_tests.exe --gtest_list_tests
OptimizationTests.
  level_0_fc
  level_1_fc
  level_2_fc
  level_3_fc
  level_4_fc
$
```

And then you can run just one of them like so:

```
$ ./run_tests.exe --gtest_filter=*level_0_fc*
```

Note the `_` instead of `-` and the `*`.
