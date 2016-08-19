#! /bin/tcsh

foreach VAR (0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.5 2.0)
  bsub -q standard -W 2400 -n 4 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J mpiexec -n 5 /share2/zyu9/miniconda/bin/python2.7 runner_hpc.py repeat_exp $VAR
end

