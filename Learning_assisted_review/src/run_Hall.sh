#! /bin/tcsh

foreach VAR (1 5)
  bsub -q standard -W 5000 -n 5 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J mpiexec -n 5 /share2/zyu9/miniconda/bin/python2.7 runner_hpc.py repeat_Hall $VAR
end

