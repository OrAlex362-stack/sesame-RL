=================================================================
00 / 003-00 SESAME RL ENVIRONMENT BASELINE
==================================================================
Physics dt        : 0.0020 s
Control dt        : 0.0200 s
Substeps          : 10
Servo tau         : 0.045 s
Servo max speed   : 600.0 deg/s
Action scale      : ±0.580 rad
Observation dim   : 33
Action dim        : 8
Episode duration  : 20.0 s
Episode steps     : 1000

=== 003-00 ENVIRONMENT CHECK ===
Observation shape : (33,)
Expected shape    : (33,)
Action shape      : (8,)
Initial height    : 0.04668 m
Initial upright   : 1.00000
Observation valid : PASS
Zero-action step  : PASS
Zero reward       : +0.010000
Zero forward vel  : -0.000000 m/s
Zero lateral vel  : +0.000000 m/s

--- VELOCITY SANITY CHECK ---
Body linear velocity : [-0.000000, +0.000000, +0.000000] m/s
Body angular velocity: [-0.000, -0.000, +0.000] deg/s

--- SMALL ACTION SANITY CHECK ---
Forward velocity : -0.000246 m/s
Lateral velocity : -0.002450 m/s
Vertical velocity: +0.000218 m/s
Roll rate        : +0.900 deg/s
Pitch rate       : -0.196 deg/s
Yaw rate         : +23.190 deg/s
SB3 check_env     : PASS

=== 003-00 RANDOM POLICY VALIDATION ===
Episodes          : 3
Episode steps     : 1000
Episode duration  : 20.0 s

Episode 00
  steps         : 1000
  reward        : -71.513
  mean forward  : +0.0110 m/s
  mean |forward|: 0.0569 m/s
  mean lateral  : +0.0005 m/s
  mean |lat|    : 0.0636 m/s
  mean |vertical|: 0.0375 m/s
  mean |roll|   : 48.66 deg/s
  mean |pitch|  : 38.31 deg/s
  mean |yaw|    : 81.23 deg/s
  min upright   : 0.9988
  min height    : 0.04635 m
  terminated    : False
  truncated     : True

Episode 01
  steps         : 1000
  reward        : -75.101
  mean forward  : +0.0075 m/s
  mean |forward|: 0.0565 m/s
  mean lateral  : +0.0049 m/s
  mean |lat|    : 0.0617 m/s
  mean |vertical|: 0.0384 m/s
  mean |roll|   : 47.32 deg/s
  mean |pitch|  : 36.14 deg/s
  mean |yaw|    : 78.48 deg/s
  min upright   : 0.9986
  min height    : 0.04627 m
  terminated    : False
  truncated     : True

Episode 02
  steps         : 1000
  reward        : -93.431
  mean forward  : -0.0003 m/s
  mean |forward|: 0.0559 m/s
  mean lateral  : +0.0049 m/s
  mean |lat|    : 0.0579 m/s
  mean |vertical|: 0.0406 m/s
  mean |roll|   : 50.39 deg/s
  mean |pitch|  : 40.08 deg/s
  mean |yaw|    : 83.59 deg/s
  min upright   : 0.9985
  min height    : 0.04626 m
  terminated    : False
  truncated     : True

======================================================================
003-00 RANDOM POLICY SUMMARY
======================================================================
Episodes                    : 3
Mean total reward           : -80.015
Mean forward velocity       : +0.00606 m/s
Mean |forward velocity|     : 0.05647 m/s
Mean |lateral velocity|     : 0.06106 m/s
Mean |vertical velocity|    : 0.03881 m/s
Mean |roll rate|            : 48.789 deg/s
Mean |pitch rate|           : 38.174 deg/s
Mean |yaw rate|             : 81.099 deg/s
Falls / terminations        : 0/3
Step CSV                    : /home/kit/sesame-RL/003_rl_results/00_random_policy_steps.csv
Episode CSV                 : /home/kit/sesame-RL/003_rl_results/00_random_policy_episodes.csv
======================================================================