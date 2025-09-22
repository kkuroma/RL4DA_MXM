I would like you to rewrite the code's repository, leaving only the ENKF implementation. The project can be described as follows: an attempt to use PPO to train agents to predict ENKF in example systems

I moved the old codebase to OLD_CODEBASE/. Either write new files (please do so for code) or cp out the files (for like config or data). The old codebase ran but the model wouldn't fit for some reason. I would like you to write one from scratch as well as a very detailed readme.md

The ENKF code is already verified to be correct as the posterior sees significant error reduction from the forecast. Two models were implemented: L63 and L96

The logic is as follows. Normally, two systems run in an ENFK - the truth system which we dont know, and N_ens ensemble systems. For each t_obs, we do t_obs/dt steps in both system and project them into the observation space, yielding Y (the obs) and x_b (the background ensemble aka forecast). The kalman filter then transforms x_b into x_a, the kalman filtered posterior, with some other values like the covariance and whatnot.

I generated a 1m long path from L96 with some random initial cond around 0, the idea being that the path stabilizes and converges, so i can later pick any starting point from the stabilized path and use for training. I then picked 100 points from the later half of the path to be used as a starting cond for the ENKF to be used as RL training data. The old code samples from the last 900k while i wanted to be sample from the last half (500k) here instead.

The idea is to run the same system but instead, an RL model is used to produce xa from xb. I currently use a lot of values (RL observation) to produce xa from xb, so think if these are overkill or justified. When xa_rl is produced, values in the next step that are normally produces xa instead uses xa_rl to enforce continuity and penalize wrong direction in the pred

These values are also normalized into -1,1 and allowed to reach -2,2 in the RL environment. Is this too small or too big? The action space is similarly normalized. The reward should be computed from the normalized space (so if like the space is in range -2,2 then rmse is [0,4])

The model is a time-based LSTM which takes a window of observation and produces the next xa_rl. The reward is compared between the real xa (precomputed) and the predicted, linearlizing the RMSE into ranges like [-1,1].

Also think if a deeper or wider LSTM network would help? 

The observation and action spaces as such should be -3,3 (the actual values are normalized to -1,1 but we leave some space for extreme values) and the reward should be smth like RMSE ([0,2]) then -1 to get into [-1,1] range.

Bottomline, only UNNORMALIZE xa_rl to produce xb for the next step, then normalize it for the RL (anything pass into the RL should be normalize)

One last strategy is curriculum learning. I halt the episode whenever xb goes too far out of range or produces nan, which is a result of xa_rl this is too far from the original path. Once a set of path completes (doesnt stop early) for 100/N episodes (N=number of paths used rn), we add one more path to the list of paths.

Path 1-99 is used for training and path 0 for eval.

I would like to keep the same idea of loading from a config file. There should still be a file for the RL env, a file to define the LSTM arch, and a train script as well as makefile. No individual file should have the power to override what is being passed to global config

Please LOG all the losses, rewards, ep length, number of included paths, to the tensorboard.

Maintain the old strategy and parameters from L96 1-3 from the old codebase. Everything should be saved, visualized, and plotted exactly as the old strategy to logs/

Directory structure: 
logs/ # data and training results as well as tensorboard
src/ # holds code for RL and ENFF
scripts/ # holds sh scripts
makefile

Start with only the 3 L96 parameters I have been training before.

The makefile should also be similar to the old one. Only rewrite the RL env agent codes since i think those are where the problems are

Trace files from train_concurrent.sh to know what is going on.

Use this python /home/kuroma/PyEnv/enkf_rl/bin/python