package BIDMach.rl.algorithms;

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.networks.layers._;
import BIDMach.networks._
import BIDMach.updaters._
import BIDMach._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import scala.util.hashing.MurmurHash3;
import java.util.HashMap;

class A3C (val opts:A3C.Options = new A3C.Options) {
  
  
  
}

object A3C {
  class Options extends Net.Options with ADAGrad.Opts {
  	var nsteps = 400000;                             // Number of steps to run (game actions per environment)
  	var npar = 16;                                   // Number of parallel environments
  	var ndqn = 5;                                    // Number of DQN steps per update
  	var target_window = 50;                          // Interval to update target estimator from q-estimator
  	var printsteps = 10000;                          // Number of steps between printouts
  	var init_moves = 4000;                           // Upper bound on random number of moves to take initially
  	var nwindow = 4;                                 // Sensing window = last n images in a state

  	var discount_factor = 0.99f;                     // Reward discount factor
  	var policygrad_weight = 0.3f;                    // Weight of policy gradient compared to regression loss
  	var entropy_weight = 1e-4f;                      // Entropy regularization weight
  	var lr_schedule = (0f \ 3e-6f on 1f \ 3e-6f);    // Learning rate schedule
  	var temp_schedule = (0f \ 1f on 1f \ 1f);        // Temperature schedule
  	var baseline_decay = 0.9999f;                    // Reward baseline decay
  	var gclip = 1f;                                  // gradient clipping
  	override var gsq_decay = 0.99f;                  // Decay factor for MSProp
  	override var vel_decay = 0.9f                    // Momentum decay

  	var nhidden = 16;                                // Number of hidden layers for estimators
  	var nhidden2 = 32;
  	var nhidden3 = 256;

  }
}