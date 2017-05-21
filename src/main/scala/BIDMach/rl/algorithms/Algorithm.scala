package BIDMach.rl.algorithms;

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.networks.layers._;
import BIDMach.networks._
import BIDMach.updaters._
import BIDMach._
import BIDMach.rl.environments._
import BIDMach.rl.estimators._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import java.util.HashMap;


@SerialVersionUID(100L)
abstract class Algorithm extends Serializable {
  def startup;
  
  def train;
}

object Algorithm {
  class Options extends Net.Options with ADAGrad.Opts { 	
  	clipByValue = 1f;                                // gradient clipping
  	gsq_decay = 0.99f;                               // Decay factor for MSProp
  	vel_decay = 0.0f;                                // Momentum decay
  	texp = 0f;
  	vexp = 1f;
  	waitsteps = -1;
  }
}