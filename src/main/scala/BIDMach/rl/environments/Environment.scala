package BIDMach.rl.environments;

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

@SerialVersionUID(100L)
abstract class Environment(val opts:Environment.Options = new Environment.Options) extends Serializable {
  
  val VALID_ACTIONS:IMat;
  
  val score_range:FMat;
  
  def step(action:Int):(FMat, Float, Boolean);
  
  def statedims:IMat;
  
  def reset();
  
}

object Environment {
  
  class Options extends BIDMat.Opts {
    
  }
}