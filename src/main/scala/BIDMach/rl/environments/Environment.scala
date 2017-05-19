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

abstract class Environment(val opts:Environment.Options = new Environment.Options) {
  
  def valid_actions:IMat;
  
  def step:(FMat, Int, Float);
  
  def stepAll:(FMat, IMat, FMat);
  
  def reset;
  
}

object Environment {
  
  class Options {
    
  }
}