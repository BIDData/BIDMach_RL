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

class AtariEnvironment(val opts:AtariEnvironment.Options = new AtariEnvironment.Options) extends Environment(opts) {
  
  val ale:ALE = new ALE;
  ale.setInt("random_seed", opts.random_seed);
  ale.loadROM(opts.rom_name);
  ale.setFloat("repeat_action_probability",opts.repeat_action_probability);
  ale.frameskip = (opts.frameskip(0), opts.frameskip(1));
  
  val VALID_ACTIONS = ale.getMinimalActionSet;
  
  def step:(FMat, Int, Float);
  
  def stepAll:(FMat, IMat, FMat);
  
  def reset;
  
}

object AtariEnvironment {
  
  class Options extends Environment.Options {

    var random_seed = 0;
    var repeat_action_probability = 0f;
    var frameskip = irow(4,4);
    var rom_name = "/code/ALE/roms/Pong.bin";
    var height = 80;
    var width = 80;
    
  }
}