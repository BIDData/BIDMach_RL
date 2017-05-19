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

class AtariEnvironment(override val opts:AtariEnvironment.Options = new AtariEnvironment.Options) extends Environment(opts) {
  
  val ale:ALE = new ALE;
  ale.setInt("random_seed", opts.random_seed);
  ale.loadROM(opts.rom_name);
  ale.setFloat("repeat_action_probability", opts.repeat_action_probability);
  ale.frameskip = (opts.frameskip(0), opts.frameskip(1));
  
  override val VALID_ACTIONS = IMat.make(ale.getMinimalActionSet);
  
  override val score_range = opts.score_range;
  
  def step(action:Int):(FMat, Float, Boolean) = ale.step(action);
  
  def statedims = {irow(opts.width, opts.height)}
  
  def reset() = ale.reset();
  
}

object AtariEnvironment {
  
  def stepAll(envs:Array[Environment], actions:IMat, obs0:Array[FMat], rewards0:FMat, dones0:FMat):(Array[FMat], FMat, FMat) = {
    ALE.stepAll(envs.map(_.asInstanceOf[AtariEnvironment].ale), actions, obs0, rewards0, dones0);
  }
  
  class Options extends Environment.Options {

    var random_seed = 0;
    var repeat_action_probability = 0f;
    var frameskip = irow(4,4);
    var rom_name = "/code/ALE/roms/Pong.bin";
    var height = 80;
    var width = 80;
    var score_range = row(-1f,1f);
    
  }
}